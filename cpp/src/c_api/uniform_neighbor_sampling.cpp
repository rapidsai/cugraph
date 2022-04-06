/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cugraph_c/algorithms.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/graph.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/visitors/generic_cascaded_dispatch.hpp>

#include <raft/handle.hpp>

namespace cugraph {
namespace c_api {

struct cugraph_sample_result_t {
  cugraph_type_erased_device_array_t* src_{nullptr};
  cugraph_type_erased_device_array_t* dst_{nullptr};
  cugraph_type_erased_device_array_t* label_{nullptr};
  cugraph_type_erased_device_array_t* index_{nullptr};
  cugraph_type_erased_host_array_t* count_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct uniform_neighbor_sampling_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_label_{nullptr};
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* fan_out_{nullptr};
  bool with_replacement_{false};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_sample_result_t* result_{nullptr};

  uniform_neighbor_sampling_functor(cugraph_resource_handle_t const* handle,
                                    cugraph_graph_t* graph,
                                    cugraph_type_erased_device_array_view_t const* start,
                                    cugraph_type_erased_device_array_view_t const* start_label,
                                    cugraph_type_erased_host_array_view_t const* fan_out,
                                    bool with_replacement,
                                    bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      start_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(start)),
      start_label_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        start_label)),
      fan_out_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)),
      with_replacement_(with_replacement),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else if constexpr (!multi_gpu) {
      unsupported();
    } else {
      // uniform_nbr_sample expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, weight_t, false, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> start(start_->size_, handle_.get_stream());
      raft::copy(start.data(), start_->as_type<vertex_t>(), start.size(), handle_.get_stream());

      //
      // Need to renumber sources
      //
      cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        start.data(),
        start.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        false);

      // C++ API wants an std::vector
      std::vector<int> fan_out(fan_out_->size_);
      std::copy_n(fan_out_->as_type<int>(), fan_out_->size_, fan_out.data());

      auto&& [tmp_tuple, counts] = cugraph::uniform_nbr_sample(handle_,
                                                               graph_view,
                                                               start.data(),
                                                               start_label_->as_type<int32_t>(),
                                                               start.size(),
                                                               fan_out,
                                                               with_replacement_);

      auto&& [srcs, dsts, labels, indices] = tmp_tuple;

      std::vector<vertex_t> vertex_partition_lasts = graph_view.vertex_partition_range_lasts();

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                            srcs.data(),
                                                            srcs.size(),
                                                            number_map->data(),
                                                            vertex_partition_lasts,
                                                            do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                            dsts.data(),
                                                            dsts.size(),
                                                            number_map->data(),
                                                            vertex_partition_lasts,
                                                            do_expensive_check_);

      result_ = new cugraph::c_api::cugraph_sample_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(srcs, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(dsts, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(labels, start_label_->type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(indices, graph_->edge_type_),
        new cugraph::c_api::cugraph_type_erased_host_array_t(counts, graph_->vertex_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_uniform_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start,
  const cugraph_type_erased_device_array_view_t* start_labels,
  const cugraph_type_erased_host_array_view_t* fan_out,
  bool_t with_replacement,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  uniform_neighbor_sampling_functor functor{
    handle, graph, start, start_labels, fan_out, with_replacement, do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_sources(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->src_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_destinations(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(internal_pointer->dst_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_start_labels(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->label_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_index(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->index_->view());
}

extern "C" cugraph_type_erased_host_array_view_t* cugraph_sample_result_get_counts(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_host_array_view_t*>(internal_pointer->count_->view());
}

extern "C" void cugraph_sample_result_free(cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t*>(result);
  delete internal_pointer->src_;
  delete internal_pointer->dst_;
  delete internal_pointer->label_;
  delete internal_pointer->index_;
  delete internal_pointer->count_;
  delete internal_pointer;
}
