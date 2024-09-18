/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "c_api/abstract_functor.hpp"
#include "c_api/coo.hpp"
#include "c_api/graph.hpp"
#include "c_api/random.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/sampling_algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/sampling_functions.hpp>

#include <raft/core/handle.hpp>

namespace {

struct negative_sampling_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertices_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* src_biases_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst_biases_{nullptr};
  size_t num_samples_;
  bool remove_duplicates_{false};
  bool remove_existing_edges_{false};
  bool exact_number_of_samples_{false};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_coo_t* result_{nullptr};

  negative_sampling_functor(const cugraph_resource_handle_t* handle,
                            cugraph_rng_state_t* rng_state,
                            cugraph_graph_t* graph,
                            const cugraph_type_erased_device_array_view_t* vertices,
                            const cugraph_type_erased_device_array_view_t* src_biases,
                            const cugraph_type_erased_device_array_view_t* dst_biases,
                            size_t num_samples,
                            bool_t remove_duplicates,
                            bool_t remove_existing_edges,
                            bool_t exact_number_of_samples,
                            bool_t do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(vertices)),
      src_biases_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        src_biases)),
      dst_biases_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
        dst_biases)),
      num_samples_(num_samples),
      remove_duplicates_(remove_duplicates),
      remove_existing_edges_(remove_existing_edges),
      exact_number_of_samples_(exact_number_of_samples),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // negative_sampling expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> vertices(0, handle_.get_stream());
      rmm::device_uvector<weight_t> src_biases(0, handle_.get_stream());
      rmm::device_uvector<weight_t> dst_biases(0, handle_.get_stream());

      if (src_biases_ != nullptr) {
        vertices.resize(vertices_->size_, handle_.get_stream());
        src_biases.resize(src_biases_->size_, handle_.get_stream());

        raft::copy(
          vertices.data(), vertices_->as_type<vertex_t>(), vertices.size(), handle_.get_stream());
        raft::copy(src_biases.data(),
                   src_biases_->as_type<weight_t>(),
                   src_biases.size(),
                   handle_.get_stream());

        src_biases = cugraph::detail::
          collect_local_vertex_values_from_ext_vertex_value_pairs<vertex_t, weight_t, multi_gpu>(
            handle_,
            std::move(vertices),
            std::move(src_biases),
            *number_map,
            graph_view.local_vertex_partition_range_first(),
            graph_view.local_vertex_partition_range_last(),
            weight_t{0},
            do_expensive_check_);
      }

      if (dst_biases_ != nullptr) {
        vertices.resize(vertices_->size_, handle_.get_stream());
        dst_biases.resize(dst_biases_->size_, handle_.get_stream());

        raft::copy(
          vertices.data(), vertices_->as_type<vertex_t>(), vertices.size(), handle_.get_stream());
        raft::copy(dst_biases.data(),
                   dst_biases_->as_type<weight_t>(),
                   dst_biases.size(),
                   handle_.get_stream());

        dst_biases = cugraph::detail::
          collect_local_vertex_values_from_ext_vertex_value_pairs<vertex_t, weight_t, multi_gpu>(
            handle_,
            std::move(vertices),
            std::move(dst_biases),
            *number_map,
            graph_view.local_vertex_partition_range_first(),
            graph_view.local_vertex_partition_range_last(),
            weight_t{0},
            do_expensive_check_);
      }

      auto&& [src, dst] = cugraph::negative_sampling(
        handle_,
        rng_state_->rng_state_,
        graph_view,
        (src_biases_ != nullptr) ? std::make_optional(raft::device_span<weight_t const>{
                                     src_biases.data(), src_biases.size()})
                                 : std::nullopt,
        (dst_biases_ != nullptr) ? std::make_optional(raft::device_span<weight_t const>{
                                     dst_biases.data(), dst_biases.size()})
                                 : std::nullopt,
        num_samples_,
        remove_duplicates_,
        remove_existing_edges_,
        exact_number_of_samples_,
        do_expensive_check_);

      std::vector<vertex_t> vertex_partition_lasts = graph_view.vertex_partition_range_lasts();

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                            src.data(),
                                                            src.size(),
                                                            number_map->data(),
                                                            vertex_partition_lasts,
                                                            do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                            dst.data(),
                                                            dst.size(),
                                                            number_map->data(),
                                                            vertex_partition_lasts,
                                                            do_expensive_check_);

      result_ = new cugraph::c_api::cugraph_coo_t{
        std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(src,
                                                                             graph_->vertex_type_),
        std::make_unique<cugraph::c_api::cugraph_type_erased_device_array_t>(dst,
                                                                             graph_->vertex_type_),
        nullptr,
        nullptr,
        nullptr};
    }
  }
};

}  // namespace

cugraph_error_code_t cugraph_negative_sampling(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* vertices,
  const cugraph_type_erased_device_array_view_t* src_biases,
  const cugraph_type_erased_device_array_view_t* dst_biases,
  size_t num_samples,
  bool_t remove_duplicates,
  bool_t remove_existing_edges,
  bool_t exact_number_of_samples,
  bool_t do_expensive_check,
  cugraph_coo_t** result,
  cugraph_error_t** error)
{
  negative_sampling_functor functor{handle,
                                    rng_state,
                                    graph,
                                    vertices,
                                    src_biases,
                                    dst_biases,
                                    num_samples,
                                    remove_duplicates,
                                    remove_existing_edges,
                                    exact_number_of_samples,
                                    do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
