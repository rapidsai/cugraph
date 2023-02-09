/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <c_api/random.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/visitors/generic_cascaded_dispatch.hpp>

#include <raft/core/handle.hpp>

namespace cugraph {
namespace c_api {

struct cugraph_sample_result_t {
  cugraph_type_erased_device_array_t* src_{nullptr};
  cugraph_type_erased_device_array_t* dst_{nullptr};
  cugraph_type_erased_device_array_t* edge_id_{nullptr};
  cugraph_type_erased_device_array_t* edge_type_{nullptr};
  cugraph_type_erased_device_array_t* wgt_{nullptr};
  cugraph_type_erased_device_array_t* hop_{nullptr};
  cugraph_type_erased_device_array_t* label_{nullptr};
  // FIXME: Will be deleted once experimental replaces current
  cugraph_type_erased_host_array_t* count_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct uniform_neighbor_sampling_functor_deprecate : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_{nullptr};
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* fan_out_{nullptr};
  bool with_replacement_{false};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_sample_result_t* result_{nullptr};

  uniform_neighbor_sampling_functor_deprecate(cugraph_resource_handle_t const* handle,
                                              cugraph_graph_t* graph,
                                              cugraph_type_erased_device_array_view_t const* start,
                                              cugraph_type_erased_host_array_view_t const* fan_out,
                                              bool with_replacement,
                                              bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      start_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(start)),
      fan_out_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)),
      with_replacement_(with_replacement),
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
      // uniform_nbr_sample expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

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

      auto&& [srcs, dsts, weights, counts] = cugraph::uniform_nbr_sample(
        handle_,
        graph_view,
        (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
        raft::device_span<vertex_t>(start.data(), start.size()),
        raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
        with_replacement_);

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
        new cugraph::c_api::cugraph_type_erased_device_array_t(
          weights, graph_->weight_type_),  // needs to be edge id...
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr};
    }
  }
};

struct uniform_neighbor_sampling_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* label_{nullptr};
  cugraph::c_api::cugraph_type_erased_host_array_view_t const* fan_out_{nullptr};
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  bool with_replacement_{false};
  bool do_expensive_check_{false};
  cugraph::c_api::cugraph_sample_result_t* result_{nullptr};

  uniform_neighbor_sampling_functor(cugraph_resource_handle_t const* handle,
                                    cugraph_graph_t* graph,
                                    cugraph_type_erased_device_array_view_t const* start,
                                    cugraph_type_erased_device_array_view_t const* label,
                                    cugraph_type_erased_host_array_view_t const* fan_out,
                                    cugraph_rng_state_t* rng_state,
                                    bool with_replacement,
                                    bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      start_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(start)),
      label_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(label)),
      fan_out_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      with_replacement_(with_replacement),
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
      // uniform_nbr_sample expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, true, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto edge_properties = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 thrust::tuple<edge_t, edge_type_t>>*>(graph_->edge_properties_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> start(start_->size_, handle_.get_stream());
      raft::copy(start.data(), start_->as_type<vertex_t>(), start.size(), handle_.get_stream());

      std::optional<rmm::device_uvector<int32_t>> label{std::nullopt};

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

      if (label_ != nullptr) {
        // FIXME: Making a copy because I couldn't get the raft::device_span of a const array
        // to construct properly.
        label = rmm::device_uvector<int32_t>(label_->size_, handle_.get_stream());
        raft::copy(label->data(), label_->as_type<int32_t>(), label->size(), handle_.get_stream());
      }

      auto&& [src, dst, wgt, edge_id, edge_type, hop, edge_label] =
        cugraph::uniform_neighbor_sample<vertex_t, edge_t, weight_t, edge_type_t, false, multi_gpu>(
          handle_,
          graph_view,
          (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
          (edge_properties != nullptr) ? std::make_optional(edge_properties->view()) : std::nullopt,
          std::move(start),
          std::move(label),
          raft::host_span<const int>(fan_out_->as_type<const int>(), fan_out_->size_),
          rng_state_->rng_state_,
          with_replacement_);

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

      result_ = new cugraph::c_api::cugraph_sample_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(src, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(dst, graph_->vertex_type_),
        (edge_id)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*edge_id, graph_->edge_type_)
          : nullptr,
        (edge_type) ? new cugraph::c_api::cugraph_type_erased_device_array_t(
                        *edge_type, graph_->edge_type_id_type_)
                    : nullptr,
        (wgt) ? new cugraph::c_api::cugraph_type_erased_device_array_t(*wgt, graph_->weight_type_)
              : nullptr,
        new cugraph::c_api::cugraph_type_erased_device_array_t(hop, INT32),
        (edge_label)
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(edge_label.value(), INT32)
          : nullptr};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_uniform_neighbor_sample(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start,
  const cugraph_type_erased_host_array_view_t* fan_out,
  bool_t with_replacement,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(start)
        ->type_,
    CUGRAPH_INVALID_INPUT,
    "vertex type of graph and start must match",
    *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)
        ->type_ == INT32,
    CUGRAPH_INVALID_INPUT,
    "fan_out should be of type int",
    *error);

  uniform_neighbor_sampling_functor_deprecate functor{
    handle, graph, start, fan_out, with_replacement, do_expensive_check};
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
  return internal_pointer->label_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->label_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_id(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->edge_id_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_id_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_type(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->edge_type_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->edge_type_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_edge_weight(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->wgt_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->wgt_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_hop(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return internal_pointer->hop_ != nullptr
           ? reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
               internal_pointer->hop_->view())
           : NULL;
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_sample_result_get_index(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->edge_id_->view());
}

extern "C" cugraph_type_erased_host_array_view_t* cugraph_sample_result_get_counts(
  const cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_host_array_view_t*>(internal_pointer->count_->view());
}

extern "C" cugraph_error_code_t cugraph_test_uniform_neighborhood_sample_result_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* srcs,
  const cugraph_type_erased_device_array_view_t* dsts,
  const cugraph_type_erased_device_array_view_t* edge_id,
  const cugraph_type_erased_device_array_view_t* edge_type,
  const cugraph_type_erased_device_array_view_t* weight,
  const cugraph_type_erased_device_array_view_t* hop,
  const cugraph_type_erased_device_array_view_t* label,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;
  size_t n_bytes{0};
  cugraph_error_code_t error_code{CUGRAPH_SUCCESS};

  if (!handle) {
    *error = reinterpret_cast<cugraph_error_t*>(
      new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
    return CUGRAPH_INVALID_HANDLE;
  }

  // Create unique_ptrs and release them during cugraph_sample_result_t
  // construction. This allows the arrays to be cleaned up if this function
  // returns early on error.
  using device_array_unique_ptr_t =
    std::unique_ptr<cugraph_type_erased_device_array_t,
                    decltype(&cugraph_type_erased_device_array_free)>;

  // copy srcs to new device array
  cugraph_type_erased_device_array_t* new_device_srcs_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, srcs, &new_device_srcs_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_srcs(new_device_srcs_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy dsts to new device array
  cugraph_type_erased_device_array_t* new_device_dsts_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, dsts, &new_device_dsts_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_dsts(new_device_dsts_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy weights to new device array
  cugraph_type_erased_device_array_t* new_device_weight_ptr{nullptr};
  error_code = cugraph_type_erased_device_array_create_from_view(
    handle, weight, &new_device_weight_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_weight(new_device_weight_ptr,
                                              &cugraph_type_erased_device_array_free);

  // copy edge ids to new device array
  cugraph_type_erased_device_array_t* new_device_edge_id_ptr{nullptr};
  error_code = cugraph_type_erased_device_array_create_from_view(
    handle, edge_id, &new_device_edge_id_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_edge_id(new_device_edge_id_ptr,
                                               &cugraph_type_erased_device_array_free);

  // copy edge types to new device array
  cugraph_type_erased_device_array_t* new_device_edge_type_ptr{nullptr};
  error_code = cugraph_type_erased_device_array_create_from_view(
    handle, edge_type, &new_device_edge_type_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_edge_type(new_device_edge_type_ptr,
                                                 &cugraph_type_erased_device_array_free);
  // copy hop ids to new device array
  cugraph_type_erased_device_array_t* new_device_hop_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, hop, &new_device_hop_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_hop(new_device_hop_ptr,
                                           &cugraph_type_erased_device_array_free);

  // copy labels to new device array
  cugraph_type_erased_device_array_t* new_device_label_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, label, &new_device_label_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_label(new_device_label_ptr,
                                             &cugraph_type_erased_device_array_free);

  // create new cugraph_sample_result_t
  *result = reinterpret_cast<cugraph_sample_result_t*>(new cugraph::c_api::cugraph_sample_result_t{
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_srcs.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_dsts.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_id.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_type.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_weight.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(new_device_hop.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_label.release())});

  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_test_sample_result_create(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* srcs,
  const cugraph_type_erased_device_array_view_t* dsts,
  const cugraph_type_erased_device_array_view_t* edge_id,
  const cugraph_type_erased_device_array_view_t* edge_type,
  const cugraph_type_erased_device_array_view_t* wgt,
  const cugraph_type_erased_device_array_view_t* hop,
  const cugraph_type_erased_device_array_view_t* label,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;
  size_t n_bytes{0};
  cugraph_error_code_t error_code{CUGRAPH_SUCCESS};

  if (!handle) {
    *error = reinterpret_cast<cugraph_error_t*>(
      new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
    return CUGRAPH_INVALID_HANDLE;
  }

  // Create unique_ptrs and release them during cugraph_sample_result_t
  // construction. This allows the arrays to be cleaned up if this function
  // returns early on error.
  using device_array_unique_ptr_t =
    std::unique_ptr<cugraph_type_erased_device_array_t,
                    decltype(&cugraph_type_erased_device_array_free)>;

  // copy srcs to new device array
  cugraph_type_erased_device_array_t* new_device_srcs_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, srcs, &new_device_srcs_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_srcs(new_device_srcs_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy dsts to new device array
  cugraph_type_erased_device_array_t* new_device_dsts_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, dsts, &new_device_dsts_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_dsts(new_device_dsts_ptr,
                                            &cugraph_type_erased_device_array_free);

  // copy edge_id to new device array
  cugraph_type_erased_device_array_t* new_device_edge_id_ptr{nullptr};

  if (edge_id != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, edge_id, &new_device_edge_id_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_edge_id(new_device_edge_id_ptr,
                                               &cugraph_type_erased_device_array_free);

  // copy edge_type to new device array
  cugraph_type_erased_device_array_t* new_device_edge_type_ptr{nullptr};

  if (edge_type != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, edge_type, &new_device_edge_type_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_edge_type(new_device_edge_type_ptr,
                                                 &cugraph_type_erased_device_array_free);

  // copy wgt to new device array
  cugraph_type_erased_device_array_t* new_device_wgt_ptr{nullptr};
  if (wgt != NULL) {
    error_code =
      cugraph_type_erased_device_array_create_from_view(handle, wgt, &new_device_wgt_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_wgt(new_device_wgt_ptr,
                                           &cugraph_type_erased_device_array_free);

  // copy hop to new device array
  cugraph_type_erased_device_array_t* new_device_hop_ptr{nullptr};
  error_code =
    cugraph_type_erased_device_array_create_from_view(handle, hop, &new_device_hop_ptr, error);
  if (error_code != CUGRAPH_SUCCESS) return error_code;

  device_array_unique_ptr_t new_device_hop(new_device_hop_ptr,
                                           &cugraph_type_erased_device_array_free);

  // copy label to new device array
  cugraph_type_erased_device_array_t* new_device_label_ptr{nullptr};

  if (label != NULL) {
    error_code = cugraph_type_erased_device_array_create_from_view(
      handle, label, &new_device_label_ptr, error);
    if (error_code != CUGRAPH_SUCCESS) return error_code;
  }

  device_array_unique_ptr_t new_device_label(new_device_label_ptr,
                                             &cugraph_type_erased_device_array_free);

  // create new cugraph_sample_result_t
  *result = reinterpret_cast<cugraph_sample_result_t*>(new cugraph::c_api::cugraph_sample_result_t{
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_srcs.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_dsts.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_id.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_edge_type.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(new_device_wgt.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(
      new_device_label.release()),
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(new_device_hop.release()),
    nullptr});

  return CUGRAPH_SUCCESS;
}

extern "C" void cugraph_sample_result_free(cugraph_sample_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_sample_result_t*>(result);
  delete internal_pointer->src_;
  delete internal_pointer->dst_;
  delete internal_pointer->edge_id_;
  delete internal_pointer->edge_type_;
  delete internal_pointer->wgt_;
  delete internal_pointer->hop_;
  delete internal_pointer->label_;
  delete internal_pointer->count_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_uniform_neighbor_sample_with_edge_properties(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start,
  const cugraph_type_erased_device_array_view_t* label,
  const cugraph_type_erased_host_array_view_t* fan_out,
  cugraph_rng_state_t* rng_state,
  bool_t with_replacement,
  bool_t do_expensive_check,
  cugraph_sample_result_t** result,
  cugraph_error_t** error)
{
  // FIXME:  We need a mechanism to specify a seed.  We should be consistent across all of the
  //   sampling/random walk algorithms (or really any algorithm that wants a seed)

  CAPI_EXPECTS(
    (label == nullptr) ||
      (reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(label)
         ->type_ == INT32),
    CUGRAPH_INVALID_INPUT,
    "label should be of type int",
    *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(start)
        ->type_,
    CUGRAPH_INVALID_INPUT,
    "vertex type of graph and start must match",
    *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_host_array_view_t const*>(fan_out)
        ->type_ == INT32,
    CUGRAPH_INVALID_INPUT,
    "fan_out should be of type int",
    *error);

  uniform_neighbor_sampling_functor functor{
    handle, graph, start, label, fan_out, rng_state, with_replacement, do_expensive_check};
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
