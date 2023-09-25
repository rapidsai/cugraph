/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cugraph_c/graph_generators.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/capi_helper.hpp>
#include <c_api/graph.hpp>
#include <c_api/induced_subgraph_result.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

namespace {

template <typename vertex_t, typename weight_t>
cugraph_error_code_t dummy_function(
  raft::handle_t const& handle,
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* src,
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst,
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights,
  cugraph::c_api::cugraph_induced_subgraph_result_t** result,
  cugraph::c_api::cugraph_error_t** error)
{
  return CUGRAPH_SUCCESS;
}

template <typename vertex_t, typename weight_t>
cugraph_error_code_t cugraph_replicate_edgelist(
  raft::handle_t const& handle,
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* src,
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* dst,
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* weights,
  cugraph::c_api::cugraph_induced_subgraph_result_t** result,
  cugraph::c_api::cugraph_error_t** error)
{
  rmm::device_uvector<vertex_t> edgelist_srcs(src->size_, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_dsts(dst->size_, handle.get_stream());

  raft::copy<vertex_t>(
    edgelist_srcs.data(), src->as_type<vertex_t>(), src->size_, handle.get_stream());
  raft::copy<vertex_t>(
    edgelist_dsts.data(), dst->as_type<vertex_t>(), dst->size_, handle.get_stream());

  std::optional<rmm::device_uvector<weight_t>> edgelist_weights =
    weights ? std::make_optional(rmm::device_uvector<weight_t>(weights->size_, handle.get_stream()))
            : std::nullopt;

  if (edgelist_weights) {
    raft::copy<weight_t>(
      edgelist_weights->data(), weights->as_type<weight_t>(), weights->size_, handle.get_stream());
  }

  try {
    auto gathered_edgelist_srcs = cugraph::c_api::detail::device_allgatherv(
      handle, raft::device_span<vertex_t const>(edgelist_srcs.data(), edgelist_srcs.size()));

    auto gathered_edgelist_dsts = cugraph::c_api::detail::device_allgatherv(
      handle, raft::device_span<vertex_t const>(edgelist_dsts.data(), edgelist_dsts.size()));

    rmm::device_uvector<size_t> edge_offsets(2, handle.get_stream());
    std::vector<size_t> h_edge_offsets{{0, gathered_edgelist_srcs.size()}};
    raft::update_device(
      edge_offsets.data(), h_edge_offsets.data(), h_edge_offsets.size(), handle.get_stream());

    // FIXME: Hnadle this case better
    if (edgelist_weights) {
      auto gathered_edgelist_weights = cugraph::c_api::detail::device_allgatherv(
        handle,
        raft::device_span<weight_t const>(edgelist_weights->data(), edgelist_weights->size()));

      *result = new cugraph::c_api::cugraph_induced_subgraph_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(gathered_edgelist_srcs, src->type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(gathered_edgelist_dsts, dst->type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(gathered_edgelist_weights,
                                                               weights->type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(edge_offsets,
                                                               cugraph_data_type_id_t::SIZE_T)};
    } else {
      *result = new cugraph::c_api::cugraph_induced_subgraph_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(gathered_edgelist_srcs, src->type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(gathered_edgelist_dsts, dst->type_),
        NULL,
        new cugraph::c_api::cugraph_type_erased_device_array_t(edge_offsets,
                                                               cugraph_data_type_id_t::SIZE_T)};
    }

  } catch (std::exception const& ex) {
    *error = new cugraph::c_api::cugraph_error_t{ex.what()};
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}

}  // namespace

// template <typename vertex_t, typename weight_t>
//  why taking out the 'extern "C"' worked.
extern "C" cugraph_error_code_t cugraph_replicate_edgelist(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* src,
  const cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* weights,
  cugraph_induced_subgraph_result_t** result,
  cugraph_error_t** error)
{
  auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
  auto p_src =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(src);
  auto p_dst =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(dst);
  auto p_weights =
    reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(weights);

  /*
  reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t**>(result),
  reinterpret_cast<cugraph::c_api::cugraph_error_t**>(error)
  */

  // FIXME: Support int64_t and float as well
  return cugraph_replicate_edgelist<int32_t, float>(
    *p_handle->handle_,
    p_src,
    p_dst,
    p_weights,
    reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t**>(result),
    reinterpret_cast<cugraph::c_api::cugraph_error_t**>(error));

  /*
  return dummy_function<int32_t, float>(
   *p_handle->handle_,
   p_src,
   p_dst,
   p_weights,
   reinterpret_cast<cugraph::c_api::cugraph_induced_subgraph_result_t**>(result),
   reinterpret_cast<cugraph::c_api::cugraph_error_t**>(error));
   */
}
