/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "c_api/array.hpp"

#include <cugraph/prims/fill_edge_property.cuh>
#include <cugraph/utilities/misc_utils.cuh>

#include <cuda/functional>

namespace cugraph {
namespace c_api {

template <typename vertex_t, typename edge_t>
rmm::device_uvector<vertex_t> expand_sparse_offsets(raft::device_span<edge_t const> offsets,
                                                    vertex_t base_vertex_id,
                                                    cuda::stream_ref const& stream)
{
  return cugraph::detail::expand_sparse_offsets(offsets, base_vertex_id, stream);
}

template <typename GraphViewType, typename T>
edge_property_t<typename GraphViewType::edge_type, T> create_constant_edge_property(
  raft::handle_t const& handle, GraphViewType const& graph_view, T constant_value)
{
  edge_property_t<typename GraphViewType::edge_type, T> edge_property(handle, graph_view);

  cugraph::fill_edge_property(handle, graph_view, edge_property.mutable_view(), constant_value);

  return edge_property;
}

template <typename new_type_t>
void copy_or_transform(raft::device_span<new_type_t> output,
                       cugraph_type_erased_device_array_view_t const* input,
                       cuda::stream_ref const& stream_view)
{
  if (((input->type_ == cugraph_data_type_id_t::INT8) && (std::is_same_v<new_type_t, int8_t>)) ||
      ((input->type_ == cugraph_data_type_id_t::INT16) && (std::is_same_v<new_type_t, int16_t>)) ||
      ((input->type_ == cugraph_data_type_id_t::INT32) && (std::is_same_v<new_type_t, int32_t>)) ||
      ((input->type_ == cugraph_data_type_id_t::INT64) && (std::is_same_v<new_type_t, int64_t>)) ||
      ((input->type_ == cugraph_data_type_id_t::FLOAT32) && (std::is_same_v<new_type_t, float>)) ||
      ((input->type_ == cugraph_data_type_id_t::FLOAT64) && (std::is_same_v<new_type_t, double>)) ||
      ((input->type_ == cugraph_data_type_id_t::SIZE_T) && (std::is_same_v<new_type_t, size_t>))) {
    // dtype match so just perform a copy
    raft::copy<new_type_t>(output.data(), input->as_type<new_type_t>(), input->size_, stream_view);
  }

  else {
    // There is a dtype mismatch
    if (input->type_ == cugraph_data_type_id_t::INT8) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input->as_type<int8_t>(),
                        input->as_type<int8_t>() + input->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    } else if (input->type_ == cugraph_data_type_id_t::INT16) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input->as_type<int16_t>(),
                        input->as_type<int16_t>() + input->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    } else if (input->type_ == cugraph_data_type_id_t::INT32) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input->as_type<int32_t>(),
                        input->as_type<int32_t>() + input->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    } else if (input->type_ == cugraph_data_type_id_t::INT64) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input->as_type<int64_t>(),
                        input->as_type<int64_t>() + input->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    } else if (input->type_ == cugraph_data_type_id_t::FLOAT32) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input->as_type<float>(),
                        input->as_type<float>() + input->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    } else if (input->type_ == cugraph_data_type_id_t::FLOAT64) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input->as_type<double>(),
                        input->as_type<double>() + input->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    } else if (input->type_ == cugraph_data_type_id_t::SIZE_T) {
      thrust::transform(rmm::exec_policy(stream_view),
                        input->as_type<size_t>(),
                        input->as_type<size_t>() + input->size_,
                        output.begin(),
                        cuda::proclaim_return_type<new_type_t>(
                          [] __device__(auto value) { return static_cast<new_type_t>(value); }));
    }
  }
}

}  // namespace c_api
}  // namespace cugraph
