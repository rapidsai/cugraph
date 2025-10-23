/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/graph_helper_impl.cuh"

namespace cugraph {
namespace c_api {

template rmm::device_uvector<int32_t> expand_sparse_offsets(
  raft::device_span<int32_t const> offsets,
  int32_t base_vertex_id,
  rmm::cuda_stream_view const& stream);

template rmm::device_uvector<int32_t> expand_sparse_offsets(
  raft::device_span<int64_t const> offsets,
  int32_t base_vertex_id,
  rmm::cuda_stream_view const& stream);

template rmm::device_uvector<int64_t> expand_sparse_offsets(
  raft::device_span<int64_t const> offsets,
  int64_t base_vertex_id,
  rmm::cuda_stream_view const& stream);

template rmm::device_uvector<int32_t> expand_sparse_offsets(raft::device_span<size_t const> offsets,
                                                            int32_t base_vertex_id,
                                                            rmm::cuda_stream_view const& stream);

template rmm::device_uvector<int64_t> expand_sparse_offsets(raft::device_span<size_t const> offsets,
                                                            int64_t base_vertex_id,
                                                            rmm::cuda_stream_view const& stream);

template void copy_or_transform(raft::device_span<int32_t> output,
                                cugraph_type_erased_device_array_view_t const* input,
                                rmm::cuda_stream_view const& stream_view);

template void copy_or_transform(raft::device_span<int64_t> output,
                                cugraph_type_erased_device_array_view_t const* input,
                                rmm::cuda_stream_view const& stream_view);

template void copy_or_transform(raft::device_span<float> output,
                                cugraph_type_erased_device_array_view_t const* input,
                                rmm::cuda_stream_view const& stream_view);

template void copy_or_transform(raft::device_span<double> output,
                                cugraph_type_erased_device_array_view_t const* input,
                                rmm::cuda_stream_view const& stream_view);

template edge_property_t<int32_t, float> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  float constant_value);

template edge_property_t<int64_t, float> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  float constant_value);

template edge_property_t<int32_t, float> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  float constant_value);

template edge_property_t<int64_t, float> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  float constant_value);

template edge_property_t<int32_t, double> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  double constant_value);

template edge_property_t<int64_t, double> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  double constant_value);

template edge_property_t<int32_t, double> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  double constant_value);

template edge_property_t<int64_t, double> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  double constant_value);

}  // namespace c_api
}  // namespace cugraph
