/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace c_api {

template <typename vertex_t, typename edge_t>
rmm::device_uvector<vertex_t> expand_sparse_offsets(raft::device_span<edge_t const> offsets,
                                                    vertex_t base_vertex_id,
                                                    rmm::cuda_stream_view const& stream);

template <typename GraphViewType, typename T>
edge_property_t<typename GraphViewType::edge_type, T> create_constant_edge_property(
  raft::handle_t const& handle, GraphViewType const& graph_view, T constant_value);

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Cast the values of a cugraph_type_erased_device_array_view_t to the new type
 *
 * @tparam      new_type_t     type of the value to operate on. Must be either int32_t or int64_t.
 *
 * @param[out]  output      device span to update with new data type
 * @param[in]   input       cugraph_type_erased_device_array_view_t with initial data type
 * @param[in]   stream_view  stream view
 *
 */
template <typename new_type_t>
void copy_or_transform(raft::device_span<new_type_t> output,
                       cugraph_type_erased_device_array_view_t const* input,
                       rmm::cuda_stream_view const& stream_view);

}  // namespace c_api
}  // namespace cugraph
