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
#pragma once

#include <cugraph/edge_partition_device_view.cuh>

namespace cugraph {

namespace detail {

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgeOp>
struct call_e_op_t {
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> const& edge_partition{};
  EdgePartitionSrcValueInputWrapper const& edge_partition_src_value_input{};
  EdgePartitionDstValueInputWrapper const& edge_partition_dst_value_input{};
  EdgePartitionEdgeValueInputWrapper const& edge_partition_e_value_input{};
  EdgeOp const& e_op{};
  typename GraphViewType::vertex_type major{};
  typename GraphViewType::vertex_type major_offset{};
  typename GraphViewType::vertex_type const* indices{nullptr};
  typename GraphViewType::edge_type edge_offset{};

  __device__ auto operator()(typename GraphViewType::edge_type i) const
  {
    auto minor        = indices[i];
    auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
    auto src          = GraphViewType::is_storage_transposed ? minor : major;
    auto dst          = GraphViewType::is_storage_transposed ? major : minor;
    auto src_offset   = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
    auto dst_offset   = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
    return e_op(src,
                dst,
                edge_partition_src_value_input.get(src_offset),
                edge_partition_dst_value_input.get(dst_offset),
                edge_partition_e_value_input.get(edge_offset + i));
  }
};

}  // namespace detail

}  // namespace cugraph
