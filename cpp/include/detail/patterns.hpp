/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <graph.hpp>


namespace cugraph {
namespace experimental {
namespace detail {

// 1-level

template <typename HandleType, typename GraphType,
          typename DstInputIterator, typename SrcOutputIterator>
void copy_dst_values_to_src(
    HandleType handle, GraphType graph,
    DstInputIterator dst_input_first, SrcOutputIterator src_output_first);

template <typename HandleType, typename GraphType, typename SrcInputIterator, typename T>
T reduce_src_v(HandelType handle, GraphType graph, SrcInputIterator src_input_first, T init);

template <typename HandleType, typename GraphType, typename DstInputIterator, typename T>
T reduce_dst_v(HandelType handle, GraphType graph, DstInputIterator dst_input_first, T init);

template <typename HandleType, typename GraphType,
          typename SrcInputIterator, typename UnaryOp, typename T>
T transform_reduce_src_v(
    HandelType handle, GraphType graph,
    SrcInputIterator src_input_first, UnaryOp v_op, T init);

template <typename HandleType, typename GraphType,
          typename DstInputIterator, typename UnaryOp, typename T>
T transform_reduce_dst_v(
    HandelType handle, GraphType graph,
    DstInputIterator dst_input_first, UnaryOp v_op, T init);

template <typename HandleType, typename GraphType,
          typename SrcInputIterator, typename DstInputIterator, typename BinaryOp, typename T>
T transform_reduce_src_dst_v(
    HandelType handle, GraphType graph,
    SrcInputIterator src_input_first, DstInputIterator dst_input_first, BinaryOp v_op, T init);

// 2-levels

template <typename HandleType, typename GraphType,
          typename SrcInputIterator, typename DstInputIterator, typename SrcOutputIterator,
          typename EdgeOp, typename T>
void transform_src_v_transform_reduce_e(
    HandelType handle, GraphType graph,
    SrcInputIterator src_input_first, DstInputIterator dst_input_first,
    SrcOutputIterator src_output_first, EdgeOp e_op, T init);

template <typename HandleType, typename GraphType,
          typename SrcInputIterator, typename DstInputIterator, typename DstOutputIterator,
          typename EdgeOp, typename T>
void transform_dst_v_transform_reduce_e(
    HandelType handle, GraphType graph,
    SrcInputIterator src_input_first, DstInputIterator dst_input_first,
    DstOutputIterator dst_output_first, EdgeOp e_op, T init);

template <typename HandleType, typename GraphType, typename SrcVertexIterator,
          typename SrcInputIterator, typename DstInputIterator, typename SrcOutputIterator,
          typename QueueOutputIterator,
          typename EdgeOp>
QueueOutputIterator for_each_src_v_expand_and_transform_if_e(
    HandelType handle, GraphType graph,
    SrcVertexIterator src_vertex_first, SrcVertexIterator src_vertex_last,
    SrcInputIterator src_input_first, DstInputIterator dst_input_first,
    SrcOutputIterator src_output_first, QueueOutputIterator queue_output_first,
    EdgeOp e_op);

template <typename HandleType, typename GraphType, typename SrcVertexIterator,
          typename SrcInputIterator, typename DstInputIterator, typename SrcOutputIterator,
          typename QueueOutputIterator,
          typename EdgeOp, typename ReduceOp>
QueueOutputIterator for_each_src_v_expand_and_transform_if_e(
    HandelType handle, GraphType graph,
    SrcVertexIterator src_vertex_first, SrcVertexIterator src_vertex_last,
    SrcInputIterator src_input_first, DstInputIterator dst_input_first,
    SrcOutputIterator src_output_first, QueueOutputIterator queue_output_first,
    EdgeOp e_op, ReduceOp reduce_op);
/*
iterating over lower triangular (or upper triangular) : triangle counting
LRB might be necessary if the cost of processing an edge (i, j) is a function of degree(i) and degree(j) : triangle counting
push-pull switching support (e.g. DOBFS), in this case, we need both CSR & CSC (trade-off execution time vs memory requirement, unless graph is symmetric)
should I take multi-GPU support as a template argument?
Add bool expensive_check = false ?
cugraph::count_if as a multi-GPU wrapper of thrust::count_if? (for expensive check)
if graph is symmetric, there will be additional optimization opportunities (e.g. in-degree == out-degree)
*/

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
