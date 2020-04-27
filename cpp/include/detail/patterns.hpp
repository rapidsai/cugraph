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
          typename DstValueInputIterator, typename SrcValueOutputIterator>
void copy_dst_values_to_src(
    HandleType handle, GraphType graph,
    DstValueInputIterator dst_value_input_first, SrcValueOutputIterator src_value_output_first);

template <typename HandleType, typename GraphType, typename SrcValueInputIterator, typename T>
T reduce_src_v(
    HandelType handle, GraphType graph, SrcValueInputIterator src_value_input_first, T init);

template <typename HandleType, typename GraphType, typename DstValueInputIterator, typename T>
T reduce_dst_v(
    HandelType handle, GraphType graph, DstValueInputIterator dst_value_input_first, T init);

template <typename HandleType, typename GraphType,
          typename SrcValueInputIterator, typename TransformOp, typename T>
T transform_reduce_src_v(
    HandelType handle, GraphType graph,
    SrcValueInputIterator src_value_input_first, TransformOp transform_op, T init);

template <typename HandleType, typename GraphType,
          typename DstValueInputIterator, typename TransformOp, typename T>
T transform_reduce_dst_v(
    HandelType handle, GraphType graph,
    DstValueInputIterator dst_value_input_first, TransformOp transform_op, T init);

template <typename HandleType, typename GraphType,
          typename SrcValueInputIterator, typename DstValueInputIterator,
          typename ReduceOp, typename T>
T transform_reduce_src_dst_v(
    HandelType handle, GraphType graph,
    SrcValueInputIterator src_value_input_first, DstValueInputIterator dst_value_input_first,
    ReduceOp reduce_op, T init);

// 2-levels

template <typename HandleType, typename GraphType,
          typename SrcValueInputIterator, typename DstValueInputIterator,
          typename SrcValueOutputIterator,
          typename EdgeOp, typename T>
void transform_src_v_transform_reduce_e(
    HandelType handle, GraphType graph,
    SrcValueInputIterator src_value_input_first, DstValueInputIterator dst_value_input_first,
    SrcValueOutputIterator src_value_output_first, EdgeOp e_op, T init);

template <typename HandleType, typename GraphType,
          typename SrcValueInputIterator, typename DstValueInputIterator,
          typename DstValueOutputIterator,
          typename EdgeOp, typename T>
void transform_dst_v_transform_reduce_e(
    HandelType handle, GraphType graph,
    SrcValueInputIterator src_value_input_first, DstValueInputIterator dst_value_input_first,
    DstValueOutputIterator dst_value_output_first, EdgeOp e_op, T init);

template <typename HandleType, typename GraphType, typename SrcVertexIterator,
          typename SrcValueInputIterator, typename DstValueInputIterator,
          typename DstValueOutputIterator,
          typename SrcQueueOutputIterator,
          typename EdgeOp>
SrcQueueOutputIterator for_each_src_v_expand_and_transform_if_e(
    HandelType handle, GraphType graph,
    SrcVertexIterator src_vertex_first, SrcVertexIterator src_vertex_last,
    SrcValueInputIterator src_value_input_first, DstValueInputIterator dst_value_input_first,
    DstValueOutputIterator dst_value_output_first,
    SrcQueueOutputIterator src_queue_output_first,
    EdgeOp e_op);

template <typename HandleType, typename GraphType, typename SrcVertexIterator,
          typename SrcValueInputIterator, typename DstValueInputIterator,
          typename DstValueOutputIterator,
          typename SrcQueueOutputIterator, typename SrcValueOutputIterator,
          typename EdgeOp, typename ReduceOp, typename TransformOp>
SrcQueueOutputIterator for_each_src_v_expand_and_transform_if_e(
    HandelType handle, GraphType graph,
    SrcVertexIterator src_vertex_first, SrcVertexIterator src_vertex_last,
    SrcValueInputIterator src_value_input_first, DstValueInputIterator dst_value_input_first,
    DstValueOutputIterator dst_value_output_first,
    SrcQueueOutputIterator src_queue_output_first, SrcValueOutputIterator src_value_output_first,
    EdgeOp e_op, ReduceOp reduce_op, TransformOp transform_op);
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
