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

// 1-level

template <typename GraphType, typename DstInputIterator, typename SrcOutputIterator>
copy_dst_values_to_src(
    GraphType graph, DstInputIterator dst_input_first, SrcOutputIterator src_output_first,

template <typename GraphType, typename SrcInputIterator, typename T>
reduce_src_v(GraphType graph, SrcInputIterator src_input_first, T init);

template <typename GraphType, typename DstInputIterator, typename T>
reduce_dst_v(GraphType graph, DstInputIterator dst_input_first, T init);

template <typename GraphType,
          typename SrcInputIterator,
          typename UnaryOp, typename T>
transform_reduce_src_v(
    GraphType graph,
    SrcInputIterator src_input_first, UnaryOp v_op, T init);

template <typename GraphType,
          typename DstInputIterator,
          typename UnaryOp, typename T>
transform_reduce_dst_v(
    GraphType graph,
    DstInputIterator dst_input_first, UnaryOp v_op, T init);

template <typename GraphType,
          typename SrcInputIterator, typename DstInputIterator,
          typename BinaryOp, typename T>
transform_reduce_src_dst_v(
    GraphType graph,
    SrcInputIterator src_input_first, DstInputIterator dst_input_first, BinaryOp v_op, T init);

// 2-levels

template <typename GraphType,
          typename SrcInputIterator, typename DstInputIterator, typename SrcOutputIterator,
          typename EdgeOp, typename T>
transform_src_v_transform_reduce_e(
    GraphType graph,
    SrcInputIterator src_input_first, DstInputIterator dst_input_first,
    SrcOutputIterator src_output_first, EdgeOp e_op, T init);

template <typename GraphType,
          typename SrcInputIterator, typename DstInputIterator, typename DstOutputIterator,
          typename EdgeOp, typename T>
transform_dst_v_transform_reduce_e(
    GraphType graph,
    SrcInputIterator src_input_first, DstInputIterator dst_input_first,
    DstOutputIterator dst_output_first, EdgeOp e_op, T init);

/*
lower triangular upper triangular (to iterate edges with i < j or i > j) target algorithms: triangle counting...

triangle counting... actual cost of processing a single edge is proportional to max(degree(i), degree(j)), not just degree(i)... how should I handle this...

Should we allow push-pull switching (e.g. direction optimized BFS), then a graph structure holding both CSR and CSC..

LRB may be beneficial for triangle counting kinda applications... 
*/

}  // namespace experimental
}  // namespace cugraph
