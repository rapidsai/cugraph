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
 #include <detail/reduce_op.cuh>
 
 #include <rmm/thrust_rmm_allocator.h>
 
 #include <cub/cub.cuh>
 #include <thrust/distance.h>
 #include <thrust/functional.h>
 #include <thrust/transform_reduce.h>
 #include <thrust/tuple.h>
 #include <thrust/type_traits/integer_sequence.h>
 
 #include <type_traits>
 #include <utility>
 

 namespace cugraph {
 namespace experimental {
 namespace detail {
 
 template <typename HandleType, typename GraphType,
           typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
           typename VertexValueInputIterator,
           typename VertexValueOutputIterator,
           typename EdgeOp, typename T>
 void transform_v_transform_reduce_e(
     HandleType handle, GraphType graph,
     AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
     AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
     VertexValueInputIterator vertex_value_input_first,
     VertexValueOutputIterator vertex_value_output_first,
     EdgeOp e_op, T init);
 
 }  // namespace detail
 }  // namespace experimental
 }  // namespace cugraph