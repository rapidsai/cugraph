/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/algorithm>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

template <typename InputIterator, typename VertexOp, typename PredOp, typename PackedBoolIterator>
struct transform_if_packed_bool_t {
  InputIterator input_first{};
  VertexOp v_op{};
  PredOp pred_op{};
  PackedBoolIterator output_first{};
  size_t num_bools{};

  __device__ void operator()(size_t i) const
  {
    auto first = i * packed_bools_per_word();
    auto last  = cuda::std::min((i + 1) * packed_bools_per_word(), num_bools);
    auto word  = *(output_first + i);
    for (auto j = first; j < last; ++j) {
      auto val = *(input_first + j);
      if (pred_op(val)) {
        auto mask = packed_bool_mask(j);
        word      = v_op(val) ? (word | mask) : (word & ~mask);
      }
    }
    *(output_first + i) = word;
  }
};

}  // namespace detail

/**
 * @brief Transform input vertex property values for which the predicate holds and write the
 * results to an output vertex property.
 *
 * This function is inspired by thrust::transform_if(). Output values for which @p pred_op
 * returns false are left unchanged. It updates in place when @p vertex_value_input and @p
 * vertex_value_output are views of the same `vertex_property_t`.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputWrapper Type of the wrapper for input vertex property values.
 * @tparam VertexOp Type of the unary operator.
 * @tparam PredOp Type of the unary predicate operator.
 * @tparam VertexValueOutputWrapper Type of the wrapper for output vertex property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input Wrapper used to access input vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::view()`.
 * @param v_op Unary operator that takes an input vertex property value and returns the output
 * vertex property value.
 * @param pred_op Unary predicate operator that takes an input vertex property value and returns
 * true if the output vertex property value should be updated.
 * @param vertex_value_output Wrapper used to store output vertex property values (for the vertices
 * assigned to this process in multi-GPU). Use `vertex_property_t::mutable_view()`.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexValueInputWrapper,
          typename VertexOp,
          typename PredOp,
          typename VertexValueOutputWrapper>
void transform_if_v(raft::handle_t const& handle,
                    GraphViewType const& graph_view,
                    VertexValueInputWrapper vertex_value_input,
                    VertexOp v_op,
                    PredOp pred_op,
                    VertexValueOutputWrapper vertex_value_output,
                    bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto input_value_first  = vertex_value_input.value_first();
  auto output_value_first = vertex_value_output.value_first();
  if constexpr (cugraph::has_packed_bool_element<typename VertexValueOutputWrapper::value_iterator,
                                                 typename VertexValueOutputWrapper::value_type>()) {
    auto num_bools = static_cast<size_t>(graph_view.local_vertex_partition_range_size());
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(packed_bool_size(num_bools)),
                     detail::transform_if_packed_bool_t<decltype(input_value_first),
                                                        VertexOp,
                                                        PredOp,
                                                        decltype(output_value_first)>{
                       input_value_first, v_op, pred_op, output_value_first, num_bools});
  } else {
    thrust::transform_if(handle.get_thrust_policy(),
                         input_value_first,
                         input_value_first + graph_view.local_vertex_partition_range_size(),
                         output_value_first,
                         v_op,
                         pred_op);
  }
}

}  // namespace CUGRAPH_EXPORT cugraph
