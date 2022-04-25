/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>

#include <cugraph-ops/aggregator.hpp>

#include <cugraph-ops/agg_concat.hpp>

#include <cugraph-ops/graph/format.hpp>

#include <cugraph-ops/cuda/stream.hpp>

#include <raft/handle.hpp>

namespace cugraph {

template <typename vertex_t>
void aggregate_forward(raft::handle_t const& handle,
                       ops::gnn::graph::mfg_csr<vertex_t> const& message_flow_graph,
                       size_t dimension,
                       size_t leading_dimension,
                       bool ignore_self,
                       ops::gnn::AggOpT op,
                       float const* input_embedding,
                       float* output_embedding,
                       std::optional<vertex_t*> output_extrema_location)
{
  cugraph::ops::gnn::cuda::stream stream(handle.get_stream());
  ops::gnn::aggregator_fwd(output_embedding,
                           ((op == ops::gnn::AggOpT::kMin) || (op == ops::gnn::AggOpT::kMax))
                             ? *output_extrema_location
                             : nullptr,
                           input_embedding,
                           dimension,
                           leading_dimension,
                           message_flow_graph,
                           op,
                           ignore_self,
                           stream);
}

template <typename vertex_t>
void aggregate_concatenate_forward(raft::handle_t const& handle,
                                   ops::gnn::graph::mfg_csr<vertex_t> const& message_flow_graph,
                                   size_t dimension,
                                   ops::gnn::AggOpT op,
                                   float const* input_embedding,
                                   float* output_embedding,
                                   std::optional<vertex_t*> output_extrema_location)
{
  cugraph::ops::gnn::cuda::stream stream(handle.get_stream());
  ops::gnn::agg_concat_fwd(output_embedding,
                           ((op == ops::gnn::AggOpT::kMin) || (op == ops::gnn::AggOpT::kMax))
                             ? *output_extrema_location
                             : nullptr,
                           input_embedding,
                           dimension,
                           message_flow_graph,
                           op,
                           stream);
}

template <typename vertex_t>
void aggregate_backward(raft::handle_t const& handle,
                        ops::gnn::graph::mfg_csr<vertex_t> const& message_flow_graph,
                        size_t dimension,
                        size_t leading_dimension,
                        bool ignore_self,
                        ops::gnn::AggOpT op,
                        float const* input_gradient,
                        std::optional<vertex_t const*> input_extrema_location,
                        float* output_gradient)
{
  cugraph::ops::gnn::cuda::stream stream(handle.get_stream());
  ops::gnn::aggregator_bwd(output_gradient,
                           input_gradient,
                           ((op == ops::gnn::AggOpT::kMin) || (op == ops::gnn::AggOpT::kMax))
                             ? *input_extrema_location
                             : nullptr,
                           dimension,
                           leading_dimension,
                           message_flow_graph,
                           op,
                           ignore_self,
                           stream);
}

template <typename vertex_t>
void aggregate_concatenate_backward(raft::handle_t const& handle,
                                    ops::gnn::graph::mfg_csr<vertex_t> const& message_flow_graph,
                                    size_t dimension,
                                    ops::gnn::AggOpT op,
                                    float const* input_gradient,
                                    std::optional<vertex_t const*> input_extrema_location,
                                    float* output_gradient)
{
  cugraph::ops::gnn::cuda::stream stream(handle.get_stream());
  ops::gnn::agg_concat_bwd(output_gradient,
                           input_gradient,
                           ((op == ops::gnn::AggOpT::kMin) || (op == ops::gnn::AggOpT::kMax))
                             ? *input_extrema_location
                             : nullptr,
                           dimension,
                           message_flow_graph,
                           op,
                           stream);
}

template void aggregate_forward<int32_t>(
  raft::handle_t const& handle,
  ops::gnn::graph::mfg_csr<int32_t> const& message_flow_graph,
  size_t dimension,
  size_t leading_dimension,
  bool ignore_self,
  ops::gnn::AggOpT op,
  float const* input_embedding,
  float* output_embedding,
  std::optional<int32_t*> output_extrema_location);

template void aggregate_forward<int64_t>(
  raft::handle_t const& handle,
  ops::gnn::graph::mfg_csr<int64_t> const& message_flow_graph,
  size_t dimension,
  size_t leading_dimension,
  bool ignore_self,
  ops::gnn::AggOpT op,
  float const* input_embedding,
  float* output_embedding,
  std::optional<int64_t*> output_extrema_location);

template void aggregate_concatenate_forward<int32_t>(
  raft::handle_t const& handle,
  ops::gnn::graph::mfg_csr<int32_t> const& message_flow_graph,
  size_t dimension,
  ops::gnn::AggOpT op,
  float const* input_embedding,
  float* output_embedding,
  std::optional<int32_t*> output_extrema_location);

template void aggregate_concatenate_forward<int64_t>(
  raft::handle_t const& handle,
  ops::gnn::graph::mfg_csr<int64_t> const& message_flow_graph,
  size_t dimension,
  ops::gnn::AggOpT op,
  float const* input_embedding,
  float* output_embedding,
  std::optional<int64_t*> output_extrema_location);

template void aggregate_backward(raft::handle_t const& handle,
                                 ops::gnn::graph::mfg_csr<int32_t> const& message_flow_graph,
                                 size_t dimension,
                                 size_t leading_dimension,
                                 bool ignore_self,
                                 ops::gnn::AggOpT op,
                                 float const* input_gradient,
                                 std::optional<int32_t const*> input_extrema_location,
                                 float* output_gradient);

template void aggregate_backward(raft::handle_t const& handle,
                                 ops::gnn::graph::mfg_csr<int64_t> const& message_flow_graph,
                                 size_t dimension,
                                 size_t leading_dimension,
                                 bool ignore_self,
                                 ops::gnn::AggOpT op,
                                 float const* input_gradient,
                                 std::optional<int64_t const*> input_extrema_location,
                                 float* output_gradient);

template void aggregate_concatenate_backward(
  raft::handle_t const& handle,
  ops::gnn::graph::mfg_csr<int32_t> const& message_flow_graph,
  size_t dimension,
  ops::gnn::AggOpT op,
  float const* input_gradient,
  std::optional<int32_t const*> input_extrema_location,
  float* output_gradient);

template void aggregate_concatenate_backward(
  raft::handle_t const& handle,
  ops::gnn::graph::mfg_csr<int64_t> const& message_flow_graph,
  size_t dimension,
  ops::gnn::AggOpT op,
  float const* input_gradient,
  std::optional<int64_t const*> input_extrema_location,
  float* output_gradient);

}  // namespace cugraph
