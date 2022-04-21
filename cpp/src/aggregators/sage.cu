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

#include <cugraph-ops/graph/format.hpp>

#include <cugraph-ops/cuda/stream.h>

#include <raft/handle.hpp>

namespace cugraph {

template <typename vertex_t>
void aggregate_forward(raft::handle_t const& handle,
                       ops::gnn::graph::mfg_csr<vertex_t> const& message_flow_graph,
                       size_t dimension,
                       size_t leading_dimension,
                       bool ignore_self,
                       ops::gnn::AggOpT& op,
                       float const* input_embeddings,
                       float* aggregated_embeddings,
                       std::optional<vertex_t*> embedding_index)
{
  cugraph::ops::gnn::cuda::stream stream(handle.get_stream());
  ops::gnn::aggregator_fwd(
    aggregated_embeddings,
    ((op == ops::gnn::AggOpT::kMin) || (op == ops::gnn::AggOpT::kMax)) ? *embedding_index : nullptr,
    input_embeddings,
    dimension,
    leading_dimension,
    message_flow_graph,
    op,
    ignore_self,
    stream);
}

template <typename vertex_t>
void aggregate_backward(raft::handle_t const& handle,
                        ops::gnn::graph::mfg_csr<vertex_t> const& message_flow_graph,
                        size_t dimension,
                        size_t leading_dimension,
                        bool ignore_self,
                        ops::gnn::AggOpT op,
                        float const* aggregated_embeddings,
                        std::optional<vertex_t const*> embedding_index,
                        float* embeddings_gradient)
{
  cugraph::ops::gnn::cuda::stream stream(handle.get_stream());
  ops::gnn::aggregator_bwd(
    embeddings_gradient,
    aggregated_embeddings,
    ((op == ops::gnn::AggOpT::kMin) || (op == ops::gnn::AggOpT::kMax)) ? *embedding_index : nullptr,
    dimension,
    leading_dimension,
    message_flow_graph,
    op,
    ignore_self,
    stream);
}

template void aggregate_forward<int32_t>(
  raft::handle_t const& handle,
  ops::gnn::graph::mfg_csr<int32_t> const& message_flow_graph,
  size_t dimension,
  size_t leading_dimension,
  bool ignore_self,
  ops::gnn::AggOpT& op,
  float const* input_embeddings,
  float* aggregated_embeddings,
  std::optional<int32_t*> embedding_index);

template void aggregate_backward(raft::handle_t const& handle,
                                 ops::gnn::graph::mfg_csr<int32_t> const& message_flow_graph,
                                 size_t dimension,
                                 size_t leading_dimension,
                                 bool ignore_self,
                                 ops::gnn::AggOpT op,
                                 float const* aggregated_embeddings,
                                 std::optional<int32_t const*> embedding_index,
                                 float* embeddings_gradient);

template void aggregate_forward<int64_t>(
  raft::handle_t const& handle,
  ops::gnn::graph::mfg_csr<int64_t> const& message_flow_graph,
  size_t dimension,
  size_t leading_dimension,
  bool ignore_self,
  ops::gnn::AggOpT& op,
  float const* input_embeddings,
  float* aggregated_embeddings,
  std::optional<int64_t*> embedding_index);

template void aggregate_backward(raft::handle_t const& handle,
                                 ops::gnn::graph::mfg_csr<int64_t> const& message_flow_graph,
                                 size_t dimension,
                                 size_t leading_dimension,
                                 bool ignore_self,
                                 ops::gnn::AggOpT op,
                                 float const* aggregated_embeddings,
                                 std::optional<int64_t const*> embedding_index,
                                 float* embeddings_gradient);

}  // namespace cugraph
