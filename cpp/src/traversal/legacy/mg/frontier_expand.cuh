/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "frontier_expand_kernels.cuh"
#include "vertex_binning.cuh"
#include <cugraph/legacy/graph.hpp>
#include <rmm/device_vector.hpp>

namespace cugraph {

namespace mg {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
class FrontierExpand {
  raft::handle_t const& handle_;
  cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph_;
  VertexBinner<vertex_t, edge_t> dist_;
  rmm::device_vector<vertex_t> reorganized_vertices_;
  edge_t vertex_begin_;
  edge_t vertex_end_;
  rmm::device_vector<edge_t> output_vertex_count_;

 public:
  FrontierExpand(raft::handle_t const& handle,
                 cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph)
    : handle_(handle), graph_(graph)
  {
    bool is_mg = (handle.comms_initialized() && (graph.local_vertices != nullptr) &&
                  (graph.local_offsets != nullptr));
    if (is_mg) {
      reorganized_vertices_.resize(graph.local_vertices[handle_.get_comms().get_rank()]);
      vertex_begin_ = graph.local_offsets[handle_.get_comms().get_rank()];
      vertex_end_   = graph.local_offsets[handle_.get_comms().get_rank()] +
                    graph.local_vertices[handle_.get_comms().get_rank()];
    } else {
      reorganized_vertices_.resize(graph.number_of_vertices);
      vertex_begin_ = 0;
      vertex_end_   = graph.number_of_vertices;
    }
    output_vertex_count_.resize(1);
  }

  // Return the size of the output_frontier
  template <typename operator_t>
  vertex_t operator()(operator_t op,
                      rmm::device_vector<vertex_t>& input_frontier,
                      vertex_t input_frontier_len,
                      rmm::device_vector<vertex_t>& output_frontier)
  {
    if (input_frontier_len == 0) { return static_cast<vertex_t>(0); }
    cudaStream_t stream     = handle_.get_stream();
    output_vertex_count_[0] = 0;
    dist_.setup(graph_.offsets, nullptr, vertex_begin_, vertex_end_);
    auto distribution =
      dist_.run(input_frontier, input_frontier_len, reorganized_vertices_, stream);

    DegreeBucket<vertex_t, edge_t> large_bucket = distribution.degreeRange(16);
    // TODO : Use other streams from handle_
    large_vertex_lb(graph_,
                    large_bucket,
                    op,
                    vertex_begin_,
                    output_frontier.data().get(),
                    output_vertex_count_.data().get(),
                    stream);

    DegreeBucket<vertex_t, edge_t> medium_bucket = distribution.degreeRange(12, 16);
    medium_vertex_lb(graph_,
                     medium_bucket,
                     op,
                     vertex_begin_,
                     output_frontier.data().get(),
                     output_vertex_count_.data().get(),
                     stream);

    DegreeBucket<vertex_t, edge_t> small_bucket_0 = distribution.degreeRange(10, 12);
    DegreeBucket<vertex_t, edge_t> small_bucket_1 = distribution.degreeRange(8, 10);
    DegreeBucket<vertex_t, edge_t> small_bucket_2 = distribution.degreeRange(6, 8);
    DegreeBucket<vertex_t, edge_t> small_bucket_3 = distribution.degreeRange(0, 6);

    small_vertex_lb(graph_,
                    small_bucket_0,
                    op,
                    vertex_begin_,
                    output_frontier.data().get(),
                    output_vertex_count_.data().get(),
                    stream);
    small_vertex_lb(graph_,
                    small_bucket_1,
                    op,
                    vertex_begin_,
                    output_frontier.data().get(),
                    output_vertex_count_.data().get(),
                    stream);
    small_vertex_lb(graph_,
                    small_bucket_2,
                    op,
                    vertex_begin_,
                    output_frontier.data().get(),
                    output_vertex_count_.data().get(),
                    stream);
    small_vertex_lb(graph_,
                    small_bucket_3,
                    op,
                    vertex_begin_,
                    output_frontier.data().get(),
                    output_vertex_count_.data().get(),
                    stream);
    return output_vertex_count_[0];
  }
};

}  // namespace detail

}  // namespace mg

}  // namespace cugraph
