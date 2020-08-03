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

#include <raft/cudart_utils.h>
#include <graph.hpp>
#include "vertex_binning.cuh"
#include "worker_kernels.cuh"
#include <utilities/high_res_timer.hpp>

namespace cugraph {

namespace mg {

namespace detail {

template <typename VT, typename ET, typename WT>
class LoadBalanceExecution {
  raft::handle_t const &handle_;
  cugraph::GraphCSRView<VT, ET, WT> const &graph_;
  VertexBinner<VT, ET> dist_;
  rmm::device_vector<VT> reorganized_vertices_;
  ET vertex_begin_;
  ET vertex_end_;
  rmm::device_vector<ET> output_vertex_count_;

  HighResTimer timer;
  HighResTimer run_timer;

 public:
  LoadBalanceExecution(raft::handle_t const &handle, cugraph::GraphCSRView<VT, ET, WT> const &graph)
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

  ~LoadBalanceExecution (void) {
    timer.display(std::cout);
    run_timer.display(std::cout);
  }

  template <typename Operator>
  void run(Operator op, unsigned *active_bitmap = nullptr)
  {
    cudaStream_t stream = handle_.get_stream();
    dist_.setup(graph_.offsets, active_bitmap, vertex_begin_, vertex_end_);
    auto distribution = dist_.run(reorganized_vertices_, stream);

    DegreeBucket<VT, ET> large_bucket = distribution.degreeRange(16);
    // TODO : Use other streams from handle_
    large_vertex_worker(graph_, large_bucket, op, stream);

    DegreeBucket<VT, ET> medium_bucket = distribution.degreeRange(12, 16);
    medium_vertex_worker(graph_, medium_bucket, op, stream);

    DegreeBucket<VT, ET> small_bucket_0 = distribution.degreeRange(10, 12);
    DegreeBucket<VT, ET> small_bucket_1 = distribution.degreeRange(8, 10);
    DegreeBucket<VT, ET> small_bucket_2 = distribution.degreeRange(6, 8);
    DegreeBucket<VT, ET> small_bucket_3 = distribution.degreeRange(0, 6);

    small_vertex_worker(graph_, small_bucket_0, op, stream);
    small_vertex_worker(graph_, small_bucket_1, op, stream);
    small_vertex_worker(graph_, small_bucket_2, op, stream);
    small_vertex_worker(graph_, small_bucket_3, op, stream);
  }

  template <typename Operator>
  void run(Operator op,
      rmm::device_vector<VT> &input_frontier)
  {
    cudaStream_t stream = handle_.get_stream();
    dist_.setup(graph_.offsets, nullptr, vertex_begin_, vertex_end_);
    auto distribution = dist_.run(
        input_frontier, reorganized_vertices_, stream);

    DegreeBucket<VT, ET> large_bucket = distribution.degreeRange(16);
    // TODO : Use other streams from handle_
    large_vertex_worker(graph_, large_bucket, op, stream);

    DegreeBucket<VT, ET> medium_bucket = distribution.degreeRange(12, 16);
    medium_vertex_worker(graph_, medium_bucket, op, stream);

    DegreeBucket<VT, ET> small_bucket_0 = distribution.degreeRange(10, 12);
    DegreeBucket<VT, ET> small_bucket_1 = distribution.degreeRange(8, 10);
    DegreeBucket<VT, ET> small_bucket_2 = distribution.degreeRange(6, 8);
    DegreeBucket<VT, ET> small_bucket_3 = distribution.degreeRange(0, 6);

    small_vertex_worker(graph_, small_bucket_0, op, stream);
    small_vertex_worker(graph_, small_bucket_1, op, stream);
    small_vertex_worker(graph_, small_bucket_2, op, stream);
    small_vertex_worker(graph_, small_bucket_3, op, stream);
  }

  //Return the size of the output_frontier
  template <typename Operator>
  VT run(Operator op,
      rmm::device_vector<VT> &input_frontier,
      VT input_frontier_len,
      rmm::device_vector<VT> &output_frontier)
  {
    if (input_frontier_len == 0) { return static_cast<VT>(0); }
    cudaStream_t stream = handle_.get_stream();
    cudaStreamSynchronize(stream);
    run_timer.start("lb : full run");
    //timer.start("lb : step 1");
    //output_frontier.resize(graph_.number_of_vertices);
    cudaStreamSynchronize(stream);
    //timer.stop();
    timer.start("lb : step 1a");
    output_vertex_count_[0] = 0;
    cudaStreamSynchronize(stream);
    timer.stop();
    cudaStreamSynchronize(stream);
    timer.start("lb : step 2");
    dist_.setup(graph_.offsets, nullptr, vertex_begin_, vertex_end_);
    cudaStreamSynchronize(stream);
    timer.stop();
    cudaStreamSynchronize(stream);
    timer.start("lb : distributer run");
    auto distribution = dist_.run(
        input_frontier, input_frontier_len, reorganized_vertices_, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    cudaStreamSynchronize(stream);
    timer.start("lb : step 3");

    DegreeBucket<VT, ET> large_bucket = distribution.degreeRange(16);
    if (large_bucket.numberOfVertices != 0) {
      std::cerr<<"large_vertex_worker\n";
    }
    // TODO : Use other streams from handle_
    //TODO : Disabled for testing
    //large_vertex_worker(graph_, large_bucket, op, stream);

    cudaStreamSynchronize(stream);
    timer.stop();
    cudaStreamSynchronize(stream);
    timer.start("lb : functor run");
    DegreeBucket<VT, ET> medium_bucket = distribution.degreeRange(12, 16);
    medium_vertex_worker(graph_, medium_bucket, op,
        output_frontier.data().get(), output_vertex_count_.data().get(), stream);

    DegreeBucket<VT, ET> small_bucket_0 = distribution.degreeRange(10, 12);
    DegreeBucket<VT, ET> small_bucket_1 = distribution.degreeRange(8, 10);
    DegreeBucket<VT, ET> small_bucket_2 = distribution.degreeRange(6, 8);
    DegreeBucket<VT, ET> small_bucket_3 = distribution.degreeRange(0, 6);

    small_vertex_worker(graph_, small_bucket_0, op,
        output_frontier.data().get(), output_vertex_count_.data().get(), stream);
    small_vertex_worker(graph_, small_bucket_1, op,
        output_frontier.data().get(), output_vertex_count_.data().get(), stream);
    small_vertex_worker(graph_, small_bucket_2, op,
        output_frontier.data().get(), output_vertex_count_.data().get(), stream);
    small_vertex_worker(graph_, small_bucket_3, op,
        output_frontier.data().get(), output_vertex_count_.data().get(), stream);
    //output_frontier.resize(output_vertex_count_[0]);
    cudaStreamSynchronize(stream);
    timer.stop();
    run_timer.stop();
    return output_vertex_count_[0];
  }
};

}  // namespace detail

}  // namespace mg

}  // namespace cugraph
