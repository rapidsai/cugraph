/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "common_utils.cuh"
#include "vertex_binning_kernels.cuh"
#include <rmm/device_vector.hpp>

#include <thrust/fill.h>
#include <thrust/host_vector.h>

namespace cugraph {

namespace mg {

namespace detail {

template <typename vertex_t, typename edge_t>
struct DegreeBucket {
  vertex_t* vertexIds;
  vertex_t numberOfVertices;
  edge_t ceilLogDegreeStart;
  edge_t ceilLogDegreeEnd;
};

template <typename vertex_t, typename edge_t>
class LogDistribution {
  vertex_t* vertex_id_begin_;
  thrust::host_vector<edge_t> bin_offsets_;

 public:
  LogDistribution(rmm::device_vector<edge_t>& vertex_id, rmm::device_vector<edge_t>& bin_offsets)
    : vertex_id_begin_(vertex_id.data().get()), bin_offsets_(bin_offsets)
  {
  }

  DegreeBucket<vertex_t, edge_t> degreeRange(
    edge_t ceilLogDegreeStart, edge_t ceilLogDegreeEnd = std::numeric_limits<edge_t>::max())
  {
    ceilLogDegreeStart = std::max(ceilLogDegreeStart, edge_t{0});
    if (ceilLogDegreeEnd > static_cast<edge_t>(bin_offsets_.size()) - 2) {
      ceilLogDegreeEnd = bin_offsets_.size() - 2;
    }
    return DegreeBucket<vertex_t, edge_t>{
      vertex_id_begin_ + bin_offsets_[ceilLogDegreeStart + 1],
      bin_offsets_[ceilLogDegreeEnd + 1] - bin_offsets_[ceilLogDegreeStart + 1],
      ceilLogDegreeStart,
      ceilLogDegreeEnd};
  }
};

template <typename vertex_t, typename edge_t>
class VertexBinner {
  edge_t* offsets_;
  uint32_t* active_bitmap_;
  vertex_t vertex_begin_;
  vertex_t vertex_end_;

  rmm::device_vector<edge_t> tempBins_;
  rmm::device_vector<edge_t> bin_offsets_;

 public:
  VertexBinner(void) : tempBins_(NumberBins<edge_t>), bin_offsets_(NumberBins<edge_t>) {}

  void setup(edge_t* offsets, uint32_t* active_bitmap, vertex_t vertex_begin, vertex_t vertex_end)
  {
    offsets_       = offsets;
    active_bitmap_ = active_bitmap;
    vertex_begin_  = vertex_begin;
    vertex_end_    = vertex_end;
  }

  LogDistribution<vertex_t, edge_t> run(rmm::device_vector<vertex_t>& reorganized_vertices,
                                        cudaStream_t stream);

  LogDistribution<vertex_t, edge_t> run(rmm::device_vector<vertex_t>& input_vertices,
                                        vertex_t input_vertices_len,
                                        rmm::device_vector<vertex_t>& reorganized_vertices,
                                        cudaStream_t stream);
};

template <typename vertex_t, typename edge_t>
LogDistribution<vertex_t, edge_t> VertexBinner<vertex_t, edge_t>::run(
  rmm::device_vector<vertex_t>& reorganized_vertices, cudaStream_t stream)
{
  thrust::fill(rmm::exec_policy(stream), bin_offsets_.begin(), bin_offsets_.end(), edge_t{0});
  thrust::fill(rmm::exec_policy(stream), tempBins_.begin(), tempBins_.end(), edge_t{0});
  bin_vertices(reorganized_vertices,
               bin_offsets_,
               tempBins_,
               active_bitmap_,
               offsets_,
               vertex_begin_,
               vertex_end_,
               stream);

  return LogDistribution<vertex_t, edge_t>(reorganized_vertices, bin_offsets_);
}

template <typename vertex_t, typename edge_t>
LogDistribution<vertex_t, edge_t> VertexBinner<vertex_t, edge_t>::run(
  rmm::device_vector<vertex_t>& input_vertices,
  vertex_t input_vertices_len,
  rmm::device_vector<vertex_t>& reorganized_vertices,
  cudaStream_t stream)
{
  bin_vertices(input_vertices,
               input_vertices_len,
               reorganized_vertices,
               bin_offsets_,
               tempBins_,
               offsets_,
               vertex_begin_,
               vertex_end_,
               stream);

  return LogDistribution<vertex_t, edge_t>(reorganized_vertices, bin_offsets_);
}

}  // namespace detail

}  // namespace mg

}  // namespace cugraph
