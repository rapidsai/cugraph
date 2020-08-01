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

#include "common_utils.cuh"
#include "vertex_binning_kernels.cuh"

namespace cugraph {

namespace mg {

namespace detail {

template <typename VT, typename ET>
struct DegreeBucket {
  VT* vertexIds;
  VT numberOfVertices;
  ET ceilLogDegreeStart;
  ET ceilLogDegreeEnd;
};

template <typename VT, typename ET>
class LogDistribution {
  VT* vertex_id_begin_;
  thrust::host_vector<ET> bin_offsets_;

 public:
  LogDistribution(rmm::device_vector<ET>& vertex_id, rmm::device_vector<ET>& bin_offsets)
    : vertex_id_begin_(vertex_id.data().get()), bin_offsets_(bin_offsets)
  {
    // If bin_offsets_ is smaller than NumberBins<ET> then resize it
    // so that the last element is repeated
    bin_offsets_.resize(NumberBins<ET>, bin_offsets_.back());
  }

  DegreeBucket<VT, ET> degreeRange(ET ceilLogDegreeStart,
                                   ET ceilLogDegreeEnd = std::numeric_limits<ET>::max())
  {
    ceilLogDegreeStart = std::max(ceilLogDegreeStart, ET{0});
    if (ceilLogDegreeEnd > static_cast<ET>(bin_offsets_.size()) - 2) {
      ceilLogDegreeEnd = bin_offsets_.size() - 2;
    }
    return DegreeBucket<VT, ET>{
      vertex_id_begin_ + bin_offsets_[ceilLogDegreeStart + 1],
      bin_offsets_[ceilLogDegreeEnd + 1] - bin_offsets_[ceilLogDegreeStart + 1],
      ceilLogDegreeStart,
      ceilLogDegreeEnd};
  }
};

template <typename VT, typename ET>
class VertexBinner {
  ET* offsets_;
  unsigned* active_bitmap_;
  VT vertex_begin_;
  VT vertex_end_;

  rmm::device_vector<ET> tempBins_;
  rmm::device_vector<ET> bin_offsets_;

 public:
  VertexBinner(void) : tempBins_(NumberBins<ET>), bin_offsets_(NumberBins<ET>) {}

  void setup(ET* offsets, unsigned* active_bitmap, VT vertex_begin, VT vertex_end)
  {
    offsets_       = offsets;
    active_bitmap_ = active_bitmap;
    vertex_begin_  = vertex_begin;
    vertex_end_    = vertex_end;
  }

  LogDistribution<VT, ET> run(rmm::device_vector<VT>& reorganized_vertices, cudaStream_t stream);

  LogDistribution<VT, ET> run(
      rmm::device_vector<VT>& input_vertices,
      rmm::device_vector<VT>& reorganized_vertices,
      cudaStream_t stream);
};

template <typename VT, typename ET>
LogDistribution<VT, ET> VertexBinner<VT, ET>::run(rmm::device_vector<VT>& reorganized_vertices,
                                                  cudaStream_t stream)
{
  thrust::fill(
    rmm::exec_policy(stream)->on(stream), bin_offsets_.begin(), bin_offsets_.end(), ET{0});
  thrust::fill(rmm::exec_policy(stream)->on(stream), tempBins_.begin(), tempBins_.end(), ET{0});
  bin_vertices(reorganized_vertices,
               bin_offsets_,
               tempBins_,
               active_bitmap_,
               offsets_,
               vertex_begin_,
               vertex_end_,
               stream);

  return LogDistribution<VT, ET>(reorganized_vertices, bin_offsets_);
}

template <typename VT, typename ET>
LogDistribution<VT, ET> VertexBinner<VT, ET>::run(
    rmm::device_vector<VT>& input_vertices,
    rmm::device_vector<VT>& reorganized_vertices,
    cudaStream_t stream)
{
  thrust::fill(
    rmm::exec_policy(stream)->on(stream), bin_offsets_.begin(), bin_offsets_.end(), ET{0});
  thrust::fill(rmm::exec_policy(stream)->on(stream), tempBins_.begin(), tempBins_.end(), ET{0});
  bin_vertices(input_vertices,
               reorganized_vertices,
               bin_offsets_,
               tempBins_,
               offsets_,
               vertex_begin_,
               vertex_end_,
               stream);

  return LogDistribution<VT, ET>(reorganized_vertices, bin_offsets_);
}

}  // namespace detail

}  // namespace mg

}  // namespace cugraph
