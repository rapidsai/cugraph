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
#include <raft/handle.hpp>
#include <graph.hpp>

namespace cugraph {

namespace detail {

namespace opg {

constexpr inline size_t
divup(const size_t& numerator, const size_t& denominator) {
  return (numerator + denominator - 1)/denominator;
}

size_t number_of_words(size_t number_of_bits)
{
  size_t numerator = number_of_bits;
  size_t denominator = BitsPWrd<unsigned>;
  return (numerator > denominator) ?
    1 + divup(numerator - denominator, denominator) :
    (denominator > 0);
}

__global__
void reduce_bitwise_or(
    unsigned** frontiers,
    size_t frontier_count,
    int my_rank,
    size_t word_count) {

  unsigned word = 0;
  size_t tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid < word_count) {
    for (int i = 0; i < frontier_count; ++i) {
      word = word | frontiers[i][tid];
    }
    frontiers[my_rank][tid] = word;
  }
}

//FIXME : This class is a stand in till nccl allreduce works with bitwise OR
//Once bitwise OR reduction is placed in raft this communicator class should
//be removed
template <typename VT, typename ET, typename WT>
class BFSCommunicator {
  raft::handle_t const &handle_;
  size_t word_count_;
  int total_gpu_count_;
  int my_gpu_rank_;
  std::vector<rmm::device_vector<unsigned>> data_;
  std::vector<unsigned*> ptr_;
  rmm::device_vector<unsigned*> ptr_device_;

  public:

  BFSCommunicator(raft::handle_t const & handle,
      size_t word_count) :
    handle_(handle),
    word_count_(word_count),
    total_gpu_count_(handle_.get_comms().get_size()),
    my_gpu_rank_(handle_.get_comms().get_rank()),
    data_(handle_.get_comms().get_size()),
    ptr_(total_gpu_count_, nullptr)
  {
    for (int i = 0; i < total_gpu_count_; ++i) {
      if (i != my_gpu_rank_) {
        data_[i].resize(word_count_);
        ptr_[i] = data_[i].data().get();
      }
    }
  }

  void allreduce(rmm::device_vector<unsigned>& frontier) {
    cudaStream_t stream = handle_.get_stream();
    ptr_[my_gpu_rank_] = frontier.data().get();
    for (int i = 0; i < total_gpu_count_; ++i) {
      handle_.get_comms().bcast(ptr_[i], frontier.size(), i, stream);
    }
    ptr_device_ = ptr_;
    reduce_bitwise_or<<<divup(frontier.size(), 512), 512>>>(
        ptr_device_.data().get(),
        ptr_device_.size(),
        my_gpu_rank_,
        frontier.size());
  }

};


}//namespace opg

}//namespace detail

}//namespace cugraph
