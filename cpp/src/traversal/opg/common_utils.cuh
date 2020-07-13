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

#include <raft/integer_utils.h>

namespace cugraph {

namespace opg {

namespace detail {

template <typename T>
constexpr inline T
number_of_words(T number_of_bits) {
  return raft::div_rounding_up_safe(number_of_bits,
      static_cast<T>(BitsPWrd<unsigned>));
}

struct bitwise_or {
  __device__ unsigned operator()(unsigned& a, unsigned & b) { return a | b; }
};

struct remove_visited {
  __device__ unsigned operator()(unsigned& visited, unsigned& output) {
    //OUTPUT AND VISITED - common bits between output and visited
    //OUTPUT AND (NOT (OUTPUT AND VISITED))
    // - remove common bits between output and visited from output
    return (output & (~( output & visited )));
  }
};

template <typename VT>
struct bfs_frontier_pred {
  unsigned * output_frontier_;
  VT * predecessors_;

  bfs_frontier_pred(
      unsigned * output_frontier,
      VT * predecessors) :
    output_frontier_(output_frontier),
    predecessors_(predecessors) {}

  __device__ void operator()(VT src, VT dst) {
    unsigned active_bit = static_cast<unsigned>(1)<<(dst % BitsPWrd<unsigned>);
    unsigned prev_word =
      atomicOr(output_frontier_ + (dst/BitsPWrd<unsigned>), active_bit);
    //If this thread activates the frontier bitmap for a destination
    //then the source is the predecessor of that destination
    if (prev_word & active_bit == 0) {
      predecessors_[dst] = src;
    }
  }
};

template <typename VT>
struct bfs_frontier_pred_dist {
  unsigned * output_frontier_;
  VT * predecessors_;
  VT * distances_;
  VT level_;

  bfs_frontier_pred_dist(
      unsigned * output_frontier,
      VT * predecessors,
      VT * distances, VT level) :
    output_frontier_(output_frontier),
    predecessors_(predecessors),
    distances_(distances),
    level_(level) {}

  __device__ void operator()(VT src, VT dst) {
    unsigned active_bit = static_cast<unsigned>(1)<<(dst % BitsPWrd<unsigned>);
    unsigned prev_word =
      atomicOr(output_frontier_ + (dst/BitsPWrd<unsigned>), active_bit);
    //If this thread activates the frontier bitmap for a destination
    //then the source is the predecessor of that destination
    if (prev_word & active_bit == 0) {
      distances_[dst] = level_;
      predecessors_[dst] = src;
    }
  }
};

struct is_not_equal {
  unsigned cmp_;
  unsigned * flag_;
  is_not_equal(unsigned cmp, unsigned * flag) : cmp_(cmp), flag_(flag) {}
  __device__ void operator()(unsigned& val) {
    if (val != cmp_) {
      *flag_ = 1;
    }
  }
};

}//namespace detail

}//namespace opg

}//namespace cugraph
