/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

// Andrei Schaffer, aschaffer@nvidia.com
//
#pragma once

#include <cugraph/experimental/graph.hpp>

#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>

namespace cugraph {
namespace experimental {

namespace detail {

template <typename T>
using device_vec_t = rmm::device_uvector<T>;

template <typename T>
using device_v_it = typename device_vec_t<T>::iterator;

template <typename value_t>
value_t* raw_ptr(device_vec_t<value_t>& dv)
{
  return dv.data();
}

template <typename value_t>
value_t const* raw_const_ptr(device_vec_t<value_t> const& dv)
{
  return dv.data();
}

template <typename value_t, typename index_t = size_t>
struct device_const_vector_view {
  device_const_vector_view(value_t const* d_buffer, index_t size) : d_buffer_(d_buffer), size_(size)
  {
  }

  device_const_vector_view(device_const_vector_view const& other) = delete;
  device_const_vector_view& operator=(device_const_vector_view const& other) = delete;

  device_const_vector_view(device_const_vector_view&& other)
  {
    d_buffer_ = other.d_buffer_;
    size_     = other.size_;
  }
  device_const_vector_view& operator=(device_const_vector_view&& other)
  {
    d_buffer_ = other.d_buffer_;
    size_     = other.size_;

    return *this;
  }

  value_t const* begin(void) const { return d_buffer_; }

  value_t const* end() const { return d_buffer_ + size_; }

  index_t size(void) const { return size_; }

 private:
  value_t const* d_buffer_{nullptr};
  index_t size_;
};

template <typename value_t>
value_t const* raw_const_ptr(device_const_vector_view<value_t>& dv)
{
  return dv.begin();
}

// classes abstracting the way the random walks path are generated:
//

// vertical traversal proxy:
// a device vector of next vertices is generated for each path;
// when a vertex is a sink the corresponding path doesn't advance anymore;
//
// smaller memory footprint;
//
struct vertical_traversal_t {
  vertical_traversal_t(size_t num_paths, size_t max_depth)
    : num_paths_(num_paths), max_depth_(max_depth)
  {
  }

  template <typename graph_t,
            typename random_walker_t,
            typename index_t,
            typename real_t,
            typename seed_t>
  void operator()(
    graph_t const& graph,                // graph being traversed
    random_walker_t const& rand_walker,  // random walker object for which traversal is driven
    seed_t seed0,                        // initial seed value
    device_vec_t<typename graph_t::vertex_type>& d_coalesced_v,  // crt coalesced vertex set
    device_vec_t<typename graph_t::weight_type>& d_coalesced_w,  // crt coalesced weight set
    device_vec_t<index_t>& d_paths_sz,                           // crt paths sizes
    device_vec_t<typename graph_t::edge_type>&
      d_crt_out_degs,                // crt out-degs for current set of vertices
    device_vec_t<real_t>& d_random,  // crt set of random real values
    device_vec_t<typename graph_t::vertex_type>&
      d_col_indx,  // crt col col indices to be used for retrieving next step
    device_vec_t<typename graph_t::vertex_type>&
      d_next_v,  // crt set of destination vertices, for next step
    device_vec_t<typename graph_t::weight_type>& d_next_w) const
  {
    // start from 1, as 0-th was initialized above:
    //
    for (decltype(max_depth_) step_indx = 1; step_indx < max_depth_; ++step_indx) {
      // take one-step in-sync for each path in parallel:
      //
      rand_walker.step(graph,
                       seed0 + static_cast<seed_t>(step_indx),
                       d_coalesced_v,
                       d_coalesced_w,
                       d_paths_sz,
                       d_crt_out_degs,
                       d_random,
                       d_col_indx,
                       d_next_v,
                       d_next_w);

      // early exit: all paths have reached sinks:
      //
      if (rand_walker.all_paths_stopped(d_crt_out_degs)) break;
    }
  }

  size_t get_random_buff_sz(void) const { return num_paths_; }

 private:
  size_t num_paths_;
  size_t max_depth_;
};

// vertical pipelined traversal proxy:
// a device vector of next vertices is generated for each path;
// when a vertex is a sink the corresponding path doesn't advance anymore;
// random vertex source is generated in parallel on CPU in a pipelined way;
//
// somewhat larger memory footprint but more efficient;
//
struct vertical_pipelined_t {
  vertical_pipelined_t(size_t num_paths, size_t max_depth)
    : num_paths_(num_paths), max_depth_(max_depth)
  {
  }

  template <typename graph_t,
            typename random_walker_t,
            typename index_t,
            typename real_t,
            typename seed_t>
  void operator()(
    graph_t const& graph,                // graph being traversed
    random_walker_t const& rand_walker,  // random walker object for which traversal is driven
    seed_t seed0,                        // initial seed value
    device_vec_t<typename graph_t::vertex_type>& d_coalesced_v,  // crt coalesced vertex set
    device_vec_t<typename graph_t::weight_type>& d_coalesced_w,  // crt coalesced weight set
    device_vec_t<index_t>& d_paths_sz,                           // crt paths sizes
    device_vec_t<typename graph_t::edge_type>&
      d_crt_out_degs,                // crt out-degs for current set of vertices
    device_vec_t<real_t>& d_random,  // crt set of random real values
    device_vec_t<typename graph_t::vertex_type>&
      d_col_indx,  // crt col col indices to be used for retrieving next step
    device_vec_t<typename graph_t::vertex_type>&
      d_next_v,  // crt set of destination vertices, for next step
    device_vec_t<typename graph_t::weight_type>& d_next_w) const
  {
    // random buffers:
    //
    real_t* p_d_rnd_work = d_random.begin();
    real_t* p_d_rnd_next = p_d_rnd_work + num_paths_;

    // start from 1, as 0-th has already been initialized:
    //
    for (decltype(max_depth_) step_indx = 1; step_indx < max_depth_; ++step_indx) {
      generate_async_rnd(p_d_rnd_next);

      // take one-step in-sync for each path in parallel:
      //
      rand_walker.step_only(graph,
                            seed0 + static_cast<seed_t>(step_indx),
                            d_coalesced_v,
                            d_coalesced_w,
                            d_paths_sz,
                            d_crt_out_degs,
                            p_d_rnd_work,
                            d_col_indx,
                            d_next_v,
                            d_next_w);

      // early exit: all paths have reached sinks:
      //
      if (rand_walker.all_paths_stopped(d_crt_out_degs)) break;

      // FIXME: needs async() wait():
      //
      std::swap(p_d_rnd_work, p_d_rnd_next);
    }
  }

  size_t get_random_buff_sz(void) const { return 2 * num_paths_; }

  template <typename real_t>
  void generate_async_rnd(real_t* p_d_rnd_next) const;

 private:
  size_t num_paths_;
  size_t max_depth_;
};

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
