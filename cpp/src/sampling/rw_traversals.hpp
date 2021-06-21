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

#include <thrust/for_each.h>

#include <algorithm>
#include <future>
#include <thread>

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
    device_vec_t<typename graph_t::weight_type>&
      d_next_w)  // set of weights between src and destination vertices, for next step
    const
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
  size_t get_tmp_buff_sz(void) const { return num_paths_; }

 private:
  size_t num_paths_;
  size_t max_depth_;
};

// horizontal traversal proxy:
// each path is generated independently from start to finish;
// when a vertex is a sink the corresponding path doesn't advance anymore;
// requires (num_paths x max_depth) precomputed real random values in [0,1];
//
// larger memory footprint, but potentially more efficient;
//
struct horizontal_traversal_t {
  horizontal_traversal_t(size_t num_paths, size_t max_depth)
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
      d_crt_out_degs,                // ignored: out-degs for the current set of vertices
    device_vec_t<real_t>& d_random,  // _entire_ set of random real values
    device_vec_t<typename graph_t::vertex_type>&
      d_col_indx,  // ignored: crt col indices to be used for retrieving next step
    device_vec_t<typename graph_t::vertex_type>&
      d_next_v,  // ignored: crt set of destination vertices, for next step (coalesced set
                 // updated directly, instead)
    device_vec_t<typename graph_t::weight_type>&
      d_next_w)  // ignored: set of weights between src and destination vertices, for next step
                 // (coalesced set updated directly, instead)
    const
  {
    using vertex_t = typename graph_t::vertex_type;
    using edge_t   = typename graph_t::edge_type;
    using weight_t = typename graph_t::weight_type;

    auto const& handle = rand_walker.get_handle();
    auto* ptr_d_random = raw_ptr(d_random);
    raft::random::Rng rng(seed0);
    rng.uniform<real_t, index_t>(
      ptr_d_random, d_random.size(), real_t{0.0}, real_t{1.0}, handle.get_stream());

    auto const* col_indices       = graph.indices();
    auto const* row_offsets       = graph.offsets();
    auto const* values            = graph.weights();
    auto* ptr_d_sizes             = raw_ptr(d_paths_sz);
    auto const& d_cached_out_degs = rand_walker.get_out_degs();

    auto rnd_to_indx_convertor = [] __device__(real_t rnd_vindx, edge_t crt_out_deg) {
      real_t max_ub     = static_cast<real_t>(crt_out_deg - 1);
      auto interp_vindx = rnd_vindx * max_ub + real_t{.5};
      vertex_t v_indx   = static_cast<vertex_t>(interp_vindx);
      return (v_indx >= crt_out_deg ? crt_out_deg - 1 : v_indx);
    };

    auto next_vw =
      [row_offsets,
       col_indices,
       values] __device__(auto v_indx,      // src vertex to find dst from
                          auto col_indx) {  // column index, in {0,...,out_deg(v_indx)-1},
        // extracted from random value in [0..1]
        auto start_row = row_offsets[v_indx];

        auto weight_value =
          (values == nullptr ? weight_t{1}
                             : values[start_row + col_indx]);  // account for un-weighted graphs
        return thrust::make_tuple(col_indices[start_row + col_indx], weight_value);
      };

    // start from 1, as 0-th was initialized above:
    //
    thrust::for_each(rmm::exec_policy(handle.get_stream_view()),
                     thrust::make_counting_iterator<index_t>(0),
                     thrust::make_counting_iterator<index_t>(num_paths_),
                     [max_depth            = max_depth_,
                      ptr_d_cache_out_degs = raw_const_ptr(d_cached_out_degs),
                      ptr_coalesced_v      = raw_ptr(d_coalesced_v),
                      ptr_coalesced_w      = raw_ptr(d_coalesced_w),
                      ptr_d_random,
                      ptr_d_sizes,
                      rnd_to_indx_convertor,
                      next_vw] __device__(auto path_index) {
                       auto chunk_offset   = path_index * max_depth;
                       vertex_t src_vertex = ptr_coalesced_v[chunk_offset];

                       for (index_t step_indx = 1; step_indx < max_depth; ++step_indx) {
                         auto crt_out_deg = ptr_d_cache_out_degs[src_vertex];
                         if (crt_out_deg == 0) break;

                         // indexing into coalesced arrays of size num_paths x (max_depth -1):
                         // (d_random, d_coalesced_w)
                         //
                         auto stepping_index = chunk_offset - path_index + step_indx - 1;

                         auto real_rnd_indx = ptr_d_random[stepping_index];

                         auto col_indx = rnd_to_indx_convertor(real_rnd_indx, crt_out_deg);
                         auto pair_vw  = next_vw(src_vertex, col_indx);

                         src_vertex      = thrust::get<0>(pair_vw);
                         auto crt_weight = thrust::get<1>(pair_vw);

                         ptr_coalesced_v[chunk_offset + step_indx] = src_vertex;
                         ptr_coalesced_w[stepping_index]           = crt_weight;
                         ptr_d_sizes[path_index]++;
                       }
                     });
  }

  size_t get_random_buff_sz(void) const { return num_paths_ * (max_depth_ - 1); }
  size_t get_tmp_buff_sz(void) const
  {
    return 0;
  }  // no need for tmp buffers
     //(see "ignored" above)

 private:
  size_t num_paths_;
  size_t max_depth_;
};  // namespace detail

// vertical pipelined traversal proxy:
// a device vector of next vertices is generated for each path;
// when a vertex is a sink the corresponding path doesn't advance anymore;
// random vertex source is generated in parallel on CPU in a pipelined way;
//
// somewhat larger memory footprint but more efficient than vertical_traversal_t;
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
    device_vec_t<typename graph_t::weight_type>&
      d_next_w)  // set of weights between src and destination vertices, for next step
    const
  {
    // random buffers:
    //
    real_t* p_d_rnd_work = d_random.begin();
    real_t* p_d_rnd_next = p_d_rnd_work + num_paths_;

    fill_rnd<index_t>(rand_walker, p_d_rnd_work, seed0);
    // start from 1, as 0-th has already been initialized:
    //
    for (decltype(max_depth_) step_indx = 1; step_indx < max_depth_; ++step_indx) {
      auto seed = seed0 + static_cast<seed_t>(step_indx);

      // async generation of next random vector:
      //
      auto&& future_res = generate_async_rnd<index_t>(rand_walker, p_d_rnd_next, seed);

      // take one-step in-sync for each path in parallel:
      //
      rand_walker.step_only(graph,
                            d_coalesced_v,
                            d_coalesced_w,
                            d_paths_sz,
                            d_crt_out_degs,
                            p_d_rnd_work,
                            d_col_indx,
                            d_next_v,
                            d_next_w);

      // block to get the next random vector:
      //
      auto res = future_res.get();

      // swap rnd vectors:
      //
      std::swap(p_d_rnd_work, p_d_rnd_next);

      // early exit: all paths have reached sinks:
      //
      if (rand_walker.all_paths_stopped(d_crt_out_degs)) break;
    }
  }

  size_t get_random_buff_sz(void) const { return 2 * num_paths_; }
  size_t get_tmp_buff_sz(void) const { return num_paths_; }

  template <typename random_walker_t, typename real_t, typename index_t, typename seed_t>
  size_t fill_rnd(random_walker_t const& rand_walker, real_t* ptr_d_rnd, seed_t seed) const
  {
    auto const& handle = rand_walker.get_handle();

    raft::random::Rng rng(seed);
    rng.uniform<real_t, index_t>(
      ptr_d_rnd, num_paths_, real_t{0.0}, real_t{1.0}, handle.get_stream());

    return num_paths_;
  }

  template <typename random_walker_t, typename index_t, typename real_t, typename seed_t>
  decltype(auto) generate_async_rnd(random_walker_t const& rand_walker,
                                    real_t* p_d_rnd,
                                    seed_t seed) const
  {
    std::future<size_t> result(
      std::async(std::launch::async, [this, rand_walker, p_d_rnd, seed](void) {
        return fill_rnd<index_t>(rand_walker, p_d_rnd, seed);
      }));

    return result;
  }

 private:
  size_t num_paths_;
  size_t max_depth_;
};

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
