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

#include <topology/topology.cuh>
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

// Biased RW selection logic:
//

template <typename vertex_t, typename edge_t, typename weight_t, typename real_t>
struct biased_selector_t {
  biased_selector_t(edge_t const* offsets,
                    vertex_t const* indices,
                    weight_t const* weights,
                    weight_t const* ptr_d_sum_weights)
    : row_offsets_(offsets),
      col_indices_(indices),
      values_(weights),
      ptr_d_sum_weights_(ptr_d_sum_weights)
  {
  }

  // pre-conditions:
  //
  // 1. (indices, weights) are assumed to be reshuflled
  //    so that weights(neighborhood(src_v)) are ordered
  //    increasingly; too expesnive to check this here;
  // 2. Sum(weights(neighborhood(src_v))) are pre-computed and
  //    stored in ptr_d_sum_weights_
  //
  __device__ thrust::optional<vertex_t> operator()(vertex_t src_v, real_t rnd_val)
  {
    weight_t run_sum_w{0};
    auto rnd_sum_weights = rnd_val * ptr_d_sum_weights_[src_v];

    auto col_indx_begin = row_offsets_[src_v];
    auto col_indx_end   = row_offsets_[src_v + 1];
    if (col_indx_begin == col_indx_end) return thrust::nullopt;  // src_v is a sink

    auto col_indx      = col_indx_begin;
    auto prev_col_indx = col_indx;

    for (; col_indx < col_indx_end; ++col_indx) {
      if (rnd_sum_weights < run_sum_w) break;

      run_sum_w += values_[col_indx];
      prev_col_indx = col_indx;
    }
    return thrust::optional<vertex_t>{col_indices_[prev_col_indx]};
  }

 private:
  edge_t const* row_offsets_;
  vertex_t const* col_indices_;
  weight_t const* values_;

  weight_t const* ptr_d_sum_weights_;
};

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
    using vertex_t        = typename graph_t::vertex_type;
    using edge_t          = typename graph_t::edge_type;
    using weight_t        = typename graph_t::weight_type;
    using random_engine_t = typename random_walker_t::rnd_engine_t;

    auto const& handle = rand_walker.get_handle();
    auto* ptr_d_random = raw_ptr(d_random);

    random_engine_t::generate_random(handle, ptr_d_random, d_random.size(), seed0);

    auto const* col_indices       = graph.get_matrix_partition_view().get_indices();
    auto const* row_offsets       = graph.get_matrix_partition_view().get_offsets();
    auto const* values            = graph.get_matrix_partition_view().get_weights()
                                      ? *(graph.get_matrix_partition_view().get_weights())
                                      : static_cast<weight_t*>(nullptr);
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

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
