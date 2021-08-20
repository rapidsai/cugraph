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

#include <cugraph/graph.hpp>
#include <cugraph/visitors/graph_envelope.hpp>
#include <cugraph/visitors/ret_terased.hpp>

#include <utilities/graph_utils.cuh>

#include <cub/cub.cuh>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <future>
#include <thread>

namespace cugraph {

namespace detail {

enum class sampling_t : int { UNIFORM = 0, BIASED };  // sampling strategy; others: NODE2VEC

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

// Uniform RW selector logic:
//
template <typename graph_type, typename real_t>
struct uniform_selector_t {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  struct sampler_t {
    sampler_t(edge_t const* ro,
              vertex_t const* ci,
              weight_t const* w,
              edge_t const* ptr_d_cache_out_degs)
      : row_offsets_(ro), col_indices_(ci), values_(w), ptr_d_cache_out_degs_(ptr_d_cache_out_degs)
    {
    }

    __device__ thrust::optional<thrust::tuple<vertex_t, weight_t>> operator()(vertex_t src_v,
                                                                              real_t rnd_val) const
    {
      auto crt_out_deg = ptr_d_cache_out_degs_[src_v];
      if (crt_out_deg == 0) return thrust::nullopt;  // src_v is a sink

      real_t max_ub     = static_cast<real_t>(crt_out_deg - 1);
      auto interp_vindx = rnd_val * max_ub;
      vertex_t v_indx   = static_cast<vertex_t>(interp_vindx);

      auto col_indx  = v_indx >= crt_out_deg ? crt_out_deg - 1 : v_indx;
      auto start_row = row_offsets_[src_v];

      auto weight_value =
        (values_ == nullptr ? weight_t{1}
                            : values_[start_row + col_indx]);  // account for un-weighted graphs
      return thrust::optional{thrust::make_tuple(col_indices_[start_row + col_indx], weight_value)};
    }

   private:
    edge_t const* row_offsets_;
    vertex_t const* col_indices_;
    weight_t const* values_;

    edge_t const* ptr_d_cache_out_degs_;
  };

  using sampler_type = sampler_t;

  uniform_selector_t(raft::handle_t const& handle, graph_type const& graph, real_t tag)
    : d_cache_out_degs_(graph.compute_out_degrees(handle)),
      sampler_{graph.get_matrix_partition_view().get_offsets(),
               graph.get_matrix_partition_view().get_indices(),
               graph.get_matrix_partition_view().get_weights()
                 ? *(graph.get_matrix_partition_view().get_weights())
                 : static_cast<weight_t*>(nullptr),
               d_cache_out_degs_.data()}
  {
  }

  device_vec_t<edge_t> const& get_cached_out_degs(void) const { return d_cache_out_degs_; }

  sampler_t const& get_strategy(void) const { return sampler_; }

 private:
  device_vec_t<edge_t> d_cache_out_degs_;  // selector-specific: selector must own this resource
  sampler_t sampler_;  // which is why the sampling must be separated into a separate object:
  // it must be captured by device a calling lambda, which is not possible for selector object,
  // because it ows a device_vec;
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename binary_op_t = thrust::plus<weight_t>>
struct visitor_aggregate_weights_t : visitors::visitor_t {
  visitor_aggregate_weights_t(
    raft::handle_t const& handle,
    size_t num_vertices,
    binary_op_t aggregate_op = thrust::plus<weight_t>{},
    weight_t initial_value   = 0)  // different aggregation ops require different initial values;
    : handle_(handle),
      d_aggregate_weights_(num_vertices, handle_.get_stream()),
      aggregator_(aggregate_op),
      initial_value_(initial_value)
  {
  }

  void visit_graph(graph_envelope_t::base_graph_t const& graph_v) override
  {
    auto const& graph_view =
      static_cast<graph_view_t<vertex_t, edge_t, weight_t, false, false> const&>(graph_v);

    auto opt_weights = graph_view.get_matrix_partition_view().get_weights();
    CUGRAPH_EXPECTS(opt_weights.has_value(), "Cannot aggregate weights of un-weighted graph.");

    size_t num_vertices = d_aggregate_weights_.size();

    edge_t const* offsets = graph_view.get_matrix_partition_view().get_offsets();

    weight_t const* values = *opt_weights;

    // Determine temporary device storage requirements:
    //
    void* ptr_d_temp_storage{nullptr};
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Reduce(ptr_d_temp_storage,
                                       temp_storage_bytes,
                                       values,
                                       d_aggregate_weights_.data(),
                                       num_vertices,
                                       offsets,
                                       offsets + 1,
                                       aggregator_,
                                       initial_value_,
                                       handle_.get_stream());
    // Allocate temporary storage
    //
    rmm::device_uvector<std::byte> d_temp_storage(temp_storage_bytes, handle_.get_stream());
    ptr_d_temp_storage = d_temp_storage.data();

    // Run reduction:
    //
    cub::DeviceSegmentedReduce::Reduce(ptr_d_temp_storage,
                                       temp_storage_bytes,
                                       values,
                                       d_aggregate_weights_.data(),
                                       num_vertices,
                                       offsets,
                                       offsets + 1,
                                       aggregator_,
                                       initial_value_,
                                       handle_.get_stream());
  }

  // no need for type-erasure, as this is only used internally:
  //
  visitors::return_t const& get_result(void) const override { return ret_unused_; }

  rmm::device_uvector<weight_t>&& get_aggregated_weights(void)
  {
    return std::move(d_aggregate_weights_);
  }

  rmm::device_uvector<weight_t> const& get_aggregated_weights(void) const
  {
    return d_aggregate_weights_;
  }

 private:
  raft::handle_t const& handle_;
  rmm::device_uvector<weight_t> d_aggregate_weights_;
  binary_op_t aggregator_;
  weight_t initial_value_{0};
  visitors::return_t ret_unused_{};  // necessary to silence a concerning warning
};

// Biased RW selection logic:
//
// FIXME:
// 1. move sum weights calculation into selector;
// 2. pass graph_view to constructor;
//
template <typename graph_type, typename real_t>
struct biased_selector_t {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  struct sampler_t {
    sampler_t(edge_t const* ro,
              vertex_t const* ci,
              weight_t const* w,
              weight_t const* ptr_d_sum_weights)
      : row_offsets_(ro), col_indices_(ci), values_(w), ptr_d_sum_weights_(ptr_d_sum_weights)
    {
    }

    // pre-condition:
    //
    // Sum(weights(neighborhood(src_v))) are pre-computed and
    // stored in ptr_d_sum_weights_ (too expensive to check, here);
    //
    __device__ thrust::optional<thrust::tuple<vertex_t, weight_t>> operator()(vertex_t src_v,
                                                                              real_t rnd_val) const
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
      return thrust::optional{
        thrust::make_tuple(col_indices_[prev_col_indx], values_[prev_col_indx])};
    }

   private:
    edge_t const* row_offsets_;
    vertex_t const* col_indices_;
    weight_t const* values_;

    weight_t const* ptr_d_sum_weights_;
  };

  using sampler_type = sampler_t;

  biased_selector_t(raft::handle_t const& handle, graph_type const& graph, real_t tag)
    : sum_calculator_(handle, graph.get_number_of_vertices()),
      sampler_{graph.get_matrix_partition_view().get_offsets(),
               graph.get_matrix_partition_view().get_indices(),
               graph.get_matrix_partition_view().get_weights()
                 ? *(graph.get_matrix_partition_view().get_weights())
                 : static_cast<weight_t*>(nullptr),
               sum_calculator_.get_aggregated_weights().data()}
  {
    graph.apply(sum_calculator_);
  }

  sampler_t const& get_strategy(void) const { return sampler_; }

  decltype(auto) get_sum_weights(void) const { return sum_calculator_.get_aggregated_weights(); }

 private:
  visitor_aggregate_weights_t<vertex_t, edge_t, weight_t> sum_calculator_;
  sampler_t sampler_;
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
            typename selector_t,
            typename index_t,
            typename real_t,
            typename seed_t>
  void operator()(
    graph_t const& graph,                // graph being traversed
    random_walker_t const& rand_walker,  // random walker object for which traversal is driven
    selector_t const& selector,          // sampling type (uniform, biased, etc.)
    seed_t seed0,                        // initial seed value
    device_vec_t<typename graph_t::vertex_type>& d_coalesced_v,  // crt coalesced vertex set
    device_vec_t<typename graph_t::weight_type>& d_coalesced_w,  // crt coalesced weight set
    device_vec_t<index_t>& d_paths_sz,                           // crt paths sizes
    device_vec_t<typename graph_t::edge_type>&
      d_crt_out_degs,                // crt out-degs for current set of vertices
    device_vec_t<real_t>& d_random,  // crt set of random real values
    device_vec_t<typename graph_t::vertex_type>&
      d_col_indx)  // crt col col indices to be used for retrieving next step
    const
  {
    auto const& handle = rand_walker.get_handle();

    // start from 1, as 0-th was initialized above:
    //
    for (decltype(max_depth_) step_indx = 1; step_indx < max_depth_; ++step_indx) {
      // take one-step in-sync for each path in parallel:
      //
      rand_walker.step(graph,
                       selector,
                       seed0 + static_cast<seed_t>(step_indx),
                       d_coalesced_v,
                       d_coalesced_w,
                       d_paths_sz,
                       d_crt_out_degs,
                       d_random,
                       d_col_indx);

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
            typename selector_t,
            typename index_t,
            typename real_t,
            typename seed_t>
  void operator()(
    graph_t const& graph,                // graph being traversed
    random_walker_t const& rand_walker,  // random walker object for which traversal is driven
    selector_t const& selector,          // sampling type (uniform, biased, etc.)
    seed_t seed0,                        // initial seed value
    device_vec_t<typename graph_t::vertex_type>& d_coalesced_v,  // crt coalesced vertex set
    device_vec_t<typename graph_t::weight_type>& d_coalesced_w,  // crt coalesced weight set
    device_vec_t<index_t>& d_paths_sz,                           // crt paths sizes
    device_vec_t<typename graph_t::edge_type>&
      d_crt_out_degs,                // ignored: out-degs for the current set of vertices
    device_vec_t<real_t>& d_random,  // _entire_ set of random real values
    device_vec_t<typename graph_t::vertex_type>&
      d_col_indx)  // ignored: crt col indices to be used for retrieving next step
                   // (Note: coalesced set updated on-the-go)
    const
  {
    using vertex_t        = typename graph_t::vertex_type;
    using edge_t          = typename graph_t::edge_type;
    using weight_t        = typename graph_t::weight_type;
    using random_engine_t = typename random_walker_t::rnd_engine_t;
    using sampler_t       = typename selector_t::sampler_type;

    auto const& handle = rand_walker.get_handle();
    auto* ptr_d_random = raw_ptr(d_random);

    random_engine_t::generate_random(handle, ptr_d_random, d_random.size(), seed0);

    auto* ptr_d_sizes = raw_ptr(d_paths_sz);

    // next step sampler functor:
    //
    sampler_t const& sampler = selector.get_strategy();

    // start from 1, as 0-th was initialized above:
    //
    thrust::for_each(rmm::exec_policy(handle.get_stream_view()),
                     thrust::make_counting_iterator<index_t>(0),
                     thrust::make_counting_iterator<index_t>(num_paths_),
                     [max_depth       = max_depth_,
                      ptr_coalesced_v = raw_ptr(d_coalesced_v),
                      ptr_coalesced_w = raw_ptr(d_coalesced_w),
                      ptr_d_random,
                      ptr_d_sizes,
                      sampler] __device__(auto path_index) {
                       auto chunk_offset   = path_index * max_depth;
                       vertex_t src_vertex = ptr_coalesced_v[chunk_offset];

                       for (index_t step_indx = 1; step_indx < max_depth; ++step_indx) {
                         // indexing into coalesced arrays of size num_paths x (max_depth -1):
                         // (d_random, d_coalesced_w)
                         //
                         auto stepping_index = chunk_offset - path_index + step_indx - 1;

                         auto real_rnd_indx = ptr_d_random[stepping_index];

                         auto opt_tpl_vn_wn = sampler(src_vertex, real_rnd_indx);
                         if (!opt_tpl_vn_wn.has_value()) break;

                         src_vertex      = thrust::get<0>(*opt_tpl_vn_wn);
                         auto crt_weight = thrust::get<1>(*opt_tpl_vn_wn);

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
}  // namespace cugraph
