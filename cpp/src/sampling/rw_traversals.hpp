/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/api_helpers.hpp>
#include <cugraph/graph.hpp>

#include <utilities/graph_utils.cuh>

#include <cub/cub.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <ctime>
#include <future>
#include <thread>

namespace cugraph {
namespace detail {
namespace original {

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

// seeding policy: time (clock) dependent,
// to avoid RW calls repeating same random data:
//
template <typename seed_t>
struct clock_seeding_t {
  clock_seeding_t(void) = default;

  seed_t operator()(void) { return static_cast<seed_t>(std::time(nullptr)); }
};

// seeding policy: fixed for debug/testing repro
//
template <typename seed_t>
struct fixed_seeding_t {
  // purposely no default cnstr.

  fixed_seeding_t(seed_t seed) : seed_(seed) {}
  seed_t operator()(void) { return seed_; }

 private:
  seed_t seed_;
};

// Uniform RW selector logic:
//
template <typename vertex_t, typename edge_t, typename weight_t, typename real_t>
struct uniform_selector_t {
  struct sampler_t {
    sampler_t(edge_t const* ro,
              vertex_t const* ci,
              weight_t const* w,
              edge_t const* ptr_d_cache_out_degs)
      : row_offsets_(ro), col_indices_(ci), values_(w), ptr_d_cache_out_degs_(ptr_d_cache_out_degs)
    {
    }

    __device__ thrust::optional<thrust::tuple<vertex_t, weight_t>> operator()(
      vertex_t src_v,
      real_t rnd_val,
      vertex_t = 0 /* not used*/,
      edge_t   = 0 /* not used*/,
      bool     = false /* not used*/) const
    {
      auto crt_out_deg = ptr_d_cache_out_degs_[src_v];
      if (crt_out_deg == 0) return thrust::nullopt;  // src_v is a sink

      vertex_t v_indx =
        static_cast<vertex_t>(rnd_val >= 1.0 ? crt_out_deg - 1 : rnd_val * crt_out_deg);
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

  uniform_selector_t(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     real_t tag)
    : d_cache_out_degs_(graph_view.compute_out_degrees(handle)),
      sampler_{
        graph_view.local_edge_partition_view().offsets().data(),
        graph_view.local_edge_partition_view().indices().data(),
        edge_weight_view ? (*edge_weight_view).value_firsts()[0] : static_cast<weight_t*>(nullptr),
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

// Biased RW selection logic:
//
template <typename vertex_t, typename edge_t, typename weight_t, typename real_t>
struct biased_selector_t {
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
    __device__ thrust::optional<thrust::tuple<vertex_t, weight_t>> operator()(
      vertex_t src_v,
      real_t rnd_val,
      vertex_t = 0 /* not used*/,
      edge_t   = 0 /* not used*/,
      bool     = false /* not used*/) const
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

  biased_selector_t(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
                    edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
                    real_t tag,
                    weight_t const* out_weight_sums)
    : sampler_{graph_view.local_edge_partition_view().offsets().data(),
               graph_view.local_edge_partition_view().indices().data(),
               edge_weight_view.value_firsts()[0],
               out_weight_sums}
  {
  }

  sampler_t const& get_strategy(void) const { return sampler_; }

 private:
  sampler_t sampler_;
};

// node2vec RW selection logic:
// uses biased selector on scaled weights,
// to be computed (and possibly cached) according to
// `node2vec` logic (see `get_alpha()`);
// works on unweighted graphs (for which unscalled weights are 1.0);
//
// TODO: need to decide logic on very 1st step of traversal
//       (which has no `prev_v` vertex);
//
template <typename vertex_t, typename edge_t, typename weight_t, typename real_t>
struct node2vec_selector_t {
  struct sampler_t {
    sampler_t(edge_t const* ro,
              vertex_t const* ci,
              weight_t const* w,
              weight_t p,
              weight_t q,
              vertex_t max_degree,
              edge_t num_paths,
              weight_t* ptr_alpha)
      : row_offsets_(ro),
        col_indices_(ci),
        values_(w),
        p_(p),
        q_(q),
        coalesced_alpha_{
          (max_degree > 0) && (num_paths > 0) && (ptr_alpha != nullptr)
            ? thrust::optional<thrust::tuple<vertex_t, edge_t, weight_t*>>{thrust::make_tuple(
                max_degree, num_paths, ptr_alpha)}
            : thrust::nullopt}
    {
    }

    // node2vec alpha scalling logic:
    // pre-condition: assume column_indices[] is seg-sorted;
    // (each row has column_indices[] sorted)
    //
    __device__ weight_t get_alpha(vertex_t prev_v, vertex_t src_v, vertex_t next_v) const
    {
      if (next_v == prev_v) {
        return 1.0 / p_;
      } else {
        // binary-search `next_v` in the adj(prev_v)
        //
        auto prev_indx_begin = row_offsets_[prev_v];
        auto prev_indx_end   = row_offsets_[prev_v + 1];

        auto found_next_from_prev = thrust::binary_search(
          thrust::seq, col_indices_ + prev_indx_begin, col_indices_ + prev_indx_end, next_v);

        if (found_next_from_prev) {
          return 1;
        } else {
          return 1.0 / q_;
        }
      }
    }

    __device__ thrust::optional<thrust::tuple<vertex_t, weight_t>> operator()(
      vertex_t src_v, real_t rnd_val, vertex_t prev_v, edge_t path_index, bool start_path) const
    {
      auto const offset_indx_begin = row_offsets_[src_v];
      auto const offset_indx_end   = row_offsets_[src_v + 1];

      weight_t sum_scaled_weights{0};
      auto offset_indx = offset_indx_begin;

      if (offset_indx_begin == offset_indx_end) return thrust::nullopt;  // src_v is a sink

      // for 1st vertex in path just use biased random selection:
      //
      if (start_path) {  // `src_v` is starting vertex in path
        for (; offset_indx < offset_indx_end; ++offset_indx) {
          weight_t crt_weight = (values_ == nullptr ? weight_t{1} : values_[offset_indx]);

          sum_scaled_weights += crt_weight;
        }

        weight_t run_sum_w{0};
        auto rnd_sum_weights  = rnd_val * sum_scaled_weights;
        offset_indx           = offset_indx_begin;
        auto prev_offset_indx = offset_indx;

        // biased sampling selection loop:
        // (Note: re-compute `scaled_weight`, since no cache is available);
        //
        for (; offset_indx < offset_indx_end; ++offset_indx) {
          if (rnd_sum_weights < run_sum_w) break;

          weight_t crt_weight = (values_ == nullptr ? weight_t{1} : values_[offset_indx]);
          run_sum_w += crt_weight;
          prev_offset_indx = offset_indx;
        }
        return thrust::optional{
          thrust::make_tuple(col_indices_[prev_offset_indx],
                             values_ == nullptr ? weight_t{1} : values_[prev_offset_indx])};
      }

      // cached solution, for increased performance, but memory expensive:
      //
      if (coalesced_alpha_.has_value()) {
        auto&& tpl = *coalesced_alpha_;

        auto max_out_deg               = thrust::get<0>(tpl);
        auto num_paths                 = thrust::get<1>(tpl);
        weight_t* ptr_d_scaled_weights = thrust::get<2>(tpl);

        // sum-scaled-weights reduction loop:
        //
        auto const start_alpha_offset = max_out_deg * path_index;
        for (vertex_t nghbr_indx = 0; offset_indx < offset_indx_end; ++offset_indx, ++nghbr_indx) {
          auto crt_alpha      = get_alpha(prev_v, src_v, col_indices_[offset_indx]);
          weight_t crt_weight = (values_ == nullptr ? weight_t{1} : values_[offset_indx]);
          auto scaled_weight  = crt_weight * crt_alpha;

          // caching is available, hence cache the alpha's for next step
          // (the actual sampling step);
          //
          ptr_d_scaled_weights[start_alpha_offset + nghbr_indx] = scaled_weight;

          sum_scaled_weights += scaled_weight;
        }

        weight_t run_sum_w{0};
        auto rnd_sum_weights  = rnd_val * sum_scaled_weights;
        offset_indx           = offset_indx_begin;
        auto prev_offset_indx = offset_indx;

        // biased sampling selection loop:
        //
        for (vertex_t nghbr_indx = 0; offset_indx < offset_indx_end; ++offset_indx, ++nghbr_indx) {
          if (rnd_sum_weights < run_sum_w) break;

          run_sum_w += ptr_d_scaled_weights[start_alpha_offset + nghbr_indx];
          prev_offset_indx = offset_indx;
        }
        return thrust::optional{
          thrust::make_tuple(col_indices_[prev_offset_indx],
                             values_ == nullptr ? weight_t{1} : values_[prev_offset_indx])};

      } else {  // uncached solution, with much lower memory footprint but not as efficient

        for (; offset_indx < offset_indx_end; ++offset_indx) {
          auto crt_alpha = get_alpha(prev_v, src_v, col_indices_[offset_indx]);

          weight_t crt_weight = (values_ == nullptr ? weight_t{1} : values_[offset_indx]);

          auto scaled_weight = crt_weight * crt_alpha;
          sum_scaled_weights += scaled_weight;
        }

        weight_t run_sum_w{0};
        auto rnd_sum_weights  = rnd_val * sum_scaled_weights;
        offset_indx           = offset_indx_begin;
        auto prev_offset_indx = offset_indx;

        // biased sampling selection loop:
        // (Note: re-compute `scaled_weight`, since no cache is available);
        //
        for (; offset_indx < offset_indx_end; ++offset_indx) {
          if (rnd_sum_weights < run_sum_w) break;

          auto crt_alpha      = get_alpha(prev_v, src_v, col_indices_[offset_indx]);
          weight_t crt_weight = (values_ == nullptr ? weight_t{1} : values_[offset_indx]);
          auto scaled_weight  = crt_weight * crt_alpha;

          run_sum_w += scaled_weight;
          prev_offset_indx = offset_indx;
        }
        return thrust::optional{
          thrust::make_tuple(col_indices_[prev_offset_indx],
                             values_ == nullptr ? weight_t{1} : values_[prev_offset_indx])};
      }
    }

    decltype(auto) get_alpha_buffer(void) const { return coalesced_alpha_; }

   private:
    edge_t const* row_offsets_;
    vertex_t const* col_indices_;
    weight_t const* values_;

    weight_t const p_;
    weight_t const q_;

    // alpha scaling coalesced buffer (per path):
    // (use as cache since the per-path alpha-buffer
    //  is used twice for each node transition:
    //  (1) for computing sum_scaled weights;
    //  (2) for using scaled_weights for the biased next vertex selection)
    // this is information related to a scratchpad buffer, used as cache, hence mutable;
    // (necessary, because get_strategy() is const)
    //
    mutable thrust::optional<thrust::tuple<vertex_t, edge_t, weight_t*>>
      coalesced_alpha_;  // tuple<max_vertex_degree,
                         // num_paths, alpha_buffer[max_vertex_degree*num_paths]>
  };

  using sampler_type = sampler_t;

  node2vec_selector_t(raft::handle_t const& handle,
                      graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
                      std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                      real_t tag,
                      weight_t p,
                      weight_t q,
                      edge_t num_paths = 0)
    : max_out_degree_(num_paths > 0 ? graph_view.compute_max_out_degree(handle) : 0),
      d_coalesced_alpha_{max_out_degree_ * num_paths, handle.get_stream()},
      sampler_{
        graph_view.local_edge_partition_view().offsets().data(),
        graph_view.local_edge_partition_view().indices().data(),
        edge_weight_view ? (*edge_weight_view).value_firsts()[0] : static_cast<weight_t*>(nullptr),
        p,
        q,
        static_cast<vertex_t>(max_out_degree_),
        num_paths,
        raw_ptr(d_coalesced_alpha_)}
  {
  }

  sampler_t const& get_strategy(void) const { return sampler_; }

  device_vec_t<weight_t> const& get_alpha_cache(void) const { return d_coalesced_alpha_; }

 private:
  size_t max_out_degree_{0};

  // alpha scaling coalesced buffer (per path):
  // (use as cache since the per-path alpha-buffer
  //  is used twice for each node transition:
  //  (1) for computing sum_scaled weights;
  //  (2) for using scaled_weights for the biased next vertex selection)
  //
  device_vec_t<weight_t> d_coalesced_alpha_;
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

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename random_walker_t,
            typename selector_t,
            typename index_t,
            typename real_t,
            typename seed_t>
  void operator()(
    graph_view_t<vertex_t, edge_t, false, false> const& graph_view,  // graph being traversed
    std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    random_walker_t const& rand_walker,     // random walker object for which traversal is driven
    selector_t const& selector,             // sampling type (uniform, biased, etc.)
    seed_t seed0,                           // initial seed value
    device_vec_t<vertex_t>& d_coalesced_v,  // crt coalesced vertex set
    device_vec_t<weight_t>& d_coalesced_w,  // crt coalesced weight set
    device_vec_t<index_t>& d_paths_sz,      // crt paths sizes
    device_vec_t<edge_t>& d_crt_out_degs,   // crt out-degs for current set of vertices
    device_vec_t<real_t>& d_random,         // crt set of random real values
    device_vec_t<vertex_t>& d_col_indx)  // crt col col indices to be used for retrieving next step
    const
  {
    auto const& handle = rand_walker.get_handle();

    // start from 1, as 0-th was initialized above:
    //
    for (decltype(max_depth_) step_indx = 1; step_indx < max_depth_; ++step_indx) {
      // take one-step in-sync for each path in parallel:
      //
      rand_walker.step(graph_view,
                       edge_weight_view,
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

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename random_walker_t,
            typename selector_t,
            typename index_t,
            typename real_t,
            typename seed_t>
  void operator()(
    graph_view_t<vertex_t, edge_t, false, false> const& graph_view,  // graph being traversed
    std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    random_walker_t const& rand_walker,     // random walker object for which traversal is driven
    selector_t const& selector,             // sampling type (uniform, biased, etc.)
    seed_t seed0,                           // initial seed value
    device_vec_t<vertex_t>& d_coalesced_v,  // crt coalesced vertex set
    device_vec_t<weight_t>& d_coalesced_w,  // crt coalesced weight set
    device_vec_t<index_t>& d_paths_sz,      // crt paths sizes
    device_vec_t<edge_t>& d_crt_out_degs,   // ignored: out-degs for the current set of vertices
    device_vec_t<real_t>& d_random,         // _entire_ set of random real values
    device_vec_t<vertex_t>& d_col_indx)  // ignored: crt col indices to be used for retrieving next
                                         // step (Note: coalesced set updated on-the-go)
    const
  {
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
    thrust::for_each(handle.get_thrust_policy(),
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
                       auto prev_v         = src_vertex;
                       bool start_path     = true;

                       for (index_t step_indx = 1; step_indx < max_depth; ++step_indx) {
                         // indexing into coalesced arrays of size num_paths x (max_depth -1):
                         // (d_random, d_coalesced_w)
                         //
                         auto stepping_index = chunk_offset - path_index + step_indx - 1;

                         auto real_rnd_indx = ptr_d_random[stepping_index];

                         auto opt_tpl_vn_wn =
                           sampler(src_vertex, real_rnd_indx, prev_v, path_index, start_path);
                         if (!opt_tpl_vn_wn.has_value()) break;

                         prev_v     = src_vertex;
                         start_path = false;

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
};

}  // namespace original
}  // namespace detail
}  // namespace cugraph
