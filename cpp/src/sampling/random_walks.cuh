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

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>

#include <utilities/graph_utils.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>

#include <cassert>
#include <cstdlib>  // FIXME: requirement for temporary std::getenv()
#include <limits>
//
#include <optional>
#include <tuple>
#include <type_traits>

#include "rw_traversals.hpp"

namespace cugraph {

namespace detail {

// raft random generator:
// (using upper-bound cached "map"
//  giving out_deg(v) for each v in [0, |V|);
//  and a pre-generated vector of float random values
//  in [0,1] to be brought into [0, d_ub[v]))
//
template <typename vertex_t,
          typename edge_t,
          typename seed_t  = uint64_t,
          typename real_t  = float,
          typename index_t = edge_t>
struct rrandom_gen_t {
  using seed_type = seed_t;
  using real_type = real_t;

  // cnstr. version that provides step-wise in-place
  // rnd generation:
  //
  rrandom_gen_t(raft::handle_t const& handle,
                index_t num_paths,
                original::device_vec_t<real_t>& d_random,  // scratch-pad, non-coalesced
                seed_t seed = seed_t{})
    : handle_(handle),
      seed_(seed),
      num_paths_(num_paths),
      d_ptr_random_(original::raw_ptr(d_random))
  {
    auto rnd_sz = d_random.size();

    CUGRAPH_EXPECTS(rnd_sz >= static_cast<decltype(rnd_sz)>(num_paths),
                    "Un-allocated random buffer.");

    // done in constructor;
    // this must be done at each step,
    // but this object is constructed at each step;
    //
    generate_random(handle, d_ptr_random_, num_paths, seed_);
  }

  // cnstr. version for the case when the
  // random vector is provided by the caller:
  //
  rrandom_gen_t(raft::handle_t const& handle,
                index_t num_paths,
                real_t* ptr_d_rnd,  // supplied
                seed_t seed = seed_t{})
    : handle_(handle), seed_(seed), num_paths_(num_paths), d_ptr_random_(ptr_d_rnd)
  {
  }

  // in place:
  // for each v in [0, num_paths) {
  // if out_deg(v) > 0
  //   d_col_indx[v] = random index in [0, out_deg(v))
  //}
  // d_crt_out_deg is non-coalesced;
  //
  void generate_col_indices(original::device_vec_t<edge_t> const& d_crt_out_deg,
                            original::device_vec_t<vertex_t>& d_col_indx) const
  {
    auto const* d_ptr_out_degs = d_crt_out_deg.data();
    thrust::transform_if(
      handle_.get_thrust_policy(),
      d_ptr_random_,
      d_ptr_random_ + num_paths_,  // input1
      d_ptr_out_degs,              // input2
      d_ptr_out_degs,              // also stencil
      d_col_indx.begin(),
      [] __device__(real_t rnd_val, edge_t crt_out_deg) {
        vertex_t v_indx =
          static_cast<vertex_t>(rnd_val >= 1.0 ? crt_out_deg - 1 : rnd_val * crt_out_deg);
        return (v_indx >= crt_out_deg ? crt_out_deg - 1 : v_indx);
      },
      [] __device__(auto crt_out_deg) { return crt_out_deg > 0; });
  }

  // abstracts away the random values generation:
  //
  static void generate_random(raft::handle_t const& handle, real_t* p_d_rnd, size_t sz, seed_t seed)
  {
    cugraph::detail::uniform_random_fill(
      handle.get_stream(), p_d_rnd, sz, real_t{0.0}, real_t{1.0}, seed);
  }

 private:
  raft::handle_t const& handle_;
  index_t num_paths_;
  real_t* d_ptr_random_;  // device buffer with real random values; size = num_paths_
  seed_t seed_;           // seed to be used for current batch
};

template <typename vertex_t, typename edge_t, typename weight_t, typename index_t>
struct col_indx_extract_t {
  col_indx_extract_t(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     edge_t* p_d_crt_out_degs,
                     index_t* p_d_sizes,
                     index_t num_paths,
                     index_t max_depth)
    : handle_(handle),
      col_indices_(graph_view.local_edge_partition_view().indices().data()),
      row_offsets_(graph_view.local_edge_partition_view().offsets().data()),
      values_(edge_weight_view
                ? std::optional<weight_t const*>{(*edge_weight_view).value_firsts()[0]}
                : std::nullopt),
      out_degs_(p_d_crt_out_degs),
      sizes_(p_d_sizes),
      num_paths_(num_paths),
      max_depth_(max_depth)
  {
  }

  // in-place extractor of next set of vertices and weights,
  // (d_v_next_vertices, d_v_next_weights),
  // given start set of vertices. d_v_src_vertices,
  // and corresponding column index set, d_v_col_indx:
  //
  // for each indx in [0, num_paths){
  //   v_indx = d_v_src_vertices[indx*max_depth + d_sizes[indx] - 1];
  //   if( out_degs_[v_indx] > 0 ) {
  //      start_row = row_offsets_[v_indx];
  //      delta = d_v_col_indx[indx];
  //      d_v_next_vertices[indx] = col_indices_[start_row + delta];
  // }
  // (use tranform_if() with transform iterator)
  //
  void operator()(
    original::device_vec_t<vertex_t> const& d_coalesced_src_v,  // in: coalesced vector of vertices
    original::device_vec_t<vertex_t> const&
      d_v_col_indx,  // in: column indices, given by stepper's random engine
    original::device_vec_t<vertex_t>&
      d_v_next_vertices,  // out: set of destination vertices, for next step
    original::device_vec_t<weight_t>&
      d_v_next_weights)  // out: set of weights between src and destination vertices, for next step
    const
  {
    thrust::transform_if(
      handle_.get_thrust_policy(),
      thrust::make_counting_iterator<index_t>(0),
      thrust::make_counting_iterator<index_t>(num_paths_),  // input1
      d_v_col_indx.begin(),                                 // input2
      out_degs_,                                            // stencil
      thrust::make_zip_iterator(
        thrust::make_tuple(d_v_next_vertices.begin(), d_v_next_weights.begin())),  // output
      [max_depth         = max_depth_,
       ptr_d_sizes       = sizes_,
       ptr_d_coalesced_v = original::raw_const_ptr(d_coalesced_src_v),
       row_offsets       = row_offsets_,
       col_indices       = col_indices_,
       values            = values_ ? thrust::optional<weight_t const*>{*values_}
                                   : thrust::nullopt] __device__(auto indx, auto col_indx) {
        auto delta     = ptr_d_sizes[indx] - 1;
        auto v_indx    = ptr_d_coalesced_v[indx * max_depth + delta];
        auto start_row = row_offsets[v_indx];

        auto weight_value = (values ? (*values)[start_row + col_indx]
                                    : weight_t{1});  // account for un-weighted graphs
        return thrust::make_tuple(col_indices[start_row + col_indx], weight_value);
      },
      [] __device__(auto crt_out_deg) { return crt_out_deg > 0; });
  }

  // Version with selector (sampling strategy):
  //
  template <typename selector_t, typename real_t>
  void operator()(
    selector_t const& selector,
    original::device_vec_t<real_t> const& d_rnd_val,  // in: random values, one per path
    original::device_vec_t<vertex_t>& d_coalesced_v,  // out: set of coalesced vertices
    original::device_vec_t<weight_t>& d_coalesced_w,  // out: set of coalesced weights
    real_t tag)  // otherwise. ambiguity with the other operator()
  {
    thrust::for_each(handle_.get_thrust_policy(),
                     thrust::make_counting_iterator<index_t>(0),
                     thrust::make_counting_iterator<index_t>(num_paths_),  // input1
                     [max_depth        = max_depth_,
                      row_offsets      = row_offsets_,
                      ptr_coalesced_v  = original::raw_ptr(d_coalesced_v),
                      ptr_coalesced_w  = original::raw_ptr(d_coalesced_w),
                      ptr_d_random     = original::raw_const_ptr(d_rnd_val),
                      ptr_d_sizes      = sizes_,
                      ptr_crt_out_degs = out_degs_,
                      sampler = selector.get_strategy()] __device__(index_t path_indx) mutable {
                       auto chunk_offset = path_indx * max_depth;
                       auto delta        = ptr_d_sizes[path_indx] - 1;
                       auto start_v_pos  = chunk_offset + delta;
                       auto start_w_pos  = chunk_offset - path_indx + delta;

                       auto src_v   = ptr_coalesced_v[start_v_pos];
                       auto rnd_val = ptr_d_random[path_indx];

                       // `node2vec` info:
                       //
                       bool start_path = true;
                       auto prev_v     = src_v;
                       if (delta > 0) {
                         start_path = false;
                         prev_v     = ptr_coalesced_v[start_v_pos - 1];
                       }

                       auto opt_tpl_vn_wn = sampler(src_v, rnd_val, prev_v, path_indx, start_path);

                       if (opt_tpl_vn_wn.has_value()) {
                         auto src_vertex = thrust::get<0>(*opt_tpl_vn_wn);
                         auto crt_weight = thrust::get<1>(*opt_tpl_vn_wn);

                         ptr_coalesced_v[start_v_pos + 1] = src_vertex;
                         ptr_coalesced_w[start_w_pos]     = crt_weight;

                         ptr_d_sizes[path_indx]++;
                         ptr_crt_out_degs[path_indx] =
                           row_offsets[src_vertex + 1] - row_offsets[src_vertex];
                       } else {
                         ptr_crt_out_degs[path_indx] = 0;
                       }
                     });
  }

 private:
  raft::handle_t const& handle_;
  vertex_t const* col_indices_;
  edge_t const* row_offsets_;
  std::optional<weight_t const*> values_;

  edge_t* out_degs_;
  index_t* sizes_;
  index_t num_paths_;
  index_t max_depth_;
};

/**
 * @brief Class abstracting the RW initialization, stepping, and stopping functionality
 *        The outline of the algorithm is as follows:
 *
 *        (1) vertex sets are coalesced into d_coalesced_v,
 *            weight sets are coalesced into d_coalesced_w;
 *            i.e., the 2 coalesced vectors are allocated to
 *            num_paths * max_depth, and num_paths * (max_depth -1), respectively
 *            (since each path has a number of edges equal one
 *             less than the number of vertices);
 *            d_coalesced_v is initialized for each i*max_depth entry
 *            (i=0,,,,num_paths-1) to the corresponding starting vertices;
 *        (2) d_sizes maintains the current size is for each path;
 *            Note that a path may end prematurely if it reaches a sink vertex;
 *        (3) d_crt_out_degs maintains the out-degree of each of the latest
 *            vertices in the path; i.e., if N(v) := set of destination
 *            vertices from v, then this vector stores |N(v)|
 *            for last v in each path; i.e.,
 *            d_crt_out_degs[i] =
 *              out-degree( d_coalesced_v[i*max_depth + d_sizes[i]-1] ),
 *            for i in {0,..., num_paths-1};
 *        (4) a set of num_paths floating point numbers between [0,1]
 *            are generated at each step; then they get translated into
 *            _indices_ k in {0,...d_crt_out_degs[i]-1};
 *        (5) the next vertex v is then picked as the k-th out-neighbor:
 *            next(v) = N(v)[k];
 *        (6) d_sizes are incremented accordingly; i.e., for those paths whose
 *            corresponding last vertex has out-degree > 0;
 *        (7) then next(v) and corresponding weight of (v, next(v)) are stored
 *            at appropriate location in their corresponding coalesced vectors;
 *        (8) the client of this class (the random_walks() function) then repeats
 *            this process max_depth times or until all paths
 *            have reached sinks; i.e., d_crt_out_degs = {0, 0,...,0},
 *            whichever comes first;
 *        (9) in the end some post-processing is done (stop()) to remove
 *            unused entries from the 2 coalesced vectors;
 *        (10) the triplet made of the 2 coalesced vectors and d_sizes is then returned;
 *
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename random_engine_t = rrandom_gen_t<vertex_t, edge_t>,
          typename index_t         = edge_t>
struct random_walker_t {
  using seed_t       = typename random_engine_t::seed_type;
  using real_t       = typename random_engine_t::real_type;
  using rnd_engine_t = random_engine_t;

  random_walker_t(raft::handle_t const& handle,
                  vertex_t num_vertices,
                  index_t num_paths,
                  index_t max_depth,
                  vertex_t v_padding_val = 0,
                  weight_t w_padding_val = 0)
    : handle_(handle),
      num_paths_(num_paths),
      max_depth_(max_depth),
      vertex_padding_value_(v_padding_val != 0 ? v_padding_val : num_vertices),
      weight_padding_value_(w_padding_val)
  {
  }

  // for each i in [0..num_paths_) {
  //   d_paths_v_set[i*max_depth] = d_src_init_v[i];
  //
  void start(original::device_const_vector_view<vertex_t, index_t>& d_src_init_v,  // in: start set
             original::device_vec_t<vertex_t>& d_paths_v_set,  // out: coalesced v
             original::device_vec_t<index_t>& d_sizes) const   // out: init sizes to {1,...}
  {
    // intialize path sizes to 1, as they contain at least one vertex each:
    // the initial set: d_src_init_v;
    //
    thrust::copy_n(handle_.get_thrust_policy(),
                   thrust::make_constant_iterator<index_t>(1),
                   num_paths_,
                   d_sizes.begin());

    // scatter d_src_init_v to coalesced vertex vector:
    //
    auto dlambda = [stride = max_depth_] __device__(auto indx) { return indx * stride; };

    // use the transform iterator as map:
    //
    auto map_it_begin =
      thrust::make_transform_iterator(thrust::make_counting_iterator<index_t>(0), dlambda);

    thrust::scatter(handle_.get_thrust_policy(),
                    d_src_init_v.begin(),
                    d_src_init_v.end(),
                    map_it_begin,
                    d_paths_v_set.begin());
  }

  // overload for start() with device_uvector d_v_start
  // (handy for testing)
  //
  void start(original::device_vec_t<vertex_t> const& d_start,  // in: start set
             original::device_vec_t<vertex_t>& d_paths_v_set,  // out: coalesced v
             original::device_vec_t<index_t>& d_sizes) const   // out: init sizes to {1,...}
  {
    original::device_const_vector_view<vertex_t, index_t> d_start_cview{
      d_start.data(), static_cast<index_t>(d_start.size())};

    start(d_start_cview, d_paths_v_set, d_sizes);
  }

  // in-place updates its arguments from one step to next
  // (to avoid copying); all "crt" arguments are updated at each step()
  // and passed as scratchpad space to avoid copying them
  // from one step to another
  //
  // take one step in sync for all paths that have not reached sinks:
  //
  template <typename selector_t>
  void step(
    graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    selector_t const& selector,
    seed_t seed,
    original::device_vec_t<vertex_t>& d_coalesced_v,  // crt coalesced vertex set
    original::device_vec_t<weight_t>& d_coalesced_w,  // crt coalesced weight set
    original::device_vec_t<index_t>& d_paths_sz,      // crt paths sizes
    original::device_vec_t<edge_t>& d_crt_out_degs,   // crt out-degs for current set of vertices
    original::device_vec_t<real_t>& d_random,         // crt set of random real values
    original::device_vec_t<vertex_t>&
      d_col_indx)  // crt col col indices to be used for retrieving next step
    const
  {
    // generate random destination indices:
    //
    random_engine_t rgen(handle_, num_paths_, d_random, seed);

    // dst extraction from dst indices:
    // (d_crt_out_degs to be maintained internally by col_extractor)
    //
    col_indx_extract_t<vertex_t, edge_t, weight_t, index_t> col_extractor(
      handle_,
      graph_view,
      edge_weight_view,
      original::raw_ptr(d_crt_out_degs),
      original::raw_ptr(d_paths_sz),
      num_paths_,
      max_depth_);

    // The following steps update the next entry in each path,
    // except the paths that reached sinks;
    //
    // for each indx in [0..num_paths) {
    //   v_indx = d_v_rnd_n_indx[indx];
    //
    //   -- get the `v_indx`-th out-vertex of d_v_paths_v_set[indx] vertex:
    //   -- also, note the size deltas increased by 1 in dst (d_sizes[]):
    //
    //   d_coalesced_v[indx*max_depth + d_sizes[indx]] =
    //       get_out_vertex(graph, d_coalesced_v[indx*max_depth + d_sizes[indx]-1)], v_indx);
    //   d_coalesced_w[indx*(max_depth-1) + d_sizes[indx] - 1] =
    //       get_out_edge_weight(graph, d_coalesced_v[indx*max_depth + d_sizes[indx]-1], v_indx);
    //
    // (1) generate actual vertex destinations;
    // (2) update path sizes;
    // (3) actual coalesced updates;
    //
    // performs steps (1) + (2) + (3) in one pass;
    //
    col_extractor(selector, d_random, d_coalesced_v, d_coalesced_w, real_t{0});
  }

  // returns true if all paths reached sinks:
  //
  bool all_paths_stopped(original::device_vec_t<edge_t> const& d_crt_out_degs) const
  {
    auto how_many_stopped =
      thrust::count_if(handle_.get_thrust_policy(),
                       d_crt_out_degs.begin(),
                       d_crt_out_degs.end(),
                       [] __device__(auto crt_out_deg) { return crt_out_deg == 0; });
    return (static_cast<size_t>(how_many_stopped) == d_crt_out_degs.size());
  }

  // wrap-up, post-process:
  // truncate v_set, w_set to actual space used
  //
  void stop(original::device_vec_t<vertex_t>& d_coalesced_v,       // coalesced vertex set
            original::device_vec_t<weight_t>& d_coalesced_w,       // coalesced weight set
            original::device_vec_t<index_t> const& d_sizes) const  // paths sizes
  {
    assert(max_depth_ > 1);  // else, no need to step; and no edges

    index_t const* ptr_d_sizes = d_sizes.data();

    auto predicate_v = [max_depth = max_depth_, ptr_d_sizes] __device__(auto indx) {
      auto row_indx = indx / max_depth;
      auto col_indx = indx % max_depth;

      return (col_indx >= ptr_d_sizes[row_indx]);
    };

    auto predicate_w = [max_depth = max_depth_, ptr_d_sizes] __device__(auto indx) {
      auto row_indx = indx / (max_depth - 1);
      auto col_indx = indx % (max_depth - 1);

      return (col_indx >= ptr_d_sizes[row_indx] - 1);
    };

    auto new_end_v = thrust::remove_if(handle_.get_thrust_policy(),
                                       d_coalesced_v.begin(),
                                       d_coalesced_v.end(),
                                       thrust::make_counting_iterator<index_t>(0),
                                       predicate_v);

    auto new_end_w = thrust::remove_if(handle_.get_thrust_policy(),
                                       d_coalesced_w.begin(),
                                       d_coalesced_w.end(),
                                       thrust::make_counting_iterator<index_t>(0),
                                       predicate_w);

    handle_.sync_stream();

    d_coalesced_v.resize(thrust::distance(d_coalesced_v.begin(), new_end_v), handle_.get_stream());
    d_coalesced_w.resize(thrust::distance(d_coalesced_w.begin(), new_end_w), handle_.get_stream());
  }

  // in-place non-static (needs handle_):
  // for indx in [0, nelems):
  //   gather d_result[indx] = d_src[d_coalesced[indx*stride + d_sizes[indx] -1]]
  //
  template <typename src_vec_t = vertex_t>
  void gather_from_coalesced(
    original::device_vec_t<vertex_t> const& d_coalesced,  // |gather map| = stride*nelems
    original::device_vec_t<src_vec_t> const& d_src,       // |gather input| = nelems
    original::device_vec_t<index_t> const& d_sizes,  // |paths sizes| = nelems, elems in [1, stride]
    original::device_vec_t<src_vec_t>& d_result,     // |output| = nelems
    index_t stride,        // stride = coalesce block size (typically max_depth)
    index_t nelems) const  // nelems = number of elements to gather (typically num_paths_)
  {
    vertex_t const* ptr_d_coalesced = original::raw_const_ptr(d_coalesced);
    index_t const* ptr_d_sizes      = original::raw_const_ptr(d_sizes);

    // delta = ptr_d_sizes[indx] - 1
    //
    auto dlambda = [stride, ptr_d_sizes, ptr_d_coalesced] __device__(auto indx) {
      auto delta = ptr_d_sizes[indx] - 1;
      return ptr_d_coalesced[indx * stride + delta];
    };

    // use the transform iterator as map:
    //
    auto map_it_begin =
      thrust::make_transform_iterator(thrust::make_counting_iterator<index_t>(0), dlambda);

    thrust::gather(handle_.get_thrust_policy(),
                   map_it_begin,
                   map_it_begin + nelems,
                   d_src.begin(),
                   d_result.begin());
  }

  // in-place non-static (needs handle_);
  // pre-condition: path sizes are assumed updated
  // to reflect new vertex additions;
  //
  // for indx in [0, nelems):
  //   if ( d_crt_out_degs[indx] > 0 )
  //     d_coalesced[indx*stride + (d_sizes[indx] - adjust)- 1] = d_src[indx]
  //
  // adjust := 0 for coalesced vertices; 1 for weights
  // (because |edges| = |vertices| - 1, in each path);
  //
  template <typename src_vec_t>
  void scatter_to_coalesced(
    original::device_vec_t<src_vec_t> const& d_src,  // |scatter input| = nelems
    original::device_vec_t<src_vec_t>& d_coalesced,  // |scatter input| = stride*nelems
    original::device_vec_t<edge_t> const&
      d_crt_out_degs,  // |current set of vertex out degrees| = nelems,
                       // to be used as stencil (don't scatter if 0)
    original::device_vec_t<index_t> const&
      d_sizes,  // paths sizes used to provide delta in coalesced paths;
                // pre-condition: assumed as updated to reflect new vertex additions;
                // also, this is the number of _vertices_ in each path;
    // hence for scattering weights this needs to be adjusted; hence the `adjust` parameter
    index_t
      stride,  // stride = coalesce block size (max_depth for vertices; max_depth-1 for weights)
    index_t nelems,  // nelems = number of elements to gather (typically num_paths_)
    index_t adjust = 0)
    const  // adjusting parameter for scattering vertices (0) or weights (1); see above for more;
  {
    index_t const* ptr_d_sizes = original::raw_const_ptr(d_sizes);

    auto dlambda = [stride, adjust, ptr_d_sizes] __device__(auto indx) {
      auto delta = ptr_d_sizes[indx] - adjust - 1;
      return indx * stride + delta;
    };

    // use the transform iterator as map:
    //
    auto map_it_begin =
      thrust::make_transform_iterator(thrust::make_counting_iterator<index_t>(0), dlambda);

    thrust::scatter_if(handle_.get_thrust_policy(),
                       d_src.begin(),
                       d_src.end(),
                       map_it_begin,
                       d_crt_out_degs.begin(),
                       d_coalesced.begin(),
                       [] __device__(auto crt_out_deg) {
                         return crt_out_deg > 0;  // predicate
                       });
  }

  // updates the entries in the corresponding coalesced vector,
  // for which out_deg > 0
  //
  void scatter_vertices(original::device_vec_t<vertex_t> const& d_src,
                        original::device_vec_t<vertex_t>& d_coalesced,
                        original::device_vec_t<edge_t> const& d_crt_out_degs,
                        original::device_vec_t<index_t> const& d_sizes) const
  {
    scatter_to_coalesced(d_src, d_coalesced, d_crt_out_degs, d_sizes, max_depth_, num_paths_);
  }
  //
  void scatter_weights(original::device_vec_t<weight_t> const& d_src,
                       original::device_vec_t<weight_t>& d_coalesced,
                       original::device_vec_t<edge_t> const& d_crt_out_degs,
                       original::device_vec_t<index_t> const& d_sizes) const
  {
    scatter_to_coalesced(
      d_src, d_coalesced, d_crt_out_degs, d_sizes, max_depth_ - 1, num_paths_, 1);
  }

  // in-place update (increment) path sizes for paths
  // that have not reached a sink; i.e., for which
  // d_crt_out_degs[indx]>0:
  //
  void update_path_sizes(original::device_vec_t<edge_t> const& d_crt_out_degs,
                         original::device_vec_t<index_t>& d_sizes) const
  {
    thrust::transform_if(
      handle_.get_thrust_policy(),
      d_sizes.begin(),
      d_sizes.end(),           // input
      d_crt_out_degs.begin(),  // stencil
      d_sizes.begin(),         // output: in-place
      [] __device__(auto crt_sz) { return crt_sz + 1; },
      [] __device__(auto crt_out_deg) { return crt_out_deg > 0; });
  }

  original::device_vec_t<edge_t> get_out_degs(
    graph_view_t<vertex_t, edge_t, false, false> const& graph_view) const
  {
    return graph_view.compute_out_degrees(handle_);
  }

  vertex_t get_vertex_padding_value(void) const { return vertex_padding_value_; }

  weight_t get_weight_padding_value(void) const { return weight_padding_value_; }

  void init_padding(original::device_vec_t<vertex_t>& d_coalesced_v,
                    original::device_vec_t<weight_t>& d_coalesced_w) const
  {
    thrust::fill(handle_.get_thrust_policy(),
                 d_coalesced_v.begin(),
                 d_coalesced_v.end(),
                 vertex_padding_value_);

    thrust::fill(handle_.get_thrust_policy(),
                 d_coalesced_w.begin(),
                 d_coalesced_w.end(),
                 weight_padding_value_);
  }

  decltype(auto) get_handle(void) const { return handle_; }

 private:
  raft::handle_t const& handle_;
  index_t num_paths_;
  index_t max_depth_;
  vertex_t const vertex_padding_value_;
  weight_t const weight_padding_value_;
};

/**
 * @brief returns random walks (RW) from starting sources, where each path is of given maximum
 * length. Single-GPU specialization.
 *
 * @tparam graph_t Type of graph (view).
 * @tparam traversal_t Traversal policy. Either horizontal (faster but requires more memory) or
 * vertical. Defaults to horizontal.
 * @tparam random_engine_t Type of random engine used to generate RW.
 * @tparam seeding_policy_t Random engine seeding policy: variable or fixed (for reproducibility).
 * Defaults to variable, clock dependent.
 * @tparam index_t Indexing type. Defaults to edge_type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to generate RW on.
 * @param d_v_start Device (view) set of starting vertex indices for the RW.
 * number(paths) == d_v_start.size().
 * @param max_depth maximum length of RWs.
 * @param use_padding (optional) specifies if return uses padded format (true), or coalesced
 * (compressed) format; when padding is used the output is a matrix of vertex paths and a matrix of
 * edges paths (weights); in this case the matrices are stored in row major order; the vertex path
 * matrix is padded with `num_vertices` values and the weight matrix is padded with `0` values;
 * @param seeder (optional) is object providing the random seeding mechanism. Defaults to local
 * clock time as initial seed.
 * @return std::tuple<device_vec_t<vertex_t>, device_vec_t<weight_t>,
 * device_vec_t<index_t>> Triplet of either padded or coalesced RW paths; in the coalesced case
 * (default), the return consists of corresponding vertex and edge weights for each, and
 * corresponding path sizes. This is meant to minimize the number of DF's to be passed to the Python
 * layer. The meaning of "coalesced" here is that a 2D array of paths of different sizes is
 * represented as a 1D contiguous array. In the padded case the return is a matrix of num_paths x
 * max_depth vertex paths; and num_paths x (max_depth-1) edge (weight) paths, with an empty array of
 * sizes. Note: if the graph is un-weighted the edge (weight) paths consists of `weight_t{1}`
 * entries;
 */
template <
  typename vertex_t,
  typename edge_t,
  typename weight_t,
  typename selector_t,
  typename traversal_t      = original::horizontal_traversal_t,
  typename random_engine_t  = rrandom_gen_t<vertex_t, edge_t>,
  typename seeding_policy_t = original::clock_seeding_t<typename random_engine_t::seed_type>,
  typename index_t          = edge_t>
std::tuple<original::device_vec_t<vertex_t>,
           original::device_vec_t<weight_t>,
           original::device_vec_t<index_t>,
           typename random_engine_t::seed_type>
random_walks_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  original::device_const_vector_view<vertex_t, index_t>& d_v_start,
  index_t max_depth,
  selector_t const& selector,
  bool use_padding        = false,
  seeding_policy_t seeder = original::clock_seeding_t<typename random_engine_t::seed_type>{})
{
  using seed_t = typename random_engine_t::seed_type;
  using real_t = typename random_engine_t::real_type;

  vertex_t num_vertices = graph_view.number_of_vertices();

  auto how_many_valid = thrust::count_if(handle.get_thrust_policy(),
                                         d_v_start.begin(),
                                         d_v_start.end(),
                                         [num_vertices] __device__(auto crt_vertex) {
                                           return (crt_vertex >= 0) && (crt_vertex < num_vertices);
                                         });

  CUGRAPH_EXPECTS(static_cast<index_t>(how_many_valid) == d_v_start.size(),
                  "Invalid set of starting vertices.");

  auto num_paths = d_v_start.size();
  auto stream    = handle.get_stream();

  random_walker_t<vertex_t, edge_t, weight_t, random_engine_t> rand_walker{
    handle,
    graph_view.number_of_vertices(),
    static_cast<index_t>(num_paths),
    static_cast<index_t>(max_depth)};

  // pre-allocate num_paths * max_depth;
  //
  original::device_vec_t<vertex_t> d_coalesced_v(num_paths * max_depth,
                                                 stream);  // coalesced vertex set
  original::device_vec_t<weight_t> d_coalesced_w(num_paths * (max_depth - 1),
                                                 stream);         // coalesced weight set
  original::device_vec_t<index_t> d_paths_sz(num_paths, stream);  // paths sizes

  // traversal policy:
  //
  traversal_t traversor(num_paths, max_depth);

  auto tmp_buff_sz = traversor.get_tmp_buff_sz();

  original::device_vec_t<edge_t> d_crt_out_degs(tmp_buff_sz, stream);  // crt vertex set out-degs
  original::device_vec_t<vertex_t> d_col_indx(tmp_buff_sz, stream);    // \in {0,..,out-deg(v)}

  // random data handling:
  //
  auto rnd_data_sz = traversor.get_random_buff_sz();
  original::device_vec_t<real_t> d_random(rnd_data_sz, stream);
  // abstracted out seed initialization:
  //
  seed_t seed0 = static_cast<seed_t>(seeder());

  // if padding used, initialize padding values:
  //
  if (use_padding) rand_walker.init_padding(d_coalesced_v, d_coalesced_w);

  // very first vertex, for each path:
  //
  rand_walker.start(d_v_start, d_coalesced_v, d_paths_sz);

  // traverse paths:
  //
  traversor(graph_view,
            edge_weight_view,
            rand_walker,
            selector,
            seed0,
            d_coalesced_v,
            d_coalesced_w,
            d_paths_sz,
            d_crt_out_degs,
            d_random,
            d_col_indx);

  // wrap-up, post-process:
  // truncate v_set, w_set to actual space used
  // unless padding is used
  //
  if (!use_padding) { rand_walker.stop(d_coalesced_v, d_coalesced_w, d_paths_sz); }

  // because device_uvector is not copy-cnstr-able:
  //
  if (!use_padding) {
    return std::make_tuple(std::move(d_coalesced_v),
                           std::move(d_coalesced_w),
                           std::move(d_paths_sz),
                           seed0);  // also return seed for repro
  } else {
    return std::make_tuple(
      std::move(d_coalesced_v),
      std::move(d_coalesced_w),
      original::device_vec_t<index_t>(0, stream),  // purposely empty size array for the padded
                                                   // case, to avoid unnecessary allocations
      seed0);                                      // also return seed for repro
  }
}

// provides conversion to (coalesced) path to COO format:
// (which in turn provides an API consistent with egonet)
//
template <typename vertex_t, typename index_t>
struct coo_convertor_t {
  coo_convertor_t(raft::handle_t const& handle, index_t num_paths)
    : handle_(handle), num_paths_(num_paths)
  {
  }

  std::tuple<original::device_vec_t<vertex_t>,
             original::device_vec_t<vertex_t>,
             original::device_vec_t<index_t>>
  operator()(original::device_const_vector_view<vertex_t>& d_coalesced_v,
             original::device_const_vector_view<index_t>& d_sizes) const
  {
    CUGRAPH_EXPECTS(static_cast<index_t>(d_sizes.size()) == num_paths_, "Invalid size vector.");

    auto tupl_fill        = fill_stencil(d_sizes);
    auto&& d_stencil      = std::move(std::get<0>(tupl_fill));
    auto total_sz_v       = std::get<1>(tupl_fill);
    auto&& d_sz_incl_scan = std::move(std::get<2>(tupl_fill));

    CUGRAPH_EXPECTS(static_cast<index_t>(d_coalesced_v.size()) == total_sz_v,
                    "Inconsistent vertex coalesced size data.");

    auto src_dst_tpl = gather_pairs(d_coalesced_v, d_stencil, total_sz_v);

    auto&& d_src = std::move(std::get<0>(src_dst_tpl));
    auto&& d_dst = std::move(std::get<1>(src_dst_tpl));

    original::device_vec_t<index_t> d_sz_w_scan(num_paths_, handle_.get_stream());

    // copy vertex path sizes that are > 1:
    // (because vertex_path_sz translates
    //  into edge_path_sz = vertex_path_sz - 1,
    //  and edge_paths_sz == 0 don't contribute
    //  anything):
    //
    auto new_end_it = thrust::copy_if(handle_.get_thrust_policy(),
                                      d_sizes.begin(),
                                      d_sizes.end(),
                                      d_sz_w_scan.begin(),
                                      [] __device__(auto sz_value) { return sz_value > 1; });

    // resize to new_end:
    //
    d_sz_w_scan.resize(thrust::distance(d_sz_w_scan.begin(), new_end_it), handle_.get_stream());

    // get paths' edge number exclusive scan
    // by transforming paths' vertex numbers that
    // are > 1, via tranaformation:
    // edge_path_sz = (vertex_path_sz-1):
    //
    thrust::transform_exclusive_scan(
      handle_.get_thrust_policy(),
      d_sz_w_scan.begin(),
      d_sz_w_scan.end(),
      d_sz_w_scan.begin(),
      [] __device__(auto sz) { return sz - 1; },
      index_t{0},
      thrust::plus<index_t>{});

    return std::make_tuple(std::move(d_src), std::move(d_dst), std::move(d_sz_w_scan));
  }

  std::tuple<original::device_vec_t<int>, index_t, original::device_vec_t<index_t>> fill_stencil(
    original::device_const_vector_view<index_t>& d_sizes) const
  {
    original::device_vec_t<index_t> d_scan(num_paths_, handle_.get_stream());
    thrust::inclusive_scan(
      handle_.get_thrust_policy(), d_sizes.begin(), d_sizes.end(), d_scan.begin());

    index_t total_sz{0};
    RAFT_CUDA_TRY(cudaMemcpy(&total_sz,
                             original::raw_ptr(d_scan) + num_paths_ - 1,
                             sizeof(index_t),
                             cudaMemcpyDeviceToHost));

    original::device_vec_t<int> d_stencil(total_sz, handle_.get_stream());

    // initialize stencil to all 1's:
    //
    thrust::copy_n(handle_.get_thrust_policy(),
                   thrust::make_constant_iterator<int>(1),
                   d_stencil.size(),
                   d_stencil.begin());

    // set to 0 entries positioned at inclusive_scan(sizes[]),
    // because those are path "breakpoints", where a path end
    // and the next one starts, hence there cannot be an edge
    // between a path ending vertex and next path starting vertex;
    //
    thrust::scatter(handle_.get_thrust_policy(),
                    thrust::make_constant_iterator(0),
                    thrust::make_constant_iterator(0) + num_paths_,
                    d_scan.begin(),
                    d_stencil.begin());

    return std::make_tuple(std::move(d_stencil), total_sz, std::move(d_scan));
  }

  std::tuple<original::device_vec_t<vertex_t>, original::device_vec_t<vertex_t>> gather_pairs(
    original::device_const_vector_view<vertex_t>& d_coalesced_v,
    original::device_vec_t<int> const& d_stencil,
    index_t total_sz_v) const
  {
    auto total_sz_w = total_sz_v - num_paths_;
    original::device_vec_t<index_t> valid_src_indx(total_sz_w, handle_.get_stream());

    // generate valid vertex src indices,
    // which is any index in {0,...,total_sz_v - 2}
    // provided the next index position; i.e., (index+1),
    // in stencil is not 0; (if it is, there's no "next"
    // or dst index, because the path has ended);
    //
    thrust::copy_if(handle_.get_thrust_policy(),
                    thrust::make_counting_iterator<index_t>(0),
                    thrust::make_counting_iterator<index_t>(total_sz_v - 1),
                    valid_src_indx.begin(),
                    [ptr_d_stencil = original::raw_const_ptr(d_stencil)] __device__(auto indx) {
                      auto dst_indx = indx + 1;
                      return ptr_d_stencil[dst_indx] == 1;
                    });

    original::device_vec_t<vertex_t> d_src_v(total_sz_w, handle_.get_stream());
    original::device_vec_t<vertex_t> d_dst_v(total_sz_w, handle_.get_stream());

    // construct pair of src[], dst[] by gathering
    // from d_coalesced_v all pairs
    // at entries (valid_src_indx, valid_src_indx+1),
    // where the set of valid_src_indx was
    // generated at the previous step;
    //
    thrust::transform(
      handle_.get_thrust_policy(),
      valid_src_indx.begin(),
      valid_src_indx.end(),
      thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin())),  // start_zip
      [ptr_d_vertex = original::raw_const_ptr(d_coalesced_v)] __device__(auto indx) {
        return thrust::make_tuple(ptr_d_vertex[indx], ptr_d_vertex[indx + 1]);
      });

    return std::make_tuple(std::move(d_src_v), std::move(d_dst_v));
  }

 private:
  raft::handle_t const& handle_;
  index_t num_paths_;
};

}  // namespace detail

/**
 * @brief returns random walks (RW) from starting sources, where each path is of given maximum
 * length. Uniform distribution is assumed for the random engine.
 *
 * @tparam graph_t Type of graph/view (typically, graph_view_t).
 * @tparam index_t Type used to store indexing and sizes.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph (view )object to generate RW on.
 * @param ptr_d_start Device pointer to set of starting vertex indices for the RW.
 * @param num_paths = number(paths).
 * @param max_depth maximum length of RWs.
 * @param use_padding (optional) specifies if return uses padded format (true), or coalesced
 * (compressed) format; when padding is used the output is a matrix of vertex paths and a matrix of
 * edges paths (weights); in this case the matrices are stored in row major order; the vertex path
 * matrix is padded with `num_vertices` values and the weight matrix is padded with `0` values;
 * @param selector_type identifier for sampling strategy: uniform, biased, etc.; defaults to
 * uniform = 0;
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>,
 * rmm::device_uvector<index_t>> Triplet of either padded or coalesced RW paths; in the coalesced
 * case (default), the return consists of corresponding vertex and edge weights for each, and
 * corresponding path sizes. This is meant to minimize the number of DF's to be passed to the Python
 * layer. The meaning of "coalesced" here is that a 2D array of paths of different sizes is
 * represented as a 1D contiguous array. In the padded case the return is a matrix of num_paths x
 * max_depth vertex paths; and num_paths x (max_depth-1) edge (weight) paths, with an empty array of
 * sizes. Note: if the graph is un-weighted the edge (weight) paths consists of `weight_t{1}`
 * entries;
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename index_t, bool multi_gpu>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>, rmm::device_uvector<index_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
               std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
               vertex_t const* ptr_d_start,
               index_t num_paths,
               index_t max_depth,
               bool use_padding,
               std::unique_ptr<sampling_params_t> sampling_strategy)
{
  using real_t = float;  // random engine type;
  // FIXME: this should not be hardcoded; at least tag-dispatched

  if constexpr (multi_gpu) { CUGRAPH_FAIL("unimplemented."); }

  // 0-copy const device view:
  //
  detail::original::device_const_vector_view<vertex_t, index_t> d_v_start{ptr_d_start, num_paths};

  // GPU memory availability:
  //
  size_t free_mem_sp_bytes{0};
  size_t total_mem_sp_bytes{0};
  cudaMemGetInfo(&free_mem_sp_bytes, &total_mem_sp_bytes);

  // GPU memory requirements:
  //
  size_t coalesced_v_count = num_paths * max_depth;
  auto coalesced_e_count   = coalesced_v_count - num_paths;
  size_t req_mem_common    = sizeof(vertex_t) * coalesced_v_count +
                          sizeof(weight_t) * coalesced_e_count +  // coalesced_v + coalesced_w
                          (sizeof(vertex_t) + sizeof(index_t)) * num_paths;  // start_v + sizes

  size_t req_mem_horizontal = req_mem_common + sizeof(real_t) * coalesced_e_count;  // + rnd_buff
  size_t req_mem_vertical =
    req_mem_common + (sizeof(edge_t) + 2 * sizeof(vertex_t) + sizeof(weight_t) + sizeof(real_t)) *
                       num_paths;  // + smaller_rnd_buff + tmp_buffs

  bool use_vertical_strategy{false};
  if (req_mem_horizontal > req_mem_vertical && req_mem_horizontal > free_mem_sp_bytes) {
    use_vertical_strategy = true;
    std::cerr
      << "WARNING: Due to GPU memory availability, slower vertical traversal will be used.\n";
  }

  int selector_type{0};
  if (sampling_strategy) selector_type = static_cast<int>(sampling_strategy->sampling_type_);

  // node2vec is only possible for weight_t being a floating-point type:
  //
  if constexpr (!std::is_floating_point_v<weight_t>) {
    CUGRAPH_EXPECTS(selector_type != static_cast<int>(sampling_strategy_t::NODE2VEC),
                    "node2vec requires floating point type for weights.");
  }

  if (use_vertical_strategy) {
    if (selector_type == static_cast<int>(sampling_strategy_t::BIASED)) {
      CUGRAPH_EXPECTS(edge_weight_view.has_value(), "biased selector requires edge weights.");
      auto out_weight_sums = compute_out_weight_sums(handle, graph_view, *edge_weight_view);
      detail::original::biased_selector_t<vertex_t, edge_t, weight_t, real_t> selector{
        handle, graph_view, *edge_weight_view, real_t{0}, out_weight_sums.data()};

      auto quad_tuple = detail::random_walks_impl<vertex_t,
                                                  edge_t,
                                                  weight_t,
                                                  decltype(selector),
                                                  detail::original::vertical_traversal_t>(
        handle, graph_view, edge_weight_view, d_v_start, max_depth, selector, use_padding);
      // ignore last element of the quad, seed,
      // since it's meant for testing / debugging, only:
      //
      return std::make_tuple(std::move(std::get<0>(quad_tuple)),
                             std::move(std::get<1>(quad_tuple)),
                             std::move(std::get<2>(quad_tuple)));
    } else if (selector_type == static_cast<int>(sampling_strategy_t::NODE2VEC)) {
      weight_t p(sampling_strategy->p_);
      weight_t q(sampling_strategy->q_);

      edge_t alpha_num_paths = sampling_strategy->use_alpha_cache_ ? num_paths : 0;

      weight_t roundoff = std::numeric_limits<weight_t>::epsilon();
      CUGRAPH_EXPECTS(p > roundoff, "node2vec p parameter is too small.");

      CUGRAPH_EXPECTS(q > roundoff, "node2vec q parameter is too small.");

      detail::original::node2vec_selector_t<vertex_t, edge_t, weight_t, real_t> selector{
        handle, graph_view, edge_weight_view, real_t{0}, p, q, alpha_num_paths};

      auto quad_tuple = detail::random_walks_impl<vertex_t,
                                                  edge_t,
                                                  weight_t,
                                                  decltype(selector),
                                                  detail::original::vertical_traversal_t>(
        handle, graph_view, edge_weight_view, d_v_start, max_depth, selector, use_padding);
      // ignore last element of the quad, seed,
      // since it's meant for testing / debugging, only:
      //
      return std::make_tuple(std::move(std::get<0>(quad_tuple)),
                             std::move(std::get<1>(quad_tuple)),
                             std::move(std::get<2>(quad_tuple)));
    } else {
      detail::original::uniform_selector_t<vertex_t, edge_t, weight_t, real_t> selector{
        handle, graph_view, edge_weight_view, real_t{0}};

      auto quad_tuple = detail::random_walks_impl<vertex_t,
                                                  edge_t,
                                                  weight_t,
                                                  decltype(selector),
                                                  detail::original::vertical_traversal_t>(
        handle, graph_view, edge_weight_view, d_v_start, max_depth, selector, use_padding);
      // ignore last element of the quad, seed,
      // since it's meant for testing / debugging, only:
      //
      return std::make_tuple(std::move(std::get<0>(quad_tuple)),
                             std::move(std::get<1>(quad_tuple)),
                             std::move(std::get<2>(quad_tuple)));
    }
  } else {  // horizontal traversal strategy
    if (selector_type == static_cast<int>(sampling_strategy_t::BIASED)) {
      CUGRAPH_EXPECTS(edge_weight_view.has_value(), "biased selector requires edge weights.");
      auto out_weight_sums = compute_out_weight_sums(handle, graph_view, *edge_weight_view);
      detail::original::biased_selector_t<vertex_t, edge_t, weight_t, real_t> selector{
        handle, graph_view, *edge_weight_view, real_t{0}, out_weight_sums.data()};

      auto quad_tuple = detail::random_walks_impl(
        handle, graph_view, edge_weight_view, d_v_start, max_depth, selector, use_padding);
      // ignore last element of the quad, seed,
      // since it's meant for testing / debugging, only:
      //
      return std::make_tuple(std::move(std::get<0>(quad_tuple)),
                             std::move(std::get<1>(quad_tuple)),
                             std::move(std::get<2>(quad_tuple)));
    } else if (selector_type == static_cast<int>(sampling_strategy_t::NODE2VEC)) {
      weight_t p(sampling_strategy->p_);
      weight_t q(sampling_strategy->q_);

      edge_t alpha_num_paths = sampling_strategy->use_alpha_cache_ ? num_paths : 0;

      weight_t roundoff = std::numeric_limits<weight_t>::epsilon();
      CUGRAPH_EXPECTS(p > roundoff, "node2vec p parameter is too small.");

      CUGRAPH_EXPECTS(q > roundoff, "node2vec q parameter is too small.");

      detail::original::node2vec_selector_t<vertex_t, edge_t, weight_t, real_t> selector{
        handle, graph_view, edge_weight_view, real_t{0}, p, q, alpha_num_paths};

      auto quad_tuple = detail::random_walks_impl(
        handle, graph_view, edge_weight_view, d_v_start, max_depth, selector, use_padding);
      // ignore last element of the quad, seed,
      // since it's meant for testing / debugging, only:
      //
      return std::make_tuple(std::move(std::get<0>(quad_tuple)),
                             std::move(std::get<1>(quad_tuple)),
                             std::move(std::get<2>(quad_tuple)));
    } else {
      detail::original::uniform_selector_t<vertex_t, edge_t, weight_t, real_t> selector{
        handle, graph_view, edge_weight_view, real_t{0}};

      auto quad_tuple = detail::random_walks_impl(
        handle, graph_view, edge_weight_view, d_v_start, max_depth, selector, use_padding);
      // ignore last element of the quad, seed,
      // since it's meant for testing / debugging, only:
      //
      return std::make_tuple(std::move(std::get<0>(quad_tuple)),
                             std::move(std::get<1>(quad_tuple)),
                             std::move(std::get<2>(quad_tuple)));
    }
  }
}

/**
 * @brief returns the COO format (src_vector, dst_vector) from the random walks (RW)
 * paths.
 *
 * @tparam vertex_t Type of vertex indices.
 * @tparam index_t Type used to store indexing and sizes.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param coalesced_sz_v coalesced vertex vector size.
 * @param num_paths number of paths.
 * @param d_coalesced_v coalesced vertex buffer.
 * @param d_sizes paths size buffer.
 * @return tuple of (src_vertex_vector, dst_Vertex_vector, path_offsets), where
 * path_offsets are the offsets where the COO set of each path starts.
 */
template <typename vertex_t, typename index_t>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<index_t>>
  convert_paths_to_coo(raft::handle_t const& handle,
                       index_t coalesced_sz_v,
                       index_t num_paths,
                       rmm::device_buffer&& d_coalesced_v,
                       rmm::device_buffer&& d_sizes)
{
  detail::coo_convertor_t<vertex_t, index_t> to_coo(handle, num_paths);

  detail::original::device_const_vector_view<vertex_t> d_v_view(
    static_cast<vertex_t const*>(d_coalesced_v.data()), coalesced_sz_v);

  detail::original::device_const_vector_view<index_t> d_sz_view(
    static_cast<index_t const*>(d_sizes.data()), num_paths);

  return to_coo(d_v_view, d_sz_view);
}

/**
 * @brief returns additional RW information on vertex paths offsets and weight path sizes and
 * offsets, for the coalesced case (the padded case does not need or provide this information)
 *
 * @tparam index_t Type used to store indexing and sizes.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param num_paths number of paths.
 * @param ptr_d_sizes sizes of vertex paths.
 * @return tuple of (vertex_path_offsets, weight_path_sizes, weight_path_offsets), where offsets are
 * exclusive scan of corresponding sizes.
 */
template <typename index_t>
std::tuple<rmm::device_uvector<index_t>, rmm::device_uvector<index_t>, rmm::device_uvector<index_t>>
query_rw_sizes_offsets(raft::handle_t const& handle, index_t num_paths, index_t const* ptr_d_sizes)
{
  rmm::device_uvector<index_t> d_vertex_offsets(num_paths, handle.get_stream());
  rmm::device_uvector<index_t> d_weight_sizes(num_paths, handle.get_stream());
  rmm::device_uvector<index_t> d_weight_offsets(num_paths, handle.get_stream());

  thrust::exclusive_scan(
    handle.get_thrust_policy(), ptr_d_sizes, ptr_d_sizes + num_paths, d_vertex_offsets.begin());

  thrust::transform(handle.get_thrust_policy(),
                    ptr_d_sizes,
                    ptr_d_sizes + num_paths,
                    d_weight_sizes.begin(),
                    [] __device__(auto vertex_path_sz) { return vertex_path_sz - 1; });

  handle.sync_stream();

  thrust::exclusive_scan(handle.get_thrust_policy(),
                         d_weight_sizes.begin(),
                         d_weight_sizes.end(),
                         d_weight_offsets.begin());

  return std::make_tuple(
    std::move(d_vertex_offsets), std::move(d_weight_sizes), std::move(d_weight_offsets));
}

}  // namespace cugraph
