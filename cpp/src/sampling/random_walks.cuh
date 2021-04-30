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

#include <experimental/graph.hpp>

#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>

#include <cassert>
#include <ctime>
#include <tuple>
#include <type_traits>

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

  rrandom_gen_t(raft::handle_t const& handle,
                index_t num_paths,
                device_vec_t<real_t>& d_random,             // scratch-pad, non-coalesced
                device_vec_t<edge_t> const& d_crt_out_deg,  // non-coalesced
                seed_t seed = seed_t{})
    : handle_(handle),
      seed_(seed),
      num_paths_(num_paths),
      d_ptr_out_degs_(raw_const_ptr(d_crt_out_deg)),
      d_ptr_random_(raw_ptr(d_random))
  {
    auto rnd_sz = d_random.size();

    CUGRAPH_EXPECTS(rnd_sz >= static_cast<decltype(rnd_sz)>(num_paths),
                    "Un-allocated random buffer.");

    // done in constructor;
    // this must be done at each step,
    // but this object is constructed at each step;
    //
    raft::random::Rng rng(seed_);
    rng.uniform<real_t, index_t>(
      d_ptr_random_, num_paths, real_t{0.0}, real_t{1.0}, handle.get_stream());
  }

  // in place:
  // for each v in [0, num_paths) {
  // if out_deg(v) > 0
  //   d_col_indx[v] = random index in [0, out_deg(v))
  //}
  void generate_col_indices(device_vec_t<vertex_t>& d_col_indx) const
  {
    thrust::transform_if(
      rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
      d_ptr_random_,
      d_ptr_random_ + num_paths_,  // input1
      d_ptr_out_degs_,             // input2
      d_ptr_out_degs_,             // also stencil
      d_col_indx.begin(),
      [] __device__(real_t rnd_vindx, edge_t crt_out_deg) {
        real_t max_ub     = static_cast<real_t>(crt_out_deg - 1);
        auto interp_vindx = rnd_vindx * max_ub + real_t{.5};
        vertex_t v_indx   = static_cast<vertex_t>(interp_vindx);
        return (v_indx >= crt_out_deg ? crt_out_deg - 1 : v_indx);
      },
      [] __device__(auto crt_out_deg) { return crt_out_deg > 0; });
  }

 private:
  raft::handle_t const& handle_;
  index_t num_paths_;
  edge_t const* d_ptr_out_degs_;  // device buffer with out-deg of current set of vertices (most
                                  // recent vertex in each path); size = num_paths_
  real_t* d_ptr_random_;          // device buffer with real random values; size = num_paths_
  seed_t seed_;                   // seed to be used for current batch
};

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

// classes abstracting the next vertex extraction mechanism:
//
// primary template, purposely undefined
template <typename graph_t,
          typename index_t  = typename graph_t::edge_type,
          typename enable_t = void>
struct col_indx_extract_t;

// specialization for single-gpu functionality:
//
template <typename graph_t, typename index_t>
struct col_indx_extract_t<graph_t, index_t, std::enable_if_t<graph_t::is_multi_gpu == false>> {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  col_indx_extract_t(raft::handle_t const& handle,
                     graph_t const& graph,
                     edge_t const* p_d_crt_out_degs,
                     index_t const* p_d_sizes,
                     index_t num_paths,
                     index_t max_depth)
    : handle_(handle),
      col_indices_(graph.indices()),
      row_offsets_(graph.offsets()),
      values_(graph.weights()),
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
    device_vec_t<vertex_t> const& d_coalesced_src_v,  // in: coalesced vector of vertices
    device_vec_t<vertex_t> const&
      d_v_col_indx,  // in: column indices, given by stepper's random engine
    device_vec_t<vertex_t>& d_v_next_vertices,  // out: set of destination vertices, for next step
    device_vec_t<weight_t>&
      d_v_next_weights)  // out: set of weights between src and destination vertices, for next step
    const
  {
    thrust::transform_if(
      rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
      thrust::make_counting_iterator<index_t>(0),
      thrust::make_counting_iterator<index_t>(num_paths_),  // input1
      d_v_col_indx.begin(),                                 // input2
      out_degs_,                                            // stencil
      thrust::make_zip_iterator(
        thrust::make_tuple(d_v_next_vertices.begin(), d_v_next_weights.begin())),  // output
      [max_depth         = max_depth_,
       ptr_d_sizes       = sizes_,
       ptr_d_coalesced_v = raw_const_ptr(d_coalesced_src_v),
       row_offsets       = row_offsets_,
       col_indices       = col_indices_,
       values            = values_] __device__(auto indx, auto col_indx) {
        auto delta     = ptr_d_sizes[indx] - 1;
        auto v_indx    = ptr_d_coalesced_v[indx * max_depth + delta];
        auto start_row = row_offsets[v_indx];

        auto weight_value =
          (values == nullptr ? weight_t{1}
                             : values[start_row + col_indx]);  // account for un-weighted graphs
        return thrust::make_tuple(col_indices[start_row + col_indx], weight_value);
      },
      [] __device__(auto crt_out_deg) { return crt_out_deg > 0; });
  }

 private:
  raft::handle_t const& handle_;
  vertex_t const* col_indices_;
  edge_t const* row_offsets_;
  weight_t const* values_;

  edge_t const* out_degs_;
  index_t const* sizes_;
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
template <typename graph_t,
          typename random_engine_t =
            rrandom_gen_t<typename graph_t::vertex_type, typename graph_t::edge_type>,
          typename index_t = typename graph_t::edge_type>
struct random_walker_t {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  using seed_t   = typename random_engine_t::seed_type;
  using real_t   = typename random_engine_t::real_type;

  random_walker_t(raft::handle_t const& handle,
                  graph_t const& graph,
                  index_t num_paths,
                  index_t max_depth)
    : handle_(handle),
      num_paths_(num_paths),
      max_depth_(max_depth),
      d_cached_out_degs_(graph.compute_out_degrees(handle_))
  {
  }

  // for each i in [0..num_paths_) {
  //   d_paths_v_set[i*max_depth] = d_src_init_v[i];
  //
  void start(device_const_vector_view<vertex_t, index_t>& d_src_init_v,  // in: start set
             device_vec_t<vertex_t>& d_paths_v_set,                      // out: coalesced v
             device_vec_t<index_t>& d_sizes) const  // out: init sizes to {1,...}
  {
    // intialize path sizes to 1, as they contain at least one vertex each:
    // the initial set: d_src_init_v;
    //
    thrust::copy_n(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
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

    thrust::scatter(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                    d_src_init_v.begin(),
                    d_src_init_v.end(),
                    map_it_begin,
                    d_paths_v_set.begin());
  }

  // overload for start() with device_uvector d_v_start
  // (handy for testing)
  //
  void start(device_vec_t<vertex_t> const& d_start,  // in: start set
             device_vec_t<vertex_t>& d_paths_v_set,  // out: coalesced v
             device_vec_t<index_t>& d_sizes) const   // out: init sizes to {1,...}
  {
    device_const_vector_view<vertex_t, index_t> d_start_cview{d_start.data(),
                                                              static_cast<index_t>(d_start.size())};

    start(d_start_cview, d_paths_v_set, d_sizes);
  }

  // in-place updates its arguments from one step to next
  // (to avoid copying); all "crt" arguments are updated at each step()
  // and passed as scratchpad space to avoid copying them
  // from one step to another
  //
  // take one step in sync for all paths that have not reached sinks:
  //
  void step(
    graph_t const& graph,
    seed_t seed,
    device_vec_t<vertex_t>& d_coalesced_v,  // crt coalesced vertex set
    device_vec_t<weight_t>& d_coalesced_w,  // crt coalesced weight set
    device_vec_t<index_t>& d_paths_sz,      // crt paths sizes
    device_vec_t<edge_t>& d_crt_out_degs,   // crt out-degs for current set of vertices
    device_vec_t<real_t>& d_random,         // crt set of random real values
    device_vec_t<vertex_t>& d_col_indx,  // crt col col indices to be used for retrieving next step
    device_vec_t<vertex_t>& d_next_v,    // crt set of destination vertices, for next step
    device_vec_t<weight_t>& d_next_w)
    const  // set of weights between src and destination vertices, for next step
  {
    // update crt snapshot of out-degs,
    // from cached out degs, using
    // latest vertex in each path as source:
    //
    gather_from_coalesced(
      d_coalesced_v, d_cached_out_degs_, d_paths_sz, d_crt_out_degs, max_depth_, num_paths_);

    // generate random destination indices:
    //
    random_engine_t rgen(handle_, num_paths_, d_random, d_crt_out_degs, seed);

    rgen.generate_col_indices(d_col_indx);

    // dst extraction from dst indices:
    //
    col_indx_extract_t<graph_t> col_extractor(handle_,
                                              graph,
                                              raw_const_ptr(d_crt_out_degs),
                                              raw_const_ptr(d_paths_sz),
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
    //   d_coalesced_v[indx*num_paths + d_sizes[indx]] =
    //       get_out_vertex(graph, d_coalesced_v[indx*num_paths + d_sizes[indx] -1)], v_indx);
    //   d_coalesced_w[indx*(num_paths-1) + d_sizes[indx] - 1] =
    //       get_out_edge_weight(graph, d_coalesced_v[indx*num_paths + d_sizes[indx]-2], v_indx);
    //
    // (1) generate actual vertex destinations:
    //
    col_extractor(d_coalesced_v, d_col_indx, d_next_v, d_next_w);

    // (2) update path sizes:
    //
    update_path_sizes(d_crt_out_degs, d_paths_sz);

    // (3) actual coalesced updates:
    //
    scatter_vertices(d_next_v, d_coalesced_v, d_crt_out_degs, d_paths_sz);
    scatter_weights(d_next_w, d_coalesced_w, d_crt_out_degs, d_paths_sz);
  }

  // returns true if all paths reached sinks:
  //
  bool all_paths_stopped(device_vec_t<edge_t> const& d_crt_out_degs) const
  {
    auto how_many_stopped =
      thrust::count_if(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                       d_crt_out_degs.begin(),
                       d_crt_out_degs.end(),
                       [] __device__(auto crt_out_deg) { return crt_out_deg == 0; });
    return (static_cast<size_t>(how_many_stopped) == d_crt_out_degs.size());
  }

  // wrap-up, post-process:
  // truncate v_set, w_set to actual space used
  //
  void stop(device_vec_t<vertex_t>& d_coalesced_v,       // coalesced vertex set
            device_vec_t<weight_t>& d_coalesced_w,       // coalesced weight set
            device_vec_t<index_t> const& d_sizes) const  // paths sizes
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

    auto new_end_v =
      thrust::remove_if(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                        d_coalesced_v.begin(),
                        d_coalesced_v.end(),
                        thrust::make_counting_iterator<index_t>(0),
                        predicate_v);

    auto new_end_w =
      thrust::remove_if(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                        d_coalesced_w.begin(),
                        d_coalesced_w.end(),
                        thrust::make_counting_iterator<index_t>(0),
                        predicate_w);

    CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));

    d_coalesced_v.resize(thrust::distance(d_coalesced_v.begin(), new_end_v), handle_.get_stream());
    d_coalesced_w.resize(thrust::distance(d_coalesced_w.begin(), new_end_w), handle_.get_stream());
  }

  // in-place non-static (needs handle_):
  // for indx in [0, nelems):
  //   gather d_result[indx] = d_src[d_coalesced[indx*stride + d_sizes[indx] -1]]
  //
  template <typename src_vec_t = vertex_t>
  void gather_from_coalesced(
    device_vec_t<vertex_t> const& d_coalesced,  // |gather map| = stride*nelems
    device_vec_t<src_vec_t> const& d_src,       // |gather input| = nelems
    device_vec_t<index_t> const& d_sizes,       // |paths sizes| = nelems, elems in [1, stride]
    device_vec_t<src_vec_t>& d_result,          // |output| = nelems
    index_t stride,        // stride = coalesce block size (typically max_depth)
    index_t nelems) const  // nelems = number of elements to gather (typically num_paths_)
  {
    vertex_t const* ptr_d_coalesced = raw_const_ptr(d_coalesced);
    index_t const* ptr_d_sizes      = raw_const_ptr(d_sizes);

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

    thrust::gather(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
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
    device_vec_t<src_vec_t> const& d_src,        // |scatter input| = nelems
    device_vec_t<src_vec_t>& d_coalesced,        // |scatter input| = stride*nelems
    device_vec_t<edge_t> const& d_crt_out_degs,  // |current set of vertex out degrees| = nelems,
                                                 // to be used as stencil (don't scatter if 0)
    device_vec_t<index_t> const&
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
    index_t const* ptr_d_sizes = raw_const_ptr(d_sizes);

    auto dlambda = [stride, adjust, ptr_d_sizes] __device__(auto indx) {
      auto delta = ptr_d_sizes[indx] - adjust - 1;
      return indx * stride + delta;
    };

    // use the transform iterator as map:
    //
    auto map_it_begin =
      thrust::make_transform_iterator(thrust::make_counting_iterator<index_t>(0), dlambda);

    thrust::scatter_if(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
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
  void scatter_vertices(device_vec_t<vertex_t> const& d_src,
                        device_vec_t<vertex_t>& d_coalesced,
                        device_vec_t<edge_t> const& d_crt_out_degs,
                        device_vec_t<index_t> const& d_sizes) const
  {
    scatter_to_coalesced(d_src, d_coalesced, d_crt_out_degs, d_sizes, max_depth_, num_paths_);
  }
  //
  void scatter_weights(device_vec_t<weight_t> const& d_src,
                       device_vec_t<weight_t>& d_coalesced,
                       device_vec_t<edge_t> const& d_crt_out_degs,
                       device_vec_t<index_t> const& d_sizes) const
  {
    scatter_to_coalesced(
      d_src, d_coalesced, d_crt_out_degs, d_sizes, max_depth_ - 1, num_paths_, 1);
  }

  // in-place update (increment) path sizes for paths
  // that have not reached a sink; i.e., for which
  // d_crt_out_degs[indx]>0:
  //
  void update_path_sizes(device_vec_t<edge_t> const& d_crt_out_degs,
                         device_vec_t<index_t>& d_sizes) const
  {
    thrust::transform_if(
      rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
      d_sizes.begin(),
      d_sizes.end(),           // input
      d_crt_out_degs.begin(),  // stencil
      d_sizes.begin(),         // output: in-place
      [] __device__(auto crt_sz) { return crt_sz + 1; },
      [] __device__(auto crt_out_deg) { return crt_out_deg > 0; });
  }

  device_vec_t<edge_t> const& get_out_degs(void) const { return d_cached_out_degs_; }

 private:
  raft::handle_t const& handle_;
  index_t num_paths_;
  index_t max_depth_;
  device_vec_t<edge_t> d_cached_out_degs_;
};

/**
 * @brief returns random walks (RW) from starting sources, where each path is of given maximum
 * length. Single-GPU specialization.
 *
 * @tparam graph_t Type of graph (view).
 * @tparam random_engine_t Type of random engine used to generate RW.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to generate RW on.
 * @param d_v_start Device (view) set of starting vertex indices for the RW.
 * number(paths) == d_v_start.size().
 * @param max_depth maximum length of RWs.
 * @return std::tuple<device_vec_t<vertex_t>, device_vec_t<weight_t>,
 * device_vec_t<index_t>, seed> Quadruplet of coalesced RW paths, with corresponding edge weights
 * for each, and corresponding path sizes. This is meant to minimize the number of DF's to be passed
 * to the Python layer. Also returning seed for testing / debugging repro. The meaning of
 * "coalesced" here is that a 2D array of paths of different sizes is represented as a 1D array.
 */
template <typename graph_t,
          typename random_engine_t =
            rrandom_gen_t<typename graph_t::vertex_type, typename graph_t::edge_type>,
          typename seeding_policy_t = clock_seeding_t<typename random_engine_t::seed_type>,
          typename index_t          = typename graph_t::edge_type>
std::enable_if_t<graph_t::is_multi_gpu == false,
                 std::tuple<device_vec_t<typename graph_t::vertex_type>,
                            device_vec_t<typename graph_t::weight_type>,
                            device_vec_t<index_t>,
                            typename random_engine_t::seed_type>>
random_walks_impl(raft::handle_t const& handle,
                  graph_t const& graph,
                  device_const_vector_view<typename graph_t::vertex_type, index_t>& d_v_start,
                  index_t max_depth,
                  seeding_policy_t seeder = clock_seeding_t<typename random_engine_t::seed_type>{})
{
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  using seed_t   = typename random_engine_t::seed_type;
  using real_t   = typename random_engine_t::real_type;

  vertex_t num_vertices = graph.get_number_of_vertices();

  auto how_many_valid =
    thrust::count_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     d_v_start.begin(),
                     d_v_start.end(),
                     [num_vertices] __device__(auto crt_vertex) {
                       return (crt_vertex >= 0) && (crt_vertex < num_vertices);
                     });

  CUGRAPH_EXPECTS(static_cast<index_t>(how_many_valid) == d_v_start.size(),
                  "Invalid set of starting vertices.");

  auto num_paths = d_v_start.size();
  auto stream    = handle.get_stream();

  random_walker_t<graph_t, random_engine_t> rand_walker{
    handle, graph, static_cast<index_t>(num_paths), static_cast<index_t>(max_depth)};

  // pre-allocate num_paths * max_depth;
  //
  auto coalesced_sz = num_paths * max_depth;
  device_vec_t<vertex_t> d_coalesced_v(coalesced_sz, stream);  // coalesced vertex set
  device_vec_t<weight_t> d_coalesced_w(coalesced_sz, stream);  // coalesced weight set
  device_vec_t<index_t> d_paths_sz(num_paths, stream);         // paths sizes
  device_vec_t<edge_t> d_crt_out_degs(num_paths, stream);  // out-degs for current set of vertices
  device_vec_t<real_t> d_random(num_paths, stream);
  device_vec_t<vertex_t> d_col_indx(num_paths, stream);
  device_vec_t<vertex_t> d_next_v(num_paths, stream);
  device_vec_t<weight_t> d_next_w(num_paths, stream);

  // abstracted out seed initialization:
  //
  seed_t seed0 = static_cast<seed_t>(seeder());

  // very first vertex, for each path:
  //
  rand_walker.start(d_v_start, d_coalesced_v, d_paths_sz);

  // start from 1, as 0-th was initialized above:
  //
  for (decltype(max_depth) step_indx = 1; step_indx < max_depth; ++step_indx) {
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

  // wrap-up, post-process:
  // truncate v_set, w_set to actual space used
  //
  rand_walker.stop(d_coalesced_v, d_coalesced_w, d_paths_sz);

  // because device_uvector is not copy-cnstr-able:
  //
  return std::make_tuple(std::move(d_coalesced_v),
                         std::move(d_coalesced_w),
                         std::move(d_paths_sz),
                         seed0);  // also return seed for repro
}

/**
 * @brief returns random walks (RW) from starting sources, where each path is of given maximum
 * length. Multi-GPU specialization.
 *
 * @tparam graph_t Type of graph (view).
 * @tparam random_engine_t Type of random engine used to generate RW.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to generate RW on.
 * @param d_v_start Device (view) set of starting vertex indices for the RW. number(RW) ==
 * d_v_start.size().
 * @param max_depth maximum length of RWs.
 * @return std::tuple<device_vec_t<vertex_t>, device_vec_t<weight_t>,
 * device_vec_t<index_t>, seed> Quadruplet of coalesced RW paths, with corresponding edge weights
 * for each, and coresponding path sizes. This is meant to minimize the number of DF's to be passed
 * to the Python layer. Also returning seed for testing / debugging repro. The meaning of
 * "coalesced" here is that a 2D array of paths of different sizes is represented as a 1D array.
 */
template <typename graph_t,
          typename random_engine_t =
            rrandom_gen_t<typename graph_t::vertex_type, typename graph_t::edge_type>,
          typename seeding_policy_t = clock_seeding_t<typename random_engine_t::seed_type>,
          typename index_t          = typename graph_t::edge_type>
std::enable_if_t<graph_t::is_multi_gpu == true,
                 std::tuple<device_vec_t<typename graph_t::vertex_type>,
                            device_vec_t<typename graph_t::weight_type>,
                            device_vec_t<index_t>,
                            typename random_engine_t::seed_type>>
random_walks_impl(raft::handle_t const& handle,
                  graph_t const& graph,
                  device_const_vector_view<typename graph_t::vertex_type, index_t>& d_v_start,
                  index_t max_depth,
                  seeding_policy_t seeder = clock_seeding_t<typename random_engine_t::seed_type>{})
{
  CUGRAPH_FAIL("Not implemented yet.");
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

  std::tuple<device_vec_t<vertex_t>, device_vec_t<vertex_t>, device_vec_t<index_t>> operator()(
    device_const_vector_view<vertex_t>& d_coalesced_v,
    device_const_vector_view<index_t>& d_sizes) const
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

    device_vec_t<index_t> d_sz_w_scan(num_paths_, handle_.get_stream());

    // copy vertex path sizes that are > 1:
    // (because vertex_path_sz translates
    //  into edge_path_sz = vertex_path_sz - 1,
    //  and edge_paths_sz == 0 don't contribute
    //  anything):
    //
    auto new_end_it =
      thrust::copy_if(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
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
      rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
      d_sz_w_scan.begin(),
      d_sz_w_scan.end(),
      d_sz_w_scan.begin(),
      [] __device__(auto sz) { return sz - 1; },
      index_t{0},
      thrust::plus<index_t>{});

    return std::make_tuple(std::move(d_src), std::move(d_dst), std::move(d_sz_w_scan));
  }

  std::tuple<device_vec_t<int>, index_t, device_vec_t<index_t>> fill_stencil(
    device_const_vector_view<index_t>& d_sizes) const
  {
    device_vec_t<index_t> d_scan(num_paths_, handle_.get_stream());
    thrust::inclusive_scan(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                           d_sizes.begin(),
                           d_sizes.end(),
                           d_scan.begin());

    index_t total_sz{0};
    CUDA_TRY(cudaMemcpy(
      &total_sz, raw_ptr(d_scan) + num_paths_ - 1, sizeof(index_t), cudaMemcpyDeviceToHost));

    device_vec_t<int> d_stencil(total_sz, handle_.get_stream());

    // initialize stencil to all 1's:
    //
    thrust::copy_n(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                   thrust::make_constant_iterator<int>(1),
                   d_stencil.size(),
                   d_stencil.begin());

    // set to 0 entries positioned at inclusive_scan(sizes[]),
    // because those are path "breakpoints", where a path end
    // and the next one starts, hence there cannot be an edge
    // between a path ending vertex and next path starting vertex;
    //
    thrust::scatter(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                    thrust::make_constant_iterator(0),
                    thrust::make_constant_iterator(0) + num_paths_,
                    d_scan.begin(),
                    d_stencil.begin());

    return std::make_tuple(std::move(d_stencil), total_sz, std::move(d_scan));
  }

  std::tuple<device_vec_t<vertex_t>, device_vec_t<vertex_t>> gather_pairs(
    device_const_vector_view<vertex_t>& d_coalesced_v,
    device_vec_t<int> const& d_stencil,
    index_t total_sz_v) const
  {
    auto total_sz_w = total_sz_v - num_paths_;
    device_vec_t<index_t> valid_src_indx(total_sz_w, handle_.get_stream());

    // generate valid vertex src indices,
    // which is any index in {0,...,total_sz_v - 2}
    // provided the next index position; i.e., (index+1),
    // in stencil is not 0; (if it is, there's no "next"
    // or dst index, because the path has ended);
    //
    thrust::copy_if(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                    thrust::make_counting_iterator<index_t>(0),
                    thrust::make_counting_iterator<index_t>(total_sz_v - 1),
                    valid_src_indx.begin(),
                    [ptr_d_stencil = raw_const_ptr(d_stencil)] __device__(auto indx) {
                      auto dst_indx = indx + 1;
                      return ptr_d_stencil[dst_indx] == 1;
                    });

    device_vec_t<vertex_t> d_src_v(total_sz_w, handle_.get_stream());
    device_vec_t<vertex_t> d_dst_v(total_sz_w, handle_.get_stream());

    // construct pair of src[], dst[] by gathering
    // from d_coalesced_v all pairs
    // at entries (valid_src_indx, valid_src_indx+1),
    // where the set of valid_src_indx was
    // generated at the previous step;
    //
    thrust::transform(
      rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
      valid_src_indx.begin(),
      valid_src_indx.end(),
      thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin())),  // start_zip
      [ptr_d_vertex = raw_const_ptr(d_coalesced_v)] __device__(auto indx) {
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
 * @tparam graph_t Type of graph (view).
 * @tparam index_t Type used to store indexing and sizes.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to generate RW on.
 * @param ptr_d_start Device pointer to set of starting vertex indices for the RW.
 * @param num_paths = number(paths).
 * @param max_depth maximum length of RWs.
 * @return std::tuple<device_vec_t<vertex_t>, device_vec_t<weight_t>,
 * device_vec_t<index_t>> Triplet of coalesced RW paths, with corresponding edge weights for
 * each, and coresponding path sizes. This is meant to minimize the number of DF's to be passed to
 * the Python layer.
 */
template <typename graph_t, typename index_t>
std::tuple<rmm::device_uvector<typename graph_t::vertex_type>,
           rmm::device_uvector<typename graph_t::weight_type>,
           rmm::device_uvector<index_t>>
random_walks(raft::handle_t const& handle,
             graph_t const& graph,
             typename graph_t::vertex_type const* ptr_d_start,
             index_t num_paths,
             index_t max_depth)
{
  using vertex_t = typename graph_t::vertex_type;

  // 0-copy const device view:
  //
  detail::device_const_vector_view<vertex_t, index_t> d_v_start{ptr_d_start, num_paths};

  auto quad_tuple = detail::random_walks_impl(handle, graph, d_v_start, max_depth);
  // ignore last element of the quad, seed,
  // since it's meant for testing / debugging, only:
  //
  return std::make_tuple(std::move(std::get<0>(quad_tuple)),
                         std::move(std::get<1>(quad_tuple)),
                         std::move(std::get<2>(quad_tuple)));
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

  detail::device_const_vector_view<vertex_t> d_v_view(
    static_cast<vertex_t const*>(d_coalesced_v.data()), coalesced_sz_v);

  detail::device_const_vector_view<index_t> d_sz_view(static_cast<index_t const*>(d_sizes.data()),
                                                      num_paths);

  return to_coo(d_v_view, d_sz_view);
}

}  // namespace experimental
}  // namespace cugraph
