/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <experimental/graph.hpp>

#include <rmm/thrust_rmm_allocator.h>
//#include <compute_partition.cuh>
//#include <experimental/shuffle.cuh>
#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <raft/random/rng.cuh>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>  //
#include <thrust/tuple.h>                  //

#include <thrust/remove.h>
#include <thrust/transform.h>

//#include <experimental/include_cuco_static_map.cuh>

#include <cassert>
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

// raft random generator:
// (using upper-bound cached "map"
//  giving out_deg(v) for each v in [0, |V|);
//  and a pre-generated vector of float random values
//  in [0,1] to be brought into [0, d_ub[v]))
//
template <typename vertex_t,
          typename edge_t,
          typename seed_t  = long,
          typename real_t  = float,
          typename index_t = vertex_t>
struct rrandom_gen_t {
  rrandom_gen_t(raft::handle_t const& handle,
                index_t nPaths,
                device_vec_t<real_t>& d_random,             // scratch-pad, non-coalesced
                device_vec_t<edge_t> const& d_crt_out_deg,  // non-coalesced
                seed_t seed = seed_t{})
    : handle_(handle),
      seed_(seed),
      num_paths_(nPaths),
      d_ptr_out_degs_(raw_const_ptr(d_crt_out_deg)),
      d_ptr_random_(raw_ptr(d_random))
  {
    CUGRAPH_EXPECTS(d_random.size() >= nPaths, "Un-allocated random buffer.");

    // FIXME: not only in constructor,
    // this must be done at each step,
    // unless this object is constructed at each step;
    //
    raft::random::Rng rng(seed_);
    rng.uniform<real_t, index_t>(
      d_ptr_random_, nPaths, real_t{0.0}, real_t{1.0}, handle.get_stream());
  }

  // in place:
  // for each v {
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
  edge_t const* d_ptr_out_degs_;  // device puffer with out-deg of current set of vertices (most
                                  // recent vertex in each path); size = num_paths_
  real_t* d_ptr_random_;          // device buffer with real random values; size = num_paths_
  seed_t seed_;                   // seed to be used for current batch
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
                     device_vec_t<vertex_t> const& d_indices,
                     device_vec_t<edge_t> const& d_offsets,
                     device_vec_t<weight_t> const& d_values,
                     device_vec_t<edge_t> const& d_crt_out_degs,
                     device_vec_t<index_t> const& d_sizes,
                     index_t num_paths,
                     index_t max_depth)
    : handle_(handle),
      col_indices_(raw_const_ptr(d_indices)),
      row_offsets_(raw_const_ptr(d_offsets)),
      values_(raw_const_ptr(d_values)),
      out_degs_(raw_const_ptr(d_crt_out_degs)),
      sizes_(raw_const_ptr(d_sizes)),
      num_paths_(num_paths),
      max_depth_(max_depth)
  {
  }

  col_indx_extract_t(raft::handle_t const& handle,
                     vertex_t const* p_d_indices,
                     edge_t const* p_d_offsets,
                     weight_t const* p_d_values,
                     edge_t const* p_d_crt_out_degs,
                     index_t const* p_d_sizes,
                     index_t num_paths,
                     index_t max_depth)
    : handle_(handle),
      col_indices_(p_d_indices),
      row_offsets_(p_d_offsets),
      values_(p_d_values),
      out_degs_(p_d_crt_out_degs),
      sizes_(p_d_sizes),
      num_paths_(num_paths),
      max_depth_(max_depth)
  {
  }

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
    device_vec_t<vertex_t> const& d_coalesced_src_v,  // coalesced vector of vertices
    device_vec_t<vertex_t> const& d_v_col_indx,  // column indices, given by stepper's random engine
    device_vec_t<vertex_t>& d_v_next_vertices,   // set of destination vertices, for next step
    device_vec_t<weight_t>&
      d_v_next_weights)  // set of weights between src and destination vertices, for next step
    const
  {
    using zip_iterator_t =
      thrust::zip_iterator<thrust::tuple<device_v_it<vertex_t>, device_v_it<weight_t>>>;

    // auto max_depth          = max_depth_;  // to avoid capturing `this`...
    // auto const* ptr_d_sizes = sizes_;

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
        return thrust::make_tuple(col_indices[start_row + col_indx], values[start_row + col_indx]);
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

// class abstracting the RW stepping algorithm:
// preprocessing, stepping, and post-processing
//
template <typename graph_t,
          typename random_engine_t,
          typename index_t = typename graph_t::edge_type>
struct random_walker_t {
  static_assert(std::is_trivially_copyable<random_engine_t>::value,
                "random engine assumed trivially copyable.");

  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  random_walker_t(raft::handle_t const& handle,
                  graph_t const& graph,
                  index_t nPaths,
                  index_t max_depth,
                  random_engine_t const& rnd)
    : handle_(handle),
      num_paths_(nPaths),
      max_depth_(max_depth),
      d_v_stopped_{static_cast<size_t>(nPaths), handle_.get_stream()},
      rnd_(rnd),
      d_cached_out_degs_(graph.compute_out_degrees(handle_))
  {
    // init d_v_stopped_ to {0} (i.e., no path is stopped):
    //
    thrust::copy_n(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                   thrust::make_constant_iterator(0),
                   nPaths,
                   d_v_stopped_.begin());
  }

  // in-place updates its arguments from one step to next
  // (to avoid copying)
  //
  // take one step in sync for all paths:
  //
  void step(graph_t const& graph,
            index_t step,
            device_vec_t<vertex_t>& d_v_paths_v_set,  // coalesced vertex set
            device_vec_t<weight_t>& d_v_paths_w_set,  // coalesced weight set
            device_vec_t<index_t>& d_v_paths_sz,      // paths sizes
            device_vec_t<edge_t>& d_crt_out_degs)     // out-degs for current set of vertices
  {
    // update crt snapshot of out-degs,
    // from cached out degs, using
    // latest vertex in each path as source:
    //
    gather_from_coalesced(
      d_v_paths_v_set, d_cached_out_degs_, d_v_paths_sz, d_crt_out_degs, max_depth_, num_paths_);

    col_indx_extract_t<graph_t> col_extractor(handle_,
                                              graph,
                                              raw_const_ptr(d_crt_out_degs),
                                              raw_const_ptr(d_v_paths_sz),
                                              num_paths_,
                                              max_depth_);
    // TODO: gather:
    // for each indx in [0..nPaths) {
    //   v_indx = d_v_rnd_n_indx[indx];
    //
    //   // get the `v_indx`-th out-vertex of d_v_paths_v_set[indx] vertex:
    //
    //   d_v_paths_v_set[indx*nPaths + step] =
    //       get_out_vertex(graph, d_v_paths_v_set[indx*nPaths + (step-1)], v_indx);
    //   d_v_paths_w_set[indx*nPaths + step] =
    //       get_out_edge_weight(graph, d_v_paths_v_set[indx*nPaths + (step-1)], v_indx);
    //   update(d_v_stopped);
    // }
  }

  bool all_stopped(void) const
  {
    // FIXME: use (all(d_crt_out_degs, (==0))
    //
    auto pos = thrust::find(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                            d_v_stopped_.begin(),
                            d_v_stopped_.end(),
                            0);

    if (pos != d_v_stopped_.end())
      return false;
    else
      return true;
  }

  // for each i in [0..num_paths_) {
  //   d_paths_v_set[i*max_depth] = d_src_init_v[i];
  //
  void initialize(device_vec_t<vertex_t> const& d_src_init_v,
                  device_vec_t<vertex_t>& d_paths_v_set,
                  device_vec_t<index_t>& d_sizes) const
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

  void defragment(device_vec_t<vertex_t>& d_coalesced_v,       // coalesced vertex set
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
    device_vec_t<edge_t> const& d_crt_out_degs,  // |current set of vertex out degrees| = nelems, to
                                                 // be used as stencil (don't scatter if 0)
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

  void scatter_vertices(device_vec_t<vertex_t> const& d_src,
                        device_vec_t<vertex_t>& d_coalesced,
                        device_vec_t<edge_t> const& d_crt_out_degs,
                        device_vec_t<index_t> const& d_sizes,
                        index_t max_depth) const
  {
    scatter_to_coalesced(d_src, d_coalesced, d_crt_out_degs, d_sizes, max_depth, num_paths_);
  }

  void scatter_weights(device_vec_t<weight_t> const& d_src,
                       device_vec_t<weight_t>& d_coalesced,
                       device_vec_t<edge_t> const& d_crt_out_degs,
                       device_vec_t<index_t> const& d_sizes,
                       index_t max_depth) const
  {
    scatter_to_coalesced(d_src, d_coalesced, d_crt_out_degs, d_sizes, max_depth - 1, num_paths_, 1);
  }

 private:
  raft::handle_t const& handle_;
  index_t num_paths_;
  index_t max_depth_;
  device_vec_t<int> d_v_stopped_;  // keeps track of paths that stopped (==1); FIXME: remove, not
                                   // needed (use p_d_crt_out_degs instead)
  random_engine_t rnd_;
  device_vec_t<edge_t> d_cached_out_degs_;
};

/**
 * @brief returns random walks (RW) from starting sources, where each path is of given maximum
 * length. Single-GPU specialization.
 *
 * @tparam graph_t Type of graph.
 * @tparam vertex_type Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_type Type of edge weights. Needs to be a floating point type.
 * @tparam random_engine_t Type of random engine used to generate RW.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to generate RW on.
 * @param start_vertex_set Set of starting vertex indices for the RW. number(RW) ==
 * start_vertex_set.size().
 * @param max_depth maximum length of RWs.
 * @param rnd_engine Random engine parameter (e.g., uniform).
 * @return std::tuple<device_vec_t<vertex_t>, device_vec_t<weight_t>,
 * device_vec_t<index_t>> Triplet of coalesced RW paths, with corresponding edge weights for
 * each, and coresponding path sizes. This is meant to minimize the number of DF's to be passed to
 * the Python layer.
 */
template <typename graph_t,
          typename random_engine_t,
          typename index_t = typename graph_t::vertex_type>
std::enable_if_t<graph_t::is_multi_gpu == false,
                 std::tuple<device_vec_t<typename graph_t::vertex_type>,
                            device_vec_t<typename graph_t::weight_type>,
                            device_vec_t<index_t>>>
random_walks(raft::handle_t const& handle,
             graph_t const& graph,
             std::vector<typename graph_t::vertex_type> const& start_vertex_set,
             index_t max_depth,
             random_engine_t& rnd_engine)
{
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  // TODO: Potentially this might change, if it's decided to pass the
  // starting vector directly on device...
  //
  auto nPaths = start_vertex_set.size();
  auto stream = handle.get_stream();

  device_vec_t<vertex_t> d_v_start{nPaths, stream};

  // Copy starting set on device:
  //
  CUDA_TRY(cudaMemcpyAsync(d_v_start.data(),
                           start_vertex_set.data(),
                           nPaths * sizeof(vertex_t),
                           cudaMemcpyHostToDevice,
                           stream));

  cudaStreamSynchronize(stream);

  random_walker_t<graph_t, random_engine_t> rand_walker{
    handle, graph, static_cast<index_t>(nPaths), static_cast<index_t>(max_depth), rnd_engine};

  // pre-allocate num_paths * max_depth;
  //
  auto coalesced_sz = nPaths * max_depth;
  device_vec_t<vertex_t> d_v_paths_v_set{coalesced_sz, stream};  // coalesced vertex set
  device_vec_t<weight_t> d_v_paths_w_set{coalesced_sz, stream};  // coalesced weight set
  device_vec_t<index_t> d_v_paths_sz{nPaths, stream};            // paths sizes
  device_vec_t<edge_t> d_crt_out_degs{nPaths, stream};  // out-degs for current set of vertices

  // very first vertex, for each path:
  //
  rand_walker.initialize(d_v_start, d_v_paths_v_set, d_v_paths_sz);

  // start from 1, as 0-th was initialized above:
  //
  for (decltype(max_depth) step_indx = 1; step_indx < max_depth; ++step_indx) {
    rand_walker.step(
      graph, step_indx, d_v_paths_v_set, d_v_paths_w_set, d_v_paths_sz, d_crt_out_degs);

    // early exit: all paths have reached sinks:
    //
    if (rand_walker.all_stopped()) break;
  }

  // truncate v_set, w_set to actual space used:
  //
  rand_walker.defragment(d_v_paths_v_set, d_v_paths_w_set, d_v_paths_sz);

  // because device_uvector is not copy-cnstr-able:
  //
  return std::make_tuple(
    std::move(d_v_paths_v_set), std::move(d_v_paths_w_set), std::move(d_v_paths_sz));
}

/**
 * @brief returns random walks (RW) from starting sources, where each path is of given maximum
 * length. Multi-GPU specialization.
 *
 * @tparam graph_t Type of graph.
 * @tparam vertex_type Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_type Type of edge weights. Needs to be a floating point type.
 * @tparam random_engine_t Type of random engine used to generate RW.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to generate RW on.
 * @param start_vertex_set Set of starting vertex indices for the RW. number(RW) ==
 * start_vertex_set.size().
 * @param max_depth maximum length of RWs.
 * @param rnd_engine Random engine parameter (e.g., uniform).
 * @return std::tuple<device_vec_t<vertex_t>, device_vec_t<weight_t>,
 * device_vec_t<index_t>> Triplet of coalesced RW paths, with corresponding edge weights for
 * each, and coresponding path sizes. This is meant to minimize the number of DF's to be passed to
 * the Python layer.
 */
template <typename graph_t,
          typename random_engine_t,
          typename index_t = typename graph_t::vertex_type>
std::enable_if_t<graph_t::is_multi_gpu == true,
                 std::tuple<device_vec_t<typename graph_t::vertex_type>,
                            device_vec_t<typename graph_t::weight_type>,
                            device_vec_t<index_t>>>
random_walks(raft::handle_t const& handle,
             graph_t const& graph,
             std::vector<typename graph_t::vertex_type> const& start_vertex_set,
             index_t max_depth,
             random_engine_t& rnd_engine)
{
  CUGRAPH_FAIL("Not implemented yet.");
}

}  // namespace detail

/**
 * @brief returns random walks (RW) from starting sources, where each path is of given maximum
 * length. Uniform distribution is assumed for the random engine.
 *
 * @tparam graph_t Type of graph.
 * @tparam vertex_type Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_type Type of edge weights. Needs to be a floating point type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph object to generate RW on.
 * @param start_vertex_set Set of starting vertex indices for the RW. number(RW) ==
 * start_vertex_set.size().
 * @param max_depth maximum length of RWs.
 * @return std::tuple<device_vec_t<vertex_t>, device_vec_t<weight_t>,
 * device_vec_t<index_t>> Triplet of coalesced RW paths, with corresponding edge weights for
 * each, and coresponding path sizes. This is meant to minimize the number of DF's to be passed to
 * the Python layer.
 */
template <typename graph_t, typename index_t = typename graph_t::vertex_type>
std::tuple<rmm::device_uvector<typename graph_t::vertex_type>,
           rmm::device_uvector<typename graph_t::weight_type>,
           rmm::device_uvector<index_t>>
random_walks(raft::handle_t const& handle,
             graph_t const& graph,
             std::vector<typename graph_t::vertex_type> const& start_vertex_set,
             index_t max_depth);
}  // namespace experimental
}  // namespace cugraph
