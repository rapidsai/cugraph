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
#include <compute_partition.cuh>
#include <experimental/shuffle.cuh>
#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/iterator/constant_iterator.h>

//#include <experimental/include_cuco_static_map.cuh>

#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

// class abstracting the RW stepping algorithm:
//
template <typename graph_t, typename random_engine_t>
struct random_walker_t {
  static_assert(std::is_trivially_copyable<random_engine_t>::value,
                "random engine assumed trivially copyable.");

  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  random_walker_t(raft::handle_t const& handle,
                  size_t nPaths,
                  vertex_t* ptr_d_current_vertices,
                  random_engine_t const& rnd)
    : handle_(handle),
      num_paths_(nPaths),
      ptr_d_vertex_set_(ptr_d_current_vertices),
      d_v_stopped_{nPaths, handle_.get_stream()},
      rnd_(rnd)
  {
    // init d_v_stopped_ to {0} (i.e., no path is stopped):
    //
    thrust::copy_n(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                   thrust::make_constant_iterator(0),
                   nPaths,
                   d_v_stopped_.begin());
  }

  // TODO: take one step in sync for all paths:
  //
  void step(void) {}

  bool all_stopped(void) const
  {
    auto pos = thrust::find(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                            d_v_stopped_.begin(),
                            d_v_stopped_.end(),
                            0);

    if (pos != d_v_stopped_.end())
      return false;
    else
      return true;
  }

 private:
  raft::handle_t const& handle_;
  size_t num_paths_;
  vertex_t* ptr_d_vertex_set_;
  rmm::device_uvector<int> d_v_stopped_;  // keeps track of paths that stopped (==1)
  random_engine_t rnd_;
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
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>,
 * rmm::device_uvector<size_t>> Triplet of coalesced RW paths, with corresponding edge weights for
 * each, and coresponding path sizes. This is meant to minimize the number of DF's to be passed to
 * the Python layer.
 */
template <typename graph_t, typename random_engine_t>
std::enable_if_t<graph_t::is_multi_gpu == false,
                 std::tuple<rmm::device_uvector<typename graph_t::vertex_type>,
                            rmm::device_uvector<typename graph_t::weight_type>,
                            rmm::device_uvector<size_t>>>
random_walks(raft::handle_t const& handle,
             graph_t const& graph,
             std::vector<typename graph_t::vertex_type> const& start_vertex_set,
             size_t max_depth,
             random_engine_t& rnd_engine)
{
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  // TODO: Potentially this might change, if it's decided to pass the
  // starting vector directly on device...
  //
  auto nPaths = start_vertex_set.size();
  auto stream = handle.get_stream();

  rmm::device_uvector<vertex_t> d_v_start{nPaths, stream};

  // Copy starting set on device:
  //
  CUDA_TRY(cudaMemcpyAsync(d_v_start.data(),
                           start_vertex_set.data(),
                           nPaths * sizeof(vertex_t),
                           cudaMemcpyHostToDevice,
                           stream));

  cudaStreamSynchronize(stream);

  random_walker_t<graph_t, random_engine_t> rand_walker{
    handle, nPaths, d_v_start.data(), rnd_engine};
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
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>,
 * rmm::device_uvector<size_t>> Triplet of coalesced RW paths, with corresponding edge weights for
 * each, and coresponding path sizes. This is meant to minimize the number of DF's to be passed to
 * the Python layer.
 */
template <typename graph_t, typename random_engine_t>
std::enable_if_t<graph_t::is_multi_gpu == true,
                 std::tuple<rmm::device_uvector<typename graph_t::vertex_type>,
                            rmm::device_uvector<typename graph_t::weight_type>,
                            rmm::device_uvector<size_t>>>
random_walks(raft::handle_t const& handle,
             graph_t const& graph,
             std::vector<typename graph_t::vertex_type> const& start_vertex_set,
             size_t max_depth,
             random_engine_t& rnd_engine);

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
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>,
 * rmm::device_uvector<size_t>> Triplet of coalesced RW paths, with corresponding edge weights for
 * each, and coresponding path sizes. This is meant to minimize the number of DF's to be passed to
 * the Python layer.
 */
template <typename graph_t>
std::tuple<rmm::device_uvector<typename graph_t::vertex_type>,
           rmm::device_uvector<typename graph_t::weight_type>,
           rmm::device_uvector<size_t>>
random_walks(raft::handle_t const& handle,
             graph_t const& graph,
             std::vector<typename graph_t::vertex_type> const& start_vertex_set,
             size_t max_depth);
}  // namespace experimental
}  // namespace cugraph
