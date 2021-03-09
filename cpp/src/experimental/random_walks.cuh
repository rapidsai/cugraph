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

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/device_atomics.cuh>

#include <experimental/include_cuco_static_map.cuh>

#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

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
             random_engine_t& rnd_engine);

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
