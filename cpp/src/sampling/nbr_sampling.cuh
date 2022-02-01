/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <vector>

namespace cugraph {

namespace detail {

}  // namespace detail

/**
 * @brief Multi-GPU Uniform Neighborhood Sampling. The outline of the algorithm:
 *
 * uniform_nbr_sample(J[p][], L, K[], flag_unique) {
 *   Out[p][] = {};                                              // initialize output result (empty)
 *
 *  loop level in {0,…, L-1} {                                   // 1 tree level / iteration
 *       n_per_level = |J| * L^ (level+1);                       // size of output per level
 *
 *       J[] = union(J[], {J[partition_row],
 *                        for partition_row same as `p`};
 *
 *      for each pair (s, _) in J[] {                            // cache out-degrees of src_v set;
 *         d_out_deg[s] = mnmg_get_out_deg(graph, s);
 *      }
 *
 *      d_indices[] = segmented_random_generator(d_out_degs[],   // sizes[] to define range to
 *                                                               // sample from;
 *                                               K[level],       // fanout per-level
 *                                               flag_unique);
 *                                                               // for each (s, _) in J[]{
 *                                                               //   generate {0,…,out-deg(s)};}
 *
 *     d_out[] = gather_nbr(J[], d_indices[], level, K[level]);  // {(s, d, r),…} MNMG prim that
 *                                                               // gathers the NBR for current
 *                                                               // level of each src_v;
 *                                                               // output is set of triplets
 *                                                               // (src_v, dst_v, rank_to_send_to)
 *     Out[p][] = union(Out[p][], d_out[]);                      // append local output to result
 *     d_out[] = shuffle(d_out[]);                               // reshuffle output to
 *                                                               // corresponding rank
 *     J[] = project(d_out[], []((s,d,r)){ return (d,r);});      // extract the (d, r) from (s,d, r)
 *                                                               // for next iter
 *    }
 *    return Out[p][];
 * }
 *
 * @tparam graph_view_t Type of graph view.
 * @tparam vertex_in_tuple_t Tuple type of the input device array;
 * typically (vertex_t source_vertex, int rank)
 * @tparam vertex_out_tuple_t Tuple type of the out device vector;
 * typically (vertex_t source_vertex, vertex_t destination_vertex, int rank)
 * @tparam index_t Type used for indexing; typically edge_t
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph View object to generate NBR Sampling on.
 * @param ptr_d_start Device array of pairs: (starting_vertex_index, rank) for the NBR Sampling.
 * @param flag_replacement boolean flag specifying if random sampling is done without replacement
 * (true); or, with replacement (false); default = true;
 */
template <typename graph_view_t,
          typename vertex_in_tuple_t,
          typename vertex_out_tuple_t = vertex_in_tuple_t,
          typename index_t            = typename graph_view_t::vertex_type>
rmm::device_vector<vertex_out_tuple_t> uniform_nbr_sample(raft::handle_t const& handle,
                                                          graph_view_t const& graph,
                                                          vertex_in_tuple_t const* ptr_d_start,
                                                          std::vector<int> const& h_fan_out,
                                                          bool flag_replacement)
{
  if constexpr (graph_view_t::is_multi_gpu) {
    // TODO: main body here
  } else {
    CUGRAPH_FAIL("Neighborhood sampling functionality is supported only for the multi-gpu case.");
  }
}

}  // namespace cugraph
