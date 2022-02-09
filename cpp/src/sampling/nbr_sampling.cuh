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

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "rw_traversals.hpp"

#include <algorithm>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace detail {

/**
 * @brief Projects output from one iteration onto the input for the next: extracts the
 (destination_vertex_id, rank_to_send_it_to) components from the output quadruplet
 * @tparam vertex_t vertex id type;
 * @tparam edge_t edge id (and vertex indexing) type;
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param[in] d_out device array of quadruplets from which new input is extracted; typically
 (vertex_t source_vertex, vertex_t destination_vertex, int rank, edge_t index)
 * @param[out] d_in next itertion input device array of pairs for next iteration; typically
 (vertex_t source_vertex, int rank)
 */
template <typename vertex_t, typename edge_t>
void project(raft::handle_t const& handle,
             thrust::tuple<vertex_t, vertex_t, int, edge_t> const* d_out,
             thrust::tuple<vertex_t, int>* d_in,
             size_t sz)
{
  thrust::transform(
    handle.get_thrust_policy(), d_out, d_out + sz, d_in, [] __device__(auto const& quad) {
      return thrust::make_tuple(thrust::get<1>(quad), thrust::get<2>(quad));  // (d, r)
    });
}

}  // namespace detail

/**
 * @brief Multi-GPU Uniform Neighborhood Sampling. The outline of the algorithm:
 *
 * uniform_nbr_sample(J[p][], L, K[], flag_unique) {
 *   Out[p][] = {};                                              // initialize output result
 * (empty)
 *
 *  loop level in {0,…, L-1} {                                   // 1 tree level / iteration
 *       n_per_level = |J| * L^ (level+1);                       // size of output per level
 *
 *       J[] = union(J[], {J[partition_row],
 *                        for partition_row same as `p`};
 *
 *      for each pair (s, _) in J[] {                            // cache out-degrees of src_v
 * set; d_out_deg[s] = mnmg_get_out_deg(graph, s);
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
 *                                                               // (src_v, dst_v,
 * rank_to_send_to) Out[p][] = union(Out[p][], d_out[]);                      // append local
 * output to result d_out[] = shuffle(d_out[]);                               // reshuffle output
 * to
 *                                                               // corresponding rank
 *     J[] = project(d_out[], []((s,d,r)){ return (d,r);});      // extract the (d, r) from (s,d,
 * r)
 *                                                               // for next iter
 *    }
 *    return Out[p][];
 * }
 *
 * @tparam graph_view_t Type of graph view.
 * @tparam vertex_in_tuple_t Tuple type of the input device array;
 * typically (vertex_t source_vertex, int rank)
 * @tparam vertex_out_tuple_t Tuple type of the out device vector;
 * typically (vertex_t source_vertex, vertex_t destination_vertex, int rank, edge_t index)
 * @tparam index_t Type used for indexing; typically edge_t
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph View object to generate NBR Sampling on.
 * @param ptr_d_start Device array of pairs: (starting_vertex_index, rank) for the NBR Sampling.
 * @param num_starting_vs size of starting vertex set
 * @param h_fan_out vector of branching out (fan-out) degree per source vertex for each level
 * @param ptr_d_partition_global_out_deg local partition of global out-degree cache; pass-through
 * parameter used for obtaining local out-degree information
 * @param flag_replacement boolean flag specifying if random sampling is done without replacement
 * (true); or, with replacement (false); default = true;
 */
template <typename graph_view_t,
          typename vertex_in_tuple_t,
          typename vertex_out_tuple_t = vertex_in_tuple_t,
          typename index_t            = typename graph_view_t::edge_type>
rmm::device_uvector<vertex_out_tuple_t> uniform_nbr_sample(
  raft::handle_t const& handle,
  graph_view_t const& graph,
  vertex_in_tuple_t const* ptr_d_start,
  size_t num_starting_vs,
  std::vector<int> const& h_fan_out,
  typename graph_view_t::edge_type const* ptr_d_partition_global_out_deg,
  bool flag_replacement)
{
  if constexpr (graph_view_t::is_multi_gpu) {
    CUGRAPH_EXPECTS((ptr_d_start != nullptr) && (num_starting_vs > 0),
                    "Invalid input argument: starting vertex set cannot be null.");

    auto num_levels = h_fan_out.size();

    CUGRAPH_EXPECTS(num_levels > 0, "Invalid input argument: number of levels must be non-zero.");

    // running input growing from one level to the next:
    //
    detail::device_vec_t<vertex_in_tuple_t> d_in(num_starting_vs);
    thrust::copy_n(handle.get_thrust_policy(), ptr_d_start, num_starting_vs, d_in.begin());
    // Output to accumulate results into:
    //
    detail::device_vec_t<vertex_out_tuple_t> d_out{};  // starts as empty

    decltype(num_levels) level{0};
    for (auto&& k_level : h_fan_out) {
      // main body:
      //{
      //}
      ++level;
    }

    return d_out;
  } else {
    CUGRAPH_FAIL("Neighborhood sampling functionality is supported only for the multi-gpu case.");
  }
}

}  // namespace cugraph
