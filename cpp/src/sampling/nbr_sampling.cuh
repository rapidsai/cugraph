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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "rw_traversals.hpp"

#include <algorithm>
#include <limits>
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

/**
 * @brief Projects output from one iteration onto the input for the next: extracts the
 (destination_vertex_id, rank_to_send_it_to) components from the output quadruplet; overload acting
 on zip-iterators;
 * @tparam zip_out_it_t zip type for the output tuple;
 * @tparam zip_in_it_t zip type for the input tuple;
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param[in] begin zip begin iterator of quadruplets from which new input is extracted; typically
 (vertex_t source_vertex, vertex_t destination_vertex, int rank, edge_t index)
 * @param[in] end zip end iterator of quadruplets from which new input is extracted;
 * @param[out] result begin of result zip iterator of pairs for next iteration; typically
 (vertex_t source_vertex, int rank)
 */
template <typename zip_out_it_t, typename zip_in_it_t>
void project(raft::handle_t const& handle, zip_out_it_t begin, zip_out_it_t end, zip_in_it_t result)
{
  thrust::transform(
    handle.get_thrust_policy(), begin, end, result, [] __device__(auto const& quad) {
      return thrust::make_tuple(thrust::get<1>(quad), thrust::get<2>(quad));
    });
}

}  // namespace detail

// Stub functions namespace:
// TODO: remove when stub functions inside are ready:
//
namespace mnmg {
using gpu_t = int;

template <typename GraphViewType, typename VertexIterator, typename GPUIdIterator>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename std::iterator_traits<GPUIdIterator>::value_type>>
gather_active_sources_in_row(raft::handle_t const& handle,
                             GraphViewType const& graph_view,
                             VertexIterator vertex_input_first,
                             VertexIterator vertex_input_last,
                             GPUIdIterator gpu_id_first);

template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> get_active_source_global_degrees(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  rmm::device_uvector<typename GraphViewType::vertex_type>& active_sources,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_out_degrees);

template <typename GraphViewType, typename EdgeIndexIterator, typename gpu_t>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<gpu_t>,
           rmm::device_uvector<typename GraphViewType::edge_type>>
gather_local_edges(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  rmm::device_uvector<typename GraphViewType::vertex_type>& active_sources_in_row,
  rmm::device_uvector<gpu_t>& active_source_gpu_ids,
  EdgeIndexIterator edge_index_first,
  typename GraphViewType::vertex_type invalid_vertex_id,
  int indices_per_source,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_degree_offsets);

namespace ops {
// see cugraph-ops/cpp/src/graph/sampling/sampling_index.cuh
template <typename IdxT>
void get_sampling_index(IdxT* index,
                        raft::random::Rng& rng,
                        const IdxT* sizes,
                        IdxT n_sizes,
                        int32_t sample_size,
                        bool replace,
                        cudaStream_t stream);
}  // namespace ops
}  // namespace mnmg

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
 * @tparam index_t Type used for indexing; typically edge_t
 * @tparam vertex_out_tuple_t Tuple type of the out device vector;
 * typically (vertex_t source_vertex, vertex_t destination_vertex, int rank, edge_t index)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param ptr_d_start Device array of pairs: (starting_vertex_index, rank) for the NBR Sampling.
 * @param num_starting_vs size of starting vertex set
 * @param h_fan_out vector of branching out (fan-out) degree per source vertex for each level
 * @param global_degree_offsets local partition of global out-degree cache; pass-through
 * parameter used for obtaining local out-degree information
 * @param flag_replacement boolean flag specifying if random sampling is done without replacement
 * (true); or, with replacement (false); default = true;
 */
template <typename graph_view_t,
          typename index_t            = typename graph_view_t::edge_type,
          typename vertex_out_tuple_t = thrust::tuple<typename graph_view_t::vertex_type,
                                                      typename graph_view_t::vertex_type,
                                                      mnmg::gpu_t,  // TODO: replace after
                                                                    // integration of PR 2064
                                                      index_t>,
          typename seeder_t =
            detail::clock_seeding_t<uint64_t>>  // TODO: move clock_... to "rw_traversal.hpp"
rmm::device_uvector<vertex_out_tuple_t> uniform_nbr_sample(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  typename graph_view_t::vertex_type const* ptr_d_start,
  mnmg::gpu_t const* ptr_d_ranks,
  size_t num_starting_vs,
  std::vector<int> const& h_fan_out,
  rmm::device_uvector<typename GraphViewType::edge_type> const& global_degree_offsets,
  bool flag_replacement)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using edge_t   = typename graph_view_t::edge_type;

  if constexpr (graph_view_t::is_multi_gpu) {
    CUGRAPH_EXPECTS((ptr_d_start != nullptr) && (num_starting_vs > 0),
                    "Invalid input argument: starting vertex set cannot be null.");

    auto num_levels = h_fan_out.size();

    CUGRAPH_EXPECTS(num_levels > 0, "Invalid input argument: number of levels must be non-zero.");

    // running input growing from one level to the next:
    // plus it gets shuffled, so need copies
    //
    detail::device_vec_t<vertex_t> d_in(num_starting_vs);
    detail::device_vec_t<mnmg::gpu_t> d_ranks(num_starting_vs);

    thrust::copy_n(handle.get_thrust_policy(), ptr_d_start, num_starting_vs, d_in.begin());
    thrust::copy_n(handle.get_thrust_policy(), ptr_d_ranks, num_starting_vs, d_ranks.begin());

    // Output to accumulate results into:
    //
    detail::device_vec_t<vertex_out_tuple_t> d_out{};  // starts as empty

    decltype(num_levels) level{0};
    for (auto&& k_level : h_fan_out) {
      // main body:
      //{

      // line 4 in specs (prep for extracting out-degs(sources)):
      //
      auto&& [d_new_in, d_new_rank] = mnmg::gather_active_sources_in_row(
        handle, graph_view, d_in.begin(), d_in.end(), d_ranks.begin());

      // line 7 in specs (extract out-degs(sources)):
      //
      auto&& d_out_degs =
        mnmg::get_active_source_global_degrees(handle, graph_view, d_in, global_degree_offsets);

      // line 8 in specs (segemented-random-generation of indices):
      //
      decltype(d_out_degs) d_indices(d_in.size() * k_level);
      seeder_t seeder{};
      raft::random::Rng rng(seeder() + k_level);  // TODO: check if this works for uniform
      mnmg::ops::get_sampling_index(detail::raw_ptr(d_indices),
                                    rng,
                                    detail::raw_const_ptr(d_out_degs),
                                    static_cast<edge_t>(d_out_degs.size()),
                                    static_cast<int32_t>(k_level),
                                    flag_replacement,
                                    handle.get_stream());

      // line 10 (gather edges):
      //
      auto&& [d_out_src, d_out_dst, d_out_ranks, d_out_indx] =
        mnmg::gather_local_edges(handle,
                                 graph_view,
                                 d_new_in.begin(),
                                 d_new_rank.begin(),
                                 d_indices.begin(),
                                 std::numeric_limits<vertex_t>::max(),
                                 static_cast<int>(k_level),
                                 global_degree_offsets);

      // resize everything:
      // d_out_degs, d_in, d_out, d_ranks
      auto old_sz = d_out.size();
      auto add_sz = d_out_dst.size();
      d_out.resize(old_sz + add_sz);

      // line 12 (union step):
      //
      auto in = thrust::make_zip_iterator(thrust::make_tuple(detail::raw_const_ptr(d_out_src),
                                                             detail::raw_const_ptr(d_out_dst),
                                                             detail::raw_const_ptr(d_out_ranks),
                                                             detail::raw_const_ptr(d_out_indx)));
      thrust::copy_n(handle.get_thrust_policy(), in, add_sz, d_out.begin() + old_sz);

      // line 13 (shuffle step):
      // zipping is necessary to preserve rank info during shuffle!
      //

      // line 14 (project output onto input):

      //}
      ++level;
    }

    return d_out;
  } else {
    CUGRAPH_FAIL("Neighborhood sampling functionality is supported only for the multi-gpu case.");
  }
}

}  // namespace cugraph
