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

#include <cugraph/detail/graph_functions.cuh>
#include <cugraph/graph.hpp>
#include <utilities/graph_utils.cuh>

#include <cugraph/utilities/shuffle_comm.cuh>

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

#include <cugraph-ops/graph/sampling.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace detail {

/**
 * @brief Projects zip input onto the lower dim zip output, where lower dimension components are
 * specified by tuple indices; e.g., extracts the (destination_vertex_id, rank_to_send_it_to)
 * components from the quadruplet (vertex_t source_vertex, vertex_t destination_vertex, int rank,
 * edge_t index) via indices {1,2};
 * @tparam vertex_index non-type template parameter specifying index in the input tuple where vertex
 * IDs are stored;
 * @tparam rank_index non-type template parameter specifying index in the input tuple where rank IDs
 * are stored;
 * @tparam zip_in_it_t zip Type for the input tuple;
 * @tparam zip_out_it_t zip Type for the output tuple;
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param[in] begin zip begin iterator of quadruplets from which new input is extracted; typically
 * (vertex_t source_vertex, vertex_t destination_vertex, int rank, edge_t index)
 * @param[in] end zip end iterator of quadruplets from which new input is extracted;
 * @param[out] result begin of result zip iterator of pairs for next iteration; typically
 * (vertex_t source_vertex, int rank)
 */
template <size_t vertex_index, size_t rank_index, typename zip_in_it_t, typename zip_out_it_t>
void project(raft::handle_t const& handle, zip_in_it_t begin, zip_in_it_t end, zip_out_it_t result)
{
  thrust::transform(handle.get_thrust_policy(), begin, end, result, [] __device__(auto const& tpl) {
    return thrust::make_tuple(thrust::get<vertex_index>(tpl), thrust::get<rank_index>(tpl));
  });
}

/**
 * @brief Shuffles zipped pairs of vertex IDs and ranks IDs to the GPU's that the vertex IDs belong
 * to. The assumption is that the return provides a per-GPU coalesced set of pairs, with
 * corresponding counts vector. To limit the result to the self-GPU one needs additional filtering
 * to extract the corresponding set from the coalesced set of sets and using the corresponding
 * counts entry;
 * @tparam graph_view_t Type of graph view.
 * @tparam zip_iterator_t zip Type for the zipped tuple<vertex_t, gpu_t> (vertexID, rank);
 * @tparam gpu_t Type used for storing GPU rank IDs;
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param[in] begin zip begin iterator of (vertexID, rank) pairs;
 * @param[in] end zip end iterator of (vertexID, rank) pairs;
 * @param[in] unnamed tag used for template tag dispatching
 * @return tuple pair of coalesced pairs and counts
 */
template <typename graph_view_t, typename zip_iterator_t, typename gpu_t>
decltype(auto) shuffle_to_gpus(raft::handle_t const& handle,
                               graph_view_t const& graph_view,
                               zip_iterator_t begin,
                               zip_iterator_t end,
                               gpu_t)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using edge_t   = typename graph_view_t::edge_type;

  auto vertex_partition_lasts = graph_view.get_vertex_partition_lasts();
  rmm::device_uvector<vertex_t> d_vertex_partition_lasts(vertex_partition_lasts.size(),
                                                         handle.get_stream());
  raft::update_device(
    d_vertex_partition_lasts.data(), vertex_partition_lasts.data(), vertex_partition_lasts.size());

  auto&& [rx_tuple, rx_counts] = groupby_gpu_ids_and_shuffle_values(
    handle.get_comms(),
    begin,
    end,
    [vertex_partition_lasts = d_vertex_partition_lasts.data(),
     num_vertex_partitions  = d_vertex_partition_lasts.size()] __device__(auto tpl_v_r) {
      auto gpu_id = static_cast<gpu_t>(
        thrust::distance(vertex_partition_lasts,
                         thrust::lower_bound(thrust::seq,
                                             vertex_partition_lasts,
                                             vertex_partition_lasts + num_vertex_partitions,
                                             thrust::get<0>(tpl_v_r))));
    },
    handle.get_stream());
  return std::make_tuple(std::move(rx_tuple), rx_counts);
}

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
 * @tparam gpu_t Type used for storing GPU rank IDs;
 * @tparam index_t Type used for indexing; typically edge_t
 * @tparam vertex_out_tuple_t Tuple type of the out device vector;
 * @tparam seeder_t Type for generating random engine seeds;
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
 * @return Device vector of vertex_out_tuple_t;
 */
template <
  typename graph_view_t,
  typename gpu_t,
  typename index_t            = typename graph_view_t::edge_type,
  typename vertex_out_tuple_t = thrust::
    tuple<typename graph_view_t::vertex_type, typename graph_view_t::vertex_type, gpu_t, index_t>,
  typename seeder_t =
    detail::clock_seeding_t<uint64_t>>  // TODO: move clock_... to "rw_traversal.hpp"
rmm::device_uvector<vertex_out_tuple_t> uniform_nbr_sample_impl(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  typename graph_view_t::vertex_type const* ptr_d_start,
  gpu_t const* ptr_d_ranks,
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
    detail::device_vec_t<vertex_t> d_in(num_starting_vs, handle.get_stream());
    detail::device_vec_t<gpu_t> d_ranks(num_starting_vs, handle.get_stream());

    thrust::copy_n(handle.get_thrust_policy(), ptr_d_start, num_starting_vs, d_in.begin());
    thrust::copy_n(handle.get_thrust_policy(), ptr_d_ranks, num_starting_vs, d_ranks.begin());

    // Output to accumulate results into:
    //
    detail::device_vec_t<vertex_out_tuple_t> d_out(0, handle.get_stream());  // starts as empty

    decltype(num_levels) level{0};
    auto const self_rank = handle.get_comms().get_rank();
    for (auto&& k_level : h_fan_out) {
      // main body:
      //{

      // line 4 in specs (prep for extracting out-degs(sources)):
      //
      auto&& [d_new_in, d_new_rank] =
        gather_active_sources_in_row(handle, graph_view, d_in.begin(), d_in.end(), d_ranks.begin());

      // line 7 in specs (extract out-degs(sources)):
      //
      auto&& d_out_degs =
        get_active_source_global_degrees(handle, graph_view, d_new_in, global_degree_offsets);

      // line 8 in specs (segemented-random-generation of indices):
      //
      decltype(d_out_degs) d_indices(d_in.size() * k_level);
      seeder_t seeder{};
      raft::random::Rng rng(seeder() + k_level);
      get_sampling_index(detail::raw_ptr(d_indices),
                         rng,
                         detail::raw_const_ptr(d_out_degs),
                         static_cast<edge_t>(d_out_degs.size()),
                         static_cast<int32_t>(k_level),
                         flag_replacement,
                         handle.get_stream());

      // line 10 (gather edges):
      //
      auto&& [d_out_src, d_out_dst, d_out_ranks] =
        gather_local_edges(handle,
                           graph_view,
                           d_new_in.begin(),
                           d_new_rank.begin(),
                           d_indices.begin(),
                           std::numeric_limits<vertex_t>::max(),
                           static_cast<int>(k_level),
                           global_degree_offsets);

      // resize d_out:
      //
      auto old_sz = d_out.size();
      auto add_sz = d_out_dst.size();
      d_out.resize(old_sz + add_sz);

      // line 12 (union step):
      //
      auto out_zip_it = thrust::make_zip_iterator(thrust::make_tuple(
        d_out_src.begin(), d_out_dst.begin(), d_out_ranks.begin(), d_indices.begin()));

      thrust::copy_n(handle.get_thrust_policy(), out_zip_it, add_sz, d_out.begin() + old_sz);

      // line 13 (shuffle step):
      // zipping is necessary to preserve rank info during shuffle!
      //
      auto next_in_zip_begin =
        thrust::make_zip_iterator(thrust::make_tuple(d_out_src.begin(), d_out_ranks.begin()));
      auto next_in_zip_end =
        thrust::make_zip_iterator(thrust::make_tuple(d_out_src.end(), d_out_ranks.end()));
      auto&& [rx_tpl_v_r, rx_counts] =
        detail::shuffle_to_gpus(handle, graph_view, next_in_zip_begin, next_in_zip_end);

      // filter rx_tpl_v_r and rx_counts vector by self_rank:
      //
      decltype(rx_counts) rx_offsets(rx_counts.size());
      std::exclusive_scan(rx_counts.begin(), rx_counts.end(), rx_offsets.begin(), 0);

      // resize d_in, d_ranks:
      //
      auto new_in_sz = rx_counts.at(self_rank);
      d_in.resize(new_in_sz);
      d_ranks.resize(new_in_sz);

      // line 14 (project output onto input):
      // zip d_in, d_ranks
      //
      auto new_in_zip = thrust::make_zip_iterator(
        thrust::make_tuple(d_in.begin(), d_ranks.begin()));  // result start_zip

      auto tpl_in_it_begin = rx_tpl_v_r.begin() + rx_offsets.at(self_rank);
      detail::project<0, 1>(handle, tpl_in_it_begin, tpl_in_it_begin + new_in_sz, new_in_zip);

      //}
      ++level;
    }

    return d_out;
  } else {
    CUGRAPH_FAIL("Neighborhood sampling functionality is supported only for the multi-gpu case.");
  }
}

}  // namespace detail

/**
 * @brief Multi-GPU Uniform Neighborhood Sampling.
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
 * parameter used for obtaining local out-degree information
 * @param flag_replacement boolean flag specifying if random sampling is done without replacement
 * (true); or, with replacement (false); default = true;
 * @return Device vector of vertex_out_tuple_t;
 */
template <
  typename graph_view_t,
  typename gpu_t,
  typename index_t            = typename graph_view_t::edge_type,
  typename vertex_out_tuple_t = thrust::
    tuple<typename graph_view_t::vertex_type, typename graph_view_t::vertex_type, gpu_t, index_t>>
decltype(auto) uniform_nbr_sample(raft::handle_t const& handle,
                                  graph_view_t const& graph_view,
                                  typename graph_view_t::vertex_type const* ptr_d_start,
                                  gpu_t const* ptr_d_ranks,
                                  size_t num_starting_vs,
                                  std::vector<int> const& h_fan_out,
                                  bool flag_replacement)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using edge_t   = typename graph_view_t::edge_type;

  auto vertex_rank_pairs_it =
    thrust::make_zip_iterator(thrust::make_tuple(ptr_d_start, ptr_d_ranks));

  // TODO: figure out h_local_counts:
  //
  std::vector<size_t> h_local_counts(num_starting_vs);  // ???

  // shuffle input data to its corresponding rank;
  // (TODO: this functionality may not exist yet in the MNMG prims;
  //  some fine-tunning might be necessary)
  //
  auto [shuffled_vertex_rank_pairs, shuffled_counts] =
    shuffle_values(handle.get_comms(), vertex_rank_pairs_it, h_local_counts, handle.get_stream());

  auto&& [d_edge_count, global_degree_offsets] = get_global_degree_information(handle, graph_view);

  return detail::uniform_nbr_sample_impl(handle,
                                         graph_view,
                                         ptr_d_start,
                                         ptr_d_ranks,
                                         num_starting_vs,
                                         h_fan_out,
                                         global_degree_offsets,
                                         flag_replacement);
}

}  // namespace cugraph
