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

#include <cugraph-ops/graph/sampling.hpp>

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
 * counts entry.
 * @tparam graph_view_t Type of graph view.
 * @tparam zip_iterator_t zip Type for the zipped tuple<vertex_t, gpu_t> (vertexID, rank);
 * @tparam gpu_t Type used for storing GPU rank IDs;
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] graph_view Graph View object to generate NBR Sampling on.
 * @param[in] begin zip begin iterator of (vertexID, rank) pairs.
 * @param[in] end zip end iterator of (vertexID, rank) pairs.
 * @param[in] unnamed tag used for template tag dispatching
 * @return tuple pair of coalesced pairs and counts
 */
template <typename graph_view_t, typename zip_iterator_t, typename gpu_t>
std::tuple<std::tuple<device_vec_t<typename graph_view_t::vertex_type>, device_vec_t<gpu_t>>,
           std::vector<size_t>>
shuffle_to_gpus(raft::handle_t const& handle,
                graph_view_t const& graph_view,
                zip_iterator_t begin,
                zip_iterator_t end,
                gpu_t)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using edge_t   = typename graph_view_t::edge_type;

  auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
  device_vec_t<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                        handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.size(),
                      handle.get_stream());

  return groupby_gpu_id_and_shuffle_values(
    handle.get_comms(),
    begin,
    end,
    [vertex_partition_range_lasts = d_vertex_partition_range_lasts.data(),
     num_vertex_partitions = d_vertex_partition_range_lasts.size()] __device__(auto tpl_v_r) {
      return static_cast<gpu_t>(
        thrust::distance(vertex_partition_range_lasts,
                         thrust::lower_bound(thrust::seq,
                                             vertex_partition_range_lasts,
                                             vertex_partition_range_lasts + num_vertex_partitions,
                                             thrust::get<0>(tpl_v_r))));
    },
    handle.get_stream());
}

/**
 * @brief Updates pair of vertex IDs and ranks IDs to the GPU's that the vertex IDs belong
 * to.
 * @tparam graph_view_t Type of graph view.
 * @tparam zip_iterator_t zip Type for the zipped tuple<vertex_t, gpu_t> (vertexID, rank).
 * @tparam gpu_t Type used for storing GPU rank IDs;
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] graph_view Graph View object to generate NBR Sampling on.
 * @param[in] begin zip begin iterator of (vertexID, rank) pairs.
 * @param[in] end zip end iterator of (vertexID, rank) pairs.
 * @param[in] rank for which data is to be extracted.
 * @param[out] d_in vertex set to be updated.
 * @param[out] d_ranks corresponding rank set to be updated.
 * @param[in] unnamed tag used for template tag dispatching.
 */
template <typename graph_view_t, typename zip_iterator_t, typename gpu_t>
void update_input_by_rank(raft::handle_t const& handle,
                          graph_view_t const& graph_view,
                          zip_iterator_t begin,
                          zip_iterator_t end,
                          size_t rank,
                          device_vec_t<typename graph_view_t::vertex_type>& d_in,
                          device_vec_t<gpu_t>& d_ranks,
                          gpu_t)
{
  auto&& [rx_tpl_v_r, rx_counts] = detail::shuffle_to_gpus(handle, graph_view, begin, end, gpu_t{});

  // filter rx_tpl_v_r and rx_counts vector by rank:
  //
  decltype(rx_counts) rx_offsets(rx_counts.size());
  std::exclusive_scan(rx_counts.begin(), rx_counts.end(), rx_offsets.begin(), 0);

  // resize d_in, d_ranks:
  //
  auto new_in_sz = rx_counts.at(rank);
  d_in.resize(new_in_sz, handle.get_stream());
  d_ranks.resize(new_in_sz, handle.get_stream());

  // project output onto input:
  // zip d_in, d_ranks
  //
  auto new_in_zip = thrust::make_zip_iterator(
    thrust::make_tuple(d_in.begin(), d_ranks.begin()));  // result start_zip

  auto&& d_new_dests = std::get<0>(rx_tpl_v_r);
  auto&& d_new_ranks = std::get<1>(rx_tpl_v_r);
  auto offset        = rx_offsets.at(rank);

  auto tpl_in_it_begin = thrust::make_zip_iterator(
    thrust::make_tuple(d_new_dests.begin() + offset, d_new_ranks.begin() + offset));
  project<0, 1>(handle, tpl_in_it_begin, tpl_in_it_begin + new_in_sz, new_in_zip);
}

/**
 * @brief Shuffles zipped tuples of (vertex_t source_vertex, vertex_t destination_vertex, int rank,
 * index_t index) to specified target GPU's.
 * @tparam vertex_t Type of vertex IDs.
 * @tparam gpu_t Type used for storing GPU rank IDs.
 * @tparam index_t Type used for indexing; typically edge_t.
 * @param[in] handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator,
 * and handles to various CUDA libraries) to run graph algorithms.
 * @param[in] d_src source vertex IDs; shuffle prims require it be mutable.
 * @param[in] d_dst destination vertex IDs; must be mutable.
 * @param[in] d_gpu_id_keys target GPU IDs (ranks); must be mutable.
 * @param[in] d_indices indices of destination vertices; must be mutable.
 * @return tuple of tuple of device vectors and counts:
 * ((vertex_t source_vertex, vertex_t destination_vertex, int rank, edge_t index), rx_counts)
 */
template <typename vertex_t, typename gpu_t, typename index_t>
std::tuple<std::tuple<device_vec_t<vertex_t>,
                      device_vec_t<vertex_t>,
                      device_vec_t<gpu_t>,
                      device_vec_t<index_t>>,
           std::vector<size_t>>
shuffle_to_target_gpu_ids(raft::handle_t const& handle,
                          device_vec_t<vertex_t>& d_src,
                          device_vec_t<vertex_t>& d_dst,
                          device_vec_t<gpu_t>& d_gpu_id_keys,
                          device_vec_t<index_t>& d_indices)
{
  auto zip_it_begin =
    thrust::make_zip_iterator(thrust::make_tuple(d_src.begin(), d_dst.begin(), d_indices.begin()));

  thrust::sort_by_key(
    handle.get_thrust_policy(), d_gpu_id_keys.begin(), d_gpu_id_keys.end(), zip_it_begin);

  rmm::device_uvector<size_t> tx_counts(handle.get_comms().get_size(), handle.get_stream());

  thrust::tabulate(
    handle.get_thrust_policy(),
    tx_counts.begin(),
    tx_counts.end(),
    [gpu_id_first = d_gpu_id_keys.begin(), gpu_id_last = d_gpu_id_keys.end()] __device__(size_t i) {
      return static_cast<size_t>(thrust::distance(
        gpu_id_first,
        thrust::upper_bound(thrust::seq, gpu_id_first, gpu_id_last, static_cast<gpu_t>(i))));
    });

  thrust::adjacent_difference(
    handle.get_thrust_policy(), tx_counts.begin(), tx_counts.end(), tx_counts.begin());

  std::vector<size_t> h_tx_counts(tx_counts.size());
  raft::update_host(h_tx_counts.data(), tx_counts.data(), tx_counts.size(), handle.get_stream());

  handle.sync_stream();

  return  // [rx_tuple, rx_counts]
    shuffle_values(handle.get_comms(),
                   thrust::make_zip_iterator(thrust::make_tuple(
                     d_src.begin(), d_dst.begin(), d_gpu_id_keys.begin(), d_indices.begin())),
                   h_tx_counts,
                   handle.get_stream());
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
 * @tparam index_t Type used for indexing; typically edge_t.
 * @tparam seeder_t Type for generating random engine seeds.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param d_in Device vector of starting vertex IDs for the NBR Sampling. Must be non-const for
 * shuffling.
 * @param d_ranks Device vector of ranks for which corresponding vertex ID data must be sent to. The
 * pairs (vertex_ID, rank) must be shuffled together. Must be non-const for shuffling.
 * @param h_fan_out vector of branching out (fan-out) degree per source vertex for each level
 * @param global_degree_offsets local partition of global out-degree cache; pass-through
 * parameter used for obtaining local out-degree information
 * @param flag_replacement boolean flag specifying if random sampling is done without replacement
 * (true); or, with replacement (false); default = true;
 * @return tuple of device vectors:
 * (vertex_t source_vertex, vertex_t destination_vertex, int rank, edge_t index)
 */
template <typename graph_view_t,
          typename gpu_t,
          typename index_t  = typename graph_view_t::edge_type,
          typename seeder_t = detail::clock_seeding_t<uint64_t>>
std::tuple<device_vec_t<typename graph_view_t::vertex_type>,
           device_vec_t<typename graph_view_t::vertex_type>,
           device_vec_t<gpu_t>,
           device_vec_t<index_t>>
uniform_nbr_sample_impl(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  device_vec_t<typename graph_view_t::vertex_type>& d_in,
  device_vec_t<gpu_t>& d_ranks,
  std::vector<int> const& h_fan_out,
  device_vec_t<typename graph_view_t::edge_type> const& global_out_degrees,
  device_vec_t<typename graph_view_t::edge_type> const& global_degree_offsets,
  device_vec_t<typename graph_view_t::edge_type> const& global_adjacency_list_offsets,
  bool flag_replacement)
{
  using vertex_t        = typename graph_view_t::vertex_type;
  using edge_t          = typename graph_view_t::edge_type;
  using return_t        = std::tuple<device_vec_t<vertex_t>,
                              device_vec_t<vertex_t>,
                              device_vec_t<gpu_t>,
                              device_vec_t<index_t>>;
  namespace cugraph_ops = cugraph::ops::gnn::graph;

  if constexpr (graph_view_t::is_multi_gpu) {
    size_t num_starting_vs = d_in.size();

    CUGRAPH_EXPECTS(num_starting_vs == d_ranks.size(),
                    "Sets of input vertices and ranks must have same sizes.");

    auto num_levels = h_fan_out.size();

    CUGRAPH_EXPECTS(num_levels > 0, "Invalid input argument: number of levels must be non-zero.");

    // Output quad of accumulators to collect results into:
    // (all start as empty)
    //
    device_vec_t<vertex_t> d_acc_src(0, handle.get_stream());
    device_vec_t<vertex_t> d_acc_dst(0, handle.get_stream());
    device_vec_t<gpu_t> d_acc_ranks(0, handle.get_stream());
    device_vec_t<index_t> d_acc_indices(0, handle.get_stream());

    auto&& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto&& row_rank = row_comm.get_rank();

    auto&& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto&& col_rank = col_comm.get_rank();

    auto const self_rank = handle.get_comms().get_rank();

    size_t level{0l};
    for (auto&& k_level : h_fan_out) {
      // prep step for extracting out-degs(sources):
      //
      auto&& [d_new_in, d_new_rank] =
        gather_active_majors(handle, graph_view, d_in.cbegin(), d_in.cend(), d_ranks.cbegin());

      rmm::device_uvector<vertex_t> d_out_src(0, handle.get_stream());
      rmm::device_uvector<vertex_t> d_out_dst(0, handle.get_stream());
      rmm::device_uvector<gpu_t> d_out_ranks(0, handle.get_stream());
      rmm::device_uvector<edge_t> d_indices(0, handle.get_stream());

      if (k_level != 0) {
        // extract out-degs(sources):
        //
        auto&& d_out_degs =
          get_active_major_global_degrees(handle, graph_view, d_new_in, global_out_degrees);

        // segemented-random-generation of indices:
        //
        device_vec_t<edge_t> d_rnd_indices(d_new_in.size() * k_level, handle.get_stream());

        cugraph_ops::Rng rng(row_rank + level);
        cugraph_ops::get_sampling_index(detail::raw_ptr(d_rnd_indices),
                                        rng,
                                        detail::raw_const_ptr(d_out_degs),
                                        static_cast<edge_t>(d_out_degs.size()),
                                        static_cast<int32_t>(k_level),
                                        flag_replacement,
                                        handle.get_stream());

        // gather edges step:
        // invalid entries (not found, etc.) filtered out in result;
        // d_indices[] filtered out in-place (to avoid copies+moves);
        //
        auto&& [temp_d_out_src, temp_d_out_dst, temp_d_out_ranks, temp_d_indices] =
          gather_local_edges(handle,
                             graph_view,
                             d_new_in,
                             d_new_rank,
                             std::move(d_rnd_indices),
                             static_cast<edge_t>(k_level),
                             global_degree_offsets,
                             global_adjacency_list_offsets);
        d_out_src   = std::move(temp_d_out_src);
        d_out_dst   = std::move(temp_d_out_dst);
        d_out_ranks = std::move(temp_d_out_ranks);
        d_indices   = std::move(temp_d_indices);
      } else {
        auto&& [temp_d_out_src, temp_d_out_dst, temp_d_out_ranks, temp_d_indices] =
          gather_one_hop_edgelist(
            handle, graph_view, d_new_in, d_new_rank, global_adjacency_list_offsets);
        d_out_src   = std::move(temp_d_out_src);
        d_out_dst   = std::move(temp_d_out_dst);
        d_out_ranks = std::move(temp_d_out_ranks);
        d_indices   = std::move(temp_d_indices);
      }

      // resize accumulators:
      //
      auto old_sz = d_acc_dst.size();
      auto add_sz = d_out_dst.size();
      auto new_sz = old_sz + add_sz;

      d_acc_src.resize(new_sz, handle.get_stream());
      d_acc_dst.resize(new_sz, handle.get_stream());
      d_acc_ranks.resize(new_sz, handle.get_stream());
      d_acc_indices.resize(new_sz, handle.get_stream());

      // zip quad; must be done after resizing,
      // because they grow from one iteration to another,
      // so iterators could be invalidated:
      //
      auto acc_zip_it =
        thrust::make_zip_iterator(thrust::make_tuple(d_acc_src.begin() + old_sz,
                                                     d_acc_dst.begin() + old_sz,
                                                     d_acc_ranks.begin() + old_sz,
                                                     d_acc_indices.begin() + old_sz));

      // union step:
      //
      auto out_zip_it = thrust::make_zip_iterator(thrust::make_tuple(
        d_out_src.begin(), d_out_dst.begin(), d_out_ranks.begin(), d_indices.begin()));

      thrust::copy_n(handle.get_thrust_policy(), out_zip_it, add_sz, acc_zip_it);

      // shuffle step: update input for self_rank
      // zipping is necessary to preserve rank info during shuffle!
      //
      auto next_in_zip_begin =
        thrust::make_zip_iterator(thrust::make_tuple(d_out_dst.begin(), d_out_ranks.begin()));
      auto next_in_zip_end =
        thrust::make_zip_iterator(thrust::make_tuple(d_out_dst.end(), d_out_ranks.end()));

      update_input_by_rank(handle,
                           graph_view,
                           next_in_zip_begin,
                           next_in_zip_end,
                           static_cast<size_t>(self_rank),
                           d_in,
                           d_ranks,
                           gpu_t{});

      ++level;
    }

    return std::make_tuple(
      std::move(d_acc_src), std::move(d_acc_dst), std::move(d_acc_ranks), std::move(d_acc_indices));
  } else {
    CUGRAPH_FAIL("Neighborhood sampling functionality is supported only for the multi-gpu case.");
  }
}

}  // namespace detail

template <typename graph_view_t, typename gpu_t, typename index_t>
std::tuple<std::tuple<rmm::device_uvector<typename graph_view_t::vertex_type>,
                      rmm::device_uvector<typename graph_view_t::vertex_type>,
                      rmm::device_uvector<gpu_t>,
                      rmm::device_uvector<index_t>>,
           std::vector<size_t>>
uniform_nbr_sample(raft::handle_t const& handle,
                   graph_view_t const& graph_view,
                   typename graph_view_t::vertex_type const* ptr_d_start,
                   gpu_t const* ptr_d_ranks,
                   size_t num_starting_vs,
                   std::vector<int> const& h_fan_out,
                   bool flag_replacement)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using edge_t   = typename graph_view_t::edge_type;

  size_t const self_rank = handle.get_comms().get_rank();

  // shuffle input data to its corresponding rank;
  // (Note: shuffle prims require mutable iterators)
  //
  detail::device_vec_t<vertex_t> d_start_vs(num_starting_vs, handle.get_stream());
  detail::device_vec_t<gpu_t> d_ranks(num_starting_vs, handle.get_stream());
  // ...hence copy required:
  //
  thrust::copy_n(handle.get_thrust_policy(), ptr_d_start, num_starting_vs, d_start_vs.begin());
  thrust::copy_n(handle.get_thrust_policy(), ptr_d_ranks, num_starting_vs, d_ranks.begin());

  // shuffle data to local rank:
  //
  auto next_in_zip_begin =
    thrust::make_zip_iterator(thrust::make_tuple(d_start_vs.begin(), d_ranks.begin()));

  auto next_in_zip_end =
    thrust::make_zip_iterator(thrust::make_tuple(d_start_vs.end(), d_ranks.end()));

  detail::update_input_by_rank(handle,
                               graph_view,
                               next_in_zip_begin,
                               next_in_zip_end,
                               self_rank,
                               d_start_vs,
                               d_ranks,
                               gpu_t{});

  // preamble step for out-degree info:
  //
  auto&& [global_degree_offsets, global_out_degrees] =
    detail::get_global_degree_information(handle, graph_view);
  auto&& global_adjacency_list_offsets = detail::get_global_adjacency_offset(
    handle, graph_view, global_degree_offsets, global_out_degrees);

  // extract output quad SOA:
  //
  auto&& [d_src, d_dst, d_gpus, d_indices] =
    detail::uniform_nbr_sample_impl(handle,
                                    graph_view,
                                    d_start_vs,
                                    d_ranks,
                                    h_fan_out,
                                    global_out_degrees,
                                    global_degree_offsets,
                                    global_adjacency_list_offsets,
                                    flag_replacement);

  // shuffle quad SOA by d_gpus:
  //
  return detail::shuffle_to_target_gpu_ids(handle, d_src, d_dst, d_gpus, d_indices);
}

}  // namespace cugraph
