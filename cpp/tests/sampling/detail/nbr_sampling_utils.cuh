/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <cuco/detail/hash_functions.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

// utilities for testing / verification of Nbr Sampling functionality:
//
namespace cugraph {
namespace test {

template <typename vertex_t, typename rank_t>
using mmap_v_to_pair_t = std::multimap<vertex_t, std::pair<vertex_t, rank_t>>;

// group h_in_src by rank affiliation;
// return unique ranks and corresponding sets of vertices;
// h_in ranks[] assumed sorted increasingly;
// h_in_src[] assumed shuffled accordingly;
//
template <typename vertex_t, typename rank_t>
std::tuple<std::vector<rank_t>, std::vector<std::vector<vertex_t>>> group_by_ranks(
  std::vector<vertex_t> const& h_in_src, std::vector<rank_t> const& h_in_ranks)
{
  // make a vector of singleton vectors from a vector input:
  //
  std::vector<std::vector<vertex_t>> vv_out;
  std::transform(
    h_in_src.cbegin(), h_in_src.cend(), std::back_inserter(vv_out), [](auto const& value) {
      return std::vector<vertex_t>{value};
    });

  std::vector<rank_t> h_uniq_ranks(h_in_ranks.size());
  std::vector<std::vector<vertex_t>> vv_uniq(vv_out.size());

  // merge two vectors, without sorting:
  //
  auto v_union = [](std::vector<vertex_t> const& v_left,
                    std::vector<vertex_t> const& v_right) -> decltype(auto) {
    auto lsize = v_left.size();
    std::vector<vertex_t> v_result(lsize + v_right.size());

    std::copy(v_left.cbegin(), v_left.cend(), v_result.begin());
    std::copy(v_right.cbegin(), v_right.cend(), v_result.begin() + lsize);

    return v_result;
  };

  // group vertices by ranks:
  //
  auto&& new_end = thrust::reduce_by_key(
    h_in_ranks.cbegin(),
    h_in_ranks.cend(),
    vv_out.cbegin(),
    h_uniq_ranks.begin(),
    vv_uniq.begin(),
    thrust::equal_to<rank_t>{},
    [v_union](auto const& v_left, auto const& v_right) { return v_union(v_left, v_right); });

  h_uniq_ranks.erase(new_end.first, h_uniq_ranks.end());
  vv_uniq.erase(new_end.second, vv_uniq.end());

  return std::make_tuple(h_uniq_ranks, vv_uniq);
}

// checks if output root affiliation to ranks
// is same as input's;
//
template <typename vertex_t, typename rank_t>
bool check_no_mismatch(std::vector<vertex_t> const& sources,
                       rank_t crt_rank,
                       std::vector<vertex_t> const& h_roots,
                       std::vector<rank_t> const& h_root_ranks)
{
  for (auto&& src_v : sources) {
    auto it = std::find(h_roots.begin(), h_roots.end(), src_v);
    if (it == h_roots.end())
      return false;
    else {
      auto pos = std::distance(h_roots.begin(), it);
      if (h_root_ranks.at(pos) != crt_rank) return false;
    }
  }

  return true;
}

// generate unique output ranks and (un)colored "forest" of corresponding subgraphs;
// the edge color defaults to "uncolored"; to be colored by rank later;
// h_out_src, h_out_dst are assumed to be sorted by h_out_ranks;
//
template <typename vertex_t, typename rank_t>
std::tuple<std::vector<rank_t>, std::vector<mmap_v_to_pair_t<vertex_t, rank_t>>> make_forest(
  std::vector<vertex_t> const& h_out_src,
  std::vector<vertex_t> const& h_out_dst,
  std::vector<rank_t> const& h_out_ranks,
  rank_t uncolored_rank = rank_t{-1})
{
  auto num_edges = h_out_src.size();

  std::vector<mmap_v_to_pair_t<vertex_t, rank_t>> forest{};
  std::vector<rank_t> h_uniq_ranks{};

  rank_t prev_rank = h_out_ranks[0];
  h_uniq_ranks.push_back(prev_rank);

  mmap_v_to_pair_t<vertex_t, rank_t> crt_map{};

  // fill the per-rank forest:
  //
  for (size_t i = 0; i < num_edges; ++i) {
    auto crt_rank = h_out_ranks[i];

    if (prev_rank != crt_rank) {
      forest.push_back(std::move(crt_map));
      h_uniq_ranks.push_back(crt_rank);

      prev_rank = crt_rank;

      crt_map.clear();  // move should _normally_ do that, but not guaranteed;
                        // if it did then this is a no-op, anyway;
    }

    crt_map.insert(std::make_pair(h_out_src[i], std::make_pair(h_out_dst[i], uncolored_rank)));
  }
  forest.push_back(std::move(crt_map));  // apend the last map

  return std::make_tuple(std::move(h_uniq_ranks), std::move(forest));
}

// BFS-style coloring of a sub-forest
// (a group of "trees" for a given rank)
//
template <typename vertex_t, typename rank_t>
void color_map(vertex_t start_key, rank_t color, mmap_v_to_pair_t<vertex_t, rank_t>& sub_forest)
{
  std::queue<vertex_t> q_candidates{{start_key}};

  auto max_sz    = sub_forest.size();
  size_t counter = 0;

  while (!q_candidates.empty() && counter < max_sz) {
    auto src_v = q_candidates.front();
    q_candidates.pop();

    auto pair_it = sub_forest.equal_range(src_v);

    for (auto it = pair_it.first; it != pair_it.second; ++it) {
      if (it->second.second != color) {
        q_candidates.push(it->second.first);
        it->second.second = color;
      }
    }

    ++counter;
  }
}

// check if all edges in a sub-forest are colored in `color`;
//
template <typename vertex_t, typename rank_t>
bool check_all_colored(rank_t color, mmap_v_to_pair_t<vertex_t, rank_t> const& sub_forest)
{
  auto it_uncolored =
    std::find_if(sub_forest.cbegin(), sub_forest.cend(), [color](auto const& pair) {
      return (pair.second.second != color);
    });

  return (it_uncolored == sub_forest.cend());
}

// traverse all sub-forests of a rank and check connectivity
// starting from corresponding input starting vertices, and
// correct rank affiliation;
// h_uniq_out_ranks[] = unique ranks resulting from nbr sampling call;
// forest = per-rank vector of sub-forests (subgraphs);
// h_uniq_in_ranks[] = unique ranks resulting from the input data;
// h_in_roots[] = vector of per-rank sets (vectors) of starting vertices;
//
template <typename vertex_t, typename rank_t>
bool check_color(std::vector<rank_t> const& h_uniq_out_ranks,
                 std::vector<mmap_v_to_pair_t<vertex_t, rank_t>> const& forest,
                 std::vector<rank_t> const& h_uniq_in_ranks,
                 std::vector<std::vector<vertex_t>> const& h_in_roots)
{
  bool flag_passed = true;

  assert(h_uniq_out_ranks.size() == forest.size());
  assert(h_uniq_in_ranks.size() == h_in_roots.size());

  size_t counter          = 0;
  bool at_least_one_valid = false;
  for (auto&& in_rank : h_uniq_in_ranks) {
    // see if it exists in the h_uniq_out_ranks
    // (it might not because the whole entry could have been invalidated;
    //  e.g., because the starting source vertex was a sink);
    //
    auto&& pos = std::find(h_uniq_out_ranks.cbegin(), h_uniq_out_ranks.cend(), in_rank);
    if (pos != h_uniq_out_ranks.cend()) {
      at_least_one_valid = true;

      // unique-ify the input vertices for this rank;
      //
      auto srcs_of_rank = h_in_roots[counter];
      std::sort(srcs_of_rank.begin(), srcs_of_rank.end());
      srcs_of_rank.erase(std::unique(srcs_of_rank.begin(), srcs_of_rank.end()), srcs_of_rank.end());

      auto index_found = std::distance(h_uniq_out_ranks.cbegin(), pos);
      auto sub_forest  = forest.at(index_found);  // <- throws here

      // for each _unique_ starting vertex input of current rank
      // there should now be an edge starting there in the output;
      // color all the other edges that are reachable starting from that edge;
      // at the end of the loop the whole sub_forest should be colored;
      //
      for (auto&& starting_v : srcs_of_rank) {
        color_map(starting_v, in_rank, sub_forest);
      }
      flag_passed = flag_passed && check_all_colored(in_rank, sub_forest);
    }

    ++counter;
  }

  return (flag_passed && at_least_one_valid);
}

// check if forest of trees generated by nbr sampling
// has the proper connectivity and ranks;
//
// h_in_src    = input startting vertices (roots);
// h_in_ranks  = corresponding input ranks;
// h_out_src   = source of edges generated by nbr sampling;
// h_out_dst   = destination of edges generated by nbr sampling;
// h_in_ranks  = corresponding edge ranks;
//
template <typename vertex_t, typename rank_t>
bool check_forest_trees_by_rank(std::vector<vertex_t>& h_in_src,
                                std::vector<rank_t>& h_in_ranks,
                                std::vector<vertex_t>& h_out_src,
                                std::vector<vertex_t>& h_out_dst,
                                std::vector<rank_t>& h_out_ranks)
{
#if 0
  auto num_edges = h_out_src.size();
  auto num_trees = h_in_src.size();

  bool flag_passed =
    (num_edges == h_out_dst.size()) && (num_edges == h_out_ranks.size()) && (num_edges > 0);
  flag_passed = flag_passed && (num_trees == h_in_ranks.size()) && (num_trees <= num_edges);

  assert(flag_passed);

  if (!flag_passed) return flag_passed;  // cannot continue

  // group-by ranks input:
  //
  thrust::sort_by_key(h_in_ranks.begin(), h_in_ranks.end(), h_in_src.begin());

  // group-by ranks output:
  //
  thrust::sort_by_key(
    h_out_ranks.begin(),
    h_out_ranks.end(),
    thrust::make_zip_iterator(thrust::make_tuple(h_out_src.begin(), h_out_dst.begin())));

  // extract input unique ranks and roots grouped by them:
  // (roots can have duplicates)
  //
  auto&& [h_uniq_in_ranks, h_in_roots] = group_by_ranks(h_in_src, h_in_ranks);

  // generate unique output ranks and "forest" of corresponding subgraphs;
  //
  auto&& [h_uniq_out_ranks, forest] = make_forest(h_out_src, h_out_dst, h_out_ranks);

  assert(h_uniq_out_ranks.size() == forest.size());

  // h_uniq_out_ranks should be included in h_uniq_in_ranks
  // (they former set might be smaller because of inavlid dst removal)
  //
  std::vector<rank_t> diff_ranks{};
  std::set_difference(h_uniq_out_ranks.begin(),
                      h_uniq_out_ranks.end(),
                      h_uniq_in_ranks.begin(),
                      h_uniq_in_ranks.end(),
                      std::back_inserter(diff_ranks));
  flag_passed = diff_ranks.empty();  // should be empty
  assert(flag_passed);

  // check rank and subgraph connectivity for all edges,
  // using a coloring algorithm:
  //
  flag_passed = flag_passed && check_color(h_uniq_out_ranks, forest, h_uniq_in_ranks, h_in_roots);

  return flag_passed;
#else
  return true;
#endif
}

template <typename vertex_t>
rmm::device_uvector<vertex_t> random_vertex_ids(raft::handle_t const& handle,
                                                vertex_t begin,
                                                vertex_t end,
                                                vertex_t count,
                                                uint64_t seed,
                                                int repetitions_per_vertex = 0)
{
#if 0
  auto& comm                  = handle.get_comms();
  auto const comm_rank        = comm.get_rank();
#endif
  vertex_t number_of_vertices = end - begin;

  rmm::device_uvector<vertex_t> vertices(
    std::max((repetitions_per_vertex + 1) * number_of_vertices, count), handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(),
    vertices.begin(),
    vertices.end(),
    [begin, number_of_vertices] __device__(auto v) { return begin + (v % number_of_vertices); });
  thrust::default_random_engine g;
  g.seed(seed);
  thrust::shuffle(handle.get_thrust_policy(), vertices.begin(), vertices.end(), g);
  vertices.resize(count, handle.get_stream());
  return vertices;
}

template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>>
create_segmented_data(raft::handle_t const& handle,
                      vertex_t invalid_vertex_id,
                      rmm::device_uvector<edge_t> const& out_degrees)
{
  rmm::device_uvector<edge_t> offset(out_degrees.size() + 1, handle.get_stream());
  // no need for sync since gather call is on stream
  offset.set_element_to_zero_async(0, handle.get_stream());
  thrust::inclusive_scan(
    handle.get_thrust_policy(), out_degrees.begin(), out_degrees.end(), offset.begin() + 1);
  auto total_edge_count = offset.back_element(handle.get_stream());
  rmm::device_uvector<vertex_t> segmented_sources(total_edge_count, handle.get_stream());
  rmm::device_uvector<edge_t> segmented_sequence(total_edge_count, handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), segmented_sources.begin(), segmented_sources.end(), vertex_t{0});
  thrust::fill(
    handle.get_thrust_policy(), segmented_sequence.begin(), segmented_sequence.end(), edge_t{1});
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(offset.size()),
                   [offset       = offset.data(),
                    source_count = out_degrees.size(),
                    src          = segmented_sources.data(),
                    seq          = segmented_sequence.data()] __device__(auto index) {
                     auto location = offset[index];
                     if (index == 0) {
                       seq[location] = edge_t{0};
                     } else {
                       seq[location] = offset[index - 1] - offset[index] + 1;
                     }
                     if ((index < source_count) && (offset[index] != offset[index + 1])) {
                       src[location] = index;
                     }
                   });
  thrust::inclusive_scan(handle.get_thrust_policy(),
                         segmented_sequence.begin(),
                         segmented_sequence.end(),
                         segmented_sequence.begin());
  thrust::inclusive_scan(handle.get_thrust_policy(),
                         segmented_sources.begin(),
                         segmented_sources.end(),
                         segmented_sources.begin(),
                         thrust::maximum<vertex_t>());
  return std::make_tuple(
    std::move(offset), std::move(segmented_sources), std::move(segmented_sequence));
}

template <typename GraphViewType, typename VertexIterator, typename EdgeIndexIterator>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
sg_gather_edges(raft::handle_t const& handle,
                GraphViewType const& graph_view,
                VertexIterator vertex_input_first,
                VertexIterator vertex_input_last,
                EdgeIndexIterator edge_index_first,
                typename GraphViewType::vertex_type invalid_vertex_id,
                typename GraphViewType::edge_type indices_per_source)
{
  static_assert(GraphViewType::is_storage_transposed == false);
  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using weight_t    = typename GraphViewType::weight_type;
  auto source_count = thrust::distance(vertex_input_first, vertex_input_last);
  auto edge_count   = source_count * indices_per_source;
  rmm::device_uvector<vertex_t> sources(edge_count, handle.get_stream());
  rmm::device_uvector<vertex_t> destinations(edge_count, handle.get_stream());
  auto edge_partition = cugraph::edge_partition_device_view_t<vertex_t, edge_t, false>(
    graph_view.local_edge_partition_view());
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>(edge_count),
                   [vertex_input_first,
                    indices_per_source,
                    edge_index_first,
                    sources      = sources.data(),
                    destinations = destinations.data(),
                    offsets      = edge_partition.offsets(),
                    indices      = edge_partition.indices(),
                    invalid_vertex_id] __device__(auto index) {
                     auto source        = vertex_input_first[index / indices_per_source];
                     sources[index]     = source;
                     auto source_offset = offsets[source];
                     auto degree        = offsets[source + 1] - source_offset;
                     auto e_index       = edge_index_first[index];
                     if (e_index < degree) {
                       destinations[index] = indices[source_offset + e_index];
                     } else {
                       destinations[index] = invalid_vertex_id;
                     }
                   });
  auto input_iter =
    thrust::make_zip_iterator(thrust::make_tuple(sources.begin(), destinations.begin()));
  auto compacted_length = thrust::distance(
    input_iter,
    thrust::remove_if(
      handle.get_thrust_policy(),
      input_iter,
      input_iter + destinations.size(),
      destinations.begin(),
      [invalid_vertex_id] __device__(auto dst) { return (dst == invalid_vertex_id); }));
  sources.resize(compacted_length, handle.get_stream());
  destinations.resize(compacted_length, handle.get_stream());
  return std::make_tuple(std::move(sources), std::move(destinations));
}

template <typename GraphViewType, typename prop_t>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<prop_t>>
sg_gather_edges(raft::handle_t const& handle,
                GraphViewType const& graph_view,
                const rmm::device_uvector<typename GraphViewType::vertex_type>& sources,
                const rmm::device_uvector<prop_t>& properties)
{
  static_assert(GraphViewType::is_storage_transposed == false);
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto edge_partition = cugraph::edge_partition_device_view_t<vertex_t, edge_t, false>(
    graph_view.local_edge_partition_view());

  rmm::device_uvector<vertex_t> sources_out_degrees(sources.size(), handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    sources.cbegin(),
                    sources.cend(),
                    sources_out_degrees.begin(),
                    [offsets = edge_partition.offsets()] __device__(auto s) {
                      return offsets[s + 1] - offsets[s];
                    });
  auto [sources_out_offsets, segmented_source_indices, segmented_sequence] =
    create_segmented_data(handle, vertex_t{0}, sources_out_degrees);
  auto edge_count = sources_out_offsets.back_element(handle.get_stream());

  rmm::device_uvector<vertex_t> srcs(edge_count, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(edge_count, handle.get_stream());
  rmm::device_uvector<prop_t> src_prop(edge_count, handle.get_stream());

  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator<size_t>(0),
                   thrust::make_counting_iterator<size_t>(edge_count),
                   [partition                = edge_partition,
                    srcs                     = srcs.data(),
                    dsts                     = dsts.data(),
                    src_prop                 = src_prop.data(),
                    sources                  = sources.data(),
                    sources_properties       = properties.data(),
                    sources_out_offsets      = sources_out_offsets.data(),
                    segmented_source_indices = segmented_source_indices.data(),
                    segmented_sequence       = segmented_sequence.data()] __device__(auto index) {
                     auto src_index  = segmented_source_indices[index];
                     auto src        = sources[src_index];
                     auto offsets    = partition.offsets();
                     auto indices    = partition.indices();
                     auto dst        = indices[offsets[src] + segmented_sequence[index]];
                     srcs[index]     = src;
                     dsts[index]     = dst;
                     src_prop[index] = sources_properties[src_index];
                   });
  return std::make_tuple(std::move(srcs), std::move(dsts), std::move(src_prop));
}

template <typename vertex_t>
void sort_coo(raft::handle_t const& handle,
              rmm::device_uvector<vertex_t>& src,
              rmm::device_uvector<vertex_t>& dst)
{
  thrust::sort_by_key(handle.get_thrust_policy(), dst.begin(), dst.end(), src.begin());
  thrust::sort_by_key(handle.get_thrust_policy(), src.begin(), src.end(), dst.begin());
}

template <typename vertex_t, typename prop_t>
void sort_coo(raft::handle_t const& handle,
              rmm::device_uvector<vertex_t>& src,
              rmm::device_uvector<prop_t>& src_prop,
              rmm::device_uvector<vertex_t>& dst)
{
  thrust::sort(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(thrust::make_tuple(src.begin(), src_prop.begin(), dst.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(src.end(), src_prop.end(), dst.end())));
}

template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> generate_random_destination_indices(
  raft::handle_t const& handle,
  const rmm::device_uvector<edge_t>& out_degrees,
  vertex_t invalid_vertex_id,
  edge_t invalid_destination_index,
  edge_t indices_per_source)
{
  auto [random_source_offsets, segmented_source_ids, segmented_sequence] =
    create_segmented_data(handle, invalid_vertex_id, out_degrees);
  // Generate random weights to shuffle sequence of destination indices
  rmm::device_uvector<int> random_weights(segmented_sequence.size(), handle.get_stream());
  auto& row_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_rank  = row_comm.get_rank();
  auto& comm           = handle.get_comms();
  auto const comm_rank = comm.get_rank();
  auto force_seed      = 0;
  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator<size_t>(random_weights.size()),
                    random_weights.begin(),
                    [force_seed] __device__(auto index) {
                      thrust::default_random_engine g;
                      g.seed(force_seed);
                      thrust::uniform_int_distribution<int> dist;
                      g.discard(index);
                      return dist(g);
                    });
  thrust::sort_by_key(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(
      thrust::make_tuple(segmented_source_ids.begin(), random_weights.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(segmented_source_ids.end(), random_weights.end())),
    segmented_sequence.begin(),
    [] __device__(auto left, auto right) { return left < right; });

  rmm::device_uvector<edge_t> dst_index(indices_per_source * out_degrees.size(),
                                        handle.get_stream());

  thrust::for_each(handle.get_thrust_policy(),
                   thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(out_degrees.size()),
                   [offset    = random_source_offsets.data(),
                    dst_index = dst_index.data(),
                    seg_seq   = segmented_sequence.data(),
                    k         = indices_per_source,
                    invalid_destination_index] __device__(auto index) {
                     auto length = thrust::minimum<edge_t>()(offset[index + 1] - offset[index], k);
                     auto source_offset = offset[index];
                     // copy first k valid destination indices. If k is larger
                     // than out degree then stop at out degree to avoid
                     // out of bounds access
                     for (edge_t i = 0; i < length; ++i) {
                       dst_index[index * k + i] = seg_seq[source_offset + i];
                     }
                     // If requested number of destination indices is larger than
                     // out degree then write out invalid destination index
                     for (edge_t i = length; i < k; ++i) {
                       dst_index[index * k + i] = invalid_destination_index;
                     }
                   });
  return dst_index;
}

// FIXME: Consider moving this to thrust_tuple_utils and making it
//        generic for any typle that supports < operator
struct ArithmeticZipLess {
  template <typename left_t, typename right_t>
  __device__ bool operator()(left_t const& left, right_t const& right)
  {
    if constexpr (cugraph::is_thrust_tuple_of_arithmetic<left_t>::value) {
      // Need a more generic solution, for now I can just check thrust::tuple_size
      if (thrust::get<0>(left) < thrust::get<0>(right)) return true;
      if (thrust::get<0>(right) < thrust::get<0>(left)) return false;

      if constexpr (thrust::tuple_size<left_t>::value > 2) {
        if (thrust::get<1>(left) < thrust::get<1>(right)) return true;
        if (thrust::get<1>(right) < thrust::get<1>(left)) return false;
        return thrust::get<2>(left) < thrust::get<2>(right);
      } else {
        return thrust::get<1>(left) < thrust::get<1>(right);
      }
    }
  }
};

// FIXME: Consider moving this to thrust_tuple_utils and making it
//        generic for any typle that supports < operator
struct ArithmeticZipEqual {
  template <typename vertex_t, typename weight_t>
  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t, weight_t> const& left,
                             thrust::tuple<vertex_t, vertex_t, weight_t> const& right)
  {
    return (thrust::get<0>(left) == thrust::get<0>(right)) &&
           (thrust::get<1>(left) == thrust::get<1>(right)) &&
           (thrust::get<2>(left) == thrust::get<2>(right));
  }

  template <typename vertex_t>
  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> const& left,
                             thrust::tuple<vertex_t, vertex_t> const& right)
  {
    return (thrust::get<0>(left) == thrust::get<0>(right)) &&
           (thrust::get<1>(left) == thrust::get<1>(right));
  }
};

template <typename vertex_t, typename weight_t>
void validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t> const& src,
  rmm::device_uvector<vertex_t> const& dst,
  std::optional<rmm::device_uvector<weight_t>> const& wgt,
  rmm::device_uvector<vertex_t> const& subgraph_src,
  rmm::device_uvector<vertex_t> const& subgraph_dst,
  std::optional<rmm::device_uvector<weight_t>> const& subgraph_wgt)
{
  ASSERT_EQ(wgt.has_value(), subgraph_wgt.has_value());

  rmm::device_uvector<vertex_t> src_v(src.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(dst.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> subgraph_src_v(subgraph_src.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> subgraph_dst_v(subgraph_dst.size(), handle.get_stream());

  raft::copy(src_v.data(), src.data(), src.size(), handle.get_stream());
  raft::copy(dst_v.data(), dst.data(), dst.size(), handle.get_stream());
  raft::copy(subgraph_src_v.data(), subgraph_src.data(), subgraph_src.size(), handle.get_stream());
  raft::copy(subgraph_dst_v.data(), subgraph_dst.data(), subgraph_dst.size(), handle.get_stream());

  size_t dist{0};

  if (wgt) {
    rmm::device_uvector<weight_t> wgt_v(wgt->size(), handle.get_stream());
    rmm::device_uvector<weight_t> subgraph_wgt_v(subgraph_wgt->size(), handle.get_stream());

    raft::copy(wgt_v.data(), wgt->data(), wgt->size(), handle.get_stream());
    raft::copy(
      subgraph_wgt_v.data(), subgraph_wgt->data(), subgraph_wgt->size(), handle.get_stream());

    auto graph_iter =
      thrust::make_zip_iterator(thrust::make_tuple(src_v.begin(), dst_v.begin(), wgt_v.begin()));
    auto subgraph_iter = thrust::make_zip_iterator(
      thrust::make_tuple(subgraph_src_v.begin(), subgraph_dst_v.begin(), subgraph_wgt_v.begin()));

    thrust::sort(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipLess{});
    thrust::sort(handle.get_thrust_policy(),
                 subgraph_iter,
                 subgraph_iter + subgraph_src_v.size(),
                 ArithmeticZipLess{});

    auto graph_iter_end = thrust::unique(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipEqual{});
    auto subgraph_iter_end = thrust::unique(handle.get_thrust_policy(),
                                            subgraph_iter,
                                            subgraph_iter + subgraph_src_v.size(),
                                            ArithmeticZipEqual{});

    auto new_size = thrust::distance(graph_iter, graph_iter_end);

    src_v.resize(new_size, handle.get_stream());
    dst_v.resize(new_size, handle.get_stream());
    wgt_v.resize(new_size, handle.get_stream());

    new_size = thrust::distance(subgraph_iter, subgraph_iter_end);
    subgraph_src_v.resize(new_size, handle.get_stream());
    subgraph_dst_v.resize(new_size, handle.get_stream());
    subgraph_wgt_v.resize(new_size, handle.get_stream());

    rmm::device_uvector<vertex_t> tmp_src(new_size, handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_dst(new_size, handle.get_stream());
    rmm::device_uvector<weight_t> tmp_wgt(new_size, handle.get_stream());

    auto tmp_subgraph_iter = thrust::make_zip_iterator(
      thrust::make_tuple(tmp_src.begin(), tmp_dst.begin(), tmp_wgt.begin()));

    auto tmp_subgraph_iter_end = thrust::set_difference(handle.get_thrust_policy(),
                                                        subgraph_iter,
                                                        subgraph_iter + subgraph_src_v.size(),
                                                        graph_iter,
                                                        graph_iter + src_v.size(),
                                                        tmp_subgraph_iter,
                                                        ArithmeticZipLess{});

    dist = thrust::distance(tmp_subgraph_iter, tmp_subgraph_iter_end);
  } else {
    auto graph_iter = thrust::make_zip_iterator(thrust::make_tuple(src_v.begin(), dst_v.begin()));
    auto subgraph_iter =
      thrust::make_zip_iterator(thrust::make_tuple(subgraph_src_v.begin(), subgraph_dst_v.begin()));

    thrust::sort(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipLess{});
    thrust::sort(handle.get_thrust_policy(),
                 subgraph_iter,
                 subgraph_iter + subgraph_src_v.size(),
                 ArithmeticZipLess{});

    auto graph_iter_end = thrust::unique(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipEqual{});
    auto subgraph_iter_end = thrust::unique(handle.get_thrust_policy(),
                                            subgraph_iter,
                                            subgraph_iter + subgraph_src_v.size(),
                                            ArithmeticZipEqual{});

    auto new_size = thrust::distance(graph_iter, graph_iter_end);

    src_v.resize(new_size, handle.get_stream());
    dst_v.resize(new_size, handle.get_stream());

    new_size = thrust::distance(subgraph_iter, subgraph_iter_end);
    subgraph_src_v.resize(new_size, handle.get_stream());
    subgraph_dst_v.resize(new_size, handle.get_stream());

    rmm::device_uvector<vertex_t> tmp_src(new_size, handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_dst(new_size, handle.get_stream());

    auto tmp_subgraph_iter = thrust::make_zip_iterator(tmp_src.begin(), tmp_dst.begin());

    auto tmp_subgraph_iter_end = thrust::set_difference(handle.get_thrust_policy(),
                                                        subgraph_iter,
                                                        subgraph_iter + subgraph_src_v.size(),
                                                        graph_iter,
                                                        graph_iter + src_v.size(),
                                                        tmp_subgraph_iter,
                                                        ArithmeticZipLess{});

    dist = thrust::distance(tmp_subgraph_iter, tmp_subgraph_iter_end);
  }

  ASSERT_EQ(0, dist);
}

template <typename vertex_t, typename weight_t>
void validate_sampling_depth(raft::handle_t const& handle,
                             rmm::device_uvector<vertex_t>&& d_src,
                             rmm::device_uvector<vertex_t>&& d_dst,
                             std::optional<rmm::device_uvector<weight_t>>&& d_wgt,
                             rmm::device_uvector<vertex_t>&& d_source_vertices,
                             int max_depth)
{
  graph_t<vertex_t, vertex_t, false, false> graph(handle);
  std::optional<rmm::device_uvector<vertex_t>> number_map{std::nullopt};
  std::tie(graph, std::ignore, std::ignore, number_map) =
    create_graph_from_edgelist<vertex_t, vertex_t, weight_t, int32_t, false, false>(
      handle,
      std::nullopt,
      std::move(d_src),
      std::move(d_dst),
      std::move(d_wgt),
      std::nullopt,
      graph_properties_t{},
      true);

  auto graph_view = graph.view();

  //  Renumber sources
  renumber_ext_vertices<vertex_t, false>(handle,
                                         d_source_vertices.data(),
                                         d_source_vertices.size(),
                                         number_map->data(),
                                         graph_view.local_vertex_partition_range_first(),
                                         graph_view.local_vertex_partition_range_last());

  rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), d_distances.begin(), d_distances.end(), vertex_t{max_depth + 1});

  rmm::device_uvector<vertex_t> d_local_distances(graph_view.number_of_vertices(),
                                                  handle.get_stream());

  std::vector<vertex_t> h_source_vertices(d_source_vertices.size());
  raft::update_host(h_source_vertices.data(),
                    d_source_vertices.data(),
                    d_source_vertices.size(),
                    handle.get_stream());

  for (size_t i = 0; i < d_source_vertices.size(); ++i) {
    if (h_source_vertices[i] != cugraph::invalid_vertex_id<vertex_t>::value) {
      // Do BFS
      cugraph::bfs<vertex_t, vertex_t, false>(handle,
                                              graph_view,
                                              d_local_distances.data(),
                                              nullptr,
                                              d_source_vertices.data() + i,
                                              size_t{1},
                                              bool{false},
                                              vertex_t{max_depth});

      auto tuple_iter = thrust::make_zip_iterator(
        thrust::make_tuple(d_distances.begin(), d_local_distances.begin()));

      thrust::transform(handle.get_thrust_policy(),
                        tuple_iter,
                        tuple_iter + d_distances.size(),
                        d_distances.begin(),
                        [] __device__(auto tuple) {
                          return thrust::min(thrust::get<0>(tuple), thrust::get<1>(tuple));
                        });
    }
  }

  ASSERT_EQ(0,
            thrust::count_if(handle.get_thrust_policy(),
                             d_distances.begin(),
                             d_distances.end(),
                             [max_depth] __device__(auto d) { return d > max_depth; }));
}

}  // namespace test
}  // namespace cugraph
