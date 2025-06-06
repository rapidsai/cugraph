/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "prims/extract_transform_if_e.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace {

// FIXME: this function (after modification) may be useful for SSSP with the near-far method to
// determine the near-far threshold.
// add new roots till the sum of the degrees first becomes no smaller than degree_sum_threshold and
// returns a triplet of (new roots, number of scanned candidates, sum of the degrees of the new
// roots)
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           typename GraphViewType::vertex_type,
           typename GraphViewType::edge_type>
accumulate_new_roots(raft::handle_t const& handle,
                     vertex_partition_device_view_t<typename GraphViewType::vertex_type,
                                                    GraphViewType::is_multi_gpu> vertex_partition,
                     typename GraphViewType::vertex_type const* components,
                     typename GraphViewType::edge_type const* degrees,
                     typename GraphViewType::vertex_type const* candidate_first,
                     typename GraphViewType::vertex_type const* candidate_last,
                     typename GraphViewType::vertex_type max_new_roots,
                     typename GraphViewType::edge_type degree_sum_threshold)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  // tuning parameter (time to scan max_scan_size elements should not take significantly longer than
  // scanning a single element)
  vertex_t max_scan_size =
    static_cast<vertex_t>(handle.get_device_properties().multiProcessorCount) * vertex_t{16384};

  rmm::device_uvector<vertex_t> new_roots(max_new_roots, handle.get_stream());
  vertex_t num_new_roots{0};
  vertex_t num_scanned{0};
  edge_t degree_sum{0};
  while ((candidate_first + num_scanned < candidate_last) && (degree_sum < degree_sum_threshold) &&
         (num_new_roots < max_new_roots)) {
    auto scan_size = std::min(
      max_scan_size,
      static_cast<vertex_t>(cuda::std::distance(candidate_first + num_scanned, candidate_last)));

    rmm::device_uvector<vertex_t> tmp_new_roots(scan_size, handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_indices(tmp_new_roots.size(), handle.get_stream());
    auto input_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
      candidate_first + num_scanned, thrust::make_counting_iterator(vertex_t{0})));
    auto output_pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(tmp_new_roots.begin(), tmp_indices.begin()));
    tmp_new_roots.resize(
      static_cast<vertex_t>(cuda::std::distance(
        output_pair_first,
        thrust::copy_if(
          handle.get_thrust_policy(),
          input_pair_first,
          input_pair_first + scan_size,
          output_pair_first,
          [vertex_partition, components] __device__(auto pair) {
            auto v = thrust::get<0>(pair);
            return (
              components[vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v)] ==
              invalid_component_id<vertex_t>::value);
          }))),
      handle.get_stream());
    tmp_indices.resize(tmp_new_roots.size(), handle.get_stream());

    if (tmp_new_roots.size() > 0) {
      rmm::device_uvector<edge_t> tmp_cumulative_degrees(tmp_new_roots.size(), handle.get_stream());
      thrust::transform(
        handle.get_thrust_policy(),
        tmp_new_roots.begin(),
        tmp_new_roots.end(),
        tmp_cumulative_degrees.begin(),
        [vertex_partition, degrees] __device__(auto v) {
          return degrees[vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v)];
        });
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             tmp_cumulative_degrees.begin(),
                             tmp_cumulative_degrees.end(),
                             tmp_cumulative_degrees.begin());
      auto last = thrust::lower_bound(handle.get_thrust_policy(),
                                      tmp_cumulative_degrees.begin(),
                                      tmp_cumulative_degrees.end(),
                                      degree_sum_threshold - degree_sum);
      if (last != tmp_cumulative_degrees.end()) { ++last; }
      auto tmp_num_new_roots =
        std::min(static_cast<vertex_t>(cuda::std::distance(tmp_cumulative_degrees.begin(), last)),
                 max_new_roots - num_new_roots);

      thrust::copy(handle.get_thrust_policy(),
                   tmp_new_roots.begin(),
                   tmp_new_roots.begin() + tmp_num_new_roots,
                   new_roots.begin() + num_new_roots);
      num_new_roots += tmp_num_new_roots;
      vertex_t tmp_num_scanned{0};
      edge_t tmp_degree_sum{0};
      if (tmp_num_new_roots == static_cast<vertex_t>(tmp_new_roots.size())) {
        tmp_num_scanned = scan_size;
      } else {
        raft::update_host(std::addressof(tmp_num_scanned),
                          tmp_indices.data() + tmp_num_new_roots,
                          size_t{1},
                          handle.get_stream());
      }
      raft::update_host(std::addressof(tmp_degree_sum),
                        tmp_cumulative_degrees.data() + (tmp_num_new_roots - 1),
                        size_t{1},
                        handle.get_stream());
      handle.sync_stream();
      num_scanned += tmp_num_scanned;
      degree_sum += tmp_degree_sum;
    } else {
      num_scanned += scan_size;
    }
  }

  new_roots.resize(num_new_roots, handle.get_stream());
  new_roots.shrink_to_fit(handle.get_stream());

  return std::make_tuple(std::move(new_roots), num_scanned, degree_sum);
}

template <typename vertex_t, typename EdgeIterator>
struct e_op_t {
  __device__ vertex_t operator()(thrust::tuple<vertex_t, vertex_t> tagged_src,
                                 vertex_t dst,
                                 cuda::std::nullopt_t,
                                 cuda::std::nullopt_t,
                                 cuda::std::nullopt_t) const
  {
    auto tag = thrust::get<1>(tagged_src);
    return tag;
  }
};

template <typename vertex_t, typename EdgeIterator>
struct pred_op_t {
  detail::edge_partition_endpoint_property_device_view_t<vertex_t, vertex_t*> dst_components{};
  vertex_t dst_first{};
  EdgeIterator edge_buffer_first{};
  size_t* num_edge_inserts{};

  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t) const
  {
    auto tag        = thrust::get<1>(tagged_src);
    auto dst_offset = dst - dst_first;
    auto old =
      dst_components.elementwise_atomic_cas(dst_offset, invalid_component_id<vertex_t>::value, tag);
    if (old != invalid_component_id<vertex_t>::value && old != tag) {  // conflict
      static_assert(sizeof(unsigned long long int) == sizeof(size_t));
      auto edge_idx = atomicAdd(reinterpret_cast<unsigned long long int*>(num_edge_inserts),
                                static_cast<unsigned long long int>(1));
      // keep only the edges in the lower triangular part
      *(edge_buffer_first + edge_idx) =
        tag >= old ? thrust::make_tuple(tag, old) : thrust::make_tuple(old, tag);
    }
    return old == invalid_component_id<vertex_t>::value;
  }
};

template <typename GraphViewType>
struct v_op_t {
  using vertex_type = typename GraphViewType::vertex_type;

  vertex_partition_device_view_t<typename GraphViewType::vertex_type, GraphViewType::is_multi_gpu>
    vertex_partition{};
  vertex_type* level_components{};
  size_t bucket_idx_next{};
  size_t bucket_idx_conflict{};  // relevant only if GraphViewType::is_multi_gpu is true

  template <bool multi_gpu = GraphViewType::is_multi_gpu>
  __device__
    std::enable_if_t<multi_gpu,
                     thrust::tuple<cuda::std::optional<size_t>, cuda::std::optional<std::byte>>>
    operator()(thrust::tuple<vertex_type, vertex_type> tagged_v, int /* v_val */) const
  {
    auto tag = thrust::get<1>(tagged_v);
    auto v_offset =
      vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(thrust::get<0>(tagged_v));
    cuda::atomic_ref<vertex_type> v_component(*(level_components + v_offset));
    auto old     = invalid_component_id<vertex_type>::value;
    bool success = v_component.compare_exchange_strong(old, tag, cuda::std::memory_order_relaxed);
    if (!success && (old != tag)) {  // conflict
      return thrust::make_tuple(cuda::std::optional<size_t>{bucket_idx_conflict},
                                cuda::std::optional<std::byte>{std::byte{0}} /* dummy */);
    } else {
      return thrust::make_tuple(
        success ? cuda::std::optional<size_t>{bucket_idx_next} : cuda::std::nullopt,
        success ? cuda::std::optional<std::byte>{std::byte{0}} /* dummy */ : cuda::std::nullopt);
    }
  }

  template <bool multi_gpu = GraphViewType::is_multi_gpu>
  __device__
    std::enable_if_t<!multi_gpu,
                     thrust::tuple<cuda::std::optional<size_t>, cuda::std::optional<std::byte>>>
    operator()(thrust::tuple<vertex_type, vertex_type> /* tagged_v */, int /* v_val */) const
  {
    return thrust::make_tuple(cuda::std::optional<size_t>{bucket_idx_next},
                              cuda::std::optional<std::byte>{std::byte{0}} /* dummy */);
  }
};

template <typename GraphViewType>
void weakly_connected_components_impl(raft::handle_t const& handle,
                                      GraphViewType const& push_graph_view,
                                      typename GraphViewType::vertex_type* components,
                                      bool do_expensive_check)
{
  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using weight_t    = float;  // dummy
  using edge_type_t = int32_t;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto const num_vertices = push_graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    push_graph_view.is_symmetric(),
    "Invalid input argument: input graph should be symmetric for weakly connected components.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. recursively run multi-root frontier expansion

  constexpr size_t bucket_idx_cur      = 0;
  constexpr size_t bucket_idx_next     = 1;
  constexpr size_t bucket_idx_conflict = 2;
  constexpr size_t num_buckets         = 3;

  // tuning parameter to balance work per iteration (should be large enough to be throughput
  // bounded) vs # conflicts between frontiers with different roots (# conflicts == # edges for the
  // next level)
  auto degree_sum_threshold =
    static_cast<edge_t>(handle.get_device_properties().multiProcessorCount) * edge_t{1024};
  if constexpr (GraphViewType::is_multi_gpu) {  // to enforce that degree_sum_threshold is same in
                                                // every GPU (e.g. there is no guarantee that every
                                                // GPU has the same number of SMs)
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    degree_sum_threshold = host_scalar_allreduce(
      comm, degree_sum_threshold, raft::comms::op_t::SUM, handle.get_stream());
    degree_sum_threshold /= static_cast<edge_t>(comm_size);
  }

  size_t num_levels{0};
  graph_t<vertex_t, edge_t, GraphViewType::is_storage_transposed, GraphViewType::is_multi_gpu>
    level_graph(handle);
  rmm::device_uvector<vertex_t> level_renumber_map(0, handle.get_stream());
  std::vector<rmm::device_uvector<vertex_t>> level_component_vectors{};
  // vertex ID in this level to the component ID in the previous level
  std::vector<rmm::device_uvector<vertex_t>> level_renumber_map_vectors{};
  std::vector<std::vector<vertex_t>> level_vertex_partition_range_last_vectors{};
  std::vector<vertex_t> level_local_vertex_first_vectors{};
  while (true) {
    auto level_graph_view = num_levels == 0 ? push_graph_view : level_graph.view();
    auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
      level_graph_view.local_vertex_partition_view());
    level_component_vectors.push_back(rmm::device_uvector<vertex_t>(
      num_levels == 0 ? vertex_t{0} : level_graph_view.local_vertex_partition_range_size(),
      handle.get_stream()));
    level_renumber_map_vectors.push_back(std::move(level_renumber_map));
    {
      auto lasts = level_graph_view.vertex_partition_range_lasts();
      level_vertex_partition_range_last_vectors.emplace_back(lasts.begin(), lasts.end());
    }
    level_local_vertex_first_vectors.push_back(
      level_graph_view.local_vertex_partition_range_first());
    auto level_components =
      num_levels == 0 ? components : level_component_vectors[num_levels].data();
    ++num_levels;
    auto degrees = level_graph_view.compute_out_degrees(handle);

    // 2-1. filter out isolated vertices

    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
      thrust::make_counting_iterator(level_graph_view.local_vertex_partition_range_first()),
      degrees.begin()));
    thrust::transform(handle.get_thrust_policy(),
                      pair_first,
                      pair_first + level_graph_view.local_vertex_partition_range_size(),
                      level_components,
                      [] __device__(auto pair) {
                        auto v      = thrust::get<0>(pair);
                        auto degree = thrust::get<1>(pair);
                        return degree > 0 ? invalid_component_id<vertex_t>::value : v;
                      });

    // 2-2. decide whether to run BFS first or not

    std::optional<vertex_t> bfs_start_vertex{std::nullopt};
    if (num_levels == 1) {  // if in level 0 and there exists a high-degree vertex, run BFS first to
                            // hopefully find the  biggest connected component (to significantly
                            // reduce the graph size and accordingly, peak memory usage)
      edge_t max_degree{0};
      vertex_t max_v = std::numeric_limits<vertex_t>::max();
      if (degrees.size() > 0) {
        auto max_it =
          thrust::max_element(handle.get_thrust_policy(), degrees.begin(), degrees.end());
        auto max_v_offset = static_cast<vertex_t>(cuda::std::distance(degrees.begin(), max_it));
        raft::update_host(std::addressof(max_degree),
                          degrees.data() + max_v_offset,
                          size_t{1},
                          handle.get_stream());
        handle.sync_stream();
        max_v = level_graph_view.local_vertex_partition_range_first() + max_v_offset;
      }
      auto threshold = degree_sum_threshold;
      if constexpr (GraphViewType::is_multi_gpu) {
        auto& comm            = handle.get_comms();
        auto const comm_size  = comm.get_size();
        auto local_max_degree = max_degree;
        max_degree =
          host_scalar_allreduce(comm, max_degree, raft::comms::op_t::MAX, handle.get_stream());
        max_v = host_scalar_allreduce(
          comm,
          local_max_degree == max_degree ? max_v : std::numeric_limits<vertex_t>::max(),
          raft::comms::op_t::MIN,
          handle.get_stream());
        threshold *= std::max(
          static_cast<edge_t>(sqrt(comm_size)),
          edge_t{
            1});  // graph size often grows proportional to comm_size but maximum degree may grow
                  // slower than graph size for many practial graphs; use sqrt() to consider this.
      }
      if (max_degree > threshold) { bfs_start_vertex = max_v; }
    }

    auto edge_buffer =
      allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(0, handle.get_stream());
    rmm::device_scalar<size_t> num_edge_inserts(size_t{0}, handle.get_stream());

    // 2-3. run BFS or multi-root expansion

    if (bfs_start_vertex) {
      // 2-3-1. run BFS and find the vertices that are reachable from the starting vertex

      rmm::device_uvector<bool> visited(level_graph_view.local_vertex_partition_range_size(),
                                        handle.get_stream());
      {
        rmm::device_uvector<vertex_t> sources(0, handle.get_stream());
        if ((*bfs_start_vertex >= level_graph_view.local_vertex_partition_range_first()) &&
            (*bfs_start_vertex < level_graph_view.local_vertex_partition_range_last())) {
          sources.resize(1, handle.get_stream());
          raft::update_device(
            sources.data(), std::addressof(*bfs_start_vertex), size_t{1}, handle.get_stream());
        }
        rmm::device_uvector<vertex_t> distances(visited.size(), handle.get_stream());
        bfs(handle,
            level_graph_view,
            distances.data(),
            static_cast<vertex_t*>(nullptr),
            sources.data(),
            sources.size(),
            true /* direction_optimizing */,
            std::numeric_limits<vertex_t>::max());
        thrust::transform(handle.get_thrust_policy(),
                          distances.begin(),
                          distances.end(),
                          visited.begin(),
                          [invalid_distance = std::numeric_limits<vertex_t>::max()] __device__(
                            auto d) { return d != invalid_distance; });
      }

      // 2-3-2. set level_components[] = start_vertex if reachable from the starting vertex and
      // itself if not reachable

      auto pair_first = thrust::make_zip_iterator(
        thrust::make_counting_iterator(level_graph_view.local_vertex_partition_range_first()),
        visited.begin());
      thrust::transform_if(handle.get_thrust_policy(),
                           pair_first,
                           pair_first + level_graph_view.local_vertex_partition_range_size(),
                           degrees.begin(),
                           level_components,
                           cuda::proclaim_return_type<vertex_t>(
                             [start_vertex = *bfs_start_vertex] __device__(auto pair) {
                               auto v       = thrust::get<0>(pair);
                               auto visited = thrust::get<1>(pair);
                               return visited ? start_vertex : v;
                             }),
                           cuda::proclaim_return_type<bool>([] __device__(auto d) {
                             return d > edge_t{0};
                           }) /* skip isolated vertices (already updated) */);

      // 2-3-3. extract edges that are unreachable from the starting vertex

      edge_src_property_t<vertex_t, bool> edge_src_visited(handle, level_graph_view);
      update_edge_src_property(
        handle, level_graph_view, visited.begin(), edge_src_visited.mutable_view());
      {
        auto tmp_graph_view = level_graph_view;
        edge_property_t<edge_t, bool> edge_mask(handle, level_graph_view);
        if (tmp_graph_view.has_edge_mask()) {
          tmp_graph_view.clear_edge_mask();
          cugraph::fill_edge_property(handle,
                                      tmp_graph_view,
                                      edge_mask.mutable_view(),
                                      false);  // if level_graph_view has an attached edge mask,
                                               // masked out values of edge_mask won't be updated.
        }
        transform_e(handle,
                    level_graph_view,
                    edge_src_visited.view(),
                    edge_dst_dummy_property_t{}.view(),
                    edge_dummy_property_t{}.view(),
                    cuda::proclaim_return_type<bool>(
                      [] __device__(auto src, auto dst, bool src_visited, auto, auto) {
                        return (src > dst) /* keep only the edges in the lower triangular part */ &&
                               !src_visited;
                      }),
                    edge_mask.mutable_view());
        tmp_graph_view.attach_edge_mask(edge_mask.view());
        std::tie(std::get<0>(edge_buffer),
                 std::get<1>(edge_buffer),
                 std::ignore,
                 std::ignore,
                 std::ignore) = decompress_to_edgelist<vertex_t,
                                                       edge_t,
                                                       weight_t,
                                                       edge_type_t,
                                                       GraphViewType::is_storage_transposed,
                                                       GraphViewType::is_multi_gpu>(
          handle, tmp_graph_view, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
      }
      thrust::sort(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(edge_buffer),
                   get_dataframe_buffer_end(edge_buffer));
      resize_dataframe_buffer(
        edge_buffer,
        cuda::std::distance(get_dataframe_buffer_begin(edge_buffer),
                            thrust::unique(handle.get_thrust_policy(),
                                           get_dataframe_buffer_begin(edge_buffer),
                                           get_dataframe_buffer_end(edge_buffer))),
        handle.get_stream());
      auto num_edges = size_dataframe_buffer(edge_buffer);
      num_edge_inserts.set_value_async(num_edges, handle.get_stream());
      handle.sync_stream();  // to ensure that the above set_value_async is completed before
                             // num_edges is freed
    } else {
      // 2-3-1. initialize new root candidates

      // Vertices are first partitioned to high-degree vertices and low-degree vertices, we can
      // reach degree_sum_threshold with fewer high-degree vertices leading to a higher compression
      // ratio. The degree threshold is set to ceil(sqrt(degree_sum_threshold * 2)); this guarantees
      // the compression ratio of at least 50% (ignoring rounding errors) even if all the selected
      // roots fall into a single connected component as there will be at least as many non-root
      // vertices in the connected component (assuming there are no multi-edges, if there are
      // multi-edges, we may not get 50% compression in # vertices but still get compression in #
      // edges). the remaining low-degree vertices will be randomly shuffled so comparable ratios of
      // vertices will be selected as roots in the remaining connected components.

      rmm::device_uvector<vertex_t> new_root_candidates(
        level_graph_view.local_vertex_partition_range_size(), handle.get_stream());
      new_root_candidates.resize(
        cuda::std::distance(
          new_root_candidates.begin(),
          thrust::copy_if(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(level_graph_view.local_vertex_partition_range_first()),
            thrust::make_counting_iterator(level_graph_view.local_vertex_partition_range_last()),
            new_root_candidates.begin(),
            [vertex_partition, level_components] __device__(auto v) {
              return level_components[vertex_partition
                                        .local_vertex_partition_offset_from_vertex_nocheck(v)] ==
                     invalid_component_id<vertex_t>::value;
            })),
        handle.get_stream());
      auto high_degree_partition_last = thrust::stable_partition(
        handle.get_thrust_policy(),
        new_root_candidates.begin(),
        new_root_candidates.end(),
        [vertex_partition,
         degrees   = degrees.data(),
         threshold = static_cast<edge_t>(
           ceil(sqrt(static_cast<double>(degree_sum_threshold) * 2.0)))] __device__(auto v) {
          return degrees[vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v)] >=
                 threshold;
        });
      thrust::shuffle(handle.get_thrust_policy(),
                      high_degree_partition_last,
                      new_root_candidates.end(),
                      thrust::default_random_engine());

      double constexpr max_new_roots_ratio =
        0.05;  // to avoid selecting all the vertices as roots leading to zero compression
      static_assert(max_new_roots_ratio > 0.0);
      auto max_new_roots = std::max(
        static_cast<vertex_t>(new_root_candidates.size() * max_new_roots_ratio), vertex_t{1});

      auto init_max_new_roots = max_new_roots;
      if (GraphViewType::is_multi_gpu) {
        auto& comm           = handle.get_comms();
        auto const comm_rank = comm.get_rank();
        auto const comm_size = comm.get_size();

        auto first_candidate_degree = thrust::transform_reduce(
          handle.get_thrust_policy(),
          new_root_candidates.begin(),
          new_root_candidates.begin() + (new_root_candidates.size() > 0 ? 1 : 0),
          cuda::proclaim_return_type<edge_t>(
            [vertex_partition, degrees = degrees.data()] __device__(auto v) {
              return degrees[vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v)];
            }),
          edge_t{0},
          thrust::plus<edge_t>{});

        auto first_candidate_degrees =
          host_scalar_gather(comm, first_candidate_degree, int{0}, handle.get_stream());
        auto new_root_candidate_counts =
          host_scalar_gather(comm, new_root_candidates.size(), int{0}, handle.get_stream());

        if (comm_rank == 0) {
          std::vector<vertex_t> init_max_new_root_counts(comm_size, vertex_t{0});

          // if there exists very high degree vertices, we can exceed degree_sum_threshold *
          // comm_size with fewer than one root per GPU
          if (std::reduce(first_candidate_degrees.begin(), first_candidate_degrees.end()) >
              degree_sum_threshold * comm_size) {
            std::vector<std::tuple<edge_t, int>> degree_gpu_id_pairs(comm_size);
            for (int i = 0; i < comm_size; ++i) {
              degree_gpu_id_pairs[i] = std::make_tuple(first_candidate_degrees[i], i);
            }
            std::sort(degree_gpu_id_pairs.begin(),
                      degree_gpu_id_pairs.end(),
                      [](auto lhs, auto rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });
            edge_t sum{0};
            for (size_t i = 0; i < degree_gpu_id_pairs.size(); ++i) {
              sum += std::get<0>(degree_gpu_id_pairs[i]);
              init_max_new_root_counts[std::get<1>(degree_gpu_id_pairs[i])] = 1;
              if (sum > degree_sum_threshold * comm_size) { break; }
            }
          }
          // to avoid selecting too many (possibly all) vertices as initial roots leading to no
          // compression in the worst case.
          else if (level_graph_view.number_of_vertices() <=
                   static_cast<vertex_t>(comm_size * ceil(1.0 / max_new_roots_ratio))) {
            std::vector<int> gpu_ids{};
            gpu_ids.reserve(
              std::reduce(new_root_candidate_counts.begin(), new_root_candidate_counts.end()));
            for (size_t i = 0; i < new_root_candidate_counts.size(); ++i) {
              gpu_ids.insert(gpu_ids.end(), new_root_candidate_counts[i], static_cast<int>(i));
            }
            std::random_device rd{};
            std::shuffle(gpu_ids.begin(), gpu_ids.end(), std::mt19937(rd()));
            gpu_ids.resize(
              std::max(static_cast<vertex_t>(gpu_ids.size() * max_new_roots_ratio), vertex_t{1}));
            for (size_t i = 0; i < gpu_ids.size(); ++i) {
              ++init_max_new_root_counts[gpu_ids[i]];
            }
          } else {
            std::fill(init_max_new_root_counts.begin(),
                      init_max_new_root_counts.end(),
                      std::numeric_limits<vertex_t>::max());
          }

          init_max_new_roots =
            host_scalar_scatter(comm, init_max_new_root_counts, int{0}, handle.get_stream());
        } else {
          init_max_new_roots =
            host_scalar_scatter(comm, std::vector<vertex_t>{}, int{0}, handle.get_stream());
        }

        handle.sync_stream();
        init_max_new_roots = std::min(init_max_new_roots, max_new_roots);
      }

      // 2-3-2. initialize vertex frontier, edge_buffer, and edge_dst_components (if
      // multi-gpu)

      vertex_frontier_t<vertex_t, vertex_t, GraphViewType::is_multi_gpu, true> vertex_frontier(
        handle, num_buckets);
      vertex_t next_candidate_offset{0};
      edge_t edge_count{0};

      auto edge_dst_components =
        GraphViewType::is_multi_gpu
          ? edge_dst_property_t<vertex_t, vertex_t>(handle, level_graph_view)
          : edge_dst_property_t<vertex_t, vertex_t>(handle) /* dummy */;
      if constexpr (GraphViewType::is_multi_gpu) {
        fill_edge_dst_property(handle,
                               level_graph_view,
                               edge_dst_components.mutable_view(),
                               invalid_component_id<vertex_t>::value);
      }

      // 2-3-3 iterate till every vertex gets visited

      size_t iter{0};
      while (true) {
        if ((edge_count < degree_sum_threshold) &&
            (next_candidate_offset < static_cast<vertex_t>(new_root_candidates.size()))) {
          auto [new_roots, num_scanned, degree_sum] = accumulate_new_roots<GraphViewType>(
            handle,
            vertex_partition,
            level_components,
            degrees.data(),
            new_root_candidates.data() + next_candidate_offset,
            new_root_candidates.data() + new_root_candidates.size(),
            iter == 0 ? init_max_new_roots : max_new_roots,
            degree_sum_threshold - edge_count);
          next_candidate_offset += num_scanned;
          edge_count += degree_sum;

          thrust::sort(handle.get_thrust_policy(), new_roots.begin(), new_roots.end());

          thrust::for_each(
            handle.get_thrust_policy(),
            new_roots.begin(),
            new_roots.end(),
            [vertex_partition, components = level_components] __device__(auto c) {
              components[vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(c)] = c;
            });

          auto pair_first =
            thrust::make_zip_iterator(thrust::make_tuple(new_roots.begin(), new_roots.begin()));
          vertex_frontier.bucket(bucket_idx_cur).insert(pair_first, pair_first + new_roots.size());
        }

        if (vertex_frontier.bucket(bucket_idx_cur).aggregate_size() == 0) { break; }

        if constexpr (GraphViewType::is_multi_gpu) {
          update_edge_dst_property(
            handle,
            level_graph_view,
            thrust::get<0>(vertex_frontier.bucket(bucket_idx_cur).begin().get_iterator_tuple()),
            thrust::get<0>(vertex_frontier.bucket(bucket_idx_cur).end().get_iterator_tuple()),
            level_components,
            edge_dst_components.mutable_view());
        }

        auto max_pushes = GraphViewType::is_multi_gpu
                            ? static_cast<edge_t>(compute_num_out_nbrs_from_frontier(
                                handle, level_graph_view, vertex_frontier.bucket(bucket_idx_cur)))
                            : edge_count;

        // FIXME: if we use cuco::static_map (no duplicates, ideally we need static_set),
        // edge_buffer size cannot exceed (# roots)^2 and we can avoid additional sort & unique (but
        // resizing the buffer may be more expensive).
        auto old_num_edge_inserts = num_edge_inserts.value(handle.get_stream());
        resize_dataframe_buffer(
          edge_buffer, old_num_edge_inserts + max_pushes, handle.get_stream());

        auto new_frontier_tagged_vertex_buffer =
          cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
            handle,
            level_graph_view,
            vertex_frontier.bucket(bucket_idx_cur),
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            edge_dummy_property_t{}.view(),
            e_op_t<vertex_t, decltype(get_dataframe_buffer_begin(edge_buffer))>{},
            reduce_op::null(),
            pred_op_t<vertex_t, decltype(get_dataframe_buffer_begin(edge_buffer))>{
              GraphViewType::is_multi_gpu
                ? detail::edge_partition_endpoint_property_device_view_t<vertex_t, vertex_t*>(
                    edge_dst_components.mutable_view())
                : detail::edge_partition_endpoint_property_device_view_t<vertex_t, vertex_t*>(
                    make_edge_dst_property_mutable_view<vertex_t, vertex_t>(
                      level_graph_view,
                      level_components,
                      static_cast<size_t>(level_graph_view.local_vertex_partition_range_size()))),
              level_graph_view.local_edge_partition_dst_range_first(),
              get_dataframe_buffer_begin(edge_buffer),
              num_edge_inserts.data()});

        auto next_frontier_bucket_indices =
          GraphViewType::is_multi_gpu ? std::vector<size_t>{bucket_idx_next, bucket_idx_conflict}
                                      : std::vector<size_t>{bucket_idx_next};
        update_v_frontier(
          handle,
          level_graph_view,
          std::move(new_frontier_tagged_vertex_buffer),
          vertex_frontier,
          raft::host_span<size_t const>(next_frontier_bucket_indices.data(),
                                        next_frontier_bucket_indices.size()),
          thrust::make_constant_iterator(0) /* dummy */,
          thrust::make_discard_iterator() /* dummy */,
          v_op_t<GraphViewType>{
            vertex_partition, level_components, bucket_idx_next, bucket_idx_conflict});

        if (GraphViewType::is_multi_gpu) {
          auto cur_num_edge_inserts = num_edge_inserts.value(handle.get_stream());
          auto& conflict_bucket     = vertex_frontier.bucket(bucket_idx_conflict);
          resize_dataframe_buffer(
            edge_buffer, cur_num_edge_inserts + conflict_bucket.size(), handle.get_stream());
          thrust::for_each(
            handle.get_thrust_policy(),
            conflict_bucket.begin(),
            conflict_bucket.end(),
            [vertex_partition,
             level_components,
             edge_buffer_first = get_dataframe_buffer_begin(edge_buffer),
             num_edge_inserts  = num_edge_inserts.data()] __device__(auto tagged_v) {
              auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(
                thrust::get<0>(tagged_v));
              auto old = *(level_components + v_offset);
              auto tag = thrust::get<1>(tagged_v);
              static_assert(sizeof(unsigned long long int) == sizeof(size_t));
              auto edge_idx = atomicAdd(reinterpret_cast<unsigned long long int*>(num_edge_inserts),
                                        static_cast<unsigned long long int>(1));
              // keep only the edges in the lower triangular part
              *(edge_buffer_first + edge_idx) =
                tag >= old ? thrust::make_tuple(tag, old) : thrust::make_tuple(old, tag);
            });
          conflict_bucket.clear();
        }

        // maintain the list of sorted unique edges (we can avoid this if we use cuco::static_map(no
        // duplicates, ideally we need static_set)).
        auto new_num_edge_inserts = num_edge_inserts.value(handle.get_stream());
        if (new_num_edge_inserts > old_num_edge_inserts) {
          auto edge_first = get_dataframe_buffer_begin(edge_buffer);
          thrust::sort(handle.get_thrust_policy(),
                       edge_first + old_num_edge_inserts,
                       edge_first + new_num_edge_inserts);
          if (old_num_edge_inserts > 0) {
            auto tmp_edge_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
              new_num_edge_inserts, handle.get_stream());
            auto tmp_edge_first = get_dataframe_buffer_begin(tmp_edge_buffer);
            thrust::merge(handle.get_thrust_policy(),
                          edge_first,
                          edge_first + old_num_edge_inserts,
                          edge_first + old_num_edge_inserts,
                          edge_first + new_num_edge_inserts,
                          tmp_edge_first);
            edge_buffer = std::move(tmp_edge_buffer);
          }
          edge_first            = get_dataframe_buffer_begin(edge_buffer);
          auto unique_edge_last = thrust::unique(
            handle.get_thrust_policy(), edge_first, edge_first + new_num_edge_inserts);
          auto num_unique_edges =
            static_cast<size_t>(cuda::std::distance(edge_first, unique_edge_last));
          num_edge_inserts.set_value_async(num_unique_edges, handle.get_stream());
          handle.sync_stream();  // to ensure that the above set_value_async is completed before
                                 // num_unique_edges is freed
        }

        vertex_frontier.bucket(bucket_idx_cur).clear();
        vertex_frontier.bucket(bucket_idx_cur).shrink_to_fit();
        vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
        edge_count = thrust::transform_reduce(
          handle.get_thrust_policy(),
          thrust::get<0>(vertex_frontier.bucket(bucket_idx_cur).begin().get_iterator_tuple()),
          thrust::get<0>(vertex_frontier.bucket(bucket_idx_cur).end().get_iterator_tuple()),
          cuda::proclaim_return_type<edge_t>(
            [vertex_partition, degrees = degrees.data()] __device__(auto v) {
              return degrees[vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v)];
            }),
          edge_t{0},
          thrust::plus<edge_t>());

        ++iter;
      }
    }

    // 2-4. construct the next level graph from the edges emitted on conflicts

    auto num_inserts           = num_edge_inserts.value(handle.get_stream());
    auto aggregate_num_inserts = num_inserts;
    if (GraphViewType::is_multi_gpu) {
      auto& comm = handle.get_comms();
      aggregate_num_inserts =
        host_scalar_allreduce(comm, num_inserts, raft::comms::op_t::SUM, handle.get_stream());
    }

    if (aggregate_num_inserts > 0) {
      resize_dataframe_buffer(
        edge_buffer, static_cast<size_t>(num_inserts * 2), handle.get_stream());
      auto input_first  = get_dataframe_buffer_begin(edge_buffer);
      auto output_first = thrust::make_zip_iterator(
                            thrust::make_tuple(thrust::get<1>(input_first.get_iterator_tuple()),
                                               thrust::get<0>(input_first.get_iterator_tuple()))) +
                          num_inserts;
      thrust::copy(
        handle.get_thrust_policy(), input_first, input_first + num_inserts, output_first);

      if (GraphViewType::is_multi_gpu) {
        std::tie(std::get<0>(edge_buffer),
                 std::get<1>(edge_buffer),
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore) =
          shuffle_ext_edges<vertex_t, edge_t, weight_t, edge_type_t, int32_t>(
            handle,
            std::move(std::get<0>(edge_buffer)),
            std::move(std::get<1>(edge_buffer)),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            GraphViewType::is_storage_transposed);
        auto edge_first = get_dataframe_buffer_begin(edge_buffer);
        auto edge_last  = get_dataframe_buffer_end(edge_buffer);
        thrust::sort(handle.get_thrust_policy(), edge_first, edge_last);
        auto unique_edge_last = thrust::unique(handle.get_thrust_policy(), edge_first, edge_last);
        resize_dataframe_buffer(
          edge_buffer,
          static_cast<size_t>(cuda::std::distance(edge_first, unique_edge_last)),
          handle.get_stream());
        shrink_to_fit_dataframe_buffer(edge_buffer, handle.get_stream());
      }

      std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
      std::tie(level_graph, std::ignore, std::ignore, std::ignore, tmp_renumber_map) =
        create_graph_from_edgelist<vertex_t,
                                   edge_t,
                                   float /* dummy */,
                                   int32_t /* dummy */,
                                   GraphViewType::is_storage_transposed,
                                   GraphViewType::is_multi_gpu>(handle,
                                                                std::nullopt,
                                                                std::move(std::get<0>(edge_buffer)),
                                                                std::move(std::get<1>(edge_buffer)),
                                                                std::nullopt,
                                                                std::nullopt,
                                                                std::nullopt,
                                                                graph_properties_t{true, false},
                                                                true);
      level_renumber_map = std::move(*tmp_renumber_map);
    } else {
      break;
    }
  }

  // 3. recursive update the current level component IDs from the next level component IDs

  for (size_t i = 0; i < num_levels - 1; ++i) {
    size_t next_level    = num_levels - 1 - i;
    size_t current_level = next_level - 1;

    unrenumber_int_vertices<vertex_t, GraphViewType::is_multi_gpu>(
      handle,
      level_component_vectors[next_level].data(),
      level_component_vectors[next_level].size(),
      level_renumber_map_vectors[next_level].data(),
      raft::host_span<vertex_t const>(
        level_vertex_partition_range_last_vectors[next_level].data(),
        level_vertex_partition_range_last_vectors[next_level]
          .size()));  // level_compponent_vectors[next_leve][]: component IDs are originally vertex
                      // IDs of next_level, convert this to the vertex IDs of current_level
    kv_store_t<vertex_t, vertex_t, true> next_level_kv_store(
      std::move(level_renumber_map_vectors[next_level]),
      std::move(level_component_vectors[next_level]),
      invalid_vertex_id_v<vertex_t>,
      false /* key sorted */,
      handle.get_stream());
    auto next_level_kv_store_view = next_level_kv_store.view();
    auto current_level_components =
      (current_level == 0) ? components : level_component_vectors[current_level].data();
    auto current_level_local_vertex_partition_range_size =
      (current_level == 0) ? push_graph_view.local_vertex_partition_range_size()
                           : level_component_vectors[current_level].size();
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();
      auto key_func              = detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
        comm_size, major_comm_size, minor_comm_size};

      rmm::device_uvector<vertex_t> sorted_unique_keys(
        current_level_local_vertex_partition_range_size, handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   current_level_components,
                   current_level_components + current_level_local_vertex_partition_range_size,
                   sorted_unique_keys.begin());
      thrust::sort(
        handle.get_thrust_policy(), sorted_unique_keys.begin(), sorted_unique_keys.end());
      sorted_unique_keys.resize(cuda::std::distance(sorted_unique_keys.begin(),
                                                    thrust::unique(handle.get_thrust_policy(),
                                                                   sorted_unique_keys.begin(),
                                                                   sorted_unique_keys.end())),
                                handle.get_stream());
      auto [unique_keys, values_for_unique_keys] = collect_values_for_unique_keys(
        handle.get_comms(),
        next_level_kv_store.view(),
        std::move(sorted_unique_keys),
        cuda::proclaim_return_type<int>([key_func] __device__(auto key) { return key_func(key); }),
        handle.get_stream());
      auto kv_pair_first =
        thrust::make_zip_iterator(unique_keys.begin(), values_for_unique_keys.begin());
      unique_keys.resize(
        cuda::std::distance(
          kv_pair_first,
          thrust::remove_if(handle.get_thrust_policy(),
                            kv_pair_first,
                            kv_pair_first + unique_keys.size(),
                            cuda::proclaim_return_type<bool>(
                              [invalid_value = invalid_vertex_id_v<vertex_t>] __device__(
                                auto pair) { return thrust::get<1>(pair) == invalid_value; }))),
        handle.get_stream());
      values_for_unique_keys.resize(unique_keys.size(), handle.get_stream());
      next_level_kv_store      = kv_store_t<vertex_t, vertex_t, true>(std::move(unique_keys),
                                                                 std::move(values_for_unique_keys),
                                                                 invalid_vertex_id_v<vertex_t>,
                                                                 false /* key sorted */,
                                                                 handle.get_stream());
      next_level_kv_store_view = next_level_kv_store.view();
    }
    auto device_view = detail::kv_binary_search_store_device_view_t(next_level_kv_store_view);
    thrust::transform(
      handle.get_thrust_policy(),
      current_level_components,
      current_level_components + current_level_local_vertex_partition_range_size,
      current_level_components,
      cuda::proclaim_return_type<vertex_t>(
        [device_view, invalid_value = invalid_vertex_id_v<vertex_t>] __device__(auto old) {
          auto val = device_view.find(old);
          return val != invalid_value ? val : old;
        }));
  }
}

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
void weakly_connected_components(raft::handle_t const& handle,
                                 graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                                 vertex_t* components,
                                 bool do_expensive_check)
{
  weakly_connected_components_impl(handle, graph_view, components, do_expensive_check);
}

}  // namespace cugraph
