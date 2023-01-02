/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <detail/graph_utils.cuh>
#include <prims/kv_store.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {
namespace detail {

template <typename vertex_t>
struct check_edge_src_and_dst_t {
  vertex_t const* sorted_majors{nullptr};
  vertex_t num_majors{0};
  vertex_t const* sorted_minors{nullptr};
  vertex_t num_minors{0};

  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> e) const
  {
    return !thrust::binary_search(
             thrust::seq, sorted_majors, sorted_majors + num_majors, thrust::get<0>(e)) ||
           !thrust::binary_search(
             thrust::seq, sorted_minors, sorted_minors + num_minors, thrust::get<1>(e));
  }
};

template <typename vertex_t>
struct find_unused_id_t {
  raft::device_span<vertex_t const> sorted_local_vertices{};
  size_t num_workers{};
  compute_gpu_id_from_ext_vertex_t<vertex_t> gpu_id_op{};
  int comm_rank{};
  vertex_t invalid_id{};

  __device__ vertex_t operator()(size_t worker_id) const
  {
    for (size_t i = worker_id; i < sorted_local_vertices.size() + size_t{1}; i += num_workers) {
      auto start = (i == size_t{0}) ? std::numeric_limits<vertex_t>::lowest()
                                    : sorted_local_vertices[i - size_t{1}];
      if (start != std::numeric_limits<vertex_t>::max()) { ++start; };  // now inclusive
      auto end = (i == sorted_local_vertices.size()) ? std::numeric_limits<vertex_t>::max()
                                                     : sorted_local_vertices[i];  // exclusive
      for (vertex_t v = start; v < end; ++v) {
        if (gpu_id_op(v) == comm_rank) { return v; }
      }
    }
    return invalid_id;
  }
};

template <typename vertex_t, typename edge_t>
struct search_and_increment_degree_t {
  vertex_t const* sorted_vertices{nullptr};
  vertex_t num_vertices{0};
  edge_t* degrees{nullptr};

  __device__ void operator()(thrust::tuple<vertex_t, edge_t> vertex_degree_pair) const
  {
    auto it = thrust::lower_bound(thrust::seq,
                                  sorted_vertices,
                                  sorted_vertices + num_vertices,
                                  thrust::get<0>(vertex_degree_pair));
    *(degrees + thrust::distance(sorted_vertices, it)) += thrust::get<1>(vertex_degree_pair);
  }
};

template <typename vertex_t>
std::optional<vertex_t> find_locally_unused_ext_vertex_id(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> sorted_local_vertices,
  bool multi_gpu)
{
  // 1. check whether we can quickly find a locally unused external vertex ID (this should be the
  // case except for some pathological cases)

  // 1.1 look for a vertex ID outside the edge source/destination range this GPU covers

  if (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto const row_comm_rank = row_comm.get_rank();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();
    auto const col_comm_rank = col_comm.get_rank();
    if ((row_comm_size < comm_size) &&
        (col_comm_size < comm_size)) {  // if neither of the edge source/destination range covers
                                        // the entire vertex range
      std::vector<bool> locally_used(comm_size, false);
      for (int i = 0; i < col_comm_size; ++i) {
        locally_used[i * row_comm_size + row_comm_rank] = true;
      }
      for (int i = 0; i < row_comm_size; ++i) {
        locally_used[col_comm_rank * row_comm_size + i] = true;
      }
      assert(std::find(locally_used.begin(), locally_used.end(), false) != locally_used.end());
      std::optional<vertex_t> ret{std::nullopt};
      vertex_t v     = std::numeric_limits<vertex_t>::lowest();
      auto gpu_id_op = compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size};
      while (true) {  // the acutal loop count should be smaller than or comparable to comm_size
        if (!locally_used[gpu_id_op(v)]) {
          ret = v;
          break;
        }
        if (v == std::numeric_limits<vertex_t>::max()) { break; }
        ++v;
      }
      auto found = static_cast<bool>(host_scalar_allreduce(
        comm, static_cast<int>(ret.has_value()), raft::comms::op_t::MIN, handle.get_stream()));
      if (found) { return ret; }
    }
  }

  // 1.2. look for a vertex ID outside the [min, max] vertex IDs used in the entire input graph

  auto min = std::numeric_limits<vertex_t>::max();
  auto max = std::numeric_limits<vertex_t>::lowest();
  if (sorted_local_vertices.size() > size_t{0}) {
    raft::update_host(&min, sorted_local_vertices.data(), size_t{1}, handle.get_stream());
    raft::update_host(&max,
                      sorted_local_vertices.data() + (sorted_local_vertices.size() - size_t{1}),
                      size_t{1},
                      handle.get_stream());
    handle.sync_stream();
  }
  if (multi_gpu && (handle.get_comms().get_size() > int{1})) {
    min =
      host_scalar_allreduce(handle.get_comms(), min, raft::comms::op_t::MIN, handle.get_stream());
    max =
      host_scalar_allreduce(handle.get_comms(), max, raft::comms::op_t::MAX, handle.get_stream());
  }
  if (min > std::numeric_limits<vertex_t>::lowest()) {
    return std::numeric_limits<vertex_t>::lowest();
  }
  if (max < std::numeric_limits<vertex_t>::max()) { return std::numeric_limits<vertex_t>::max(); }

  // 2. in case the vertex ID range covers [std::numeric_limits<vertex_t>::lowest(),
  // std::numeric_limits<vertex_t>::max()] (this is very unlikely to be the case in reality, but for
  // completeness)

  auto num_workers =
    std::min(static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * size_t{1024},
             sorted_local_vertices.size() + size_t{1});
  auto gpu_id_op =
    compute_gpu_id_from_ext_vertex_t<vertex_t>{multi_gpu ? handle.get_comms().get_size() : int{1}};
  auto unused_id = thrust::transform_reduce(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(num_workers),
    find_unused_id_t<vertex_t>{sorted_local_vertices,
                               num_workers,
                               gpu_id_op,
                               multi_gpu ? handle.get_comms().get_rank() : int{0},
                               std::numeric_limits<vertex_t>::max()},
    std::numeric_limits<vertex_t>::max(),  // already taken in the step 1.2, so this can't be a
                                           // valid answer
    thrust::minimum<vertex_t>{});

  if (multi_gpu && (handle.get_comms().get_size() > int{1})) {
    unused_id = host_scalar_allreduce(
      handle.get_comms(), unused_id, raft::comms::op_t::MIN, handle.get_stream());
  }

  return (unused_id != std::numeric_limits<vertex_t>::max())
           ? std::make_optional<vertex_t>(unused_id)
           : std::nullopt /* if the entire range of vertex_t is used */;
}

// returns renumber map and segment_offsets
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<vertex_t>, vertex_t> compute_renumber_map(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  std::vector<vertex_t const*> const& edgelist_majors,
  std::vector<vertex_t const*> const& edgelist_minors,
  std::vector<edge_t> const& edgelist_edge_counts)
{
  rmm::device_uvector<vertex_t> sorted_local_vertices(0, handle.get_stream());

  edge_t num_local_edges = std::reduce(edgelist_edge_counts.begin(), edgelist_edge_counts.end());

  // 1. if local_vertices.has_value() is false, find unique vertices from edge majors (to construct
  // local_vertices)

  rmm::device_uvector<vertex_t> sorted_unique_majors(0, handle.get_stream());
  if (!local_vertices) {
    sorted_unique_majors.resize(num_local_edges, handle.get_stream());
    size_t major_offset{0};
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_majors[i],
                   edgelist_majors[i] + edgelist_edge_counts[i],
                   sorted_unique_majors.begin() + major_offset);
      thrust::sort(handle.get_thrust_policy(),
                   sorted_unique_majors.begin() + major_offset,
                   sorted_unique_majors.begin() + major_offset + edgelist_edge_counts[i]);
      major_offset += static_cast<size_t>(thrust::distance(
        sorted_unique_majors.begin() + major_offset,
        thrust::unique(handle.get_thrust_policy(),
                       sorted_unique_majors.begin() + major_offset,
                       sorted_unique_majors.begin() + major_offset + edgelist_edge_counts[i])));
    }
    sorted_unique_majors.resize(major_offset, handle.get_stream());

    if (edgelist_majors.size() > 1) {
      thrust::sort(
        handle.get_thrust_policy(), sorted_unique_majors.begin(), sorted_unique_majors.end());
    }
    sorted_unique_majors.shrink_to_fit(handle.get_stream());
  }

  // 2. if local_vertices.has_value() is false, find unique vertices from edge minors (to construct
  // local_vertices)

  rmm::device_uvector<vertex_t> sorted_unique_minors(0, handle.get_stream());
  if (!local_vertices) {
    sorted_unique_minors.resize(num_local_edges, handle.get_stream());
    size_t minor_offset{0};
    for (size_t i = 0; i < edgelist_minors.size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_minors[i],
                   edgelist_minors[i] + edgelist_edge_counts[i],
                   sorted_unique_minors.begin() + minor_offset);
      thrust::sort(handle.get_thrust_policy(),
                   sorted_unique_minors.begin() + minor_offset,
                   sorted_unique_minors.begin() + minor_offset + edgelist_edge_counts[i]);
      minor_offset += static_cast<size_t>(thrust::distance(
        sorted_unique_minors.begin() + minor_offset,
        thrust::unique(handle.get_thrust_policy(),
                       sorted_unique_minors.begin() + minor_offset,
                       sorted_unique_minors.begin() + minor_offset + edgelist_edge_counts[i])));
    }
    sorted_unique_minors.resize(minor_offset, handle.get_stream());
    if (edgelist_minors.size() > 1) {
      thrust::sort(
        handle.get_thrust_policy(), sorted_unique_minors.begin(), sorted_unique_minors.end());
      sorted_unique_minors.resize(thrust::distance(sorted_unique_minors.begin(),
                                                   thrust::unique(handle.get_thrust_policy(),
                                                                  sorted_unique_minors.begin(),
                                                                  sorted_unique_minors.end())),
                                  handle.get_stream());
    }
    sorted_unique_minors.shrink_to_fit(handle.get_stream());
  }

  // 3. update sorted_local_vertices.
  // if local_vertices.has_value() is false, reconstruct local_vertices first

  if (local_vertices) {
    sorted_local_vertices = std::move(*local_vertices);
    thrust::sort(
      handle.get_thrust_policy(), sorted_local_vertices.begin(), sorted_local_vertices.end());
  } else {
    sorted_local_vertices.resize(sorted_unique_majors.size() + sorted_unique_minors.size(),
                                 handle.get_stream());

    thrust::merge(handle.get_thrust_policy(),
                  sorted_unique_majors.begin(),
                  sorted_unique_majors.end(),
                  sorted_unique_minors.begin(),
                  sorted_unique_minors.end(),
                  sorted_local_vertices.begin());

    sorted_unique_majors.resize(0, handle.get_stream());
    sorted_unique_majors.shrink_to_fit(handle.get_stream());
    sorted_unique_minors.resize(0, handle.get_stream());
    sorted_unique_minors.shrink_to_fit(handle.get_stream());

    sorted_local_vertices.resize(thrust::distance(sorted_local_vertices.begin(),
                                                  thrust::unique(handle.get_thrust_policy(),
                                                                 sorted_local_vertices.begin(),
                                                                 sorted_local_vertices.end())),
                                 handle.get_stream());
    sorted_local_vertices.shrink_to_fit(handle.get_stream());

    if constexpr (multi_gpu) {
      sorted_local_vertices =
        cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
          handle, std::move(sorted_local_vertices));
      thrust::sort(
        handle.get_thrust_policy(), sorted_local_vertices.begin(), sorted_local_vertices.end());
      sorted_local_vertices.resize(thrust::distance(sorted_local_vertices.begin(),
                                                    thrust::unique(handle.get_thrust_policy(),
                                                                   sorted_local_vertices.begin(),
                                                                   sorted_local_vertices.end())),
                                   handle.get_stream());
      sorted_local_vertices.shrink_to_fit(handle.get_stream());
    }
  }

  auto locally_unused_vertex_id = find_locally_unused_ext_vertex_id(
    handle,
    raft::device_span<vertex_t const>(sorted_local_vertices.data(), sorted_local_vertices.size()),
    multi_gpu);
  CUGRAPH_EXPECTS(locally_unused_vertex_id.has_value(),
                  "Invalid input arguments: there is no unused value in the entire range of "
                  "vertex_t, increase vertex_t to 64 bit.");

  // 4. compute global degrees for the sorted local vertices

  rmm::device_uvector<edge_t> sorted_local_vertex_degrees(0, handle.get_stream());
  std::optional<std::vector<size_t>> stream_pool_indices{
    std::nullopt};  // FIXME: move this inside the if statement
  if constexpr (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    auto constexpr num_chunks = size_t{
      2};  // tuning parameter, this trade-offs # binary searches (up to num_chunks times more
           // binary searches can be necessary if num_unique_majors << edgelist_edge_counts[i]) and
           // temporary buffer requirement (cut by num_chunks times), currently set to 2 to avoid
           // peak memory usage happening in this part (especially when col_comm_size is small)

    assert(edgelist_majors.size() == col_comm_size);

    auto edge_partition_major_range_sizes =
      host_scalar_allgather(col_comm, sorted_local_vertices.size(), handle.get_stream());

    if ((col_comm_size >= 2) && (handle.get_stream_pool_size() >= 2)) {
      auto vertex_edge_counts = host_scalar_allreduce(
        comm,
        thrust::make_tuple(static_cast<vertex_t>(sorted_local_vertices.size()), num_local_edges),
        raft::comms::op_t::SUM,
        handle.get_stream());
      // memory footprint vs parallelism trade-off
      // peak memory requirement per loop is approximately
      //   (V/P) * (sizeof(vertex_t) + sizeof(edge_t)) +
      //   (E / (comm_size * col_comm_size)) / num_chunks * sizeof(vertex_t) * 2 +
      //   std::min(V/P, (E / (comm_size * col_comm_size)) / num_chunks) * (sizeof(vertex_t) +
      //   sizeof(edge_t))
      // and limit temporary memory requirement to (E / comm_size) * sizeof(vertex_t)
      auto avg_vertex_degree = thrust::get<0>(vertex_edge_counts) > 0
                                 ? static_cast<double>(thrust::get<1>(vertex_edge_counts)) /
                                     static_cast<double>(thrust::get<0>(vertex_edge_counts))
                                 : double{0.0};
      auto num_streams       = static_cast<size_t>(
        (avg_vertex_degree * sizeof(vertex_t)) /
        (static_cast<double>(sizeof(vertex_t) + sizeof(edge_t)) +
         (((avg_vertex_degree / col_comm_size) / num_chunks) * sizeof(vertex_t) * 2) +
         (std::min(1.0, ((avg_vertex_degree / col_comm_size) / num_chunks)) *
          (sizeof(vertex_t) + sizeof(edge_t)))));
      if (num_streams >= 2) {
        stream_pool_indices = std::vector<size_t>(num_streams);
        std::iota((*stream_pool_indices).begin(), (*stream_pool_indices).end(), size_t{0});
        handle.sync_stream();
      }
    }

    for (int i = 0; i < col_comm_size; ++i) {
      auto loop_stream = stream_pool_indices
                           ? handle.get_stream_from_stream_pool(i % (*stream_pool_indices).size())
                           : handle.get_stream();

      rmm::device_uvector<vertex_t> sorted_majors(edge_partition_major_range_sizes[i], loop_stream);
      device_bcast(col_comm,
                   sorted_local_vertices.data(),
                   sorted_majors.data(),
                   edge_partition_major_range_sizes[i],
                   i,
                   loop_stream);

      rmm::device_uvector<edge_t> sorted_major_degrees(sorted_majors.size(), loop_stream);
      thrust::fill(rmm::exec_policy(loop_stream),
                   sorted_major_degrees.begin(),
                   sorted_major_degrees.end(),
                   edge_t{0});

      rmm::device_uvector<vertex_t> tmp_majors(
        (static_cast<size_t>(edgelist_edge_counts[i]) + (num_chunks - 1)) / num_chunks,
        handle.get_stream());
      size_t offset{0};
      for (size_t j = 0; j < num_chunks; ++j) {
        size_t this_chunk_size =
          std::min(tmp_majors.size(), static_cast<size_t>(edgelist_edge_counts[i]) - offset);
        thrust::copy(rmm::exec_policy(loop_stream),
                     edgelist_majors[i] + offset,
                     edgelist_majors[i] + offset + this_chunk_size,
                     tmp_majors.begin());
        thrust::sort(
          rmm::exec_policy(loop_stream), tmp_majors.begin(), tmp_majors.begin() + this_chunk_size);
        auto num_unique_majors =
          thrust::count_if(rmm::exec_policy(loop_stream),
                           thrust::make_counting_iterator(size_t{0}),
                           thrust::make_counting_iterator(this_chunk_size),
                           is_first_in_run_t<vertex_t const*>{tmp_majors.data()});
        rmm::device_uvector<vertex_t> tmp_keys(num_unique_majors, loop_stream);
        rmm::device_uvector<edge_t> tmp_values(num_unique_majors, loop_stream);
        thrust::reduce_by_key(rmm::exec_policy(loop_stream),
                              tmp_majors.begin(),
                              tmp_majors.begin() + this_chunk_size,
                              thrust::make_constant_iterator(edge_t{1}),
                              tmp_keys.begin(),
                              tmp_values.begin());

        auto kv_pair_first =
          thrust::make_zip_iterator(thrust::make_tuple(tmp_keys.begin(), tmp_values.begin()));
        thrust::for_each(rmm::exec_policy(loop_stream),
                         kv_pair_first,
                         kv_pair_first + tmp_keys.size(),
                         search_and_increment_degree_t<vertex_t, edge_t>{
                           sorted_majors.data(),
                           static_cast<vertex_t>(sorted_majors.size()),
                           sorted_major_degrees.data()});
        offset += this_chunk_size;
      }

      device_reduce(col_comm,
                    sorted_major_degrees.begin(),
                    sorted_major_degrees.begin(),
                    edge_partition_major_range_sizes[i],
                    raft::comms::op_t::SUM,
                    i,
                    loop_stream);
      if (i == col_comm_rank) { sorted_local_vertex_degrees = std::move(sorted_major_degrees); }
    }

    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
  } else {
    assert(edgelist_majors.size() == 1);

    rmm::device_uvector<vertex_t> tmp_majors(edgelist_edge_counts[0], handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_majors[0],
                 edgelist_majors[0] + edgelist_edge_counts[0],
                 tmp_majors.begin());
    thrust::sort(handle.get_thrust_policy(), tmp_majors.begin(), tmp_majors.end());
    auto num_unique_majors =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(tmp_majors.size()),
                       is_first_in_run_t<vertex_t const*>{tmp_majors.data()});
    rmm::device_uvector<vertex_t> tmp_keys(num_unique_majors, handle.get_stream());
    rmm::device_uvector<edge_t> tmp_values(num_unique_majors, handle.get_stream());
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          tmp_majors.begin(),
                          tmp_majors.end(),
                          thrust::make_constant_iterator(edge_t{1}),
                          tmp_keys.begin(),
                          tmp_values.begin());

    tmp_majors.resize(0, handle.get_stream());
    tmp_majors.shrink_to_fit(handle.get_stream());

    sorted_local_vertex_degrees.resize(sorted_local_vertices.size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 sorted_local_vertex_degrees.begin(),
                 sorted_local_vertex_degrees.end(),
                 edge_t{0});

    auto kv_pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(tmp_keys.begin(), tmp_values.begin()));
    thrust::for_each(handle.get_thrust_policy(),
                     kv_pair_first,
                     kv_pair_first + tmp_keys.size(),
                     search_and_increment_degree_t<vertex_t, edge_t>{
                       sorted_local_vertices.data(),
                       static_cast<vertex_t>(sorted_local_vertices.size()),
                       sorted_local_vertex_degrees.data()});
  }

  // 4. sort local vertices by degree (descending)

  thrust::sort_by_key(handle.get_thrust_policy(),
                      sorted_local_vertex_degrees.begin(),
                      sorted_local_vertex_degrees.end(),
                      sorted_local_vertices.begin(),
                      thrust::greater<edge_t>());

  // 5. compute segment_offsets

  static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
  static_assert((detail::low_degree_threshold <= detail::mid_degree_threshold) &&
                (detail::mid_degree_threshold <= std::numeric_limits<edge_t>::max()));
  static_assert(detail::low_degree_threshold >= 1);
  static_assert((detail::hypersparse_threshold_ratio >= 0.0) &&
                (detail::hypersparse_threshold_ratio <= 1.0));
  size_t mid_degree_threshold{detail::mid_degree_threshold};
  size_t low_degree_threshold{detail::low_degree_threshold};
  size_t hypersparse_degree_threshold{0};
  if (multi_gpu) {
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();
    mid_degree_threshold *= col_comm_size;
    low_degree_threshold *= col_comm_size;
    hypersparse_degree_threshold =
      static_cast<size_t>(col_comm_size * detail::hypersparse_threshold_ratio);
  }
  auto num_segments_per_vertex_partition =
    detail::num_sparse_segments_per_vertex_partition +
    (hypersparse_degree_threshold > 0 ? size_t{2} : size_t{1});  // last is 0-degree segment
  rmm::device_uvector<edge_t> d_thresholds(num_segments_per_vertex_partition - 1,
                                           handle.get_stream());
  auto h_thresholds =
    hypersparse_degree_threshold > 0
      ? std::vector<edge_t>{static_cast<edge_t>(mid_degree_threshold),
                            static_cast<edge_t>(low_degree_threshold),
                            static_cast<edge_t>(hypersparse_degree_threshold),
                            std::min(static_cast<edge_t>(hypersparse_degree_threshold), edge_t{1})}
      : std::vector<edge_t>{static_cast<edge_t>(mid_degree_threshold),
                            static_cast<edge_t>(low_degree_threshold),
                            edge_t{1}};
  raft::update_device(
    d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), handle.get_stream());

  rmm::device_uvector<vertex_t> d_segment_offsets(num_segments_per_vertex_partition + 1,
                                                  handle.get_stream());

  auto vertex_count = static_cast<vertex_t>(sorted_local_vertices.size());
  d_segment_offsets.set_element_to_zero_async(0, handle.get_stream());
  d_segment_offsets.set_element_async(
    num_segments_per_vertex_partition, vertex_count, handle.get_stream());

  thrust::upper_bound(handle.get_thrust_policy(),
                      sorted_local_vertex_degrees.begin(),
                      sorted_local_vertex_degrees.end(),
                      d_thresholds.begin(),
                      d_thresholds.end(),
                      d_segment_offsets.begin() + 1,
                      thrust::greater<edge_t>{});

  std::vector<vertex_t> h_segment_offsets(d_segment_offsets.size());
  raft::update_host(h_segment_offsets.data(),
                    d_segment_offsets.data(),
                    d_segment_offsets.size(),
                    handle.get_stream());
  handle.sync_stream();

  return std::make_tuple(
    std::move(sorted_local_vertices), h_segment_offsets, *locally_unused_vertex_id);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void expensive_check_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>> const& local_vertices,
  std::vector<vertex_t const*> const& edgelist_majors,
  std::vector<vertex_t const*> const& edgelist_minors,
  std::vector<edge_t> const& edgelist_edge_counts,
  std::optional<std::vector<std::vector<edge_t>>> const& edgelist_intra_partition_segment_offsets)
{
  std::optional<rmm::device_uvector<vertex_t>> sorted_local_vertices{std::nullopt};
  if (local_vertices) {
    sorted_local_vertices =
      rmm::device_uvector<vertex_t>((*local_vertices).size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 (*local_vertices).begin(),
                 (*local_vertices).end(),
                 (*sorted_local_vertices).begin());
    thrust::sort(
      handle.get_thrust_policy(), (*sorted_local_vertices).begin(), (*sorted_local_vertices).end());
    CUGRAPH_EXPECTS(
      static_cast<size_t>(thrust::distance((*sorted_local_vertices).begin(),
                                           thrust::unique(handle.get_thrust_policy(),
                                                          (*sorted_local_vertices).begin(),
                                                          (*sorted_local_vertices).end()))) ==
        (*sorted_local_vertices).size(),
      "Invalid input argument: (local_)vertices should not have duplicates.");
  }

  if constexpr (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto const row_comm_rank = row_comm.get_rank();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();
    auto const col_comm_rank = col_comm.get_rank();

    CUGRAPH_EXPECTS((edgelist_majors.size() == edgelist_minors.size()) &&
                      (edgelist_majors.size() == static_cast<size_t>(col_comm_size)),
                    "Invalid input argument: both edgelist_majors.size() & "
                    "edgelist_minors.size() should coincide with col_comm_size.");

    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      auto edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors[i], edgelist_minors[i]));
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + edgelist_edge_counts[i],
          [comm_size,
           comm_rank,
           row_comm_rank,
           col_comm_size,
           col_comm_rank,
           i,
           gpu_id_key_func =
             detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
               comm_size, row_comm_size, col_comm_size},
           partition_id_key_func =
             detail::compute_partition_id_from_ext_edge_endpoints_t<vertex_t>{
               comm_size, row_comm_size, col_comm_size}] __device__(auto edge) {
            return (gpu_id_key_func(thrust::get<0>(edge), thrust::get<1>(edge)) != comm_rank) ||
                   (partition_id_key_func(thrust::get<0>(edge), thrust::get<1>(edge)) !=
                    row_comm_rank * col_comm_size + col_comm_rank + i * comm_size);
          }) == 0,
        "Invalid input argument: edgelist_majors & edgelist_minors should be "
        "pre-shuffled.");

      if (edgelist_intra_partition_segment_offsets) {
        for (int j = 0; j < row_comm_size; ++j) {
          CUGRAPH_EXPECTS(
            thrust::count_if(
              handle.get_thrust_policy(),
              edgelist_minors[i] + (*edgelist_intra_partition_segment_offsets)[i][j],
              edgelist_minors[i] + (*edgelist_intra_partition_segment_offsets)[i][j + 1],
              [row_comm_size,
               col_comm_rank,
               j,
               gpu_id_key_func =
                 detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
                   comm_size}] __device__(auto minor) {
                return gpu_id_key_func(minor) != col_comm_rank * row_comm_size + j;
              }) == 0,
            "Invalid input argument: if edgelist_intra_partition_segment_offsets.has_value() is "
            "true, edgelist_majors & edgelist_minors should be properly grouped "
            "within each local partition.");
        }
      }
    }

    if (sorted_local_vertices) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          (*sorted_local_vertices).begin(),
          (*sorted_local_vertices).end(),
          [comm_rank,
           key_func =
             detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size}] __device__(auto val) {
            return key_func(val) != comm_rank;
          }) == 0,
        "Invalid input argument: local_vertices should be pre-shuffled.");

      auto major_range_sizes =
        host_scalar_allgather(col_comm, (*sorted_local_vertices).size(), handle.get_stream());

      rmm::device_uvector<vertex_t> sorted_minors(0, handle.get_stream());
      auto recvcounts =
        host_scalar_allgather(row_comm, (*sorted_local_vertices).size(), handle.get_stream());
      std::vector<size_t> displacements(recvcounts.size(), size_t{0});
      std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
      sorted_minors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
      device_allgatherv(row_comm,
                        (*sorted_local_vertices).data(),
                        sorted_minors.data(),
                        recvcounts,
                        displacements,
                        handle.get_stream());
      thrust::sort(handle.get_thrust_policy(), sorted_minors.begin(), sorted_minors.end());

      for (size_t i = 0; i < edgelist_majors.size(); ++i) {
        rmm::device_uvector<vertex_t> sorted_majors(0, handle.get_stream());
        {
          sorted_majors.resize(major_range_sizes[i], handle.get_stream());
          device_bcast(col_comm,
                       (*sorted_local_vertices).begin(),
                       sorted_majors.begin(),
                       major_range_sizes[i],
                       i,
                       handle.get_stream());
        }

        auto edge_first =
          thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors[i], edgelist_minors[i]));

        CUGRAPH_EXPECTS(thrust::count_if(handle.get_thrust_policy(),
                                         edge_first,
                                         edge_first + edgelist_edge_counts[i],
                                         check_edge_src_and_dst_t<vertex_t>{
                                           sorted_majors.data(),
                                           static_cast<vertex_t>(sorted_majors.size()),
                                           sorted_minors.data(),
                                           static_cast<vertex_t>(sorted_minors.size())}) == 0,
                        "Invalid input argument: edgelist_majors and/or edgelist_minors have "
                        "invalid vertex ID(s).");
      }
    }
  } else {
    assert(edgelist_majors.size() == 1);
    assert(edgelist_minors.size() == 1);

    if (sorted_local_vertices) {
      auto edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors[0], edgelist_minors[0]));
      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         edge_first,
                         edge_first + edgelist_edge_counts[0],
                         check_edge_src_and_dst_t<vertex_t>{
                           (*sorted_local_vertices).data(),
                           static_cast<vertex_t>((*sorted_local_vertices).size()),
                           (*sorted_local_vertices).data(),
                           static_cast<vertex_t>((*sorted_local_vertices).size())}) == 0,
        "Invalid input argument: edgelist_majors and/or edgelist_minors have "
        "invalid vertex ID(s).");
    }

    CUGRAPH_EXPECTS(
      edgelist_intra_partition_segment_offsets.has_value() == false,
      "Invalid input argument: edgelist_intra_partition_segment_offsets.has_value() should "
      "be false for single-GPU.");
  }
}

template <typename vertex_t>
std::vector<vertex_t> aggregate_segment_offsets(raft::handle_t const& handle,
                                                std::vector<vertex_t> const& segment_offsets)
{
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  rmm::device_uvector<vertex_t> d_segment_offsets(segment_offsets.size(), handle.get_stream());
  raft::update_device(
    d_segment_offsets.data(), segment_offsets.data(), segment_offsets.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_aggregate_segment_offsets(
    col_comm_size * d_segment_offsets.size(), handle.get_stream());
  col_comm.allgather(d_segment_offsets.data(),
                     d_aggregate_segment_offsets.data(),
                     d_segment_offsets.size(),
                     handle.get_stream());

  std::vector<vertex_t> h_aggregate_segment_offsets(d_aggregate_segment_offsets.size(),
                                                    vertex_t{0});
  raft::update_host(h_aggregate_segment_offsets.data(),
                    d_aggregate_segment_offsets.data(),
                    d_aggregate_segment_offsets.size(),
                    handle.get_stream());

  handle.sync_stream();  // this is necessary as h_aggregate_offsets can be used right after return.

  return h_aggregate_segment_offsets;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>>
renumber_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  std::vector<vertex_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<vertex_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<edge_t> const& edgelist_edge_counts,
  std::optional<std::vector<std::vector<edge_t>>> const& edgelist_intra_partition_segment_offsets,
  bool store_transposed,
  bool do_expensive_check)
{
  auto edgelist_majors = store_transposed ? edgelist_dsts : edgelist_srcs;
  auto edgelist_minors = store_transposed ? edgelist_srcs : edgelist_dsts;

  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto const comm_rank     = comm.get_rank();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto const row_comm_rank = row_comm.get_rank();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();
  auto const col_comm_rank = col_comm.get_rank();

  CUGRAPH_EXPECTS(edgelist_majors.size() == static_cast<size_t>(col_comm_size),
                  "Invalid input arguments: erroneous edgelist_majors.size().");
  CUGRAPH_EXPECTS(edgelist_minors.size() == static_cast<size_t>(col_comm_size),
                  "Invalid input arguments: erroneous edgelist_minors.size().");
  CUGRAPH_EXPECTS(edgelist_edge_counts.size() == static_cast<size_t>(col_comm_size),
                  "Invalid input arguments: erroneous edgelist_edge_counts.size().");
  if (edgelist_intra_partition_segment_offsets) {
    CUGRAPH_EXPECTS(
      (*edgelist_intra_partition_segment_offsets).size() == static_cast<size_t>(col_comm_size),
      "Invalid input arguments: erroneous (*edgelist_intra_partition_segment_offsets).size().");
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      CUGRAPH_EXPECTS(
        (*edgelist_intra_partition_segment_offsets)[i].size() ==
          static_cast<size_t>(row_comm_size + 1),
        "Invalid input arguments: erroneous (*edgelist_intra_partition_segment_offsets)[].size().");
      CUGRAPH_EXPECTS(
        std::is_sorted((*edgelist_intra_partition_segment_offsets)[i].begin(),
                       (*edgelist_intra_partition_segment_offsets)[i].end()),
        "Invalid input arguments: (*edgelist_intra_partition_segment_offsets)[] is not sorted.");
      CUGRAPH_EXPECTS(
        ((*edgelist_intra_partition_segment_offsets)[i][0] == 0) &&
          ((*edgelist_intra_partition_segment_offsets)[i].back() == edgelist_edge_counts[i]),
        "Invalid input arguments: (*edgelist_intra_partition_segment_offsets)[][0] should be 0 and "
        "(*edgelist_intra_partition_segment_offsets)[].back() should coincide with "
        "edgelist_edge_counts[].");
    }
  }

  std::vector<vertex_t const*> edgelist_const_majors(edgelist_majors.size());
  std::vector<vertex_t const*> edgelist_const_minors(edgelist_const_majors.size());
  for (size_t i = 0; i < edgelist_const_majors.size(); ++i) {
    edgelist_const_majors[i] = edgelist_majors[i];
    edgelist_const_minors[i] = edgelist_minors[i];
  }

  if (do_expensive_check) {
    detail::expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      local_vertices,
      edgelist_const_majors,
      edgelist_const_minors,
      edgelist_edge_counts,
      edgelist_intra_partition_segment_offsets);
  }

  // 1. compute renumber map

  auto [renumber_map_labels, vertex_partition_segment_offsets, locally_unused_vertex_id] =
    detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(handle,
                                                              std::move(local_vertices),
                                                              edgelist_const_majors,
                                                              edgelist_const_minors,
                                                              edgelist_edge_counts);

  // 2. initialize partition_t object, number_of_vertices, and number_of_edges

  auto vertex_counts = host_scalar_allgather(
    comm, static_cast<vertex_t>(renumber_map_labels.size()), handle.get_stream());
  std::vector<vertex_t> vertex_partition_range_offsets(comm_size + 1, 0);
  std::partial_sum(
    vertex_counts.begin(), vertex_counts.end(), vertex_partition_range_offsets.begin() + 1);

  partition_t<vertex_t> partition(
    vertex_partition_range_offsets, row_comm_size, col_comm_size, row_comm_rank, col_comm_rank);

  auto number_of_vertices = vertex_partition_range_offsets.back();
  auto number_of_edges    = host_scalar_allreduce(
    comm,
    std::accumulate(edgelist_edge_counts.begin(), edgelist_edge_counts.end(), edge_t{0}),
    raft::comms::op_t::SUM,
    handle.get_stream());

  // 3. renumber edges

  {
    vertex_t max_edge_partition_major_range_size{0};
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      max_edge_partition_major_range_size = std::max(
        max_edge_partition_major_range_size, partition.local_edge_partition_major_range_size(i));
    }
    rmm::device_uvector<vertex_t> renumber_map_major_labels(max_edge_partition_major_range_size,
                                                            handle.get_stream());
    // FIXME: we may run this in parallel if memory is sufficient
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      device_bcast(col_comm,
                   renumber_map_labels.data(),
                   renumber_map_major_labels.data(),
                   partition.local_edge_partition_major_range_size(i),
                   i,
                   handle.get_stream());

      kv_store_t<vertex_t, vertex_t, false> renumber_map(
        renumber_map_major_labels.begin(),
        renumber_map_major_labels.begin() + partition.local_edge_partition_major_range_size(i),
        thrust::make_counting_iterator(partition.local_edge_partition_major_range_first(i)),
        locally_unused_vertex_id,
        invalid_vertex_id<vertex_t>::value,
        handle.get_stream());
      auto renumber_map_view = renumber_map.view();
      renumber_map_view.find(edgelist_majors[i],
                             edgelist_majors[i] + edgelist_edge_counts[i],
                             edgelist_majors[i],
                             handle.get_stream());
    }
  }

  if ((static_cast<double>(partition.local_edge_partition_minor_range_size() *
                           2.5 /* tuning parameter */) >=
       static_cast<double>(number_of_edges / comm_size)) &&
      edgelist_intra_partition_segment_offsets) {  // memory footprint dominated by the O(V/sqrt(P))
                                                   // part than the O(E/P) part
    vertex_t max_segment_size{0};
    for (int i = 0; i < row_comm_size; ++i) {
      max_segment_size = std::max(
        max_segment_size, partition.vertex_partition_range_size(col_comm_rank * row_comm_size + i));
    }
    rmm::device_uvector<vertex_t> renumber_map_minor_labels(max_segment_size, handle.get_stream());
    for (int i = 0; i < row_comm_size; ++i) {
      auto segment_size = partition.vertex_partition_range_size(col_comm_rank * row_comm_size + i);
      device_bcast(row_comm,
                   renumber_map_labels.data(),
                   renumber_map_minor_labels.data(),
                   segment_size,
                   i,
                   handle.get_stream());

      kv_store_t<vertex_t, vertex_t, false> renumber_map(
        renumber_map_minor_labels.begin(),
        renumber_map_minor_labels.begin() + segment_size,
        thrust::make_counting_iterator(
          partition.vertex_partition_range_first(col_comm_rank * row_comm_size + i)),
        locally_unused_vertex_id,
        invalid_vertex_id<vertex_t>::value,
        handle.get_stream());
      auto renumber_map_view = renumber_map.view();
      for (size_t j = 0; j < edgelist_minors.size(); ++j) {
        renumber_map_view.find(
          edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i],
          edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i + 1],
          edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i],
          handle.get_stream());
      }
    }
  } else {
    rmm::device_uvector<vertex_t> renumber_map_minor_labels(
      partition.local_edge_partition_minor_range_size(), handle.get_stream());
    std::vector<size_t> recvcounts(row_comm_size);
    for (int i = 0; i < row_comm_size; ++i) {
      recvcounts[i] = partition.vertex_partition_range_size(col_comm_rank * row_comm_size + i);
    }
    std::vector<size_t> displacements(recvcounts.size(), 0);
    std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
    device_allgatherv(row_comm,
                      renumber_map_labels.begin(),
                      renumber_map_minor_labels.begin(),
                      recvcounts,
                      displacements,
                      handle.get_stream());

    kv_store_t<vertex_t, vertex_t, false> renumber_map(
      renumber_map_minor_labels.begin(),
      renumber_map_minor_labels.begin() + renumber_map_minor_labels.size(),
      thrust::make_counting_iterator(partition.local_edge_partition_minor_range_first()),
      locally_unused_vertex_id,
      invalid_vertex_id<vertex_t>::value,
      handle.get_stream());
    auto renumber_map_view = renumber_map.view();
    for (size_t i = 0; i < edgelist_minors.size(); ++i) {
      renumber_map_view.find(edgelist_minors[i],
                             edgelist_minors[i] + edgelist_edge_counts[i],
                             edgelist_minors[i],
                             handle.get_stream());
    }
  }

  auto edge_partition_segment_offsets =
    detail::aggregate_segment_offsets(handle, vertex_partition_segment_offsets);

  return std::make_tuple(
    std::move(renumber_map_labels),
    renumber_meta_t<vertex_t, edge_t, multi_gpu>{
      number_of_vertices, number_of_edges, partition, edge_partition_segment_offsets});
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>>
renumber_edgelist(raft::handle_t const& handle,
                  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
                  vertex_t* edgelist_srcs /* [INOUT] */,
                  vertex_t* edgelist_dsts /* [INOUT] */,
                  edge_t num_edgelist_edges,
                  bool store_transposed,
                  bool do_expensive_check)
{
  auto edgelist_majors = store_transposed ? edgelist_dsts : edgelist_srcs;
  auto edgelist_minors = store_transposed ? edgelist_srcs : edgelist_dsts;

  if (do_expensive_check) {
    detail::expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      vertices,
      std::vector<vertex_t const*>{edgelist_majors},
      std::vector<vertex_t const*>{edgelist_minors},
      std::vector<edge_t>{num_edgelist_edges},
      std::nullopt);
  }

  auto [renumber_map_labels, segment_offsets, locally_unused_vertex_id] =
    detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(
      handle,
      std::move(vertices),
      std::vector<vertex_t const*>{edgelist_majors},
      std::vector<vertex_t const*>{edgelist_minors},
      std::vector<edge_t>{num_edgelist_edges});

  kv_store_t<vertex_t, vertex_t, false> renumber_map(
    renumber_map_labels.begin(),
    renumber_map_labels.begin() + renumber_map_labels.size(),
    thrust::make_counting_iterator(vertex_t{0}),
    locally_unused_vertex_id,
    invalid_vertex_id<vertex_t>::value,
    handle.get_stream());
  auto renumber_map_view = renumber_map.view();
  renumber_map_view.find(
    edgelist_majors, edgelist_majors + num_edgelist_edges, edgelist_majors, handle.get_stream());
  renumber_map_view.find(
    edgelist_minors, edgelist_minors + num_edgelist_edges, edgelist_minors, handle.get_stream());

  return std::make_tuple(std::move(renumber_map_labels),
                         renumber_meta_t<vertex_t, edge_t, multi_gpu>{segment_offsets});
}

}  // namespace cugraph
