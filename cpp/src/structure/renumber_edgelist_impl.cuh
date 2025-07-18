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

#include "detail/graph_partition_utils.cuh"
#include "prims/kv_store.cuh"

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

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
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

#include <cuco/hash_functions.cuh>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {
namespace detail {

template <typename vertex_t>
struct check_edge_src_and_dst_t {
  raft::device_span<vertex_t const> sorted_majors{};
  raft::device_span<vertex_t const> sorted_minors{};

  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> e) const
  {
    return !thrust::binary_search(
             thrust::seq, sorted_majors.begin(), sorted_majors.end(), thrust::get<0>(e)) ||
           !thrust::binary_search(
             thrust::seq, sorted_minors.begin(), sorted_minors.end(), thrust::get<1>(e));
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
    *(degrees + cuda::std::distance(sorted_vertices, it)) += thrust::get<1>(vertex_degree_pair);
  }
};

template <typename vertex_t>
rmm::device_uvector<vertex_t> find_uniques(raft::handle_t const& handle,
                                           raft::device_span<vertex_t const> vertices,
                                           size_t mem_frugal_threshold,
                                           std::optional<large_buffer_type_t> large_buffer_type)
{
  if (vertices.size() > mem_frugal_threshold) {  // halve the temporary memory usage compared to the
                                                 // simple copy & sort & unqiue approach
    size_t first_half_size = vertices.size() / 2;

    /* find uqniues in the firs half */

    rmm::device_uvector<vertex_t> first_half_uniques(first_half_size, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 vertices.begin(),
                 vertices.begin() + first_half_size,
                 first_half_uniques.begin());
    thrust::sort(handle.get_thrust_policy(), first_half_uniques.begin(), first_half_uniques.end());
    first_half_uniques.resize(cuda::std::distance(first_half_uniques.begin(),
                                                  thrust::unique(handle.get_thrust_policy(),
                                                                 first_half_uniques.begin(),
                                                                 first_half_uniques.end())),
                              handle.get_stream());
    first_half_uniques.shrink_to_fit(handle.get_stream());

    /* find uqniues in the second half */

    rmm::device_uvector<vertex_t> second_half_uniques(vertices.size() - first_half_size,
                                                      handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 vertices.begin() + first_half_size,
                 vertices.end(),
                 second_half_uniques.begin());
    thrust::sort(
      handle.get_thrust_policy(), second_half_uniques.begin(), second_half_uniques.end());
    second_half_uniques.resize(cuda::std::distance(second_half_uniques.begin(),
                                                   thrust::unique(handle.get_thrust_policy(),
                                                                  second_half_uniques.begin(),
                                                                  second_half_uniques.end())),
                               handle.get_stream());
    second_half_uniques.shrink_to_fit(handle.get_stream());

    /* find the final uqniues */

    rmm::device_uvector<vertex_t> uniques(first_half_uniques.size() + second_half_uniques.size(),
                                          handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 first_half_uniques.begin(),
                 first_half_uniques.end(),
                 uniques.begin());
    thrust::copy(handle.get_thrust_policy(),
                 second_half_uniques.begin(),
                 second_half_uniques.end(),
                 uniques.begin() + first_half_uniques.size());
    thrust::sort(handle.get_thrust_policy(), uniques.begin(), uniques.end());
    uniques.resize(cuda::std::distance(
                     uniques.begin(),
                     thrust::unique(handle.get_thrust_policy(), uniques.begin(), uniques.end())),
                   handle.get_stream());
    uniques.shrink_to_fit(handle.get_stream());

    return uniques;
  } else {
    auto uniques = large_buffer_type
                     ? large_buffer_manager::allocate_memory_buffer<vertex_t>(vertices.size(),
                                                                              handle.get_stream())
                     : rmm::device_uvector<vertex_t>(vertices.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(), vertices.begin(), vertices.end(), uniques.begin());
    thrust::sort(handle.get_thrust_policy(), uniques.begin(), uniques.end());
    uniques.resize(cuda::std::distance(
                     uniques.begin(),
                     thrust::unique(handle.get_thrust_policy(), uniques.begin(), uniques.end())),
                   handle.get_stream());
    uniques.shrink_to_fit(handle.get_stream());

    return uniques;
  }
}

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
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto const major_comm_rank = major_comm.get_rank();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    auto const minor_comm_rank = minor_comm.get_rank();
    if ((major_comm_size < comm_size) &&
        (minor_comm_size < comm_size)) {  // if neither of the edge source/destination range covers
                                          // the entire vertex range
      std::vector<bool> locally_used(comm_size, false);
      for (int i = 0; i < minor_comm_size; ++i) {
        auto major_range_vertex_partition_id =
          compute_local_edge_partition_major_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        locally_used[major_range_vertex_partition_id] = true;
      }
      for (int i = 0; i < major_comm_size; ++i) {
        auto minor_range_vertex_partition_id =
          compute_local_edge_partition_minor_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        locally_used[minor_range_vertex_partition_id] = true;
      }
      assert(std::find(locally_used.begin(), locally_used.end(), false) != locally_used.end());
      std::optional<vertex_t> ret{std::nullopt};
      vertex_t v = std::numeric_limits<vertex_t>::lowest();
      auto vertex_partition_id_op =
        compute_vertex_partition_id_from_ext_vertex_t<vertex_t>{comm_size};
      while (true) {  // the acutal loop count should be smaller than or comparable to comm_size
        if (!locally_used[vertex_partition_id_op(v)]) {
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
  auto gpu_id_op = compute_gpu_id_from_ext_vertex_t<vertex_t>{int{1}, int{1}, int{1}};
  if (multi_gpu && (handle.get_comms().get_size() > int{1})) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    gpu_id_op =
      compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size, major_comm_size, minor_comm_size};
  }
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

// to be used when sorted_local_major_degrees is located on a large buffer (where atomic operations
// are slow)
template <typename vertex_t, typename edge_t>
void compute_sorted_local_major_degrees_without_atomics(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> sorted_local_majors,
  raft::device_span<vertex_t const> edgelist_majors,
  raft::device_span<edge_t> sorted_local_major_degrees)
{
  thrust::fill(handle.get_thrust_policy(),
               sorted_local_major_degrees.begin(),
               sorted_local_major_degrees.end(),
               edge_t{0});

  auto max_edges_to_process_per_iteration =
    std::min(static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * 4096,
             edgelist_majors.size());
  auto max_cache_size =
    std::min(static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20),
             sorted_local_majors.size());

  rmm::device_uvector<vertex_t> indices(max_edges_to_process_per_iteration, handle.get_stream());
  rmm::device_uvector<vertex_t> output_indices(max_edges_to_process_per_iteration,
                                               handle.get_stream());
  rmm::device_uvector<edge_t> output_counts(max_edges_to_process_per_iteration,
                                            handle.get_stream());

  auto cache_stride =
    std::max((sorted_local_majors.size() + (max_cache_size - 1)) / max_cache_size, size_t{1});
  rmm::device_uvector<vertex_t> sorted_local_major_cache(
    (sorted_local_majors.size() + (cache_stride - 1)) / cache_stride, handle.get_stream());
  auto gather_index_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(size_t{0}), detail::multiplier_t<size_t>{cache_stride});
  thrust::gather(handle.get_thrust_policy(),
                 gather_index_first,
                 gather_index_first + sorted_local_major_cache.size(),
                 sorted_local_majors.begin(),
                 sorted_local_major_cache.begin());

  size_t num_edges_processed{0};
  while (num_edges_processed < edgelist_majors.size()) {
    auto num_edges_to_process =
      std::min(edgelist_majors.size() - num_edges_processed, max_edges_to_process_per_iteration);
    thrust::transform(
      handle.get_thrust_policy(),
      edgelist_majors.begin() + num_edges_processed,
      edgelist_majors.begin() + (num_edges_processed + num_edges_to_process),
      indices.begin(),
      cuda::proclaim_return_type<vertex_t>(
        [sorted_local_major_cache = raft::device_span<vertex_t const>(
           sorted_local_major_cache.data(), sorted_local_major_cache.size()),
         sorted_local_majors = raft::device_span<vertex_t const>(sorted_local_majors.data(),
                                                                 sorted_local_majors.size()),
         cache_stride] __device__(auto major) {
          auto it    = thrust::upper_bound(thrust::seq,
                                        sorted_local_major_cache.begin() + 1,
                                        sorted_local_major_cache.end(),
                                        major);
          auto idx   = cuda::std::distance(sorted_local_major_cache.begin() + 1, it);
          auto first = sorted_local_majors.begin() + idx * cache_stride;
          auto last  = sorted_local_majors.begin() +
                      cuda::std::min((idx + 1) * cache_stride, sorted_local_majors.size());
          return static_cast<vertex_t>(cuda::std::distance(
            sorted_local_majors.begin(), thrust::lower_bound(thrust::seq, first, last, major)));
        }));
    thrust::sort(
      handle.get_thrust_policy(), indices.begin(), indices.begin() + num_edges_to_process);
    auto it          = thrust::reduce_by_key(handle.get_thrust_policy(),
                                    indices.begin(),
                                    indices.begin() + num_edges_to_process,
                                    thrust::make_constant_iterator(edge_t{1}),
                                    output_indices.begin(),
                                    output_counts.begin());
    auto input_first = thrust::make_zip_iterator(output_indices.begin(), output_counts.begin());
    thrust::for_each(handle.get_thrust_policy(),
                     input_first,
                     input_first + cuda::std::distance(output_indices.begin(), thrust::get<0>(it)),
                     [degrees = raft::device_span<edge_t>(
                        sorted_local_major_degrees.data(),
                        sorted_local_major_degrees.size())] __device__(auto pair) {
                       degrees[thrust::get<0>(pair)] += thrust::get<1>(pair);
                     });
    num_edges_processed += num_edges_to_process;
  }
}

// returns renumber map, segment_offsets, and hypersparse_degree_offsets
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           std::vector<vertex_t>,
           std::optional<std::vector<vertex_t>>,
           vertex_t>
compute_renumber_map(raft::handle_t const& handle,
                     std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
                     std::vector<raft::device_span<vertex_t const>> const& edgelist_majors,
                     std::vector<raft::device_span<vertex_t const>> const& edgelist_minors,
                     std::optional<large_buffer_type_t> large_vertex_buffer_type,
                     std::optional<large_buffer_type_t> large_edge_buffer_type)
{
  CUGRAPH_EXPECTS((!large_vertex_buffer_type && !large_edge_buffer_type) ||
                    cugraph::large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  // 1. if local_vertices.has_value() is false, find unique vertices from edge majors & minors (to
  // construct local_vertices)

  auto sorted_local_vertices =
    large_vertex_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<vertex_t>(0, handle.get_stream())
      : rmm::device_uvector<vertex_t>(0, handle.get_stream());
  if (!local_vertices) {
    auto sorted_unique_majors =
      large_edge_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<vertex_t>(0, handle.get_stream())
        : rmm::device_uvector<vertex_t>(0, handle.get_stream());
    {
      auto mem_frugal_threshold = std::numeric_limits<size_t>::max();
      if (!large_edge_buffer_type) {
        auto total_global_mem = handle.get_device_properties().totalGlobalMem;
        auto constexpr mem_frugal_ratio =
          0.1;  // if expected temporary buffer size exceeds the mem_Frugal_ratio of the
                // total_global_mem, switch to the memory frugal approach
        mem_frugal_threshold =
          static_cast<size_t>(static_cast<double>(total_global_mem / sizeof(vertex_t)) *
                              mem_frugal_ratio) /
          2 /* tmp_minors & temporary memory use in sort */;
      }

      std::vector<rmm::device_uvector<vertex_t>> edge_partition_tmp_majors{};
      edge_partition_tmp_majors.reserve(edgelist_majors.size());
      for (size_t i = 0; i < edgelist_majors.size(); ++i) {
        auto tmp_majors = find_uniques(
          handle,
          raft::device_span<vertex_t const>(edgelist_majors[i].data(), edgelist_majors[i].size()),
          mem_frugal_threshold,
          large_edge_buffer_type);
        edge_partition_tmp_majors.push_back(std::move(tmp_majors));
      }
      if constexpr (multi_gpu) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size = minor_comm.get_size();
        if (minor_comm_size > 1) {
          std::vector<size_t> tx_counts(minor_comm_size);
          for (int i = 0; i < minor_comm_size; ++i) {
            tx_counts[i] = edge_partition_tmp_majors[i].size();
          }
          sorted_unique_majors.resize(std::reduce(tx_counts.begin(), tx_counts.end()),
                                      handle.get_stream());
          size_t output_offset{0};
          for (size_t i = 0; i < edge_partition_tmp_majors.size(); ++i) {
            thrust::copy(handle.get_thrust_policy(),
                         edge_partition_tmp_majors[i].begin(),
                         edge_partition_tmp_majors[i].end(),
                         sorted_unique_majors.begin() + output_offset);
            output_offset += edge_partition_tmp_majors[i].size();
          }
          edge_partition_tmp_majors.clear();
          sorted_unique_majors = shuffle_and_unique_segment_sorted_values(
            minor_comm,
            sorted_unique_majors.begin(),
            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
            handle.get_stream(),
            large_edge_buffer_type);
        } else {
          sorted_unique_majors = std::move(edge_partition_tmp_majors[0]);
        }
      } else {
        sorted_unique_majors = std::move(edge_partition_tmp_majors[0]);
      }
    }

    auto sorted_unique_minors =
      large_edge_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<vertex_t>(0, handle.get_stream())
        : rmm::device_uvector<vertex_t>(0, handle.get_stream());
    {
      auto mem_frugal_threshold = std::numeric_limits<size_t>::max();
      if (!large_edge_buffer_type) {
        auto total_global_mem = handle.get_device_properties().totalGlobalMem;
        auto constexpr mem_frugal_ratio =
          0.1;  // if expected temporary buffer size exceeds the mem_Frugal_ratio of the
                // total_global_mem, switch to the memory frugal approach
        mem_frugal_threshold =
          static_cast<size_t>(static_cast<double>(total_global_mem / sizeof(vertex_t)) *
                              mem_frugal_ratio) /
          2 /* tmp_minors & temporary memory use in sort */;
      }

      std::vector<rmm::device_uvector<vertex_t>> edge_partition_tmp_minors{};
      edge_partition_tmp_minors.reserve(edgelist_minors.size());
      for (size_t i = 0; i < edgelist_minors.size(); ++i) {
        auto tmp_minors = find_uniques(
          handle,
          raft::device_span<vertex_t const>(edgelist_minors[i].data(), edgelist_minors[i].size()),
          mem_frugal_threshold,
          large_edge_buffer_type);
        edge_partition_tmp_minors.push_back(std::move(tmp_minors));
      }
      if (edge_partition_tmp_minors.size() == 1) {
        sorted_unique_minors = std::move(edge_partition_tmp_minors[0]);
      } else {
        edge_t aggregate_size{0};
        for (size_t i = 0; i < edge_partition_tmp_minors.size(); ++i) {
          aggregate_size += edge_partition_tmp_minors[i].size();
        }
        sorted_unique_minors.resize(aggregate_size, handle.get_stream());
        size_t output_offset{0};
        for (size_t i = 0; i < edge_partition_tmp_minors.size(); ++i) {
          thrust::copy(handle.get_thrust_policy(),
                       edge_partition_tmp_minors[i].begin(),
                       edge_partition_tmp_minors[i].end(),
                       sorted_unique_minors.begin() + output_offset);
          output_offset += edge_partition_tmp_minors[i].size();
        }
        edge_partition_tmp_minors.clear();
        thrust::sort(
          handle.get_thrust_policy(), sorted_unique_minors.begin(), sorted_unique_minors.end());
        sorted_unique_minors.resize(cuda::std::distance(sorted_unique_minors.begin(),
                                                        thrust::unique(handle.get_thrust_policy(),
                                                                       sorted_unique_minors.begin(),
                                                                       sorted_unique_minors.end())),
                                    handle.get_stream());
        sorted_unique_minors.shrink_to_fit(handle.get_stream());
      }
      if constexpr (multi_gpu) {
        auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
        auto const major_comm_size = major_comm.get_size();
        if (major_comm_size > 1) {
          auto& comm           = handle.get_comms();
          auto const comm_size = comm.get_size();
          auto& minor_comm     = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
          auto const minor_comm_size = minor_comm.get_size();
          compute_gpu_id_from_ext_vertex_t<vertex_t> gpu_id_func{
            comm_size, major_comm_size, minor_comm_size};
          auto d_tx_counts = groupby_and_count(
            sorted_unique_minors.begin(),
            sorted_unique_minors.end(),
            cuda::proclaim_return_type<int>(
              [major_comm_size, minor_comm_size, gpu_id_func] __device__(auto v) {
                return partition_manager::compute_major_comm_rank_from_global_comm_rank(
                  major_comm_size, minor_comm_size, gpu_id_func(v));
              }),
            major_comm_size,
            std::numeric_limits<size_t>::max(),
            handle.get_stream(),
            large_edge_buffer_type);
          std::vector<size_t> h_tx_counts(d_tx_counts.size());
          raft::update_host(
            h_tx_counts.data(), d_tx_counts.data(), d_tx_counts.size(), handle.get_stream());
          handle.sync_stream();
          std::vector<size_t> tx_displacements(h_tx_counts.size());
          std::exclusive_scan(
            h_tx_counts.begin(), h_tx_counts.end(), tx_displacements.begin(), size_t{0});
          for (int j = 0; j < major_comm_size; ++j) {
            thrust::sort(handle.get_thrust_policy(),
                         sorted_unique_minors.begin() + tx_displacements[j],
                         sorted_unique_minors.begin() + (tx_displacements[j] + h_tx_counts[j]));
          }
          sorted_unique_minors = shuffle_and_unique_segment_sorted_values(
            major_comm,
            sorted_unique_minors.begin(),
            raft::host_span<size_t const>(h_tx_counts.data(), h_tx_counts.size()),
            handle.get_stream(),
            large_edge_buffer_type);
        }
      }
    }

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
    sorted_local_vertices.resize(cuda::std::distance(sorted_local_vertices.begin(),
                                                     thrust::unique(handle.get_thrust_policy(),
                                                                    sorted_local_vertices.begin(),
                                                                    sorted_local_vertices.end())),
                                 handle.get_stream());
    sorted_local_vertices.shrink_to_fit(handle.get_stream());
  } else {
    sorted_local_vertices = std::move(*local_vertices);
    thrust::sort(
      handle.get_thrust_policy(), sorted_local_vertices.begin(), sorted_local_vertices.end());
  }

  // 2. find an unused vertex ID

  auto locally_unused_vertex_id = find_locally_unused_ext_vertex_id(
    handle,
    raft::device_span<vertex_t const>(sorted_local_vertices.data(), sorted_local_vertices.size()),
    multi_gpu);
  CUGRAPH_EXPECTS(locally_unused_vertex_id.has_value(),
                  "Invalid input arguments: there is no unused value in the entire range of "
                  "vertex_t, increase vertex_t to 64 bit.");

  // 3. compute global degrees for the sorted local vertices

  auto sorted_local_vertex_degrees =
    large_vertex_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<vertex_t>(0, handle.get_stream())
      : rmm::device_uvector<vertex_t>(0, handle.get_stream());
  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    assert(edgelist_majors.size() == minor_comm_size);

    auto edge_partition_major_range_sizes =
      host_scalar_allgather(minor_comm, sorted_local_vertices.size(), handle.get_stream());

    for (int i = 0; i < minor_comm_size; ++i) {
      auto sorted_majors =
        large_vertex_buffer_type
          ? large_buffer_manager::allocate_memory_buffer<vertex_t>(
              edge_partition_major_range_sizes[i], handle.get_stream())
          : rmm::device_uvector<vertex_t>(edge_partition_major_range_sizes[i], handle.get_stream());
      device_bcast(minor_comm,
                   sorted_local_vertices.data(),
                   sorted_majors.data(),
                   edge_partition_major_range_sizes[i],
                   i,
                   handle.get_stream());

      auto sorted_major_degrees =
        large_vertex_buffer_type
          ? large_buffer_manager::allocate_memory_buffer<edge_t>(sorted_majors.size(),
                                                                 handle.get_stream())
          : rmm::device_uvector<edge_t>(sorted_majors.size(), handle.get_stream());
      if (large_vertex_buffer_type) {
        compute_sorted_local_major_degrees_without_atomics(
          handle,
          raft::device_span<vertex_t const>(sorted_majors.data(), sorted_majors.size()),
          raft::device_span<vertex_t const>(edgelist_majors[i].data(), edgelist_majors[i].size()),
          raft::device_span<edge_t>(sorted_major_degrees.data(), sorted_major_degrees.size()));
      } else {
        thrust::fill(handle.get_thrust_policy(),
                     sorted_major_degrees.begin(),
                     sorted_major_degrees.end(),
                     edge_t{0});
        thrust::for_each(
          handle.get_thrust_policy(),
          edgelist_majors[i].begin(),
          edgelist_majors[i].end(),
          [sorted_majors =
             raft::device_span<vertex_t const>(sorted_majors.data(), sorted_majors.size()),
           sorted_major_degrees = raft::device_span<edge_t>(
             sorted_major_degrees.data(), sorted_major_degrees.size())] __device__(auto major) {
            auto it =
              thrust::lower_bound(thrust::seq, sorted_majors.begin(), sorted_majors.end(), major);
            assert((it != sorted_majors.end()) && (*it == major));
            cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(
              sorted_major_degrees[cuda::std::distance(sorted_majors.begin(), it)]);
            atomic_counter.fetch_add(edge_t{1}, cuda::std::memory_order_relaxed);
          });
      }

      device_reduce(minor_comm,
                    sorted_major_degrees.begin(),
                    sorted_major_degrees.begin(),
                    edge_partition_major_range_sizes[i],
                    raft::comms::op_t::SUM,
                    i,
                    handle.get_stream());
      if (i == minor_comm_rank) { sorted_local_vertex_degrees = std::move(sorted_major_degrees); }
    }
  } else {
    assert(edgelist_majors.size() == 1);

    sorted_local_vertex_degrees.resize(sorted_local_vertices.size(), handle.get_stream());
    if (large_vertex_buffer_type) {
      compute_sorted_local_major_degrees_without_atomics(
        handle,
        raft::device_span<vertex_t const>(sorted_local_vertices.data(),
                                          sorted_local_vertices.size()),
        raft::device_span<vertex_t const>(edgelist_majors[0].data(), edgelist_majors[0].size()),
        raft::device_span<edge_t>(sorted_local_vertex_degrees.data(),
                                  sorted_local_vertex_degrees.size()));
    } else {
      thrust::fill(handle.get_thrust_policy(),
                   sorted_local_vertex_degrees.begin(),
                   sorted_local_vertex_degrees.end(),
                   edge_t{0});
      thrust::for_each(handle.get_thrust_policy(),
                       edgelist_majors[0].begin(),
                       edgelist_majors[0].end(),
                       [sorted_majors = raft::device_span<vertex_t const>(
                          sorted_local_vertices.data(), sorted_local_vertices.size()),
                        sorted_major_degrees = raft::device_span<edge_t>(
                          sorted_local_vertex_degrees.data(),
                          sorted_local_vertex_degrees.size())] __device__(auto major) {
                         auto it = thrust::lower_bound(
                           thrust::seq, sorted_majors.begin(), sorted_majors.end(), major);
                         assert((it != sorted_majors.end()) && (*it == major));
                         cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(
                           sorted_major_degrees[cuda::std::distance(sorted_majors.begin(), it)]);
                         atomic_counter.fetch_add(edge_t{1}, cuda::std::memory_order_relaxed);
                       });
    }
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
  size_t hypersparse_degree_threshold{1};
  if (multi_gpu) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    mid_degree_threshold *= minor_comm_size;
    low_degree_threshold *= minor_comm_size;
    hypersparse_degree_threshold = std::max(
      static_cast<size_t>(minor_comm_size * detail::hypersparse_threshold_ratio), size_t{1});
  }

  std::vector<vertex_t> h_segment_offsets{};
  std::optional<std::vector<vertex_t>> h_hypersparse_degree_offsets{};
  {
    auto num_partitions = detail::num_sparse_segments_per_vertex_partition /* high, mid, low */ +
                          (hypersparse_degree_threshold > 1
                             ? hypersparse_degree_threshold - size_t{1}
                             /* one partition per each global degree in the hypersparse region */
                             : size_t{0}) +
                          size_t{1} /* zero */;
    rmm::device_uvector<edge_t> d_thresholds(num_partitions - 1, handle.get_stream());
    thrust::tabulate(handle.get_thrust_policy(),
                     d_thresholds.begin(),
                     d_thresholds.end(),
                     [mid_degree_threshold,
                      low_degree_threshold,
                      hypersparse_degree_threshold] __device__(size_t i) {
                       if (i == 0) {
                         return mid_degree_threshold;  // high,mid boundary
                       } else if (i == 1) {
                         return low_degree_threshold;  // mid, low boundary
                       } else {
                         assert(hypersparse_degree_threshold > (i - 2));
                         return hypersparse_degree_threshold - (i - 2);
                       }
                     });
    rmm::device_uvector<vertex_t> d_offsets(num_partitions + 1, handle.get_stream());
    d_offsets.set_element_to_zero_async(0, handle.get_stream());
    auto vertex_count = static_cast<vertex_t>(sorted_local_vertices.size());
    d_offsets.set_element(num_partitions, vertex_count, handle.get_stream());
    thrust::upper_bound(handle.get_thrust_policy(),
                        sorted_local_vertex_degrees.begin(),
                        sorted_local_vertex_degrees.end(),
                        d_thresholds.begin(),
                        d_thresholds.end(),
                        d_offsets.begin() + 1,
                        thrust::greater<edge_t>{});
    std::vector<vertex_t> h_offsets(d_offsets.size());
    raft::update_host(h_offsets.data(), d_offsets.data(), d_offsets.size(), handle.get_stream());
    handle.sync_stream();

    auto num_segments_per_vertex_partition =
      detail::num_sparse_segments_per_vertex_partition +
      (hypersparse_degree_threshold > 1 ? size_t{2} : size_t{1});  // last is 0-degree segment
    h_segment_offsets.resize(num_segments_per_vertex_partition + 1);
    std::copy(h_offsets.begin(),
              h_offsets.begin() + num_sparse_segments_per_vertex_partition + 1,
              h_segment_offsets.begin());
    *(h_segment_offsets.rbegin()) = *(h_offsets.rbegin());
    if (hypersparse_degree_threshold > 1) {
      *(h_segment_offsets.rbegin() + 1) = *(h_offsets.rbegin() + 1);

      h_hypersparse_degree_offsets = std::vector<vertex_t>(hypersparse_degree_threshold);
      std::copy(h_offsets.begin() + num_sparse_segments_per_vertex_partition,
                h_offsets.begin() + num_sparse_segments_per_vertex_partition +
                  (hypersparse_degree_threshold - 1),
                (*h_hypersparse_degree_offsets).begin());
      auto shift = (*h_hypersparse_degree_offsets)[0];
      std::transform((*h_hypersparse_degree_offsets).begin(),
                     (*h_hypersparse_degree_offsets).end(),
                     (*h_hypersparse_degree_offsets).begin(),
                     [shift](auto offset) { return offset - shift; });
      *((*h_hypersparse_degree_offsets).rbegin()) = *(h_offsets.rbegin() + 1);
    }
  }

  return std::make_tuple(std::move(sorted_local_vertices),
                         h_segment_offsets,
                         h_hypersparse_degree_offsets,
                         *locally_unused_vertex_id);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void expensive_check_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>> const& local_vertices,
  std::vector<raft::device_span<vertex_t const>> const& edgelist_majors,
  std::vector<raft::device_span<vertex_t const>> const& edgelist_minors,
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
      static_cast<size_t>(cuda::std::distance((*sorted_local_vertices).begin(),
                                              thrust::unique(handle.get_thrust_policy(),
                                                             (*sorted_local_vertices).begin(),
                                                             (*sorted_local_vertices).end()))) ==
        (*sorted_local_vertices).size(),
      "Invalid input argument: (local_)vertices should not have duplicates.");
  }

  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto const comm_rank       = comm.get_rank();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto const major_comm_rank = major_comm.get_rank();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    auto const minor_comm_rank = minor_comm.get_rank();

    CUGRAPH_EXPECTS((edgelist_majors.size() == edgelist_minors.size()) &&
                      (edgelist_majors.size() == static_cast<size_t>(minor_comm_size)),
                    "Invalid input argument: both edgelist_majors.size() & "
                    "edgelist_minors.size() should coincide with minor_comm_size.");

    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      CUGRAPH_EXPECTS(edgelist_majors[i].size() == edgelist_minors[i].size(),
                      "Invalid input argument: edgelist_majors[].size() and "
                      "edgelist_minors[i].size() should coincide.");

      auto edge_first =
        thrust::make_zip_iterator(edgelist_majors[i].begin(), edgelist_minors[i].begin());
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + edgelist_majors[i].size(),
          [comm_size,
           comm_rank,
           major_comm_rank,
           minor_comm_size,
           minor_comm_rank,
           i,
           gpu_id_key_func =
             detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
               comm_size, major_comm_size, minor_comm_size},
           local_edge_partition_id_key_func =
             detail::compute_local_edge_partition_id_from_ext_edge_endpoints_t<vertex_t>{
               comm_size, major_comm_size, minor_comm_size}] __device__(auto edge) {
            return (gpu_id_key_func(thrust::get<0>(edge), thrust::get<1>(edge)) != comm_rank) ||
                   (local_edge_partition_id_key_func(edge) != i);
          }) == 0,
        "Invalid input argument: edgelist_majors & edgelist_minors should be "
        "pre-shuffled.");

      if (edgelist_intra_partition_segment_offsets) {
        for (int j = 0; j < major_comm_size; ++j) {
          CUGRAPH_EXPECTS(
            thrust::count_if(
              handle.get_thrust_policy(),
              edgelist_minors[i].begin() + (*edgelist_intra_partition_segment_offsets)[i][j],
              edgelist_minors[i].begin() + (*edgelist_intra_partition_segment_offsets)[i][j + 1],
              [major_comm_size,
               minor_comm_rank,
               j,
               vertex_partition_id_key_func =
                 detail::compute_vertex_partition_id_from_ext_vertex_t<vertex_t>{comm_size},
               minor_range_vertex_partition_id =
                 detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
                   major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(
                   j)] __device__(auto minor) {
                return vertex_partition_id_key_func(minor) != minor_range_vertex_partition_id;
              }) == 0,
            "Invalid input argument: if edgelist_intra_partition_segment_offsets.has_value() is "
            "true, edgelist_majors & edgelist_minors should be properly grouped "
            "within each local partition.");
        }
      }
    }

    if (sorted_local_vertices) {
      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         (*sorted_local_vertices).begin(),
                         (*sorted_local_vertices).end(),
                         [comm_rank,
                          key_func =
                            detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
                              comm_size, major_comm_size, minor_comm_size}] __device__(auto val) {
                           return key_func(val) != comm_rank;
                         }) == 0,
        "Invalid input argument: local_vertices should be pre-shuffled.");

      rmm::device_uvector<vertex_t> sorted_minors(0, handle.get_stream());
      auto recvcounts =
        host_scalar_allgather(major_comm, (*sorted_local_vertices).size(), handle.get_stream());
      std::vector<size_t> displacements(recvcounts.size(), size_t{0});
      std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
      sorted_minors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
      device_allgatherv(major_comm,
                        (*sorted_local_vertices).data(),
                        sorted_minors.data(),
                        raft::host_span<size_t const>(recvcounts.data(), recvcounts.size()),
                        raft::host_span<size_t const>(displacements.data(), displacements.size()),
                        handle.get_stream());
      thrust::sort(handle.get_thrust_policy(), sorted_minors.begin(), sorted_minors.end());

      auto major_range_sizes =
        host_scalar_allgather(minor_comm, (*sorted_local_vertices).size(), handle.get_stream());
      for (size_t i = 0; i < edgelist_majors.size(); ++i) {
        rmm::device_uvector<vertex_t> sorted_majors(0, handle.get_stream());
        {
          sorted_majors.resize(major_range_sizes[i], handle.get_stream());
          device_bcast(minor_comm,
                       (*sorted_local_vertices).begin(),
                       sorted_majors.begin(),
                       major_range_sizes[i],
                       i,
                       handle.get_stream());
        }

        auto edge_first =
          thrust::make_zip_iterator(edgelist_majors[i].begin(), edgelist_minors[i].begin());

        CUGRAPH_EXPECTS(
          thrust::count_if(
            handle.get_thrust_policy(),
            edge_first,
            edge_first + edgelist_majors[i].size(),
            check_edge_src_and_dst_t<vertex_t>{
              raft::device_span<vertex_t const>(sorted_majors.data(),
                                                static_cast<vertex_t>(sorted_majors.size())),
              raft::device_span<vertex_t const>(sorted_minors.data(),
                                                static_cast<vertex_t>(sorted_minors.size()))}) == 0,
          "Invalid input argument: edgelist_majors and/or edgelist_minors have "
          "invalid vertex ID(s).");
      }
    }
  } else {
    assert(edgelist_majors.size() == 1);
    assert(edgelist_minors.size() == 1);

    if (sorted_local_vertices) {
      auto edge_first =
        thrust::make_zip_iterator(edgelist_majors[0].begin(), edgelist_minors[0].begin());
      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         edge_first,
                         edge_first + edgelist_majors[0].size(),
                         check_edge_src_and_dst_t<vertex_t>{
                           raft::device_span<vertex_t const>(
                             (*sorted_local_vertices).data(),
                             static_cast<vertex_t>((*sorted_local_vertices).size())),
                           raft::device_span<vertex_t const>(
                             (*sorted_local_vertices).data(),
                             static_cast<vertex_t>((*sorted_local_vertices).size()))}) == 0,
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
std::vector<vertex_t> aggregate_offset_vectors(raft::handle_t const& handle,
                                               std::vector<vertex_t> const& offsets)
{
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  rmm::device_uvector<vertex_t> d_offsets(offsets.size(), handle.get_stream());
  raft::update_device(d_offsets.data(), offsets.data(), offsets.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_aggregate_offset_vectors(minor_comm_size * d_offsets.size(),
                                                           handle.get_stream());
  minor_comm.allgather(
    d_offsets.data(), d_aggregate_offset_vectors.data(), d_offsets.size(), handle.get_stream());

  std::vector<vertex_t> h_aggregate_offset_vectors(d_aggregate_offset_vectors.size(), vertex_t{0});
  raft::update_host(h_aggregate_offset_vectors.data(),
                    d_aggregate_offset_vectors.data(),
                    d_aggregate_offset_vectors.size(),
                    handle.get_stream());

  handle.sync_stream();  // this is necessary as h_aggregate_offsets can be used right after return.

  return h_aggregate_offset_vectors;
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
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check)
{
  auto edgelist_majors = store_transposed ? edgelist_dsts : edgelist_srcs;
  auto edgelist_minors = store_transposed ? edgelist_srcs : edgelist_dsts;

  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto const comm_rank       = comm.get_rank();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto const major_comm_rank = major_comm.get_rank();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();
  auto const minor_comm_rank = minor_comm.get_rank();

  CUGRAPH_EXPECTS(edgelist_majors.size() == static_cast<size_t>(minor_comm_size),
                  "Invalid input arguments: erroneous edgelist_majors.size().");
  CUGRAPH_EXPECTS(edgelist_minors.size() == static_cast<size_t>(minor_comm_size),
                  "Invalid input arguments: erroneous edgelist_minors.size().");
  CUGRAPH_EXPECTS(edgelist_edge_counts.size() == static_cast<size_t>(minor_comm_size),
                  "Invalid input arguments: erroneous edgelist_edge_counts.size().");
  if (edgelist_intra_partition_segment_offsets) {
    CUGRAPH_EXPECTS(
      (*edgelist_intra_partition_segment_offsets).size() == static_cast<size_t>(minor_comm_size),
      "Invalid input arguments: erroneous (*edgelist_intra_partition_segment_offsets).size().");
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      CUGRAPH_EXPECTS((*edgelist_intra_partition_segment_offsets)[i].size() ==
                        static_cast<size_t>(major_comm_size + 1),
                      "Invalid input arguments: erroneous "
                      "(*edgelist_intra_partition_segment_offsets)[].size().");
      CUGRAPH_EXPECTS(
        std::is_sorted((*edgelist_intra_partition_segment_offsets)[i].begin(),
                       (*edgelist_intra_partition_segment_offsets)[i].end()),
        "Invalid input arguments: (*edgelist_intra_partition_segment_offsets)[] is not sorted.");
      CUGRAPH_EXPECTS(
        ((*edgelist_intra_partition_segment_offsets)[i][0] == 0) &&
          ((*edgelist_intra_partition_segment_offsets)[i].back() == edgelist_edge_counts[i]),
        "Invalid input arguments: (*edgelist_intra_partition_segment_offsets)[][0] should be 0 "
        "and (*edgelist_intra_partition_segment_offsets)[].back() should coincide with "
        "edgelist_edge_counts[].");
    }
  }
  CUGRAPH_EXPECTS((!large_vertex_buffer_type && !large_edge_buffer_type) ||
                    cugraph::large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  std::vector<raft::device_span<vertex_t const>> edgelist_const_majors(edgelist_majors.size());
  std::vector<raft::device_span<vertex_t const>> edgelist_const_minors(
    edgelist_const_majors.size());
  for (size_t i = 0; i < edgelist_const_majors.size(); ++i) {
    edgelist_const_majors[i] =
      raft::device_span<vertex_t const>(edgelist_majors[i], edgelist_edge_counts[i]);
    edgelist_const_minors[i] =
      raft::device_span<vertex_t const>(edgelist_minors[i], edgelist_edge_counts[i]);
  }

  if (do_expensive_check) {
    detail::expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      local_vertices,
      edgelist_const_majors,
      edgelist_const_minors,
      edgelist_intra_partition_segment_offsets);
  }

  // 1. compute renumber map

  auto [renumber_map_labels,
        vertex_partition_segment_offsets,
        vertex_partition_hypersparse_degree_offsets,
        locally_unused_vertex_id] =
    detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(handle,
                                                              std::move(local_vertices),
                                                              edgelist_const_majors,
                                                              edgelist_const_minors,
                                                              large_vertex_buffer_type,
                                                              large_edge_buffer_type);

  // 2. initialize partition_t object, number_of_vertices, and number_of_edges

  auto vertex_counts = host_scalar_allgather(
    comm, static_cast<vertex_t>(renumber_map_labels.size()), handle.get_stream());
  auto vertex_partition_ids =
    host_scalar_allgather(comm,
                          partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
                            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank),
                          handle.get_stream());

  std::vector<vertex_t> vertex_partition_range_offsets(comm_size + 1, 0);
  for (int i = 0; i < comm_size; ++i) {
    vertex_partition_range_offsets[vertex_partition_ids[i]] = vertex_counts[i];
  }
  std::exclusive_scan(vertex_partition_range_offsets.begin(),
                      vertex_partition_range_offsets.end(),
                      vertex_partition_range_offsets.begin(),
                      vertex_t{0});

  partition_t<vertex_t> partition(vertex_partition_range_offsets,
                                  major_comm_size,
                                  minor_comm_size,
                                  major_comm_rank,
                                  minor_comm_rank);

  auto number_of_vertices = vertex_partition_range_offsets.back();
  auto number_of_edges    = host_scalar_allreduce(
    comm,
    std::accumulate(edgelist_edge_counts.begin(), edgelist_edge_counts.end(), edge_t{0}),
    raft::comms::op_t::SUM,
    handle.get_stream());

  // 3. renumber edges

  {
    // FIXME: we may run this in parallel if memory is sufficient
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      auto major_range_size = partition.local_edge_partition_major_range_size(i);
      auto renumber_map_major_labels =
        large_vertex_buffer_type
          ? large_buffer_manager::allocate_memory_buffer<vertex_t>(major_range_size,
                                                                   handle.get_stream())
          : rmm::device_uvector<vertex_t>(major_range_size, handle.get_stream());
      device_bcast(minor_comm,
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

  if (edgelist_intra_partition_segment_offsets) {
    for (int i = 0; i < major_comm_size; ++i) {
      auto minor_range_vertex_partition_id =
        detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
          major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
      auto segment_size = partition.vertex_partition_range_size(minor_range_vertex_partition_id);
      auto renumber_map_minor_labels =
        large_vertex_buffer_type ? large_buffer_manager::allocate_memory_buffer<vertex_t>(
                                     segment_size, handle.get_stream())
                                 : rmm::device_uvector<vertex_t>(segment_size, handle.get_stream());
      device_bcast(major_comm,
                   renumber_map_labels.data(),
                   renumber_map_minor_labels.data(),
                   segment_size,
                   i,
                   handle.get_stream());

      kv_store_t<vertex_t, vertex_t, false> renumber_map(
        renumber_map_minor_labels.begin(),
        renumber_map_minor_labels.end(),
        thrust::make_counting_iterator(
          partition.vertex_partition_range_first(minor_range_vertex_partition_id)),
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
    auto minor_range_size = partition.local_edge_partition_minor_range_size();
    auto renumber_map_minor_labels =
      large_vertex_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<vertex_t>(minor_range_size,
                                                                 handle.get_stream())
        : rmm::device_uvector<vertex_t>(minor_range_size, handle.get_stream());
    std::vector<size_t> recvcounts(major_comm_size);
    for (int i = 0; i < major_comm_size; ++i) {
      auto minor_range_vertex_partition_id =
        detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
          major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
      recvcounts[i] = partition.vertex_partition_range_size(minor_range_vertex_partition_id);
    }
    std::vector<size_t> displacements(recvcounts.size(), 0);
    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), displacements.begin(), size_t{0});
    device_allgatherv(major_comm,
                      renumber_map_labels.data(),
                      renumber_map_minor_labels.data(),
                      raft::host_span<size_t const>(recvcounts.data(), recvcounts.size()),
                      raft::host_span<size_t const>(displacements.data(), displacements.size()),
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
    detail::aggregate_offset_vectors(handle, vertex_partition_segment_offsets);
  auto edge_partition_hypersparse_degree_offsets =
    vertex_partition_hypersparse_degree_offsets
      ? std::make_optional(
          detail::aggregate_offset_vectors(handle, *vertex_partition_hypersparse_degree_offsets))
      : std::nullopt;

  return std::make_tuple(
    std::move(renumber_map_labels),
    renumber_meta_t<vertex_t, edge_t, multi_gpu>{number_of_vertices,
                                                 number_of_edges,
                                                 partition,
                                                 edge_partition_segment_offsets,
                                                 edge_partition_hypersparse_degree_offsets});
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
                  std::optional<large_buffer_type_t> large_vertex_buffer_type,
                  std::optional<large_buffer_type_t> large_edge_buffer_type,
                  bool do_expensive_check)
{
  CUGRAPH_EXPECTS((!large_vertex_buffer_type && !large_edge_buffer_type) ||
                    cugraph::large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto edgelist_majors = store_transposed ? edgelist_dsts : edgelist_srcs;
  auto edgelist_minors = store_transposed ? edgelist_srcs : edgelist_dsts;

  if (do_expensive_check) {
    detail::expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      vertices,
      std::vector<raft::device_span<vertex_t const>>{
        raft::device_span<vertex_t const>(edgelist_majors, num_edgelist_edges)},
      std::vector<raft::device_span<vertex_t const>>{
        raft::device_span<vertex_t const>(edgelist_minors, num_edgelist_edges)},
      std::nullopt);
  }

  auto [renumber_map_labels,
        segment_offsets,
        hypersparse_degree_offsets,
        locally_unused_vertex_id] =
    detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(
      handle,
      std::move(vertices),
      std::vector<raft::device_span<vertex_t const>>{
        raft::device_span<vertex_t const>(edgelist_majors, num_edgelist_edges)},
      std::vector<raft::device_span<vertex_t const>>{
        raft::device_span<vertex_t const>(edgelist_minors, num_edgelist_edges)},
      large_vertex_buffer_type,
      large_edge_buffer_type);

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

  return std::make_tuple(
    std::move(renumber_map_labels),
    renumber_meta_t<vertex_t, edge_t, multi_gpu>{segment_offsets, hypersparse_degree_offsets});
}

}  // namespace cugraph
