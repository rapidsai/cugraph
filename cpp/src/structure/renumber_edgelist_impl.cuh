/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

// returns renumber map and segment_offsets
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<vertex_t>, vertex_t> compute_renumber_map(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  std::vector<vertex_t const*> const& edgelist_majors,
  std::vector<vertex_t const*> const& edgelist_minors,
  std::vector<edge_t> const& edgelist_edge_counts)
{
  // 1. if local_vertices.has_value() is false, find unique vertices from edge majors & minors (to
  // construct local_vertices)

  rmm::device_uvector<vertex_t> sorted_local_vertices(0, handle.get_stream());
  if (!local_vertices) {
    rmm::device_uvector<vertex_t> sorted_unique_majors(0, handle.get_stream());
    if (edgelist_majors.size() > 1) {
      constexpr size_t num_bins{
        8};  // increase the number of bins to cut peak memory usage (at the expense of additional
             // computing), limit the maximum temporary memory usage to "size of local edge list
             // majors & minors" / "# bins".
      constexpr uint32_t hash_seed =
        1;  // shouldn't be 0 (in that case this hash function will coincide with the hash function
            // used to map vertices to GPUs, and we may not see the expected randomization)

      assert(multi_gpu);
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& minor_comm     = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      assert(static_cast<size_t>(minor_comm_size) == edgelist_majors.size());

      auto edge_count_vectors = num_bins > 1
                                  ? std::make_optional<std::vector<std::vector<edge_t>>>(
                                      edgelist_majors.size(), std::vector<edge_t>(num_bins))
                                  : std::nullopt;
      if (edge_count_vectors) {
        for (size_t i = 0; i < edgelist_majors.size(); ++i) {
          rmm::device_uvector<edge_t> d_edge_counts(num_bins, handle.get_stream());
          thrust::fill(
            handle.get_thrust_policy(), d_edge_counts.begin(), d_edge_counts.end(), edge_t{0});
          thrust::for_each(handle.get_thrust_policy(),
                           edgelist_majors[i],
                           edgelist_majors[i] + edgelist_edge_counts[i],
                           [counts = raft::device_span<edge_t>(
                              d_edge_counts.data(), d_edge_counts.size())] __device__(auto v) {
                             cuco::detail::MurmurHash3_32<vertex_t> hash_func{hash_seed};
                             cuda::atomic_ref<edge_t, cuda::thread_scope_device> atomic_counter(
                               counts[hash_func(v) % num_bins]);
                             atomic_counter.fetch_add(edge_t{1}, cuda::std::memory_order_relaxed);
                           });
          raft::update_host((*edge_count_vectors)[i].data(),
                            d_edge_counts.data(),
                            d_edge_counts.size(),
                            handle.get_stream());
        }
        handle.sync_stream();
      }

      for (size_t i = 0; i < num_bins; ++i) {
        std::vector<rmm::device_uvector<vertex_t>>
          edge_partition_sorted_unique_majors{};  // for bin "i"
        edge_partition_sorted_unique_majors.reserve(edgelist_majors.size());
        std::vector<size_t> tx_counts(minor_comm_size);
        for (size_t j = 0; j < edgelist_majors.size(); ++j) {
          rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
          if (num_bins > 1) {
            majors.resize((*edge_count_vectors)[j][i], handle.get_stream());
            thrust::copy_if(handle.get_thrust_policy(),
                            edgelist_majors[j],
                            edgelist_majors[j] + edgelist_edge_counts[j],
                            majors.begin(),
                            [i] __device__(auto v) {
                              cuco::detail::MurmurHash3_32<vertex_t> hash_func{hash_seed};
                              return (static_cast<size_t>(hash_func(v) % num_bins) == i);
                            });
          } else {
            majors.resize(edgelist_edge_counts[j], handle.get_stream());
            thrust::copy(handle.get_thrust_policy(),
                         edgelist_majors[j],
                         edgelist_majors[j] + edgelist_edge_counts[j],
                         majors.begin());
          }
          thrust::sort(handle.get_thrust_policy(), majors.begin(), majors.end());
          majors.resize(thrust::distance(
                          majors.begin(),
                          thrust::unique(handle.get_thrust_policy(), majors.begin(), majors.end())),
                        handle.get_stream());
          majors.shrink_to_fit(handle.get_stream());

          tx_counts[j] = majors.size();
          edge_partition_sorted_unique_majors.push_back(std::move(majors));
        }

        rmm::device_uvector<vertex_t> tmp_majors(std::reduce(tx_counts.begin(), tx_counts.end()),
                                                 handle.get_stream());
        size_t offset{0};
        for (size_t j = 0; j < edge_partition_sorted_unique_majors.size(); ++j) {
          thrust::copy(handle.get_thrust_policy(),
                       edge_partition_sorted_unique_majors[j].begin(),
                       edge_partition_sorted_unique_majors[j].end(),
                       tmp_majors.begin() + offset);
          offset += edge_partition_sorted_unique_majors[j].size();
        }
        edge_partition_sorted_unique_majors.clear();

        std::tie(tmp_majors, std::ignore) =
          shuffle_values(minor_comm, tmp_majors.begin(), tx_counts, handle.get_stream());

        thrust::sort(handle.get_thrust_policy(), tmp_majors.begin(), tmp_majors.end());
        tmp_majors.resize(
          thrust::distance(
            tmp_majors.begin(),
            thrust::unique(handle.get_thrust_policy(), tmp_majors.begin(), tmp_majors.end())),
          handle.get_stream());
        tmp_majors.shrink_to_fit(handle.get_stream());
        if (i == 0) {
          sorted_unique_majors = std::move(tmp_majors);
        } else {
          rmm::device_uvector<vertex_t> merged_majors(
            sorted_unique_majors.size() + tmp_majors.size(), handle.get_stream());
          thrust::merge(handle.get_thrust_policy(),
                        sorted_unique_majors.begin(),
                        sorted_unique_majors.end(),
                        tmp_majors.begin(),
                        tmp_majors.end(),
                        merged_majors.begin());
          sorted_unique_majors = std::move(merged_majors);
          tmp_majors.resize(0, handle.get_stream());
          tmp_majors.shrink_to_fit(handle.get_stream());
          sorted_unique_majors.resize(thrust::distance(sorted_unique_majors.begin(),
                                                       thrust::unique(handle.get_thrust_policy(),
                                                                      sorted_unique_majors.begin(),
                                                                      sorted_unique_majors.end())),
                                      handle.get_stream());
          sorted_unique_majors.shrink_to_fit(handle.get_stream());
        }
      }
    } else {
      rmm::device_uvector<vertex_t> majors(edgelist_edge_counts[0], handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_majors[0],
                   edgelist_majors[0] + edgelist_edge_counts[0],
                   majors.begin());
      thrust::sort(handle.get_thrust_policy(), majors.begin(), majors.end());
      majors.resize(
        thrust::distance(majors.begin(),
                         thrust::unique(handle.get_thrust_policy(), majors.begin(), majors.end())),
        handle.get_stream());
      majors.shrink_to_fit(handle.get_stream());
      sorted_unique_majors = std::move(majors);
    }

    rmm::device_uvector<vertex_t> sorted_unique_minors(0, handle.get_stream());
    for (size_t i = 0; i < edgelist_minors.size();
         ++i) {  // limit the maximum temporary memory usage to "size of local edge list majors &
                 // minors" / "# local edge partitions" (FXIME: we can further cut peak memory usage
                 // by applying binning here as well; fewer bins than the edge list major case)
      rmm::device_uvector<vertex_t> tmp_minors(edgelist_edge_counts[i], handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_minors[i],
                   edgelist_minors[i] + edgelist_edge_counts[i],
                   tmp_minors.begin());
      thrust::sort(handle.get_thrust_policy(), tmp_minors.begin(), tmp_minors.end());
      tmp_minors.resize(
        thrust::distance(
          tmp_minors.begin(),
          thrust::unique(handle.get_thrust_policy(), tmp_minors.begin(), tmp_minors.end())),
        handle.get_stream());
      tmp_minors.shrink_to_fit(handle.get_stream());

      if constexpr (multi_gpu) {
        auto& comm           = handle.get_comms();
        auto const comm_size = comm.get_size();
        auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
        auto const major_comm_size = major_comm.get_size();
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size = minor_comm.get_size();

        compute_gpu_id_from_ext_vertex_t<vertex_t> gpu_id_func{
          comm_size, major_comm_size, minor_comm_size};
        std::tie(tmp_minors, std::ignore) = groupby_gpu_id_and_shuffle_values(
          major_comm,
          tmp_minors.begin(),
          tmp_minors.end(),
          [major_comm_size, minor_comm_size, gpu_id_func] __device__(auto v) {
            auto comm_rank = gpu_id_func(v);
            return partition_manager::compute_major_comm_rank_from_global_comm_rank(
              major_comm_size, minor_comm_size, comm_rank);
          },
          handle.get_stream());
        thrust::sort(handle.get_thrust_policy(), tmp_minors.begin(), tmp_minors.end());
        tmp_minors.resize(
          thrust::distance(
            tmp_minors.begin(),
            thrust::unique(handle.get_thrust_policy(), tmp_minors.begin(), tmp_minors.end())),
          handle.get_stream());
        tmp_minors.shrink_to_fit(handle.get_stream());
      }

      if (i == 0) {
        sorted_unique_minors = std::move(tmp_minors);
      } else {
        rmm::device_uvector<vertex_t> merged_minors(sorted_unique_minors.size() + tmp_minors.size(),
                                                    handle.get_stream());
        thrust::merge(handle.get_thrust_policy(),
                      sorted_unique_minors.begin(),
                      sorted_unique_minors.end(),
                      tmp_minors.begin(),
                      tmp_minors.end(),
                      merged_minors.begin());
        sorted_unique_minors = std::move(merged_minors);
        tmp_minors.resize(0, handle.get_stream());
        tmp_minors.shrink_to_fit(handle.get_stream());
        sorted_unique_minors.resize(thrust::distance(sorted_unique_minors.begin(),
                                                     thrust::unique(handle.get_thrust_policy(),
                                                                    sorted_unique_minors.begin(),
                                                                    sorted_unique_minors.end())),
                                    handle.get_stream());
        sorted_unique_minors.shrink_to_fit(handle.get_stream());
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
    sorted_local_vertices.resize(thrust::distance(sorted_local_vertices.begin(),
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

  rmm::device_uvector<edge_t> sorted_local_vertex_degrees(0, handle.get_stream());
  std::optional<std::vector<size_t>> stream_pool_indices{
    std::nullopt};  // FIXME: move this inside the if statement

  auto constexpr num_chunks = size_t{
    2};  // tuning parameter, this trade-offs # binary searches (up to num_chunks times more binary
         // searches can be necessary if num_unique_majors << edgelist_edge_counts[i]) and temporary
         // buffer requirement (cut by num_chunks times), currently set to 2 to avoid peak memory
         // usage happening in this part (especially when minor_comm_size is small)

  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    assert(edgelist_majors.size() == minor_comm_size);

    auto edge_partition_major_range_sizes =
      host_scalar_allgather(minor_comm, sorted_local_vertices.size(), handle.get_stream());

    if ((minor_comm_size >= 2) && (handle.get_stream_pool_size() >= 2)) {
      edge_t num_local_edges =
        std::reduce(edgelist_edge_counts.begin(), edgelist_edge_counts.end());

      auto vertex_edge_counts = host_scalar_allreduce(
        comm,
        thrust::make_tuple(static_cast<vertex_t>(sorted_local_vertices.size()), num_local_edges),
        raft::comms::op_t::SUM,
        handle.get_stream());
      // memory footprint vs parallelism trade-off
      // peak memory requirement per loop is approximately
      //   (V/P) * (sizeof(vertex_t) + sizeof(edge_t)) +
      //   (E / (comm_size * minor_comm_size)) / num_chunks * sizeof(vertex_t) * 2 +
      //   std::min(V/P, (E / (comm_size * minor_comm_size)) / num_chunks) * (sizeof(vertex_t) +
      //   sizeof(edge_t))
      // and limit temporary memory requirement to (E / comm_size) * sizeof(vertex_t)
      auto avg_vertex_degree = thrust::get<0>(vertex_edge_counts) > 0
                                 ? static_cast<double>(thrust::get<1>(vertex_edge_counts)) /
                                     static_cast<double>(thrust::get<0>(vertex_edge_counts))
                                 : double{0.0};
      auto num_streams       = static_cast<size_t>(
        (avg_vertex_degree * sizeof(vertex_t)) /
        (static_cast<double>(sizeof(vertex_t) + sizeof(edge_t)) +
         (((avg_vertex_degree / minor_comm_size) / num_chunks) * sizeof(vertex_t) * 2) +
         (std::min(1.0, ((avg_vertex_degree / minor_comm_size) / num_chunks)) *
          (sizeof(vertex_t) + sizeof(edge_t)))));
      if (num_streams >= 2) {
        stream_pool_indices = std::vector<size_t>(num_streams);
        std::iota((*stream_pool_indices).begin(), (*stream_pool_indices).end(), size_t{0});
        handle.sync_stream();
      }
    }

    for (int i = 0; i < minor_comm_size; ++i) {
      auto loop_stream = stream_pool_indices
                           ? handle.get_stream_from_stream_pool(i % (*stream_pool_indices).size())
                           : handle.get_stream();

      rmm::device_uvector<vertex_t> sorted_majors(edge_partition_major_range_sizes[i], loop_stream);
      device_bcast(minor_comm,
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

      rmm::device_uvector<vertex_t> tmp_majors(0, loop_stream);
      tmp_majors.reserve(
        (static_cast<size_t>(edgelist_edge_counts[i]) + (num_chunks - 1)) / num_chunks,
        loop_stream);
      size_t offset{0};
      for (size_t j = 0; j < num_chunks; ++j) {
        size_t this_chunk_size =
          std::min(tmp_majors.capacity(), static_cast<size_t>(edgelist_edge_counts[i]) - offset);
        tmp_majors.resize(this_chunk_size, loop_stream);
        thrust::copy(rmm::exec_policy(loop_stream),
                     edgelist_majors[i] + offset,
                     edgelist_majors[i] + offset + tmp_majors.size(),
                     tmp_majors.begin());
        thrust::sort(rmm::exec_policy(loop_stream), tmp_majors.begin(), tmp_majors.end());
        auto num_unique_majors =
          thrust::count_if(rmm::exec_policy(loop_stream),
                           thrust::make_counting_iterator(size_t{0}),
                           thrust::make_counting_iterator(tmp_majors.size()),
                           is_first_in_run_t<vertex_t const*>{tmp_majors.data()});
        rmm::device_uvector<vertex_t> tmp_keys(num_unique_majors, loop_stream);
        rmm::device_uvector<edge_t> tmp_values(num_unique_majors, loop_stream);
        thrust::reduce_by_key(rmm::exec_policy(loop_stream),
                              tmp_majors.begin(),
                              tmp_majors.end(),
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

      device_reduce(minor_comm,
                    sorted_major_degrees.begin(),
                    sorted_major_degrees.begin(),
                    edge_partition_major_range_sizes[i],
                    raft::comms::op_t::SUM,
                    i,
                    loop_stream);
      if (i == minor_comm_rank) { sorted_local_vertex_degrees = std::move(sorted_major_degrees); }
    }

    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
  } else {
    assert(edgelist_majors.size() == 1);

    sorted_local_vertex_degrees.resize(sorted_local_vertices.size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 sorted_local_vertex_degrees.begin(),
                 sorted_local_vertex_degrees.end(),
                 edge_t{0});

    rmm::device_uvector<vertex_t> tmp_majors(0, handle.get_stream());
    tmp_majors.reserve(static_cast<size_t>(edgelist_edge_counts[0] + (num_chunks - 1)) / num_chunks,
                       handle.get_stream());
    size_t offset{0};
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t this_chunk_size =
        std::min(tmp_majors.capacity(), static_cast<size_t>(edgelist_edge_counts[0]) - offset);
      tmp_majors.resize(this_chunk_size, handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_majors[0] + offset,
                   edgelist_majors[0] + offset + tmp_majors.size(),
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

      auto kv_pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(tmp_keys.begin(), tmp_values.begin()));
      thrust::for_each(handle.get_thrust_policy(),
                       kv_pair_first,
                       kv_pair_first + tmp_keys.size(),
                       search_and_increment_degree_t<vertex_t, edge_t>{
                         sorted_local_vertices.data(),
                         static_cast<vertex_t>(sorted_local_vertices.size()),
                         sorted_local_vertex_degrees.data()});
      offset += this_chunk_size;
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
  size_t hypersparse_degree_threshold{0};
  if (multi_gpu) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    mid_degree_threshold *= minor_comm_size;
    low_degree_threshold *= minor_comm_size;
    hypersparse_degree_threshold =
      static_cast<size_t>(minor_comm_size * detail::hypersparse_threshold_ratio);
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
  d_segment_offsets.set_element(
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
      auto edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors[i], edgelist_minors[i]));
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + edgelist_edge_counts[i],
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
              edgelist_minors[i] + (*edgelist_intra_partition_segment_offsets)[i][j],
              edgelist_minors[i] + (*edgelist_intra_partition_segment_offsets)[i][j + 1],
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
                        recvcounts,
                        displacements,
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
          thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors[i], edgelist_minors[i]));

        CUGRAPH_EXPECTS(
          thrust::count_if(
            handle.get_thrust_policy(),
            edge_first,
            edge_first + edgelist_edge_counts[i],
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
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors[0], edgelist_minors[0]));
      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         edge_first,
                         edge_first + edgelist_edge_counts[0],
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
std::vector<vertex_t> aggregate_segment_offsets(raft::handle_t const& handle,
                                                std::vector<vertex_t> const& segment_offsets)
{
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  rmm::device_uvector<vertex_t> d_segment_offsets(segment_offsets.size(), handle.get_stream());
  raft::update_device(
    d_segment_offsets.data(), segment_offsets.data(), segment_offsets.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_aggregate_segment_offsets(
    minor_comm_size * d_segment_offsets.size(), handle.get_stream());
  minor_comm.allgather(d_segment_offsets.data(),
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
        "and "
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
    vertex_t max_edge_partition_major_range_size{0};
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      max_edge_partition_major_range_size = std::max(
        max_edge_partition_major_range_size, partition.local_edge_partition_major_range_size(i));
    }
    rmm::device_uvector<vertex_t> renumber_map_major_labels(max_edge_partition_major_range_size,
                                                            handle.get_stream());
    // FIXME: we may run this in parallel if memory is sufficient
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
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

  if ((static_cast<double>(partition.local_edge_partition_minor_range_size() *
                           2.5 /* tuning parameter */) >=
       static_cast<double>(number_of_edges / comm_size)) &&
      edgelist_intra_partition_segment_offsets) {  // memory footprint dominated by the O(V/sqrt(P))
                                                   // part than the O(E/P) part
    vertex_t max_segment_size{0};
    for (int i = 0; i < major_comm_size; ++i) {
      auto minor_range_vertex_partition_id =
        detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
          major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
      max_segment_size = std::max(
        max_segment_size, partition.vertex_partition_range_size(minor_range_vertex_partition_id));
    }
    rmm::device_uvector<vertex_t> renumber_map_minor_labels(max_segment_size, handle.get_stream());
    for (int i = 0; i < major_comm_size; ++i) {
      auto minor_range_vertex_partition_id =
        detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
          major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
      auto segment_size = partition.vertex_partition_range_size(minor_range_vertex_partition_id);
      device_bcast(major_comm,
                   renumber_map_labels.data(),
                   renumber_map_minor_labels.data(),
                   segment_size,
                   i,
                   handle.get_stream());

      kv_store_t<vertex_t, vertex_t, false> renumber_map(
        renumber_map_minor_labels.begin(),
        renumber_map_minor_labels.begin() + segment_size,
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
    rmm::device_uvector<vertex_t> renumber_map_minor_labels(
      partition.local_edge_partition_minor_range_size(), handle.get_stream());
    std::vector<size_t> recvcounts(major_comm_size);
    for (int i = 0; i < major_comm_size; ++i) {
      auto minor_range_vertex_partition_id =
        detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
          major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
      recvcounts[i] = partition.vertex_partition_range_size(minor_range_vertex_partition_id);
    }
    std::vector<size_t> displacements(recvcounts.size(), 0);
    std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
    device_allgatherv(major_comm,
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
