/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <experimental/include_cuco_static_map.cuh>

#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph.hpp>
#include <experimental/graph_functions.hpp>
#include <utilities/collect_comm.cuh>
#include <utilities/error.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace experimental {

template <typename vertex_t, bool multi_gpu>
void renumber_ext_vertices(raft::handle_t const& handle,
                           vertex_t* vertices /* [INOUT] */,
                           size_t num_vertices,
                           vertex_t const* renumber_map_labels,
                           vertex_t local_int_vertex_first,
                           vertex_t local_int_vertex_last,
                           bool do_expensive_check)
{
  double constexpr load_factor = 0.7;

  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "renumber_vertices() not supported on Pascal and older architectures.");

#ifdef CUCO_STATIC_MAP_DEFINED
  if (do_expensive_check) {
    rmm::device_uvector<vertex_t> labels(local_int_vertex_last - local_int_vertex_first,
                                         handle.get_stream());
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 renumber_map_labels,
                 renumber_map_labels + labels.size(),
                 labels.begin());
    thrust::sort(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()), labels.begin(), labels.end());
    CUGRAPH_EXPECTS(thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                   labels.begin(),
                                   labels.end()) == labels.end(),
                    "Invalid input arguments: renumber_map_labels have duplicate elements.");
  }

  auto renumber_map_ptr = std::make_unique<cuco::static_map<vertex_t, vertex_t>>(
    size_t{0}, invalid_vertex_id<vertex_t>::value, invalid_vertex_id<vertex_t>::value);
  if (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    rmm::device_uvector<vertex_t> sorted_unique_ext_vertices(num_vertices, handle.get_stream());
    sorted_unique_ext_vertices.resize(
      thrust::distance(
        sorted_unique_ext_vertices.begin(),
        thrust::copy_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        vertices,
                        vertices + num_vertices,
                        sorted_unique_ext_vertices.begin(),
                        [] __device__(auto v) { return v != invalid_vertex_id<vertex_t>::value; })),
      handle.get_stream());
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 sorted_unique_ext_vertices.begin(),
                 sorted_unique_ext_vertices.end());
    sorted_unique_ext_vertices.resize(
      thrust::distance(
        sorted_unique_ext_vertices.begin(),
        thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       sorted_unique_ext_vertices.begin(),
                       sorted_unique_ext_vertices.end())),
      handle.get_stream());

    auto int_vertices_for_sorted_unique_ext_vertices = collect_values_for_unique_keys(
      comm,
      renumber_map_labels,
      renumber_map_labels + (local_int_vertex_last - local_int_vertex_first),
      thrust::make_counting_iterator(local_int_vertex_first),
      sorted_unique_ext_vertices.begin(),
      sorted_unique_ext_vertices.end(),
      detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size},
      handle.get_stream());

    handle.get_stream_view().synchronize();  // cuco::static_map currently does not take stream

    renumber_map_ptr.reset();

    renumber_map_ptr = std::make_unique<cuco::static_map<vertex_t, vertex_t>>(
      // FIXME: std::max(..., size_t{1}) as a temporary workaround for
      // https://github.com/NVIDIA/cuCollections/issues/72
      std::max(
        static_cast<size_t>(static_cast<double>(sorted_unique_ext_vertices.size()) / load_factor),
        size_t{1}),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value);

    auto kv_pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(thrust::make_tuple(
        sorted_unique_ext_vertices.begin(), int_vertices_for_sorted_unique_ext_vertices.begin())),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });
    renumber_map_ptr->insert(kv_pair_first, kv_pair_first + sorted_unique_ext_vertices.size());
  } else {
    renumber_map_ptr.reset();

    renumber_map_ptr = std::make_unique<cuco::static_map<vertex_t, vertex_t>>(
      // FIXME: std::max(..., size_t{1}) as a temporary workaround for
      // https://github.com/NVIDIA/cuCollections/issues/72
      std::max(static_cast<size_t>(
                 static_cast<double>(local_int_vertex_last - local_int_vertex_first) / load_factor),
               size_t{1}),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value);

    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(renumber_map_labels, thrust::make_counting_iterator(vertex_t{0}))),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });
    renumber_map_ptr->insert(pair_first,
                             pair_first + (local_int_vertex_last - local_int_vertex_first));
  }

  if (do_expensive_check) {
    rmm::device_uvector<bool> contains(num_vertices, handle.get_stream());
    renumber_map_ptr->contains(vertices, vertices + num_vertices, contains.begin());
    auto vc_pair_first = thrust::make_zip_iterator(thrust::make_tuple(vertices, contains.begin()));
    CUGRAPH_EXPECTS(thrust::count_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                     vc_pair_first,
                                     vc_pair_first + num_vertices,
                                     [] __device__(auto pair) {
                                       auto v = thrust::get<0>(pair);
                                       auto c = thrust::get<1>(pair);
                                       return v == invalid_vertex_id<vertex_t>::value
                                                ? (c == true)
                                                : (c == false);
                                     }) == 0,
                    "Invalid input arguments: vertices have elements that are missing in "
                    "(aggregate) renumber_map_labels.");
  }

  renumber_map_ptr->find(vertices, vertices + num_vertices, vertices);
#endif
}

template <typename vertex_t>
void unrenumber_local_int_vertices(
  raft::handle_t const& handle,
  vertex_t* vertices /* [INOUT] */,
  size_t num_vertices,
  vertex_t const* renumber_map_labels /* size = local_int_vertex_last - local_int_vertex_first */,
  vertex_t local_int_vertex_first,
  vertex_t local_int_vertex_last,
  bool do_expensive_check)
{
  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "unrenumber_local_vertices() not supported on Pascal and older architectures.");

#ifdef CUCO_STATIC_MAP_DEFINED
  if (do_expensive_check) {
    CUGRAPH_EXPECTS(
      thrust::count_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       vertices,
                       vertices + num_vertices,
                       [local_int_vertex_first, local_int_vertex_last] __device__(auto v) {
                         return v != invalid_vertex_id<vertex_t>::value &&
                                (v < local_int_vertex_first || v >= local_int_vertex_last);
                       }) == 0,
      "Invalid input arguments: there are non-local vertices in [vertices, vertices "
      "+ num_vertices).");
  }

  thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    vertices,
                    vertices + num_vertices,
                    vertices,
                    [renumber_map_labels, local_int_vertex_first] __device__(auto v) {
                      return v == invalid_vertex_id<vertex_t>::value
                               ? v
                               : renumber_map_labels[v - local_int_vertex_first];
                    });
#endif
}

template <typename vertex_t, bool multi_gpu>
void unrenumber_int_vertices(raft::handle_t const& handle,
                             vertex_t* vertices /* [INOUT] */,
                             size_t num_vertices,
                             vertex_t const* renumber_map_labels,
                             vertex_t local_int_vertex_first,
                             vertex_t local_int_vertex_last,
                             std::vector<vertex_t>& vertex_partition_lasts,
                             bool do_expensive_check)
{
  double constexpr load_factor = 0.7;

  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "unrenumber_vertices() not supported on Pascal and older architectures.");

#ifdef CUCO_STATIC_MAP_DEFINED
  if (do_expensive_check) {
    CUGRAPH_EXPECTS(
      thrust::count_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       vertices,
                       vertices + num_vertices,
                       [num_vertices = vertex_partition_lasts.back()] __device__(auto v) {
                         return v != invalid_vertex_id<vertex_t>::value &&
                                !is_valid_vertex(num_vertices, v);
                       }) == 0,
      "Invalid input arguments: there are out-of-range vertices in [vertices, vertices "
      "+ num_vertices).");
  }

  if (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    rmm::device_uvector<vertex_t> sorted_unique_int_vertices(num_vertices, handle.get_stream());
    sorted_unique_int_vertices.resize(
      thrust::distance(
        sorted_unique_int_vertices.begin(),
        thrust::copy_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        vertices,
                        vertices + num_vertices,
                        sorted_unique_int_vertices.begin(),
                        [] __device__(auto v) { return v != invalid_vertex_id<vertex_t>::value; })),
      handle.get_stream());
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 sorted_unique_int_vertices.begin(),
                 sorted_unique_int_vertices.end());
    sorted_unique_int_vertices.resize(
      thrust::distance(
        sorted_unique_int_vertices.begin(),
        thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       sorted_unique_int_vertices.begin(),
                       sorted_unique_int_vertices.end())),
      handle.get_stream());

    rmm::device_uvector<vertex_t> d_vertex_partition_lasts(vertex_partition_lasts.size(),
                                                           handle.get_stream());
    raft::update_device(d_vertex_partition_lasts.data(),
                        vertex_partition_lasts.data(),
                        vertex_partition_lasts.size(),
                        handle.get_stream());
    rmm::device_uvector<size_t> d_tx_int_vertex_offsets(d_vertex_partition_lasts.size(),
                                                        handle.get_stream());
    thrust::lower_bound(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        sorted_unique_int_vertices.begin(),
                        sorted_unique_int_vertices.end(),
                        d_vertex_partition_lasts.begin(),
                        d_vertex_partition_lasts.end(),
                        d_tx_int_vertex_offsets.begin());
    std::vector<size_t> h_tx_int_vertex_counts(d_tx_int_vertex_offsets.size());
    raft::update_host(h_tx_int_vertex_counts.data(),
                      d_tx_int_vertex_offsets.data(),
                      d_tx_int_vertex_offsets.size(),
                      handle.get_stream());
    handle.get_stream_view().synchronize();
    std::adjacent_difference(
      h_tx_int_vertex_counts.begin(), h_tx_int_vertex_counts.end(), h_tx_int_vertex_counts.begin());

    rmm::device_uvector<vertex_t> rx_int_vertices(0, handle.get_stream());
    std::vector<size_t> rx_int_vertex_counts{};
    std::tie(rx_int_vertices, rx_int_vertex_counts) = shuffle_values(
      comm, sorted_unique_int_vertices.begin(), h_tx_int_vertex_counts, handle.get_stream());

    auto tx_ext_vertices = std::move(rx_int_vertices);
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      tx_ext_vertices.begin(),
                      tx_ext_vertices.end(),
                      tx_ext_vertices.begin(),
                      [renumber_map_labels, local_int_vertex_first] __device__(auto v) {
                        return renumber_map_labels[v - local_int_vertex_first];
                      });

    rmm::device_uvector<vertex_t> rx_ext_vertices_for_sorted_unique_int_vertices(
      0, handle.get_stream());
    std::tie(rx_ext_vertices_for_sorted_unique_int_vertices, std::ignore) =
      shuffle_values(comm, tx_ext_vertices.begin(), rx_int_vertex_counts, handle.get_stream());

    handle.get_stream_view().synchronize();  // cuco::static_map currently does not take stream

    cuco::static_map<vertex_t, vertex_t> unrenumber_map(
      // FIXME: std::max(..., size_t{1}) as a temporary workaround for
      // https://github.com/NVIDIA/cuCollections/issues/72
      std::max(
        static_cast<size_t>(static_cast<double>(sorted_unique_int_vertices.size()) / load_factor),
        size_t{1}),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value);

    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(sorted_unique_int_vertices.begin(),
                           rx_ext_vertices_for_sorted_unique_int_vertices.begin())),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });
    unrenumber_map.insert(pair_first, pair_first + sorted_unique_int_vertices.size());
    unrenumber_map.find(vertices, vertices + num_vertices, vertices);
  } else {
    unrenumber_local_int_vertices(handle,
                                  vertices,
                                  num_vertices,
                                  renumber_map_labels,
                                  local_int_vertex_first,
                                  local_int_vertex_last,
                                  do_expensive_check);
  }
#endif
}

// explicit instantiation

template void renumber_ext_vertices<int32_t, false>(raft::handle_t const& handle,
                                                    int32_t* vertices,
                                                    size_t num_vertices,
                                                    int32_t const* renumber_map_labels,
                                                    int32_t local_int_vertex_first,
                                                    int32_t local_int_vertex_last,
                                                    bool do_expensive_check);

template void renumber_ext_vertices<int32_t, true>(raft::handle_t const& handle,
                                                   int32_t* vertices,
                                                   size_t num_vertices,
                                                   int32_t const* renumber_map_labels,
                                                   int32_t local_int_vertex_first,
                                                   int32_t local_int_vertex_last,
                                                   bool do_expensive_check);

template void renumber_ext_vertices<int64_t, false>(raft::handle_t const& handle,
                                                    int64_t* vertices,
                                                    size_t num_vertices,
                                                    int64_t const* renumber_map_labels,
                                                    int64_t local_int_vertex_first,
                                                    int64_t local_int_vertex_last,
                                                    bool do_expensive_check);

template void renumber_ext_vertices<int64_t, true>(raft::handle_t const& handle,
                                                   int64_t* vertices,
                                                   size_t num_vertices,
                                                   int64_t const* renumber_map_labels,
                                                   int64_t local_int_vertex_first,
                                                   int64_t local_int_vertex_last,
                                                   bool do_expensive_check);

template void unrenumber_local_int_vertices<int32_t>(raft::handle_t const& handle,
                                                     int32_t* vertices,
                                                     size_t num_vertices,
                                                     int32_t const* renumber_map_labels,
                                                     int32_t local_int_vertex_first,
                                                     int32_t local_int_vertex_last,
                                                     bool do_expensive_check);

template void unrenumber_local_int_vertices<int64_t>(raft::handle_t const& handle,
                                                     int64_t* vertices,
                                                     size_t num_vertices,
                                                     int64_t const* renumber_map_labels,
                                                     int64_t local_int_vertex_first,
                                                     int64_t local_int_vertex_last,
                                                     bool do_expensive_check);

template void unrenumber_int_vertices<int32_t, false>(raft::handle_t const& handle,
                                                      int32_t* vertices,
                                                      size_t num_vertices,
                                                      int32_t const* renumber_map_labels,
                                                      int32_t local_int_vertex_first,
                                                      int32_t local_int_vertex_last,
                                                      std::vector<int32_t>& vertex_partition_lasts,
                                                      bool do_expensive_check);

template void unrenumber_int_vertices<int32_t, true>(raft::handle_t const& handle,
                                                     int32_t* vertices,
                                                     size_t num_vertices,
                                                     int32_t const* renumber_map_labels,
                                                     int32_t local_int_vertex_first,
                                                     int32_t local_int_vertex_last,
                                                     std::vector<int32_t>& vertex_partition_lasts,
                                                     bool do_expensive_check);

template void unrenumber_int_vertices<int64_t, false>(raft::handle_t const& handle,
                                                      int64_t* vertices,
                                                      size_t num_vertices,
                                                      int64_t const* renumber_map_labels,
                                                      int64_t local_int_vertex_first,
                                                      int64_t local_int_vertex_last,
                                                      std::vector<int64_t>& vertex_partition_lasts,
                                                      bool do_expensive_check);

template void unrenumber_int_vertices<int64_t, true>(raft::handle_t const& handle,
                                                     int64_t* vertices,
                                                     size_t num_vertices,
                                                     int64_t const* renumber_map_labels,
                                                     int64_t local_int_vertex_first,
                                                     int64_t local_int_vertex_last,
                                                     std::vector<int64_t>& vertex_partition_lasts,
                                                     bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
