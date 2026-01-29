/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/sampling_utils.hpp"
#include "utilities/collect_comm.cuh"

#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<std::optional<rmm::device_uvector<vertex_t>>,
           std::optional<rmm::device_uvector<int32_t>>>
update_dst_visited_vertices_and_labels(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<rmm::device_uvector<vertex_t>>&& visited_vertices,
  std::optional<rmm::device_uvector<int32_t>>&& visited_vertex_labels,
  raft::device_span<vertex_t const> sampled_vertices,
  std::optional<raft::device_span<int32_t const>> sampled_vertex_labels)
{
  CUGRAPH_EXPECTS(visited_vertices.has_value(), "Invalid input: visited_vertices must be provided");
  CUGRAPH_EXPECTS(
    visited_vertex_labels.has_value() == sampled_vertex_labels.has_value(),
    "Invalid input: visited_vertex_labels and sampled_vertex_labels must have the same presence");

  // 1) Shuffle sampled items to the owning GPU by vertex partitioning (minor path context)
  rmm::device_uvector<vertex_t> new_samples(sampled_vertices.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               sampled_vertices.begin(),
               sampled_vertices.end(),
               new_samples.begin());

  std::vector<cugraph::arithmetic_device_uvector_t> props{};
  std::optional<rmm::device_uvector<int32_t>> new_sample_labels{std::nullopt};
  if (sampled_vertex_labels) {
    new_sample_labels =
      rmm::device_uvector<int32_t>(sampled_vertex_labels->size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 sampled_vertex_labels->begin(),
                 sampled_vertex_labels->end(),
                 new_sample_labels->begin());
    props.push_back(std::move(*new_sample_labels));
  }

  if constexpr (multi_gpu) {
    std::tie(new_samples, props) = cugraph::shuffle_int_vertices(
      handle, std::move(new_samples), std::move(props), graph_view.vertex_partition_range_lasts());
  }

  if (props.size() > 0) {
    new_sample_labels = std::move(std::get<rmm::device_uvector<int32_t>>(props[0]));
  }

  // 2) Sort and dedupe the new sampled items (reduce comm and storage)
  if (new_sample_labels) {
    thrust::sort(handle.get_thrust_policy(),
                 thrust::make_zip_iterator(new_samples.begin(), new_sample_labels->begin()),
                 thrust::make_zip_iterator(new_samples.end(), new_sample_labels->end()));
    auto new_zip_end =
      thrust::unique(handle.get_thrust_policy(),
                     thrust::make_zip_iterator(new_samples.begin(), new_sample_labels->begin()),
                     thrust::make_zip_iterator(new_samples.end(), new_sample_labels->end()));
    size_t new_size =
      static_cast<size_t>(thrust::get<0>(new_zip_end.get_iterator_tuple()) - new_samples.begin());
    new_samples.resize(new_size, handle.get_stream());
    new_sample_labels->resize(new_size, handle.get_stream());
  } else {
    thrust::sort(handle.get_thrust_policy(), new_samples.begin(), new_samples.end());
    auto new_end =
      thrust::unique(handle.get_thrust_policy(), new_samples.begin(), new_samples.end());
    size_t n_keep = static_cast<size_t>(new_end - new_samples.begin());
    new_samples.resize(n_keep, handle.get_stream());
  }

  // 3) Aggregate new samples across minor_comm (unify per-minor partition)
  if constexpr (multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    new_samples =
      device_allgatherv(handle,
                        minor_comm,
                        raft::device_span<vertex_t const>{new_samples.data(), new_samples.size()});
    if (new_sample_labels) {
      new_sample_labels = device_allgatherv(
        handle,
        minor_comm,
        raft::device_span<int32_t const>{new_sample_labels->data(), new_sample_labels->size()});
    }
  }

  // 4) Single-GPU: append local new samples (already deduped) at tail and sort
  auto const orig_size = static_cast<size_t>(visited_vertices->size());
  visited_vertices->resize(orig_size + new_samples.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               new_samples.begin(),
               new_samples.end(),
               visited_vertices->begin() + orig_size);

  if (new_sample_labels) {
    visited_vertex_labels->resize(orig_size + new_sample_labels->size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 new_sample_labels->begin(),
                 new_sample_labels->end(),
                 visited_vertex_labels->begin() + orig_size);

    thrust::sort(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(visited_vertices->begin(), visited_vertex_labels->begin()),
      thrust::make_zip_iterator(visited_vertices->end(), visited_vertex_labels->end()));
  } else {
    thrust::sort(handle.get_thrust_policy(), visited_vertices->begin(), visited_vertices->end());
  }

  return std::make_tuple(std::move(visited_vertices), std::move(visited_vertex_labels));
}

// Explicit instantiations for common configurations
template std::tuple<std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
update_dst_visited_vertices_and_labels<int32_t, int32_t, false>(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, false> const&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  raft::device_span<int32_t const>,
  std::optional<raft::device_span<int32_t const>>);

template std::tuple<std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
update_dst_visited_vertices_and_labels<int32_t, int32_t, true>(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  raft::device_span<int32_t const>,
  std::optional<raft::device_span<int32_t const>>);

template std::tuple<std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
update_dst_visited_vertices_and_labels<int64_t, int64_t, false>(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, false> const&,
  std::optional<rmm::device_uvector<int64_t>>&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  raft::device_span<int64_t const>,
  std::optional<raft::device_span<int32_t const>>);

template std::tuple<std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
update_dst_visited_vertices_and_labels<int64_t, int64_t, true>(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, true> const&,
  std::optional<rmm::device_uvector<int64_t>>&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  raft::device_span<int64_t const>,
  std::optional<raft::device_span<int32_t const>>);

}  // namespace detail
}  // namespace cugraph
