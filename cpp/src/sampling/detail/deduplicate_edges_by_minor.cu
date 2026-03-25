/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/sampling_utils.hpp"
#include "utilities/collect_comm.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/copy.hpp>

#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <variant>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<vertex_t>>,
           std::optional<rmm::device_uvector<int32_t>>>
deduplicate_edges_by_minor(raft::handle_t const& handle,
                           graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                           rmm::device_uvector<vertex_t>&& result_majors,
                           rmm::device_uvector<vertex_t>&& result_minors,
                           arithmetic_device_uvector_t&& tmp_edge_indices,
                           std::optional<rmm::device_uvector<int32_t>>&& result_labels,
                           bool call_from_sampling)
{
  std::optional<rmm::device_uvector<vertex_t>> resample_majors{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> resample_major_labels{std::nullopt};

  size_t total_edges = result_majors.size();

  if constexpr (multi_gpu) {
    total_edges = host_scalar_allreduce(
      handle.get_comms(), total_edges, raft::comms::op_t::SUM, handle.get_stream());
  }

  if (total_edges == 0) {
    return std::make_tuple(std::move(result_majors),
                           std::move(result_minors),
                           std::move(tmp_edge_indices),
                           std::move(result_labels),
                           std::move(resample_majors),
                           std::move(resample_major_labels));
  }

  // 1. Shuffle the edges to GPUs by minor vertex id if multi-gpu
  std::optional<rmm::device_uvector<int>> keep_ranks{std::nullopt};
  rmm::device_uvector<vertex_t> keep_majors(result_minors.size(), handle.get_stream());
  std::optional<rmm::device_uvector<int32_t>> keep_labels{std::nullopt};
  rmm::device_uvector<vertex_t> keep_minors(result_minors.size(), handle.get_stream());
  rmm::device_uvector<size_t> keep_positions(result_minors.size(), handle.get_stream());

  raft::copy(keep_majors.begin(), result_majors.begin(), result_majors.size(), handle.get_stream());
  raft::copy(keep_minors.begin(), result_minors.begin(), result_minors.size(), handle.get_stream());

  if (result_labels) {
    keep_labels =
      std::make_optional<rmm::device_uvector<int32_t>>(result_labels->size(), handle.get_stream());
    raft::copy(
      keep_labels->begin(), result_labels->begin(), result_labels->size(), handle.get_stream());
  }

  thrust::sequence(
    handle.get_thrust_policy(), keep_positions.begin(), keep_positions.end(), size_t{0});

  if constexpr (multi_gpu) {
    keep_ranks =
      std::make_optional<rmm::device_uvector<int>>(keep_positions.size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 keep_ranks->begin(),
                 keep_ranks->end(),
                 handle.get_comms().get_rank());

    std::vector<arithmetic_device_uvector_t> shuffle_properties{};
    shuffle_properties.push_back(std::move(keep_majors));
    shuffle_properties.push_back(std::move(*keep_ranks));
    shuffle_properties.push_back(std::move(keep_positions));
    if (result_labels) { shuffle_properties.push_back(std::move(*keep_labels)); }

    std::tie(keep_minors, shuffle_properties) =
      shuffle_int_vertices(handle,
                           std::move(keep_minors),
                           std::move(shuffle_properties),
                           graph_view.vertex_partition_range_lasts(),
                           std::nullopt);

    keep_majors    = std::move(std::get<rmm::device_uvector<vertex_t>>(shuffle_properties[0]));
    keep_ranks     = std::move(std::get<rmm::device_uvector<int>>(shuffle_properties[1]));
    keep_positions = std::move(std::get<rmm::device_uvector<size_t>>(shuffle_properties[2]));
    if (result_labels) {
      keep_labels = std::move(std::get<rmm::device_uvector<int32_t>>(shuffle_properties[3]));
    }
  }

  // 2. Now all edges that lead to a visited minor are on this GPU, we can
  //    Sort by minor vertex id and identify duplicates.
  rmm::device_uvector<vertex_t> local_positions(keep_minors.size(), handle.get_stream());
  thrust::sequence(
    handle.get_thrust_policy(), local_positions.begin(), local_positions.end(), size_t{0});

  // FIXME: After we refactor sample_edges_to_unvisited_neighbors to use the partial results we
  //        can remove the majors from this sort.  Until then we need to sort by (label, minor,
  //        major, position) to guarantee that each iteration of the outer loop makes progress.
  if (keep_labels) {
    thrust::sort(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(
        keep_labels->begin(), keep_minors.begin(), keep_majors.begin(), local_positions.begin()),
      thrust::make_zip_iterator(
        keep_labels->end(), keep_minors.end(), keep_majors.end(), local_positions.end()));
  } else {
    thrust::sort(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(keep_minors.begin(), keep_majors.begin(), local_positions.begin()),
      thrust::make_zip_iterator(keep_minors.end(), keep_majors.end(), local_positions.end()));
  }

  // 3. Mark the edges to keep locally
  size_t keep_count{0};
  rmm::device_uvector<uint32_t> keep_flags(0, handle.get_stream());

  // We'll keep the first edge for each (label, minor) pair.
  if (keep_labels) {
    std::tie(keep_count, keep_flags) =
      detail::mark_entries(handle,
                           keep_minors.size(),
                           detail::is_first_in_run_t<decltype(thrust::make_zip_iterator(
                             keep_labels->begin(), keep_minors.begin()))>{
                             thrust::make_zip_iterator(keep_labels->begin(), keep_minors.begin())});
  } else {
    std::tie(keep_count, keep_flags) = detail::mark_entries(
      handle,
      keep_minors.size(),
      detail::is_first_in_run_t<decltype(keep_minors.begin())>{keep_minors.begin()});
  }

  size_t global_remove_count{keep_minors.size() - keep_count};
  if constexpr (multi_gpu) {
    global_remove_count = host_scalar_allreduce(handle.get_comms(),
                                                (keep_minors.size() - keep_count),
                                                raft::comms::op_t::SUM,
                                                handle.get_stream());
  }

  // 4. If we have any duplicates on any GPU we need to remove them
  if (global_remove_count > 0) {
    bool skip_shuffle_back_to_ranks{false};

    if (call_from_sampling) {
      // When called from sampling, we need to skip all edges that come from any major vertex that
      // we are going to skip.
      resample_majors =
        std::make_optional<rmm::device_uvector<vertex_t>>(keep_majors.size(), handle.get_stream());
      raft::copy(
        resample_majors->begin(), keep_majors.begin(), keep_majors.size(), handle.get_stream());

      if (keep_labels) {
        resample_major_labels = std::make_optional<rmm::device_uvector<int32_t>>(
          keep_labels->size(), handle.get_stream());
        raft::copy(resample_major_labels->begin(),
                   keep_labels->begin(),
                   keep_labels->size(),
                   handle.get_stream());
      }

      detail::invert_flags(handle,
                           raft::device_span<uint32_t>{keep_flags.data(), keep_flags.size()},
                           keep_minors.size());

      resample_majors = detail::keep_marked_entries(
        handle,
        std::move(*resample_majors),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        (keep_minors.size() - keep_count));
      if (resample_major_labels) {
        resample_major_labels = detail::keep_marked_entries(
          handle,
          std::move(*resample_major_labels),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          (keep_minors.size() - keep_count));
      }

      if constexpr (multi_gpu) {
        std::vector<cugraph::arithmetic_device_uvector_t> shuffle_properties{};
        if (resample_major_labels) {
          shuffle_properties.push_back(std::move(*resample_major_labels));
        }
        std::tie(resample_majors, shuffle_properties) =
          cugraph::shuffle_int_vertices(handle,
                                        std::move(*resample_majors),
                                        std::move(shuffle_properties),
                                        graph_view.vertex_partition_range_lasts());
        if (resample_major_labels) {
          resample_major_labels =
            std::move(std::get<rmm::device_uvector<int32_t>>(shuffle_properties[0]));
        }
      }

      if (resample_major_labels) {
        auto new_begin =
          thrust::make_zip_iterator(resample_major_labels->begin(), resample_majors->begin());
        thrust::sort(handle.get_thrust_policy(), new_begin, new_begin + resample_majors->size());
        auto new_end = thrust::unique(
          handle.get_thrust_policy(), new_begin, new_begin + resample_majors->size());
        resample_majors->resize(cuda::std::distance(new_begin, new_end), handle.get_stream());
        resample_major_labels->resize(cuda::std::distance(new_begin, new_end), handle.get_stream());
      } else {
        thrust::sort(handle.get_thrust_policy(), resample_majors->begin(), resample_majors->end());
        auto new_end = thrust::unique(
          handle.get_thrust_policy(), resample_majors->begin(), resample_majors->end());
        resample_majors->resize(cuda::std::distance(resample_majors->begin(), new_end),
                                handle.get_stream());
      }

      rmm::device_uvector<vertex_t> resample_majors_gathered(0, handle.get_stream());
      rmm::device_uvector<int32_t> resample_major_labels_gathered(0, handle.get_stream());
      raft::device_span<vertex_t const> resample_majors_span{resample_majors->data(),
                                                             resample_majors->size()};
      raft::device_span<int32_t const> resample_major_labels_span{
        resample_major_labels ? resample_major_labels->data() : nullptr,
        resample_major_labels ? resample_major_labels->size() : size_t{0}};

      if constexpr (multi_gpu) {
        auto& major_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

        resample_majors_gathered =
          cugraph::device_allgatherv(handle, major_comm, resample_majors_span);
        resample_majors_span = raft::device_span<vertex_t const>{resample_majors_gathered.data(),
                                                                 resample_majors_gathered.size()};
        if (resample_major_labels) {
          resample_major_labels_gathered = cugraph::device_allgatherv(
            handle,
            major_comm,
            raft::device_span<int32_t const>{resample_major_labels->data(),
                                             resample_major_labels->size()});
          resample_major_labels_span = raft::device_span<int32_t const>{
            resample_major_labels_gathered.data(), resample_major_labels_gathered.size()};
        }
      }

      // Now we'll regenerate keep_flags to remove all entries that come from an entry in
      // resample_majors This will include all of the current marked entries and potentially some
      // new entries

      if (resample_major_labels) {
        std::tie(keep_count, keep_flags) =
          detail::mark_entries(handle,
                               result_majors.size(),
                               [majors = result_majors.data(),
                                labels = result_labels->data(),
                                resample_majors_span,
                                resample_major_labels_span] __device__(auto index) {
                                 return !thrust::binary_search(
                                   thrust::seq,
                                   thrust::make_zip_iterator(resample_major_labels_span.begin(),
                                                             resample_majors_span.begin()),
                                   thrust::make_zip_iterator(resample_major_labels_span.end(),
                                                             resample_majors_span.end()),
                                   cuda::std::make_tuple(labels[index], majors[index]));
                               });
      } else {
        std::tie(keep_count, keep_flags) = detail::mark_entries(
          handle,
          result_majors.size(),
          [majors = result_majors.data(), resample_majors_span] __device__(auto index) {
            return !thrust::binary_search(
              thrust::seq, resample_majors_span.begin(), resample_majors_span.end(), majors[index]);
          });
      }

      if constexpr (multi_gpu) {
        resample_majors_gathered.resize(0, handle.get_stream());
        resample_majors_gathered.shrink_to_fit(handle.get_stream());
        if (resample_major_labels) {
          resample_major_labels_gathered.resize(0, handle.get_stream());
          resample_major_labels_gathered.shrink_to_fit(handle.get_stream());
        }
      }

      raft::device_span<uint32_t const> const result_keep_flags{keep_flags.data(),
                                                                keep_flags.size()};
      result_majors = detail::keep_marked_entries(
        handle, std::move(result_majors), result_keep_flags, keep_count);
      result_minors = detail::keep_marked_entries(
        handle, std::move(result_minors), result_keep_flags, keep_count);
      if (!std::holds_alternative<std::monostate>(tmp_edge_indices)) {
        cugraph::variant_type_dispatch(
          tmp_edge_indices, [&handle, result_keep_flags, keep_count](auto& property) {
            property = detail::keep_marked_entries(
              handle, std::move(property), result_keep_flags, keep_count);
          });
      }
      if (result_labels) {
        *result_labels = detail::keep_marked_entries(
          handle, std::move(*result_labels), result_keep_flags, keep_count);
      }
      skip_shuffle_back_to_ranks = true;
    } else {
      // Gather to reflect the original positions of the edges
      rmm::device_uvector<size_t> tmp(local_positions.size(), handle.get_stream());
      thrust::gather(handle.get_thrust_policy(),
                     local_positions.begin(),
                     local_positions.end(),
                     keep_positions.data(),
                     tmp.begin());
      keep_positions = std::move(tmp);
    }

    if (!skip_shuffle_back_to_ranks) {
      // 5. Now keep_ranks and keep_positions need to be updated to reflect the new keep_flags and
      // shuffled back to the original ranks (for multi-gpu)
      if (keep_count < keep_minors.size()) {
        if constexpr (multi_gpu) {
          keep_ranks = detail::keep_marked_entries(
            handle,
            std::move(*keep_ranks),
            raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
            keep_count);
        }
        keep_positions = detail::keep_marked_entries(
          handle,
          std::move(keep_positions),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          keep_count);
      }

      if constexpr (multi_gpu) {
        std::tie(std::ignore, keep_positions, std::ignore) =
          groupby_gpu_id_and_shuffle_kv_pairs(handle.get_comms(),
                                              keep_ranks->begin(),
                                              keep_ranks->end(),
                                              keep_positions.begin(),
                                              cuda::std::identity{},
                                              handle.get_stream());
      }

      // 6. Now we can remove values from results arrays
      {
        rmm::device_uvector<vertex_t> tmp(keep_positions.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       keep_positions.begin(),
                       keep_positions.end(),
                       result_majors.data(),
                       tmp.begin());
        result_majors = std::move(tmp);
      }
      {
        rmm::device_uvector<vertex_t> tmp(keep_positions.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       keep_positions.begin(),
                       keep_positions.end(),
                       result_minors.data(),
                       tmp.begin());
        result_minors = std::move(tmp);
      }

      if (!std::holds_alternative<std::monostate>(tmp_edge_indices)) {
        cugraph::variant_type_dispatch(
          tmp_edge_indices, [&handle, &keep_positions](auto& property) {
            using T = typename std::remove_reference<decltype(property)>::type::value_type;
            rmm::device_uvector<T> tmp(keep_positions.size(), handle.get_stream());
            thrust::gather(handle.get_thrust_policy(),
                           keep_positions.begin(),
                           keep_positions.end(),
                           property.data(),
                           tmp.begin());
            property = std::move(tmp);
          });
      }
      if (result_labels) {
        rmm::device_uvector<int32_t> tmp(keep_positions.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       keep_positions.begin(),
                       keep_positions.end(),
                       result_labels->begin(),
                       tmp.begin());
        *result_labels = std::move(tmp);
      }
    }
  }

  return std::make_tuple(std::move(result_majors),
                         std::move(result_minors),
                         std::move(tmp_edge_indices),
                         std::move(result_labels),
                         std::move(resample_majors),
                         std::move(resample_major_labels));
}

// Explicit instantiations for common configurations
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    arithmetic_device_uvector_t,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
deduplicate_edges_by_minor<int32_t, int32_t, false>(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, false> const&,
  rmm::device_uvector<int32_t>&&,
  rmm::device_uvector<int32_t>&&,
  arithmetic_device_uvector_t&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  bool);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    arithmetic_device_uvector_t,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
deduplicate_edges_by_minor<int32_t, int32_t, true>(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  rmm::device_uvector<int32_t>&&,
  rmm::device_uvector<int32_t>&&,
  arithmetic_device_uvector_t&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  bool);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    arithmetic_device_uvector_t,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
deduplicate_edges_by_minor<int64_t, int64_t, false>(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, false> const&,
  rmm::device_uvector<int64_t>&&,
  rmm::device_uvector<int64_t>&&,
  arithmetic_device_uvector_t&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  bool);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    arithmetic_device_uvector_t,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
deduplicate_edges_by_minor<int64_t, int64_t, true>(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, true> const&,
  rmm::device_uvector<int64_t>&&,
  rmm::device_uvector<int64_t>&&,
  arithmetic_device_uvector_t&&,
  std::optional<rmm::device_uvector<int32_t>>&&,
  bool);

}  // namespace detail
}  // namespace cugraph
