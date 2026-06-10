/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sampling/detail/sampling_utils.hpp"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/device_comm_wrapper.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers.hpp>

#include <raft/core/copy.hpp>

#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <variant>
#include <vector>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
deduplicate_edges_by_minor(raft::handle_t const& handle,
                           graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                           rmm::device_uvector<vertex_t>&& result_majors,
                           rmm::device_uvector<vertex_t>&& result_minors,
                           arithmetic_device_uvector_t&& result_edge_property,
                           arithmetic_device_uvector_t&& result_types,
                           std::optional<rmm::device_uvector<int32_t>>&& result_labels)
{
  bool const has_edge_property = !std::holds_alternative<std::monostate>(result_edge_property);
  bool const has_types         = !std::holds_alternative<std::monostate>(result_types);
  bool const has_labels        = result_labels.has_value();
  if (has_types) {
    CUGRAPH_EXPECTS(std::holds_alternative<rmm::device_uvector<int32_t>>(result_types),
                    "result_types must be rmm::device_uvector<int32_t> when present.");
  }
  if (has_edge_property) {
    CUGRAPH_EXPECTS(std::holds_alternative<rmm::device_uvector<edge_t>>(result_edge_property),
                    "result_edge_property must be rmm::device_uvector<edge_t> when present.");
  }

  size_t total_edges = result_majors.size();

  if constexpr (multi_gpu) {
    total_edges = host_scalar_allreduce(
      handle.get_comms(), total_edges, raft::comms::op_t::SUM, handle.get_stream());
  }

  if (total_edges == 0) {
    return std::make_tuple(std::move(result_majors),
                           std::move(result_minors),
                           std::move(result_edge_property),
                           std::move(result_labels),
                           rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                           rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                           arithmetic_device_uvector_t{std::monostate{}},
                           arithmetic_device_uvector_t{std::monostate{}},
                           std::optional<rmm::device_uvector<int32_t>>{std::nullopt});
  }

  // 1. Shuffle the edges to GPUs by minor vertex id if multi-gpu
  if constexpr (multi_gpu) {
    std::vector<arithmetic_device_uvector_t> shuffle_properties{};
    shuffle_properties.push_back(std::move(result_majors));
    if (has_edge_property) { shuffle_properties.push_back(std::move(result_edge_property)); }
    if (has_types) { shuffle_properties.push_back(std::move(result_types)); }
    if (has_labels) { shuffle_properties.push_back(std::move(*result_labels)); }

    std::tie(result_minors, shuffle_properties) =
      shuffle_int_vertices(handle,
                           std::move(result_minors),
                           std::move(shuffle_properties),
                           graph_view.vertex_partition_range_lasts(),
                           std::nullopt);

    result_majors = std::move(std::get<rmm::device_uvector<vertex_t>>(shuffle_properties[0]));
    size_t shuffle_prop_idx{1};
    if (has_edge_property) {
      result_edge_property = std::move(shuffle_properties[shuffle_prop_idx++]);
    }
    if (has_types) { result_types = std::move(shuffle_properties[shuffle_prop_idx++]); }
    if (has_labels) {
      result_labels =
        std::move(std::get<rmm::device_uvector<int32_t>>(shuffle_properties[shuffle_prop_idx++]));
    }
  }

  // 2. Sort the edges by minor vertex id and identify duplicates
  if (has_edge_property) {
    auto& property = std::get<rmm::device_uvector<edge_t>>(result_edge_property);
    if (has_types) {
      auto& types = std::get<rmm::device_uvector<int32_t>>(result_types);
      if (has_labels) {
        cugraph::sort_wrapper(handle.get_thrust_policy(),
                              thrust::make_zip_iterator(result_labels->begin(),
                                                        result_minors.begin(),
                                                        result_majors.begin(),
                                                        property.begin(),
                                                        types.begin()),
                              thrust::make_zip_iterator(result_labels->end(),
                                                        result_minors.end(),
                                                        result_majors.end(),
                                                        property.end(),
                                                        types.end()));
      } else {
        cugraph::sort_wrapper(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(
            result_minors.begin(), result_majors.begin(), property.begin(), types.begin()),
          thrust::make_zip_iterator(
            result_minors.end(), result_majors.end(), property.end(), types.end()));
      }
    } else if (has_labels) {
      cugraph::sort_wrapper(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(
          result_labels->begin(), result_minors.begin(), result_majors.begin(), property.begin()),
        thrust::make_zip_iterator(
          result_labels->end(), result_minors.end(), result_majors.end(), property.end()));
    } else {
      cugraph::sort_wrapper(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(result_minors.begin(), result_majors.begin(), property.begin()),
        thrust::make_zip_iterator(result_minors.end(), result_majors.end(), property.end()));
    }
  } else {
    if (has_types) {
      auto& types = std::get<rmm::device_uvector<int32_t>>(result_types);
      if (has_labels) {
        cugraph::sort_wrapper(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(
            result_labels->begin(), result_minors.begin(), result_majors.begin(), types.begin()),
          thrust::make_zip_iterator(
            result_labels->end(), result_minors.end(), result_majors.end(), types.end()));
      } else {
        cugraph::sort_wrapper(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(result_minors.begin(), result_majors.begin(), types.begin()),
          thrust::make_zip_iterator(result_minors.end(), result_majors.end(), types.end()));
      }
    } else if (has_labels) {
      cugraph::sort_wrapper(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(
          result_labels->begin(), result_minors.begin(), result_majors.begin()),
        thrust::make_zip_iterator(result_labels->end(), result_minors.end(), result_majors.end()));
    } else {
      cugraph::sort_wrapper(handle.get_thrust_policy(),
                            thrust::make_zip_iterator(result_minors.begin(), result_majors.begin()),
                            thrust::make_zip_iterator(result_minors.end(), result_majors.end()));
    }
  }

  // 3. Mark the edges to keep locally
  size_t keep_count{0};
  rmm::device_uvector<uint32_t> keep_flags(0, handle.get_stream());
  if (has_labels) {
    std::tie(keep_count, keep_flags) = detail::mark_entries(
      result_minors.size(),
      detail::is_first_in_run_t<decltype(thrust::make_zip_iterator(result_labels->begin(),
                                                                   result_minors.begin()))>{
        thrust::make_zip_iterator(result_labels->begin(), result_minors.begin())},
      handle.get_stream());
  } else {
    std::tie(keep_count, keep_flags) = detail::mark_entries(
      result_minors.size(),
      detail::is_first_in_run_t<decltype(result_minors.begin())>{result_minors.begin()},
      handle.get_stream());
  }

  // split to new result_majors and discarded_majors, then minors, edge_property, types and labels
  rmm::device_uvector<vertex_t> discarded_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> discarded_minors(0, handle.get_stream());
  arithmetic_device_uvector_t discarded_edge_property{std::monostate{}};
  arithmetic_device_uvector_t discarded_types{std::monostate{}};
  if (has_edge_property) {
    discarded_edge_property = rmm::device_uvector<edge_t>(0, handle.get_stream());
  }
  if (has_types) { discarded_types = rmm::device_uvector<int32_t>(0, handle.get_stream()); }
  std::optional<rmm::device_uvector<int32_t>> discarded_labels{std::nullopt};

  if (keep_count < result_minors.size()) {
    size_t const discard_count = result_minors.size() - keep_count;
    raft::device_span<uint32_t const> const keep_flags_span{keep_flags.data(), keep_flags.size()};

    discarded_majors.resize(discard_count, handle.get_stream());
    detail::copy_if_mask_unset(handle,
                               result_majors.begin(),
                               result_majors.end(),
                               keep_flags.begin(),
                               discarded_majors.begin());
    result_majors =
      detail::keep_marked_entries(handle, std::move(result_majors), keep_flags_span, keep_count);

    discarded_minors.resize(discard_count, handle.get_stream());
    detail::copy_if_mask_unset(handle,
                               result_minors.begin(),
                               result_minors.end(),
                               keep_flags.begin(),
                               discarded_minors.begin());
    result_minors =
      detail::keep_marked_entries(handle, std::move(result_minors), keep_flags_span, keep_count);

    if (has_edge_property) {
      auto& property = std::get<rmm::device_uvector<edge_t>>(result_edge_property);
      rmm::device_uvector<edge_t> discarded(discard_count, handle.get_stream());
      detail::copy_if_mask_unset(
        handle, property.begin(), property.end(), keep_flags.begin(), discarded.begin());
      property =
        detail::keep_marked_entries(handle, std::move(property), keep_flags_span, keep_count);
      discarded_edge_property = std::move(discarded);
    }

    if (has_types) {
      auto& types     = std::get<rmm::device_uvector<int32_t>>(result_types);
      auto& discarded = std::get<rmm::device_uvector<int32_t>>(discarded_types);
      discarded.resize(discard_count, handle.get_stream());
      detail::copy_if_mask_unset(
        handle, types.begin(), types.end(), keep_flags.begin(), discarded.begin());
    }

    if (has_labels) {
      discarded_labels =
        std::make_optional(rmm::device_uvector<int32_t>(discard_count, handle.get_stream()));
      detail::copy_if_mask_unset(handle,
                                 result_labels->begin(),
                                 result_labels->end(),
                                 keep_flags.begin(),
                                 discarded_labels->begin());
      *result_labels =
        detail::keep_marked_entries(handle, std::move(*result_labels), keep_flags_span, keep_count);
    }
  }

  if (has_types) {
    auto& types = std::get<rmm::device_uvector<int32_t>>(result_types);
    types.resize(0, handle.get_stream());
    types.shrink_to_fit(handle.get_stream());
  }

  // 4. Shuffle edges back to the source-owner GPU (inverse of step 1's shuffle by minor).
  if constexpr (multi_gpu) {
    std::vector<arithmetic_device_uvector_t> shuffle_properties{};
    shuffle_properties.push_back(std::move(result_minors));
    if (has_edge_property) { shuffle_properties.push_back(std::move(result_edge_property)); }
    if (has_labels) { shuffle_properties.push_back(std::move(*result_labels)); }

    std::tie(result_majors, shuffle_properties) =
      shuffle_int_vertices(handle,
                           std::move(result_majors),
                           std::move(shuffle_properties),
                           graph_view.vertex_partition_range_lasts(),
                           std::nullopt);

    result_minors = std::move(std::get<rmm::device_uvector<vertex_t>>(shuffle_properties[0]));
    size_t shuffle_prop_idx{1};
    if (has_edge_property) {
      result_edge_property = std::move(shuffle_properties[shuffle_prop_idx++]);
    }
    if (has_labels) {
      result_labels =
        std::move(std::get<rmm::device_uvector<int32_t>>(shuffle_properties[shuffle_prop_idx++]));
    }

    shuffle_properties.clear();
    shuffle_properties.push_back(std::move(discarded_minors));
    if (has_edge_property) { shuffle_properties.push_back(std::move(discarded_edge_property)); }
    if (has_types) { shuffle_properties.push_back(std::move(discarded_types)); }
    if (has_labels) { shuffle_properties.push_back(std::move(*discarded_labels)); }

    std::tie(discarded_majors, shuffle_properties) =
      shuffle_int_vertices(handle,
                           std::move(discarded_majors),
                           std::move(shuffle_properties),
                           graph_view.vertex_partition_range_lasts(),
                           std::nullopt);

    discarded_minors = std::move(std::get<rmm::device_uvector<vertex_t>>(shuffle_properties[0]));
    shuffle_prop_idx = 1;
    if (has_edge_property) {
      discarded_edge_property = std::move(shuffle_properties[shuffle_prop_idx++]);
    }
    if (has_types) { discarded_types = std::move(shuffle_properties[shuffle_prop_idx++]); }
    if (has_labels) {
      discarded_labels =
        std::move(std::get<rmm::device_uvector<int32_t>>(shuffle_properties[shuffle_prop_idx++]));
    }
  }

  return std::make_tuple(std::move(result_majors),
                         std::move(result_minors),
                         std::move(result_edge_property),
                         std::move(result_labels),
                         std::move(discarded_majors),
                         std::move(discarded_minors),
                         std::move(discarded_edge_property),
                         std::move(discarded_types),
                         std::move(discarded_labels));
}

}  // namespace detail
}  // namespace cugraph
