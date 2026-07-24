/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {
namespace detail {

// Concatenate a set of same-typed device spans (carried as an arithmetic variant) into a single
// contiguous device vector, in the order given.  The element type is inferred from the first span;
// every span in @p spans must hold that same alternative.  @p result_size is the total element
// count across all spans.  Returns a monostate variant when @p spans is empty.
//
// Declared here and defined in sampling_result_utils.cpp (compiled once into cugraph_common) so the
// non-template body is not emitted inline in every translation unit that includes this header.
CUGRAPH_EXPORT arithmetic_device_uvector_t
concatenate_spans(raft::handle_t const& handle,
                  std::vector<arithmetic_device_span_t> const& spans,
                  size_t result_size);

// Assemble the final sampling output from the per-hop edge lists produced during sampling.  Each
// entry of @p produced_edge_lists owns (source vertices, destination vertices, edge-property
// columns in schema order, optional per-edge labels, hop index).  This concatenates, in edge-list
// order, the sources, destinations, every property column, the labels (when present), and — when @p
// return_hops is set — a per-edge hop column.  The property columns are returned in the same schema
// order they appear in each edge list, so the caller can map them back to its named columns by
// index.
template <typename vertex_t, typename label_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<rmm::device_uvector<int32_t>>>
concatenate_produced_edge_list_properties(
  raft::handle_t const& handle,
  std::vector<std::tuple<rmm::device_uvector<vertex_t>,
                         rmm::device_uvector<vertex_t>,
                         std::vector<cugraph::arithmetic_device_uvector_t>,
                         std::optional<rmm::device_uvector<label_t>>,
                         int32_t>>&& produced_edge_lists,
  bool return_hops)
{
  size_t const num_props =
    produced_edge_lists.empty() ? 0 : std::get<2>(produced_edge_lists.front()).size();
  bool const has_labels =
    !produced_edge_lists.empty() && std::get<3>(produced_edge_lists.front()).has_value();

  size_t result_size{0};
  std::vector<size_t> edge_list_sizes{};
  std::vector<arithmetic_device_span_t> src_spans{};
  std::vector<arithmetic_device_span_t> dst_spans{};
  std::vector<std::vector<arithmetic_device_span_t>> property_spans(num_props);
  std::vector<arithmetic_device_span_t> label_spans{};
  std::vector<int32_t> hop_values{};

  for (auto& [srcs, dsts, properties, labels, hop] : produced_edge_lists) {
    result_size += srcs.size();
    edge_list_sizes.push_back(srcs.size());
    src_spans.push_back(raft::device_span<vertex_t>{srcs.data(), srcs.size()});
    dst_spans.push_back(raft::device_span<vertex_t>{dsts.data(), dsts.size()});
    for (size_t j = 0; j < num_props; ++j) {
      property_spans[j].push_back(make_arithmetic_device_span(properties[j]));
    }
    if (has_labels) {
      label_spans.push_back(raft::device_span<label_t>{labels->data(), labels->size()});
    }
    hop_values.push_back(hop);
  }

  auto result_srcs =
    std::get<rmm::device_uvector<vertex_t>>(concatenate_spans(handle, src_spans, result_size));
  auto result_dsts =
    std::get<rmm::device_uvector<vertex_t>>(concatenate_spans(handle, dst_spans, result_size));

  std::vector<arithmetic_device_uvector_t> result_properties{};
  result_properties.reserve(num_props);
  for (auto& column : property_spans) {
    result_properties.push_back(concatenate_spans(handle, column, result_size));
  }

  std::optional<rmm::device_uvector<label_t>> result_labels{std::nullopt};
  if (has_labels) {
    result_labels =
      std::get<rmm::device_uvector<label_t>>(concatenate_spans(handle, label_spans, result_size));
  }

  std::optional<rmm::device_uvector<int32_t>> result_hops{std::nullopt};
  if (return_hops) {
    result_hops          = rmm::device_uvector<int32_t>(result_size, handle.get_stream());
    size_t output_offset = 0;
    for (size_t i = 0; i < hop_values.size(); ++i) {
      cugraph::fill(handle.get_thrust_policy(),
                    result_hops->data() + output_offset,
                    result_hops->data() + output_offset + edge_list_sizes[i],
                    hop_values[i]);
      output_offset += edge_list_sizes[i];
    }
  }

  produced_edge_lists.clear();

  return std::make_tuple(std::move(result_srcs),
                         std::move(result_dsts),
                         std::move(result_properties),
                         std::move(result_labels),
                         std::move(result_hops));
}

}  // namespace detail
}  // namespace cugraph
