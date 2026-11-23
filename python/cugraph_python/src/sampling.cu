/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Uniform, biased, homogeneous, and heterogeneous neighbor sampling, with optional temporal
 * filters. The parameter list follows cugraph::neighbor_sample. Properties named as roles occupy
 * the sampler's columns; the edge-property vector is returned in caller order.
 * Sampled endpoints are unrenumbered to external IDs.
 * The compiled neighbor_sample symbols are single-GPU and store_transposed=false.
 */
#include "cugraph_python.hpp"

#include <cugraph/sampling_functions.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/host_span.hpp>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph_python {
namespace {

using neighbor_sample_result_t = std::tuple<device_array_t,
                                            device_array_t,
                                            std::optional<device_array_t>,
                                            std::vector<device_array_t>,
                                            std::optional<device_array_t>>;

enum class property_kind_t { int32, int64, float32, float64 };
enum class property_role_t { none, weight, edge_id, edge_type, start_time, end_time, bias };
enum class sampled_column_t { weight, edge_id, edge_type, start_time, end_time };

struct narrow_offset_t {
  __host__ __device__ std::int64_t operator()(std::size_t value) const
  {
    return static_cast<std::int64_t>(value);
  }
};

template <typename value_t>
std::vector<value_t> narrow_ids(std::vector<std::int64_t> const& values, char const* what)
{
  std::vector<value_t> narrowed;
  narrowed.reserve(values.size());
  for (auto value : values) {
    if (value < static_cast<std::int64_t>(std::numeric_limits<value_t>::min()) ||
        value > static_cast<std::int64_t>(std::numeric_limits<value_t>::max())) {
      throw std::invalid_argument(std::string(what) + " does not fit in the graph's type");
    }
    narrowed.push_back(static_cast<value_t>(value));
  }
  return narrowed;
}

template <typename value_t>
device_array_t device_array_from(rmm::device_uvector<value_t>&& values)
{
  return device_array_t{std::make_shared<rmm::device_uvector<value_t>>(std::move(values))};
}

cugraph::temporal_sampling_comparison_t comparison_of(temporal_comparison_id_t temporal)
{
  switch (temporal) {
    case temporal_comparison_id_t::strictly_increasing:
      return cugraph::temporal_sampling_comparison_t::STRICTLY_INCREASING;
    case temporal_comparison_id_t::monotonically_increasing:
      return cugraph::temporal_sampling_comparison_t::MONOTONICALLY_INCREASING;
    case temporal_comparison_id_t::strictly_decreasing:
      return cugraph::temporal_sampling_comparison_t::STRICTLY_DECREASING;
    case temporal_comparison_id_t::monotonically_decreasing:
      return cugraph::temporal_sampling_comparison_t::MONOTONICALLY_DECREASING;
    case temporal_comparison_id_t::none:
      break;
  }
  throw std::invalid_argument("neighbor_sample: temporal comparison is unset");
}

cugraph::prior_sources_behavior_t prior_sources_of(prior_sources_behavior_id_t behavior)
{
  switch (behavior) {
    case prior_sources_behavior_id_t::defaults:
      return cugraph::prior_sources_behavior_t::DEFAULT;
    case prior_sources_behavior_id_t::carry_over:
      return cugraph::prior_sources_behavior_t::CARRY_OVER;
    case prior_sources_behavior_id_t::exclude:
      return cugraph::prior_sources_behavior_t::EXCLUDE;
  }
  throw std::invalid_argument("neighbor_sample: unknown prior-sources behavior");
}

cugraph::neighbor_selection_t selection_of(neighbor_selection_id_t selection)
{
  switch (selection) {
    case neighbor_selection_id_t::random: return cugraph::neighbor_selection_t::RANDOM;
    case neighbor_selection_id_t::last: return cugraph::neighbor_selection_t::LAST;
  }
  throw std::invalid_argument("neighbor_sample: unknown neighbor selection");
}

template <typename edge_t>
edge_property_handle_t<edge_t> const& property_handle(edge_property_t const& property,
                                                      char const* what)
{
  auto const* handle = std::get_if<edge_property_handle_t<edge_t>>(&property.impl);
  if (handle == nullptr) {
    throw std::invalid_argument(std::string("neighbor_sample: ") + what +
                                " type does not match the graph");
  }
  return *handle;
}

template <typename edge_t>
void check_property_graph(edge_property_handle_t<edge_t> const& handle,
                          std::uint64_t graph_id,
                          char const* what)
{
  if (handle.state->graph_id != graph_id) {
    throw std::invalid_argument(std::string("neighbor_sample: ") + what +
                                " was created for a different graph");
  }
}

template <typename edge_t>
property_kind_t kind_of(cugraph::edge_arithmetic_property_t<edge_t> const& property)
{
  return std::visit(
    [](auto const& value) -> property_kind_t {
      using property_t = std::decay_t<decltype(value)>;
      if constexpr (std::is_same_v<property_t, std::monostate>) {
        throw std::invalid_argument("neighbor_sample: an edge property is empty");
      } else {
        using value_t = typename property_t::value_type;
        if constexpr (std::is_same_v<value_t, std::int32_t>) {
          return property_kind_t::int32;
        } else if constexpr (std::is_same_v<value_t, std::int64_t>) {
          return property_kind_t::int64;
        } else if constexpr (std::is_same_v<value_t, float>) {
          return property_kind_t::float32;
        } else if constexpr (std::is_same_v<value_t, double>) {
          return property_kind_t::float64;
        } else {
          throw std::invalid_argument(
            "neighbor_sample: edge properties must be int32, int64, float32, or float64");
        }
      }
    },
    property);
}

template <typename value_t, typename edge_t>
std::optional<cugraph::edge_property_view_t<edge_t, value_t const*>> view_as(
  cugraph::edge_arithmetic_property_t<edge_t> const& property)
{
  return std::visit(
    [](auto const& value)
      -> std::optional<cugraph::edge_property_view_t<edge_t, value_t const*>> {
      using property_t = std::decay_t<decltype(value)>;
      if constexpr (std::is_same_v<property_t, std::monostate>) {
        throw std::invalid_argument("neighbor_sample: an edge property is empty");
      } else if constexpr (std::is_same_v<typename property_t::value_type, value_t>) {
        return std::make_optional(value.view());
      } else {
        return std::nullopt;
      }
    },
    property);
}

template <typename edge_t>
struct classified_property_t {
  cugraph::edge_arithmetic_property_t<edge_t> const* property{};
  std::size_t index{};
  property_kind_t kind{};
  property_role_t role{property_role_t::none};
};

template <typename edge_t>
struct role_binding_t {
  cugraph::edge_arithmetic_property_t<edge_t> const* property{};
  std::optional<std::size_t> output_index{};
};

template <typename edge_t>
struct bound_roles_t {
  role_binding_t<edge_t> weight;
  role_binding_t<edge_t> edge_id;
  role_binding_t<edge_t> edge_type;
  role_binding_t<edge_t> start_time;
  role_binding_t<edge_t> end_time;
  role_binding_t<edge_t> bias;
};

template <typename edge_t>
role_binding_t<edge_t> bind_role(
  std::optional<edge_property_t> const& role,
  std::shared_ptr<edge_properties_state_t<edge_t>> const* vector_state,
  std::uint64_t graph_id,
  char const* what)
{
  if (!role) { return {}; }
  auto const& handle = property_handle<edge_t>(*role, what);
  check_property_graph(handle, graph_id, what);
  role_binding_t<edge_t> bound;
  bound.property = &handle.state->values[handle.index];
  if (vector_state != nullptr && handle.state == *vector_state) {
    bound.output_index = handle.index;
  }
  return bound;
}

template <typename weight_t, typename time_stamp_t, typename edge_t>
struct sample_views_t {
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> weight;
  std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>> edge_id;
  std::optional<cugraph::edge_property_view_t<edge_t, std::int32_t const*>> edge_type;
  std::optional<cugraph::edge_property_view_t<edge_t, time_stamp_t const*>> start_time;
  std::optional<cugraph::edge_property_view_t<edge_t, time_stamp_t const*>> end_time;
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> bias;
  std::vector<std::pair<std::size_t, sampled_column_t>> columns;
};

template <typename value_t, typename edge_t>
cugraph::edge_property_view_t<edge_t, value_t const*> require_view(
  cugraph::edge_arithmetic_property_t<edge_t> const& property,
  char const* what)
{
  auto view = view_as<value_t>(property);
  if (!view) { throw std::invalid_argument(what); }
  return std::move(*view);
}

template <typename weight_t, typename time_stamp_t, typename edge_t>
sample_views_t<weight_t, time_stamp_t, edge_t> assign_views(
  std::vector<classified_property_t<edge_t>> const& properties,
  bound_roles_t<edge_t> const& roles)
{
  sample_views_t<weight_t, time_stamp_t, edge_t> views;
  auto add_column = [&](role_binding_t<edge_t> const& role, sampled_column_t column) {
    if (role.output_index) { views.columns.emplace_back(*role.output_index, column); }
  };

  if (roles.weight.property != nullptr) {
    views.weight = require_view<weight_t>(
      *roles.weight.property, "neighbor_sample: edge weights must be float32 or float64");
    add_column(roles.weight, sampled_column_t::weight);
  }
  if (roles.edge_id.property != nullptr) {
    views.edge_id = require_view<edge_t>(
      *roles.edge_id.property, "neighbor_sample: edge ids must use the graph integer type");
    add_column(roles.edge_id, sampled_column_t::edge_id);
  }
  if (roles.edge_type.property != nullptr) {
    views.edge_type = require_view<std::int32_t>(
      *roles.edge_type.property, "neighbor_sample: edge types must be int32");
    add_column(roles.edge_type, sampled_column_t::edge_type);
  }
  if (roles.start_time.property != nullptr) {
    views.start_time = require_view<time_stamp_t>(
      *roles.start_time.property, "neighbor_sample: edge start times must be int32 or int64");
    add_column(roles.start_time, sampled_column_t::start_time);
  }
  if (roles.end_time.property != nullptr) {
    views.end_time = require_view<time_stamp_t>(
      *roles.end_time.property, "neighbor_sample: edge end times must match the start-time dtype");
    add_column(roles.end_time, sampled_column_t::end_time);
  }
  if (roles.bias.property != nullptr) {
    views.bias = require_view<weight_t>(
      *roles.bias.property, "neighbor_sample: edge biases must match the edge-weight dtype");
  }

  bool saw_weight = roles.weight.property != nullptr;
  bool saw_id     = roles.edge_id.property != nullptr;
  bool saw_type   = roles.edge_type.property != nullptr;
  std::vector<bool> placed(properties.size(), false);
  for (auto const& [index, column] : views.columns) {
    (void)column;
    placed[index] = true;
  }

  for (auto const& item : properties) {
    if (item.role != property_role_t::none || placed[item.index]) { continue; }
    bool const floating =
      item.kind == property_kind_t::float32 || item.kind == property_kind_t::float64;
    if (floating) {
      bool const matches =
        (std::is_same_v<weight_t, float> && item.kind == property_kind_t::float32) ||
        (std::is_same_v<weight_t, double> && item.kind == property_kind_t::float64);
      if (!matches) {
        throw std::invalid_argument(
          "neighbor_sample: float32 and float64 edge properties cannot be sampled together");
      }
      if (saw_weight) {
        throw std::invalid_argument(
          "neighbor_sample: only one floating-point edge property can be sampled");
      }
      views.weight = require_view<weight_t>(
        *item.property, "neighbor_sample: edge weights must be float32 or float64");
      views.columns.emplace_back(item.index, sampled_column_t::weight);
      saw_weight = true;
      continue;
    }

    bool const matches_edge_id =
      (std::is_same_v<edge_t, std::int32_t> && item.kind == property_kind_t::int32) ||
      (std::is_same_v<edge_t, std::int64_t> && item.kind == property_kind_t::int64);
    if (matches_edge_id && !saw_id) {
      views.edge_id = require_view<edge_t>(
        *item.property, "neighbor_sample: edge ids must use the graph integer type");
      views.columns.emplace_back(item.index, sampled_column_t::edge_id);
      saw_id = true;
      continue;
    }
    if (item.kind == property_kind_t::int32 && !saw_type) {
      views.edge_type = require_view<std::int32_t>(*item.property,
                                                   "neighbor_sample: edge types must be int32");
      views.columns.emplace_back(item.index, sampled_column_t::edge_type);
      saw_type = true;
      continue;
    }
    throw std::invalid_argument(
      "neighbor_sample: the sampler can return one floating-point property, one property of the "
      "graph integer type, one int32 property, and the start-time and end-time properties");
  }
  return views;
}

struct sampling_request_t {
  std::optional<std::vector<std::int64_t>> const* starting_vertex_start_times{};
  std::optional<std::vector<std::int64_t>> const* starting_vertex_end_times{};
  std::optional<std::vector<std::int32_t>> const* starting_vertex_labels{};
  std::optional<std::int32_t> num_edge_types{};
  prior_sources_behavior_id_t prior_sources_behavior{prior_sources_behavior_id_t::defaults};
  bool return_hops{true};
  bool dedupe_sources{false};
  bool with_replacement{false};
  temporal_comparison_id_t temporal{temporal_comparison_id_t::none};
  bool disjoint_sampling{false};
  neighbor_selection_id_t neighbor_selection{neighbor_selection_id_t::random};
  bool fixed_window{false};
  bool do_expensive_check{false};
  raft::random::RngState* rng_state{};
};

template <typename weight_t,
          typename time_stamp_t,
          typename vertex_t,
          typename edge_t,
          bool store_transposed,
          bool multi_gpu>
neighbor_sample_result_t neighbor_sample_call(
  graph_instance_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph,
  rmm::device_uvector<vertex_t>&& internal_starts,
  std::vector<std::int32_t> const& fan_out,
  std::vector<classified_property_t<edge_t>> const& properties,
  bound_roles_t<edge_t> const& roles,
  sampling_request_t const& request)
{
  static_assert(!store_transposed && !multi_gpu,
                "neighbor_sample is compiled for this orientation only");

  auto views = assign_views<weight_t, time_stamp_t>(properties, roles);

  auto& handle           = graph.context->handle;
  auto stream            = handle.get_stream();
  auto view              = graph.native.view();
  auto const& number_map = *graph.number_map;

  cugraph::sampling_options_t options;
  options.prior_sources_behavior = prior_sources_of(request.prior_sources_behavior);
  options.return_hops            = request.return_hops;
  options.dedupe_sources         = request.dedupe_sources;
  options.with_replacement       = request.with_replacement;
  options.disjoint_sampling      = request.disjoint_sampling;
  options.neighbor_selection     = selection_of(request.neighbor_selection);
  options.fixed_window           = request.fixed_window;
  if (request.temporal != temporal_comparison_id_t::none) {
    options.temporal_sampling_comparison = comparison_of(request.temporal);
  }

  auto vertex_times = [&](std::optional<std::vector<std::int64_t>> const* times, char const* what) {
    std::optional<rmm::device_uvector<time_stamp_t>> buffer;
    std::optional<raft::device_span<time_stamp_t const>> span;
    if (times != nullptr && *times) {
      if ((*times)->size() != internal_starts.size()) {
        throw std::invalid_argument(std::string("neighbor_sample: ") + what +
                                    " and starting vertices differ in length");
      }
      auto host_times = narrow_ids<time_stamp_t>(**times, (std::string("neighbor_sample: ") + what).c_str());
      buffer.emplace(host_times.size(), stream);
      raft::copy(buffer->data(), host_times.data(), host_times.size(), stream);
      span = raft::device_span<time_stamp_t const>(buffer->data(), buffer->size());
    }
    return std::make_pair(std::move(buffer), span);
  };
  // Keep the buffers alive for the spans passed into neighbor_sample.
  auto [start_time_buffer, start_time_span] =
    vertex_times(request.starting_vertex_start_times, "starting_vertex_start_times");
  auto [end_time_buffer, end_time_span] =
    vertex_times(request.starting_vertex_end_times, "starting_vertex_end_times");
  (void)start_time_buffer;
  (void)end_time_buffer;

  std::optional<rmm::device_uvector<std::int32_t>> label_buffer;
  std::optional<raft::device_span<std::int32_t const>> label_span;
  if (request.starting_vertex_labels != nullptr && *request.starting_vertex_labels) {
    auto const& labels = **request.starting_vertex_labels;
    if (labels.size() != internal_starts.size()) {
      throw std::invalid_argument(
        "neighbor_sample: starting_vertex_labels and starting vertices differ in length");
    }
    label_buffer.emplace(labels.size(), stream);
    raft::copy(label_buffer->data(), labels.data(), labels.size(), stream);
    label_span = raft::device_span<std::int32_t const>(label_buffer->data(), label_buffer->size());
  }

  auto [sources, destinations, sampled_weights, sampled_ids, sampled_types, sampled_start_times,
        sampled_end_times, hops, offsets] =
    cugraph::neighbor_sample<vertex_t,
                             edge_t,
                             weight_t,
                             std::int32_t,
                             time_stamp_t,
                             store_transposed,
                             multi_gpu>(
      handle,
      *request.rng_state,
      view,
      std::move(views.weight),
      std::move(views.edge_id),
      std::move(views.edge_type),
      std::move(views.start_time),
      std::move(views.end_time),
      std::move(views.bias),
      raft::device_span<vertex_t const>(internal_starts.data(), internal_starts.size()),
      start_time_span,
      end_time_span,
      label_span,
      std::optional<raft::device_span<std::int32_t const>>{},
      raft::host_span<std::int32_t const>(fan_out.data(), fan_out.size()),
      request.num_edge_types,
      options,
      request.do_expensive_check);

  auto labels = cugraph::vertex_property_view_t<vertex_t, vertex_t const*>(number_map.data(),
                                                                           number_map.size());
  auto external_sources =
    cugraph::unrenumber_int_vertices(handle, view, std::move(sources), labels, false);
  auto external_destinations =
    cugraph::unrenumber_int_vertices(handle, view, std::move(destinations), labels, false);

  std::optional<device_array_t> hop_array;
  if (hops) {
    hop_array = device_array_from(std::move(*hops));
  } else if (request.return_hops) {
    throw std::logic_error("neighbor_sample: hops were requested but not returned");
  }

  std::optional<device_array_t> offset_array;
  if (offsets) {
    rmm::device_uvector<std::int64_t> converted(offsets->size(), stream);
    thrust::transform(handle.get_thrust_policy(),
                      offsets->begin(),
                      offsets->end(),
                      converted.begin(),
                      narrow_offset_t{});
    offset_array = device_array_from(std::move(converted));
  }

  // Columns were recorded as explicit roles first, then leftover properties. Reorder to caller
  // index. device_array_t is not default-constructible, so build the vector by index order.
  std::vector<std::pair<std::size_t, device_array_t>> tagged;
  tagged.reserve(views.columns.size());
  for (auto const& [index, column] : views.columns) {
    switch (column) {
      case sampled_column_t::weight:
        if (!sampled_weights) {
          throw std::logic_error("neighbor_sample: a weight column was requested but not returned");
        }
        tagged.emplace_back(index, device_array_from(std::move(*sampled_weights)));
        break;
      case sampled_column_t::edge_id:
        if (!sampled_ids) {
          throw std::logic_error(
            "neighbor_sample: an edge-id column was requested but not returned");
        }
        tagged.emplace_back(index, device_array_from(std::move(*sampled_ids)));
        break;
      case sampled_column_t::edge_type:
        if (!sampled_types) {
          throw std::logic_error(
            "neighbor_sample: an edge-type column was requested but not returned");
        }
        tagged.emplace_back(index, device_array_from(std::move(*sampled_types)));
        break;
      case sampled_column_t::start_time:
        if (!sampled_start_times) {
          throw std::logic_error(
            "neighbor_sample: a start-time column was requested but not returned");
        }
        tagged.emplace_back(index, device_array_from(std::move(*sampled_start_times)));
        break;
      case sampled_column_t::end_time:
        if (!sampled_end_times) {
          throw std::logic_error(
            "neighbor_sample: an end-time column was requested but not returned");
        }
        tagged.emplace_back(index, device_array_from(std::move(*sampled_end_times)));
        break;
    }
  }
  std::sort(tagged.begin(), tagged.end(), [](auto const& left, auto const& right) {
    return left.first < right.first;
  });
  std::vector<device_array_t> sampled_properties;
  sampled_properties.reserve(tagged.size());
  for (auto& [index, column] : tagged) {
    if (index != sampled_properties.size()) {
      throw std::logic_error("neighbor_sample: sampled property columns are out of order");
    }
    sampled_properties.push_back(std::move(column));
  }

  handle.sync_stream();
  return {device_array_from(std::move(external_sources)),
          device_array_from(std::move(external_destinations)),
          std::move(hop_array),
          std::move(sampled_properties),
          std::move(offset_array)};
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
neighbor_sample_result_t neighbor_sample_impl(
  graph_instance_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph,
  rng_state_t& rng_state,
  std::optional<edge_property_t> const& edge_weights,
  std::optional<edge_property_t> const& edge_ids,
  std::optional<edge_property_t> const& edge_types,
  std::optional<edge_property_t> const& edge_start_times,
  std::optional<edge_property_t> const& edge_end_times,
  std::optional<edge_property_t> const& edge_biases,
  std::vector<std::int64_t> const& starting_vertices,
  std::optional<std::vector<std::int64_t>> const& starting_vertex_start_times,
  std::optional<std::vector<std::int64_t>> const& starting_vertex_end_times,
  std::optional<std::vector<std::int32_t>> const& starting_vertex_labels,
  std::optional<std::vector<std::int32_t>> const& label_to_output_comm_rank,
  std::vector<std::int32_t> const& fan_out,
  std::optional<std::int32_t> num_edge_types,
  std::optional<edge_properties_t> const& edge_properties,
  sampling_options_t const& sampling_options,
  bool do_expensive_check)
{
  if constexpr (store_transposed || multi_gpu) {
    throw std::invalid_argument(
      "neighbor_sample: requires a graph created with store_transposed=false");
  } else {
    if (starting_vertices.empty()) {
      throw std::invalid_argument("neighbor_sample: starting vertices must not be empty");
    }
    if (fan_out.empty()) {
      throw std::invalid_argument("neighbor_sample: fan_out must not be empty");
    }
    if (std::any_of(fan_out.begin(), fan_out.end(), [](std::int32_t fan) { return fan < 0; })) {
      throw std::invalid_argument("neighbor_sample: fan_out values must be non-negative");
    }
    if (!graph.number_map) {
      throw std::logic_error("neighbor_sample: graph has no number map (renumbering disabled)");
    }
    if (label_to_output_comm_rank) {
      throw std::invalid_argument(
        "neighbor_sample: label_to_output_comm_rank is not available for single-GPU sampling");
    }
    bool const temporal_sampling =
      sampling_options.temporal_sampling_comparison != temporal_comparison_id_t::none;
    if (!temporal_sampling &&
        (edge_start_times || edge_end_times || starting_vertex_start_times ||
         starting_vertex_end_times)) {
      throw std::invalid_argument(
        "neighbor_sample: edge and starting-vertex times require a temporal comparison");
    }
    if (temporal_sampling && !edge_start_times) {
      throw std::invalid_argument("neighbor_sample: temporal sampling requires edge_start_times");
    }
    if (num_edge_types && *num_edge_types < 1) {
      throw std::invalid_argument("neighbor_sample: num_edge_types must be positive");
    }
    if (num_edge_types && !edge_types) {
      throw std::invalid_argument("neighbor_sample: heterogeneous sampling requires edge_types");
    }
    auto const length_matches = [&](auto const& values, char const* what) {
      if (values && values->size() != starting_vertices.size()) {
        throw std::invalid_argument(std::string("neighbor_sample: ") + what +
                                    " and starting vertices differ in length");
      }
    };
    length_matches(starting_vertex_start_times, "starting_vertex_start_times");
    length_matches(starting_vertex_end_times, "starting_vertex_end_times");
    length_matches(starting_vertex_labels, "starting_vertex_labels");

    bool const disjoint = sampling_options.disjoint_sampling;
    if (temporal_sampling && !disjoint) {
      throw std::invalid_argument("neighbor_sample: temporal sampling requires disjoint_sampling");
    }
    if (temporal_sampling && sampling_options.with_replacement) {
      throw std::invalid_argument(
        "neighbor_sample: temporal sampling does not support with_replacement");
    }
    if (disjoint && sampling_options.with_replacement) {
      throw std::invalid_argument(
        "neighbor_sample: disjoint sampling and sampling with replacement are mutually exclusive");
    }

    std::shared_ptr<edge_properties_state_t<edge_t>> const* vector_state = nullptr;
    std::shared_ptr<edge_properties_state_t<edge_t>> state_holder;
    std::vector<classified_property_t<edge_t>> classified;
    if (edge_properties) {
      auto const* state =
        std::get_if<std::shared_ptr<edge_properties_state_t<edge_t>>>(&edge_properties->impl);
      if (state == nullptr) {
        throw std::invalid_argument("neighbor_sample: edge property type does not match the graph");
      }
      if ((*state)->graph_id != graph.graph_id) {
        throw std::invalid_argument(
          "neighbor_sample: edge properties were created for a different graph");
      }
      state_holder = *state;
      vector_state = &state_holder;
      classified.reserve((*state)->values.size());
      for (std::size_t index = 0; index < (*state)->values.size(); ++index) {
        classified.push_back(classified_property_t<edge_t>{
          &(*state)->values[index], index, kind_of((*state)->values[index]), property_role_t::none});
      }
    }

    bound_roles_t<edge_t> roles;
    roles.weight = bind_role<edge_t>(edge_weights, vector_state, graph.graph_id, "edge weights");
    roles.edge_id = bind_role<edge_t>(edge_ids, vector_state, graph.graph_id, "edge ids");
    roles.edge_type = bind_role<edge_t>(edge_types, vector_state, graph.graph_id, "edge types");
    roles.start_time =
      bind_role<edge_t>(edge_start_times, vector_state, graph.graph_id, "edge start times");
    roles.end_time =
      bind_role<edge_t>(edge_end_times, vector_state, graph.graph_id, "edge end times");
    roles.bias = bind_role<edge_t>(edge_biases, vector_state, graph.graph_id, "edge biases");

    std::vector<cugraph::edge_arithmetic_property_t<edge_t> const*> claimed;
    auto claim = [&](cugraph::edge_arithmetic_property_t<edge_t> const* property) {
      if (property == nullptr) { return; }
      if (std::find(claimed.begin(), claimed.end(), property) != claimed.end()) {
        throw std::invalid_argument(
          "neighbor_sample: an edge property was passed as more than one sampling input");
      }
      claimed.push_back(property);
    };
    claim(roles.weight.property);
    claim(roles.edge_id.property);
    claim(roles.edge_type.property);
    claim(roles.start_time.property);
    claim(roles.end_time.property);
    if (roles.bias.property != nullptr &&
        std::find(claimed.begin(), claimed.end(), roles.bias.property) == claimed.end()) {
      claimed.push_back(roles.bias.property);
    }

    auto mark = [&](role_binding_t<edge_t> const& role, property_role_t kind) {
      if (role.output_index) { classified[*role.output_index].role = kind; }
    };
    mark(roles.weight, property_role_t::weight);
    mark(roles.edge_id, property_role_t::edge_id);
    mark(roles.edge_type, property_role_t::edge_type);
    mark(roles.start_time, property_role_t::start_time);
    mark(roles.end_time, property_role_t::end_time);

    auto floating_kind = [](property_kind_t kind) {
      return kind == property_kind_t::float32 || kind == property_kind_t::float64;
    };
    std::optional<property_kind_t> weight_kind;
    if (roles.weight.property != nullptr) {
      weight_kind = kind_of(*roles.weight.property);
    } else if (roles.bias.property != nullptr) {
      weight_kind = kind_of(*roles.bias.property);
    }
    for (auto const& item : classified) {
      if (!floating_kind(item.kind) || item.role != property_role_t::none) { continue; }
      if (weight_kind && *weight_kind != item.kind) {
        throw std::invalid_argument(
          "neighbor_sample: float32 and float64 edge properties cannot be sampled together");
      }
      if (!weight_kind) { weight_kind = item.kind; }
    }
    if (weight_kind && !floating_kind(*weight_kind)) {
      throw std::invalid_argument("neighbor_sample: edge weights must be float32 or float64");
    }
    if (roles.bias.property != nullptr && weight_kind &&
        kind_of(*roles.bias.property) != *weight_kind) {
      throw std::invalid_argument("neighbor_sample: edge biases must match the edge-weight dtype");
    }
    bool const weight_is_double = weight_kind && *weight_kind == property_kind_t::float64;

    bool time_is_int64 = false;
    if (roles.start_time.property != nullptr) {
      auto kind = kind_of(*roles.start_time.property);
      if (kind == property_kind_t::int64) {
        time_is_int64 = true;
      } else if (kind != property_kind_t::int32) {
        throw std::invalid_argument("neighbor_sample: edge start times must be int32 or int64");
      }
    }
    if (roles.end_time.property != nullptr) {
      auto kind = kind_of(*roles.end_time.property);
      bool const end_is_int64 = kind == property_kind_t::int64;
      if (kind != property_kind_t::int32 && kind != property_kind_t::int64) {
        throw std::invalid_argument("neighbor_sample: edge end times must be int32 or int64");
      }
      if (roles.start_time.property != nullptr && end_is_int64 != time_is_int64) {
        throw std::invalid_argument(
          "neighbor_sample: edge end times must match the start-time dtype");
      }
      time_is_int64 = end_is_int64;
    }
    if (roles.edge_id.property != nullptr) {
      auto kind = kind_of(*roles.edge_id.property);
      bool const matches = (std::is_same_v<edge_t, std::int32_t> && kind == property_kind_t::int32) ||
                           (std::is_same_v<edge_t, std::int64_t> && kind == property_kind_t::int64);
      if (!matches) {
        throw std::invalid_argument("neighbor_sample: edge ids must use the graph integer type");
      }
    }
    if (roles.edge_type.property != nullptr &&
        kind_of(*roles.edge_type.property) != property_kind_t::int32) {
      throw std::invalid_argument("neighbor_sample: edge types must be int32");
    }

    auto& handle           = graph.context->handle;
    auto stream            = handle.get_stream();
    auto view              = graph.native.view();
    auto const& number_map = *graph.number_map;

    auto host_starts = narrow_ids<vertex_t>(starting_vertices, "neighbor_sample: a starting vertex ID");
    rmm::device_uvector<vertex_t> external(host_starts.size(), stream);
    raft::copy(external.data(), host_starts.data(), host_starts.size(), stream);
    auto internal = cugraph::renumber_ext_vertices(
      handle,
      view,
      std::move(external),
      cugraph::vertex_property_view_t<vertex_t, vertex_t const*>(number_map.data(),
                                                                 number_map.size()),
      /*do_expensive_check=*/false);

    std::vector<vertex_t> host_internal(internal.size());
    raft::copy(host_internal.data(), internal.data(), internal.size(), stream);
    handle.sync_stream();
    if (std::any_of(host_internal.begin(), host_internal.end(), [](vertex_t vertex) {
          return vertex == cugraph::invalid_vertex_id<vertex_t>::value;
        })) {
      throw std::invalid_argument(
        "neighbor_sample: a starting vertex ID is not present in the graph");
    }

    sampling_request_t request;
    request.starting_vertex_start_times = &starting_vertex_start_times;
    request.starting_vertex_end_times   = &starting_vertex_end_times;
    request.starting_vertex_labels      = &starting_vertex_labels;
    request.num_edge_types         = num_edge_types;
    request.prior_sources_behavior = sampling_options.prior_sources_behavior;
    request.return_hops            = sampling_options.return_hops;
    request.dedupe_sources         = sampling_options.dedupe_sources;
    request.with_replacement       = sampling_options.with_replacement;
    request.temporal               = sampling_options.temporal_sampling_comparison;
    request.disjoint_sampling      = disjoint;
    request.neighbor_selection     = sampling_options.neighbor_selection;
    request.fixed_window           = sampling_options.fixed_window;
    request.do_expensive_check = do_expensive_check;
    request.rng_state          = &rng_state.state();

    if (weight_is_double && time_is_int64) {
      return neighbor_sample_call<double, std::int64_t>(
        graph, std::move(internal), fan_out, classified, roles, request);
    }
    if (weight_is_double) {
      return neighbor_sample_call<double, std::int32_t>(
        graph, std::move(internal), fan_out, classified, roles, request);
    }
    if (time_is_int64) {
      return neighbor_sample_call<float, std::int64_t>(
        graph, std::move(internal), fan_out, classified, roles, request);
    }
    return neighbor_sample_call<float, std::int32_t>(
      graph, std::move(internal), fan_out, classified, roles, request);
  }
}

}  // namespace

std::tuple<device_array_t,
           device_array_t,
           std::optional<device_array_t>,
           std::vector<device_array_t>,
           std::optional<device_array_t>>
neighbor_sample(graph_t const& graph,
                rng_state_t& rng_state,
                std::optional<edge_property_t> edge_weights,
                std::optional<edge_property_t> edge_ids,
                std::optional<edge_property_t> edge_types,
                std::optional<edge_property_t> edge_start_times,
                std::optional<edge_property_t> edge_end_times,
                std::optional<edge_property_t> edge_biases,
                std::vector<std::int64_t> const& starting_vertices,
                std::optional<std::vector<std::int64_t>> const& starting_vertex_start_times,
                std::optional<std::vector<std::int64_t>> const& starting_vertex_end_times,
                std::optional<std::vector<std::int32_t>> const& starting_vertex_labels,
                std::optional<std::vector<std::int32_t>> const& label_to_output_comm_rank,
                std::vector<std::int32_t> const& fan_out,
                std::optional<std::int32_t> num_edge_types,
                std::optional<edge_properties_t> edge_properties,
                sampling_options_t sampling_options,
                bool do_expensive_check)
{
  return std::visit(
    [&](auto const& instance) {
      return neighbor_sample_impl(instance,
                                  rng_state,
                                  edge_weights,
                                  edge_ids,
                                  edge_types,
                                  edge_start_times,
                                  edge_end_times,
                                  edge_biases,
                                  starting_vertices,
                                  starting_vertex_start_times,
                                  starting_vertex_end_times,
                                  starting_vertex_labels,
                                  label_to_output_comm_rank,
                                  fan_out,
                                  num_edge_types,
                                  edge_properties,
                                  sampling_options,
                                  do_expensive_check);
    },
    graph.impl);
}

}  // namespace cugraph_python
