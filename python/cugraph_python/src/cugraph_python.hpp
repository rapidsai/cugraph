/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Prototype: nanobind-facing types for a directed, simple graph.
 *
 * The C++ graph is a template on vertex type, edge type, store_transposed, and multi_gpu.
 * Python selects those at runtime. graph_t stores the compiled instantiation in a variant.
 * The compiled set is (int32, int32) and (int64, int64), each with store_transposed false and true,
 * and multi_gpu=false. BFS and neighbor sampling require store_transposed=false. The current
 * PageRank API requires store_transposed=true. create_graph() rejects multi_gpu=true.
 *
 * owned_coo_t and prepared_vertices_t vary only by vertex type. Edge index type follows the
 * vertex type. An optional edge-weight array is float32 or float64 and is independent of the
 * index type. The two bools are arguments to create_graph().
 *
 * Algorithms return std::tuple. bfs() returns (vertices, distances, predecessors). pagerank()
 * returns (vertices, pageranks, iterations, converged). neighbor_sample() returns (sources,
 * destinations, hops, sampled properties, label offsets). create_graph() returns (graph, edge
 * properties). Array elements are device_array_t. predecessors, hops, and label offsets are
 * std::nullopt when they were not requested. PageRank scores use the weight dtype, and are
 * float32 when no weight property is passed. The edge-property vector keeps the order the
 * caller supplied. Sampling returns one sampled column per property, in that order. Omitting a
 * temporal comparison selects non-temporal sampling.
 *
 * Ownership model
 *   context_t            Execution resources (raft handle, stream). Shared by all objects built
 *                        from it. Not templated.
 *   rng_state_t          Host RNG state. neighbor_sample updates it, so one object continues the
 *                        random stream across calls.
 *   owned_coo_t          Exclusive owner of endpoint buffers and the edge-property buffers, in
 *                        caller order. create_graph() consumes it; the buffers are moved into the
 *                        cuGraph builder (no element copy). All Python aliases observe the
 *                        consumed state; a second consumption fails.
 *   graph_t              Owns one instantiated native graph and the internal-to-external number
 *                        map. Native storage is destroyed before the context reference.
 *   edge_properties_t    Edge properties returned beside the graph, in the order the caller
 *                        supplied. Holds the context, so the buffers outlive the graph, and records
 *                        the graph identity so an algorithm can reject a property from another
 *                        graph. edge_property_t is one element of that vector.
 *   prepared_vertices_t  Internal IDs translated once with a graph's number map. Records the graph
 *                        identity, numbering-map identity, partition, and version, so bfs() can
 *                        reject use with a different graph.
 *   device_array_t       One device array, type-erased over int32, int64, float32, and float64.
 *                        Each DLPack export owns its buffer, so the data outlives the graph and
 *                        the context.
 *
 * Vertex IDs
 *   Internal IDs are the contiguous IDs used by cuGraph. External IDs are the user's IDs. The
 *   number map (internal -> external) has unspecified order.
 *
 * Python names are applied in the nanobind bindings. These C++ types stay snake_case with a _t
 * suffix, so a variable can use the name without the suffix.
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>
#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cugraph_python {

enum class arithmetic_type_id_t { int32, int64, float32, float64 };

// One device array copied into an OwnedCOO. The pointer is only read during copy_from_arrays().
struct device_property_input_t {
  void const* data{};
  arithmetic_type_id_t dtype{};
};

// none leaves temporal sampling off. Any other value requires an edge start-time property.
enum class temporal_comparison_id_t {
  none = 0,
  strictly_increasing,
  monotonically_increasing,
  strictly_decreasing,
  monotonically_decreasing
};

// Matches cugraph::prior_sources_behavior_t.
enum class prior_sources_behavior_id_t { defaults = 0, carry_over, exclude };

// Matches cugraph::neighbor_selection_t. last is temporal only.
enum class neighbor_selection_id_t { random = 0, last };

// Same fields and defaults as cugraph::sampling_options_t. temporal_sampling_comparison
// uses none for the unset C++ optional.
struct sampling_options_t {
  prior_sources_behavior_id_t prior_sources_behavior{prior_sources_behavior_id_t::defaults};
  bool return_hops{false};
  bool dedupe_sources{false};
  bool with_replacement{true};
  temporal_comparison_id_t temporal_sampling_comparison{temporal_comparison_id_t::none};
  bool disjoint_sampling{false};
  neighbor_selection_id_t neighbor_selection{neighbor_selection_id_t::random};
  bool fixed_window{false};
};

struct context_t {
  raft::handle_t handle;
};

// Owns one raft::random::RngState. Sampling writes the updated subsequence back into this object.
class rng_state_t {
 public:
  explicit rng_state_t(std::uint64_t seed) : state_(seed) {}

  [[nodiscard]] std::uint64_t seed() const noexcept { return state_.seed; }
  [[nodiscard]] std::uint64_t base_subsequence() const noexcept { return state_.base_subsequence; }
  [[nodiscard]] raft::random::RngState& state() noexcept { return state_; }

 private:
  raft::random::RngState state_;
};

template <typename vertex_t>
class owned_coo_instance_t {
 public:
  using vertex_type = vertex_t;

  owned_coo_instance_t(
    std::shared_ptr<context_t> context,
    rmm::device_uvector<vertex_t> sources,
    rmm::device_uvector<vertex_t> destinations,
    std::vector<cugraph::arithmetic_device_uvector_t> edge_properties = {})
    : context_(std::move(context)),
      sources_(std::move(sources)),
      destinations_(std::move(destinations)),
      edge_properties_(std::move(edge_properties))
  {
    if (sources_.size() != destinations_.size()) {
      throw std::invalid_argument("OwnedCOO: sources and destinations differ in length");
    }
    for (auto const& property : edge_properties_) {
      if (cugraph::arithmetic_device_uvector_size(property) != sources_.size()) {
        throw std::invalid_argument("OwnedCOO: an edge property differs in length from the endpoints");
      }
    }
  }

  [[nodiscard]] bool consumed() const noexcept { return consumed_; }
  [[nodiscard]] std::size_t number_of_edges() const noexcept { return sources_.size(); }
  [[nodiscard]] std::shared_ptr<context_t> const& context() const noexcept { return context_; }

  // Moves the endpoint buffers and edge properties out. The consumed flag is set before any
  // transfer, so repeated calls fail and every alias sees the consumed state even if a later step
  // throws.
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::vector<cugraph::arithmetic_device_uvector_t>>
  take()
  {
    if (consumed_) { throw std::runtime_error("OwnedCOO has already been consumed"); }
    consumed_ = true;
    return {std::move(sources_), std::move(destinations_), std::move(edge_properties_)};
  }

 private:
  std::shared_ptr<context_t> context_;
  rmm::device_uvector<vertex_t> sources_;
  rmm::device_uvector<vertex_t> destinations_;
  // Caller order. copy_from_arrays copies each supplied edge property array into this vector.
  std::vector<cugraph::arithmetic_device_uvector_t> edge_properties_;
  bool consumed_{false};
};

struct owned_coo_t {
  using variant_t =
    std::variant<owned_coo_instance_t<std::int32_t>, owned_coo_instance_t<std::int64_t>>;

  variant_t impl;

  template <typename vertex_t>
  explicit owned_coo_t(owned_coo_instance_t<vertex_t> instance)
    : impl(std::in_place_type<owned_coo_instance_t<vertex_t>>, std::move(instance))
  {
  }

  [[nodiscard]] bool consumed() const
  {
    return std::visit([](auto const& instance) { return instance.consumed(); }, impl);
  }

  [[nodiscard]] std::size_t number_of_edges() const
  {
    return std::visit([](auto const& instance) { return instance.number_of_edges(); }, impl);
  }
};

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
struct graph_instance_t {
  using vertex_type = vertex_t;
  using edge_type   = edge_t;
  static constexpr bool is_storage_transposed = store_transposed;
  static constexpr bool is_multi_gpu          = multi_gpu;

  // Declaration order matters: members are destroyed in reverse order, so `native` (and the
  // device storage it owns) is destroyed before `context` is released.
  std::shared_ptr<context_t> context;
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> native;
  std::optional<rmm::device_uvector<vertex_t>> number_map;
  std::uint64_t graph_id{};
  std::uint64_t numbering_id{};
  std::uint64_t version{};
};

struct graph_t {
  using variant_t = std::variant<graph_instance_t<std::int32_t, std::int32_t, false, false>,
                                 graph_instance_t<std::int64_t, std::int64_t, false, false>,
                                 graph_instance_t<std::int32_t, std::int32_t, true, false>,
                                 graph_instance_t<std::int64_t, std::int64_t, true, false>>;

  variant_t impl;

  template <typename instance_t>
  explicit graph_t(instance_t&& instance)
    : impl(std::in_place_type<std::decay_t<instance_t>>, std::forward<instance_t>(instance))
  {
  }

  [[nodiscard]] std::int64_t number_of_vertices() const
  {
    return std::visit(
      [](auto const& instance) -> std::int64_t {
        return static_cast<std::int64_t>(instance.native.number_of_vertices());
      },
      impl);
  }

  [[nodiscard]] std::int64_t number_of_edges() const
  {
    return std::visit(
      [](auto const& instance) -> std::int64_t {
        return static_cast<std::int64_t>(instance.native.number_of_edges());
      },
      impl);
  }
};

// Edge properties are not members of the graph. create_graph() returns them in caller order.
// The state is shared so an indexed edge_property_t keeps the buffers and the context alive
// after the vector itself is dropped.
template <typename edge_t>
struct edge_properties_state_t {
  using edge_type = edge_t;

  std::shared_ptr<context_t> context;
  std::vector<cugraph::edge_arithmetic_property_t<edge_t>> values;
  std::uint64_t graph_id{};
};

template <typename edge_t>
struct edge_property_handle_t {
  using edge_type = edge_t;

  std::shared_ptr<edge_properties_state_t<edge_t>> state;
  std::size_t index{};
};

struct edge_property_t {
  using variant_t =
    std::variant<edge_property_handle_t<std::int32_t>, edge_property_handle_t<std::int64_t>>;

  variant_t impl;

  template <typename edge_t>
  explicit edge_property_t(edge_property_handle_t<edge_t> handle)
    : impl(std::in_place_type<edge_property_handle_t<edge_t>>, std::move(handle))
  {
  }
};

struct edge_properties_t {
  using variant_t = std::variant<std::shared_ptr<edge_properties_state_t<std::int32_t>>,
                                 std::shared_ptr<edge_properties_state_t<std::int64_t>>>;

  variant_t impl;

  template <typename edge_t>
  explicit edge_properties_t(std::shared_ptr<edge_properties_state_t<edge_t>> state)
    : impl(std::in_place_type<std::shared_ptr<edge_properties_state_t<edge_t>>>, std::move(state))
  {
  }

  [[nodiscard]] std::size_t size() const
  {
    return std::visit([](auto const& state) { return state->values.size(); }, impl);
  }

  [[nodiscard]] edge_property_t at(std::size_t index) const
  {
    if (index >= size()) { throw std::out_of_range("EdgeProperties index out of range"); }
    return std::visit(
      [&](auto const& state) -> edge_property_t {
        using edge_t = typename std::decay_t<decltype(*state)>::edge_type;
        return edge_property_t{edge_property_handle_t<edge_t>{state, index}};
      },
      impl);
  }
};

template <typename vertex_t>
struct prepared_vertices_instance_t {
  using vertex_type = vertex_t;

  std::shared_ptr<context_t> context;
  rmm::device_uvector<vertex_t> internal_ids;
  std::uint64_t graph_id{};
  std::uint64_t numbering_id{};
  std::uint64_t version{};
  int partition_rank{};
  int partition_count{};
};

struct prepared_vertices_t {
  using variant_t = std::variant<prepared_vertices_instance_t<std::int32_t>,
                                 prepared_vertices_instance_t<std::int64_t>>;

  variant_t impl;

  template <typename vertex_t>
  explicit prepared_vertices_t(prepared_vertices_instance_t<vertex_t> instance)
    : impl(std::in_place_type<prepared_vertices_instance_t<vertex_t>>, std::move(instance))
  {
  }

  [[nodiscard]] std::size_t number_of_vertices() const
  {
    return std::visit([](auto const& instance) { return instance.internal_ids.size(); }, impl);
  }
};

struct device_array_t {
  using variant_t = std::variant<std::shared_ptr<rmm::device_uvector<std::int32_t>>,
                                 std::shared_ptr<rmm::device_uvector<std::int64_t>>,
                                 std::shared_ptr<rmm::device_uvector<float>>,
                                 std::shared_ptr<rmm::device_uvector<double>>>;

  variant_t buffer;

  template <typename value_t>
  explicit device_array_t(std::shared_ptr<rmm::device_uvector<value_t>> buffer)
    : buffer(std::in_place_type<std::shared_ptr<rmm::device_uvector<value_t>>>, std::move(buffer))
  {
  }
};

// Factories. Each synchronizes the stream before returning.
std::unique_ptr<owned_coo_t> make_cycle_coo(std::shared_ptr<context_t> context, std::int64_t size);

// Endpoint dtypes select the vertex type. Both must be int32 or both int64, and both arrays have
// `num_edges` values. `edge_properties` is copied in caller order. Each array has one value per
// edge and is int32, int64, float32, or float64. An empty vector builds an unweighted COO.
std::unique_ptr<owned_coo_t> copy_from_arrays(
  std::shared_ptr<context_t> context,
  device_property_input_t sources,
  device_property_input_t destinations,
  std::size_t num_edges,
  std::vector<device_property_input_t> const& edge_properties);

// Consumes `coo`. Returns (graph, edge properties). The vertex type comes from `coo`. Edge type
// matches it. The property vector preserves the order supplied on `coo`. store_transposed selects
// the BFS orientation (false) or the PageRank orientation (true). Throws std::invalid_argument
// when multi_gpu is true, and std::runtime_error when `coo` was already consumed.
std::tuple<std::unique_ptr<graph_t>, std::unique_ptr<edge_properties_t>>
create_graph(owned_coo_t& coo, bool store_transposed, bool multi_gpu);

std::unique_ptr<prepared_vertices_t> prepare_vertices(graph_t const& graph,
                                                      std::vector<std::int64_t> const& external_ids);

// (vertices, distances, predecessors). vertices[i] is the external ID of internal vertex i.
// distances[i] is the hop count; unreached vertices hold numeric_limits<vertex_t>::max().
// predecessors[i] is the external ID of the BFS parent, or -1 for starts and unreached vertices.
// predecessors is std::nullopt unless compute_predecessors is true.
std::tuple<device_array_t, device_array_t, std::optional<device_array_t>>
bfs(graph_t const& graph, prepared_vertices_t const& starts, bool compute_predecessors);

// (vertices, pageranks, iterations, converged). vertices[i] is the external ID of internal
// vertex i. pageranks[i] is the score of that vertex. Requires store_transposed=true. `weights`
// is one element of the edge-property vector from create_graph(); std::nullopt means every edge
// weight is 1 and scores are float32. Otherwise the score dtype matches the property. alpha and
// epsilon are cast to that dtype. Scores use the vertex_property_t API.
std::tuple<device_array_t, device_array_t, std::size_t, bool> pagerank(
  graph_t const& graph,
  std::optional<edge_property_t> weights,
  double alpha,
  double epsilon,
  std::size_t max_iterations);

// (sources, destinations, hops, sampled properties, label offsets). Parameter order matches
// cugraph::neighbor_sample: rng_state, edge weight, id, type, start time, end time, bias,
// starting vertices, per-vertex start times, end times, labels, label-to-rank, fan_out,
// num_edge_types, sampling_options, do_expensive_check. edge_properties is the prototype's
// caller-order vector and sits after num_edge_types. rng_state is updated in place.
//
// sources[i] and destinations[i] are external vertex IDs of one sampled edge. hops[i] is the hop
// that produced it, or std::nullopt when sampling_options.return_hops is false.
// sampled_properties[k] is the values of edge_properties[k] on those edges. label offsets are
// int64 and are std::nullopt unless starting_vertex_labels is set. Requires store_transposed=false.
//
// starting_vertex_start_times is one lower bound per starting vertex (the hop-0 frontier time
// for an increasing walk). starting_vertex_end_times is the matching upper bound. Heterogeneous
// sampling sets num_edge_types and requires edge_types; fan_out then holds one value per
// (hop, edge type). edge_biases selects biased sampling and uses the same dtype as edge weights.
// label_to_output_comm_rank is rejected: this build is single-GPU.
//
// temporal_sampling_comparison == none rejects edge start/end times and per-vertex times. Any
// other comparison requires edge_start_times (int32 or int64) and disjoint_sampling.
//
// The compiled sampler returns one floating-point column, one column of the graph's integer
// type, one int32 column, and, when temporal, the start-time and optional end-time columns.
// Explicit role arguments occupy those columns. Remaining properties in the vector fill free
// columns. A vector that does not fit is rejected. Columns are still returned in caller order.
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
                bool do_expensive_check);

}  // namespace cugraph_python
