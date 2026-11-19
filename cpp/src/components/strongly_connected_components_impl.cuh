/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/reduce_v.cuh>
#include <cugraph/prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/prims/update_v_frontier.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

/// @cond

namespace cugraph {
namespace detail {
namespace ecl_scc {

// Gather arbitrary internal vertex IDs, including remote owners in MG. Every
// rank must participate even if its local query range is empty.
template <typename GraphViewType, typename VertexIterator, typename ValueType>
rmm::device_uvector<ValueType> collect_vertex_values(raft::handle_t const& handle,
                                                     GraphViewType const& graph_view,
                                                     VertexIterator first,
                                                     size_t count,
                                                     ValueType const* values)
{
  if constexpr (GraphViewType::is_multi_gpu) {
    return collect_values_for_int_vertices(handle,
                                           first,
                                           first + count,
                                           values,
                                           graph_view.vertex_partition_range_lasts(),
                                           graph_view.local_vertex_partition_range_first());
  } else {
    rmm::device_uvector<ValueType> result(count, handle.get_stream());
    cugraph::gather(handle.get_thrust_policy(), first, first + count, values, result.begin());
    return result;
  }
}

template <typename GraphViewType>
size_t global_frontier_size(raft::handle_t const& handle, GraphViewType const&, size_t local_size)
{
  if constexpr (GraphViewType::is_multi_gpu) {
    return host_scalar_allreduce(
      handle.get_comms(), local_size, raft::comms::op_t::SUM, handle.get_stream());
  } else {
    return local_size;
  }
}

// Initialize labels, resolving vertices with zero in-degree or out-degree as singleton SCCs.
// This is one pass over the original degrees, without iterative peeling.
template <typename GraphViewType>
void trim_trivial_singletons(raft::handle_t const& handle,
                             GraphViewType const& graph_view,
                             raft::device_span<typename GraphViewType::edge_type const> in_degrees,
                             raft::device_span<typename GraphViewType::edge_type const> out_degrees,
                             raft::device_span<typename GraphViewType::vertex_type> labels)
{
  using vertex_t = typename GraphViewType::vertex_type;
  auto first     = graph_view.local_vertex_partition_range_first();
  thrust::tabulate(handle.get_thrust_policy(),
                   labels.begin(),
                   labels.end(),
                   [in_degrees, out_degrees, first] __device__(auto i) -> vertex_t {
                     return (in_degrees[i] == 0 || out_degrees[i] == 0)
                              ? static_cast<vertex_t>(first + i)
                              : invalid_component_id<vertex_t>::value;
                   });
}

// Colors always use forward-graph IDs. Reverse traversal uses its own native
// vertex IDs, so parent lookups must translate colors before collecting values.
template <typename GraphViewType>
void shortcut_frontier(raft::handle_t const& handle,
                       GraphViewType const& graph_view,
                       GraphViewType const& forward_view,
                       typename GraphViewType::vertex_type const* active,
                       size_t count,
                       typename GraphViewType::vertex_type* signatures,
                       typename GraphViewType::vertex_type seed,
                       bool reverse_direction,
                       typename GraphViewType::vertex_type const* reverse_to_forward,
                       typename GraphViewType::vertex_type const* forward_to_reverse)
{
  using vertex_t = typename GraphViewType::vertex_type;
  auto first     = graph_view.local_vertex_partition_range_first();
  auto n         = graph_view.number_of_vertices();
  auto query_first =
    thrust::make_transform_iterator(active, [=] __device__(vertex_t v) -> vertex_t {
      auto offset    = v - first;
      auto canonical = reverse_to_forward ? reverse_to_forward[offset] : v;
      auto signature = signatures[offset];
      auto query     = signature > canonical ? signature : canonical;
      // N names the physical seed's color; it is never a vertex lookup ID.
      return query == n ? seed : query;
    });
  rmm::device_uvector<vertex_t> shortcuts(0, handle.get_stream());
  if (reverse_direction) {
    auto reverse_queries =
      collect_vertex_values(handle, forward_view, query_first, count, forward_to_reverse);
    shortcuts = collect_vertex_values(
      handle, graph_view, reverse_queries.begin(), count, static_cast<vertex_t const*>(signatures));
  } else {
    shortcuts = collect_vertex_values(
      handle, graph_view, query_first, count, static_cast<vertex_t const*>(signatures));
  }
  auto policy = handle.get_thrust_policy();
  thrust::transform(
    policy,
    shortcuts.begin(),
    shortcuts.end(),
    active,
    shortcuts.begin(),
    [=] __device__(vertex_t value, vertex_t v) { return signatures[v - first] == n ? n : value; });
  auto offsets = thrust::make_transform_iterator(
    active, [first] __device__(vertex_t v) -> vertex_t { return v - first; });
  thrust::scatter(policy, shortcuts.begin(), shortcuts.end(), offsets, signatures);
}

template <typename vertex_t>
struct accept_owner_max_t {
  using result_t = cuda::std::tuple<cuda::std::optional<size_t>, cuda::std::optional<vertex_t>>;
  __device__ result_t operator()(vertex_t, vertex_t previous, vertex_t candidate) const
  {
    return candidate > previous ? result_t{size_t{1}, candidate}
                                : result_t{cuda::std::nullopt, cuda::std::nullopt};
  }
};

// Cache endpoint state for one partition round. Partition metadata is immutable
// within that round; subsequent steps publish only changed owner colors.
template <typename GraphViewType>
struct propagation_cache {
  using vertex_t    = typename GraphViewType::vertex_type;
  using partition_t = cuda::std::tuple<vertex_t, vertex_t>;
  using metadata_t  = cuda::std::tuple<vertex_t, vertex_t, vertex_t>;
  edge_src_property_t<vertex_t, vertex_t> source_colors;
  edge_src_property_t<vertex_t, partition_t> source_partitions;
  edge_dst_property_t<vertex_t, metadata_t> destination_metadata;
  edge_dst_property_t<vertex_t, vertex_t> destination_colors;
  bool initialized{false};

  propagation_cache(raft::handle_t const& handle, GraphViewType const& graph_view)
    : source_colors(GraphViewType::is_multi_gpu
                      ? edge_src_property_t<vertex_t, vertex_t>(handle, graph_view)
                      : edge_src_property_t<vertex_t, vertex_t>(handle)),
      source_partitions(GraphViewType::is_multi_gpu
                          ? edge_src_property_t<vertex_t, partition_t>(handle, graph_view)
                          : edge_src_property_t<vertex_t, partition_t>(handle)),
      destination_metadata(GraphViewType::is_multi_gpu
                             ? edge_dst_property_t<vertex_t, metadata_t>(handle, graph_view)
                             : edge_dst_property_t<vertex_t, metadata_t>(handle)),
      destination_colors(GraphViewType::is_multi_gpu
                           ? edge_dst_property_t<vertex_t, vertex_t>(handle, graph_view)
                           : edge_dst_property_t<vertex_t, vertex_t>(handle))
  {
  }
};

// Graph primitives own edge traversal, mask handling, reduction, and frontier
// deduplication. The callbacks specify only ECL partition and signature rules.
template <typename GraphViewType, typename FrontierType>
void propagate_frontier(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        FrontierType& frontier,
                        typename GraphViewType::vertex_type* signatures,
                        typename GraphViewType::vertex_type const* labels,
                        typename GraphViewType::vertex_type const* partition_in,
                        typename GraphViewType::vertex_type const* partition_out,
                        propagation_cache<GraphViewType>& cache)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using property_t    = cuda::std::tuple<vertex_t, vertex_t, vertex_t>;
  using destination_t = cuda::std::tuple<vertex_t, vertex_t, vertex_t, vertex_t>;
  auto count          = graph_view.local_vertex_partition_range_size();
  auto source_values  = thrust::make_zip_iterator(
    static_cast<vertex_t const*>(signatures), partition_in, partition_out);
  auto destination_values = thrust::make_zip_iterator(
    labels, partition_in, partition_out, static_cast<vertex_t const*>(signatures));
  if constexpr (GraphViewType::is_multi_gpu) {
    if (!cache.initialized) {
      update_edge_src_property(
        handle,
        graph_view,
        source_values,
        view_concat(cache.source_colors.mutable_view(), cache.source_partitions.mutable_view()));
      update_edge_dst_property(handle,
                               graph_view,
                               destination_values,
                               view_concat(cache.destination_metadata.mutable_view(),
                                           cache.destination_colors.mutable_view()));
      cache.initialized = true;
    } else {
      auto& bucket = frontier.bucket(0);
      auto first   = graph_view.local_vertex_partition_range_first();
      // The frontier contains every owner improved by the preceding traversal.
      // Load colors here, after pointer jumping has updated those same owners.
      auto colors = thrust::make_transform_iterator(
        bucket.begin(), [=] __device__(vertex_t v) -> vertex_t { return signatures[v - first]; });
      // Every rank participates, including ranks with an empty local frontier.
      update_edge_src_property(handle,
                               graph_view,
                               bucket.begin(),
                               bucket.end(),
                               colors,
                               cache.source_colors.mutable_view());
      update_edge_dst_property(handle,
                               graph_view,
                               bucket.begin(),
                               bucket.end(),
                               colors,
                               cache.destination_colors.mutable_view());
    }
  }
  auto source_view = [&] {
    if constexpr (GraphViewType::is_multi_gpu) {
      return view_concat(cache.source_colors.view(), cache.source_partitions.view());
    } else {
      return make_edge_src_property_view<vertex_t, property_t>(graph_view, source_values, count);
    }
  }();
  auto destination_view = [&] {
    if constexpr (GraphViewType::is_multi_gpu) {
      return view_concat(cache.destination_metadata.view(), cache.destination_colors.view());
    } else {
      return make_edge_dst_property_view<vertex_t, destination_t>(
        graph_view, destination_values, count);
    }
  }();
  auto [vertices, values] = cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
    handle,
    graph_view,
    frontier.bucket(0),
    source_view,
    destination_view,
    edge_dummy_property_t{}.view(),
    [] __device__(vertex_t, vertex_t, property_t source, destination_t, cuda::std::nullopt_t)
      -> vertex_t { return cuda::std::get<0>(source); },
    reduce_op::maximum<vertex_t>{},
    [] __device__(
      vertex_t, vertex_t, property_t source, destination_t destination, cuda::std::nullopt_t)
      -> bool {
      return cuda::std::get<0>(destination) == invalid_component_id<vertex_t>::value &&
             cuda::std::get<1>(source) == cuda::std::get<1>(destination) &&
             cuda::std::get<2>(source) == cuda::std::get<2>(destination) &&
             cuda::std::get<0>(source) > cuda::std::get<3>(destination);
    });
  std::array<size_t, 1> next_buckets{1};
  cugraph::update_v_frontier(
    handle,
    graph_view,
    std::move(vertices),
    std::move(values),
    frontier,
    raft::host_span<size_t const>(next_buckets.data(), next_buckets.size()),
    signatures,
    signatures,
    accept_owner_max_t<vertex_t>{});
  frontier.bucket(0).clear();
  frontier.swap_buckets(0, 1);
}

}  // namespace ecl_scc
}  // namespace detail

template <typename GraphViewType>
void strongly_connected_components_impl(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type> components,
  bool do_expensive_check)
{
  // ECL-SCC: Alabandi et al., SC 2023, doi:10.1145/3581784.3607071.
  // Both directions propagate colors in the forward graph's global ID space.
  // Equal signatures resolve an SCC; unequal pairs define the next partition.
  using vertex_t           = typename GraphViewType::vertex_type;
  using edge_t             = typename GraphViewType::edge_type;
  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;
  static_assert(std::is_integral_v<vertex_t>);
  static_assert(!GraphViewType::is_storage_transposed, "SCC requires push storage.");
  using namespace detail::ecl_scc;
  auto stream = handle.get_stream();
  auto policy = handle.get_thrust_policy();
  auto n      = graph_view.number_of_vertices();
  CUGRAPH_EXPECTS(
    components.size() == static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
    "Invalid local component size.");
  if (n == 0) { return; }
  CUGRAPH_EXPECTS(!graph_view.is_symmetric(),
                  "Call weakly_connected_components instead for symmetric graphs.");

  // Native reverse construction retains the map to forward-graph IDs.
  // MG additionally redistributes vertices and reversed edges before building.
  graph_t<vertex_t, edge_t, false, multi_gpu> reverse(handle);
  std::optional<rmm::device_uvector<vertex_t>> reverse_number_map;
  {
    rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(), stream);
    cugraph::sequence(
      policy, vertices.begin(), vertices.end(), graph_view.local_vertex_partition_range_first());
    if constexpr (multi_gpu) {
      std::tie(vertices, std::ignore) = shuffle_ext_vertices(
        handle, std::move(vertices), std::vector<arithmetic_device_uvector_t>{});
    }
    auto [sources, destinations, weights, ids, types] =
      decompress_to_edgelist<vertex_t, edge_t, float, int32_t, false, multi_gpu>(
        handle, graph_view, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    std::swap(sources, destinations);
    if constexpr (multi_gpu) {
      std::tie(sources, destinations, std::ignore) =
        shuffle_ext_edges(handle,
                          std::move(sources),
                          std::move(destinations),
                          std::vector<arithmetic_device_uvector_t>{},
                          false);
    }
    std::tie(reverse, std::ignore, reverse_number_map) =
      create_graph_from_edgelist<vertex_t, edge_t, false, multi_gpu>(
        handle,
        std::move(vertices),
        std::move(sources),
        std::move(destinations),
        std::vector<arithmetic_device_uvector_t>{},
        graph_properties_t{graph_view.is_symmetric(), graph_view.is_multigraph()},
        true,
        std::nullopt,
        std::nullopt,
        do_expensive_check);
  }
  auto reverse_view = reverse.view();
  CUGRAPH_EXPECTS(
    reverse_number_map && reverse_number_map->size() ==
                            static_cast<size_t>(reverse_view.local_vertex_partition_range_size()),
    "Invalid reverse vertex map size.");
  auto first         = graph_view.local_vertex_partition_range_first();
  auto reverse_first = reverse_view.local_vertex_partition_range_first();
  auto local_n       = graph_view.local_vertex_partition_range_size();
  auto reverse_n     = reverse_view.local_vertex_partition_range_size();
  auto labels        = components;
  CUGRAPH_EXPECTS(reverse_view.number_of_vertices() == n, "Reverse graph vertex count differs.");

  rmm::device_uvector<vertex_t> forward_to_reverse(local_n, stream);
  auto canonical_reverse = reverse_number_map->data();
  rmm::device_uvector<vertex_t> partition_in(local_n, stream), partition_out(local_n, stream);
  rmm::device_uvector<vertex_t> incoming(local_n, stream), outgoing(reverse_n, stream);
  rmm::device_uvector<vertex_t> reverse_labels(reverse_n, stream);
  rmm::device_uvector<vertex_t> reverse_pin(reverse_n, stream);
  rmm::device_uvector<vertex_t> reverse_pout(reverse_n, stream);
  rmm::device_uvector<vertex_t> active(local_n, stream), reverse_active(reverse_n, stream);
  cugraph::fill(policy, partition_in.begin(), partition_in.end(), vertex_t{0});
  cugraph::fill(policy, partition_out.begin(), partition_out.end(), vertex_t{0});

  auto first_vertex = thrust::make_counting_iterator(first);
  vertex_t seed{};
  {
    auto out_degrees     = graph_view.compute_out_degrees(handle);
    auto reverse_degrees = reverse_view.compute_out_degrees(handle);
    rmm::device_uvector<edge_t> in_degrees(local_n, stream);
    {
      // Route reverse IDs and reverse out-degrees to the corresponding forward
      // owners. This simultaneously builds the inverse map and forward in-degrees.
      rmm::device_uvector<vertex_t> forward_ids(reverse_n, stream), reverse_ids(reverse_n, stream);
      thrust::copy(policy, canonical_reverse, canonical_reverse + reverse_n, forward_ids.begin());
      cugraph::sequence(policy, reverse_ids.begin(), reverse_ids.end(), reverse_first);
      std::vector<arithmetic_device_uvector_t> properties;
      properties.emplace_back(std::move(reverse_ids));
      properties.emplace_back(std::move(reverse_degrees));
      if constexpr (multi_gpu) {
        std::tie(forward_ids, properties) =
          shuffle_int_vertices(handle,
                               std::move(forward_ids),
                               std::move(properties),
                               graph_view.vertex_partition_range_lasts());
      }
      CUGRAPH_EXPECTS(forward_ids.size() == static_cast<size_t>(local_n),
                      "Invalid forward mapping size.");
      auto offsets = thrust::make_transform_iterator(
        forward_ids.begin(), [first] __device__(vertex_t v) -> vertex_t { return v - first; });
      auto values =
        thrust::make_zip_iterator(std::get<rmm::device_uvector<vertex_t>>(properties[0]).begin(),
                                  std::get<rmm::device_uvector<edge_t>>(properties[1]).begin());
      thrust::scatter(policy,
                      values,
                      values + local_n,
                      offsets,
                      thrust::make_zip_iterator(forward_to_reverse.begin(), in_degrees.begin()));
    }
    using seed_t    = cuda::std::tuple<double, vertex_t>;
    auto candidates = thrust::make_transform_iterator(
      first_vertex,
      [in = in_degrees.data(), out = out_degrees.data(), first] __device__(vertex_t v) -> seed_t {
        auto i = v - first;
        return {-static_cast<double>(in[i]) * static_cast<double>(out[i]), v};
      });
    seed = cuda::std::get<1>(cugraph::reduce_v(
      handle,
      graph_view,
      candidates,
      seed_t{std::numeric_limits<double>::max(), std::numeric_limits<vertex_t>::max()},
      reduce_op::minimum<seed_t>{}));

    trim_trivial_singletons(handle,
                            graph_view,
                            raft::device_span<edge_t const>(in_degrees.data(), in_degrees.size()),
                            raft::device_span<edge_t const>(out_degrees.data(), out_degrees.size()),
                            labels);
  }
  auto end = thrust::copy_if(policy,
                             first_vertex,
                             first_vertex + local_n,
                             active.begin(),
                             [labels = labels.data(), first] __device__(vertex_t v) {
                               return labels[v - first] == invalid_component_id<vertex_t>::value;
                             });
  active.resize(static_cast<size_t>(end - active.begin()), stream);
  // Wrappers can load a trimmed endpoint before testing the edge predicate.
  cugraph::sequence(policy, incoming.begin(), incoming.end(), first);
  thrust::copy(policy, canonical_reverse, canonical_reverse + reverse_n, outgoing.begin());
  bool first_round{true};
  while (global_frontier_size(handle, graph_view, active.size()) > 0) {
    reverse_labels       = collect_vertex_values(handle,
                                           graph_view,
                                           canonical_reverse,
                                           reverse_n,
                                           static_cast<vertex_t const*>(labels.data()));
    reverse_pin          = collect_vertex_values(handle,
                                        graph_view,
                                        canonical_reverse,
                                        reverse_n,
                                        static_cast<vertex_t const*>(partition_in.data()));
    reverse_pout         = collect_vertex_values(handle,
                                         graph_view,
                                         canonical_reverse,
                                         reverse_n,
                                         static_cast<vertex_t const*>(partition_out.data()));
    auto backward_labels = reverse_labels.data();
    auto backward_pin    = reverse_pin.data();
    auto backward_pout   = reverse_pout.data();
    reverse_active.resize(reverse_n, stream);
    auto reverse_begin = thrust::make_counting_iterator(reverse_first);
    auto reverse_end   = thrust::copy_if(policy,
                                       reverse_begin,
                                       reverse_begin + reverse_n,
                                       reverse_active.begin(),
                                       [backward_labels, reverse_first] __device__(vertex_t v) {
                                         return backward_labels[v - reverse_first] ==
                                                invalid_component_id<vertex_t>::value;
                                       });
    reverse_active.resize(static_cast<size_t>(reverse_end - reverse_active.begin()), stream);
    auto forward_colors = thrust::make_transform_iterator(
      active.begin(),
      [=] __device__(vertex_t v) -> vertex_t { return first_round && v == seed ? n : v; });
    auto forward_offsets = thrust::make_transform_iterator(
      active.begin(), [first] __device__(vertex_t v) -> vertex_t { return v - first; });
    thrust::scatter(
      policy, forward_colors, forward_colors + active.size(), forward_offsets, incoming.begin());
    auto backward_colors = thrust::make_transform_iterator(
      reverse_active.begin(), [=] __device__(vertex_t v) -> vertex_t {
        auto canonical = canonical_reverse[v - reverse_first];
        return first_round && canonical == seed ? n : canonical;
      });
    auto backward_offsets = thrust::make_transform_iterator(
      reverse_active.begin(),
      [reverse_first] __device__(vertex_t v) -> vertex_t { return v - reverse_first; });
    thrust::scatter(policy,
                    backward_colors,
                    backward_colors + reverse_active.size(),
                    backward_offsets,
                    outgoing.begin());
    first_round = false;
    vertex_frontier_t<vertex_t, void, multi_gpu, true> forward_frontier(handle, 2),
      backward_frontier(handle, 2);
    forward_frontier.bucket(0).insert(active.begin(), active.end());
    backward_frontier.bucket(0).insert(reverse_active.begin(), reverse_active.end());
    // Reinitialize caches after each split; retain them across inner steps.
    propagation_cache<GraphViewType> forward_cache(handle, graph_view);
    propagation_cache<GraphViewType> backward_cache(handle, reverse_view);
    while (true) {
      auto forward_size  = forward_frontier.bucket(0).aggregate_size();
      auto backward_size = backward_frontier.bucket(0).aggregate_size();
      if (forward_size == 0 && backward_size == 0) break;
      if (forward_size > 0) {
        propagate_frontier(handle,
                           graph_view,
                           forward_frontier,
                           incoming.data(),
                           labels.data(),
                           partition_in.data(),
                           partition_out.data(),
                           forward_cache);
        if (forward_frontier.bucket(0).aggregate_size() > 0) {
          auto& next = forward_frontier.bucket(0);
          shortcut_frontier(handle,
                            graph_view,
                            graph_view,
                            next.begin(),
                            next.size(),
                            incoming.data(),
                            seed,
                            false,
                            static_cast<vertex_t const*>(nullptr),
                            forward_to_reverse.data());
        }
      }
      if (backward_size > 0) {
        propagate_frontier(handle,
                           reverse_view,
                           backward_frontier,
                           outgoing.data(),
                           backward_labels,
                           backward_pin,
                           backward_pout,
                           backward_cache);
        if (backward_frontier.bucket(0).aggregate_size() > 0) {
          auto& next = backward_frontier.bucket(0);
          shortcut_frontier(handle,
                            reverse_view,
                            graph_view,
                            next.begin(),
                            next.size(),
                            outgoing.data(),
                            seed,
                            true,
                            canonical_reverse,
                            forward_to_reverse.data());
        }
      }
    }
    auto aligned_outgoing = collect_vertex_values(handle,
                                                  reverse_view,
                                                  forward_to_reverse.begin(),
                                                  local_n,
                                                  static_cast<vertex_t const*>(outgoing.data()));
    auto backward_values  = aligned_outgoing.data();
    thrust::for_each(policy,
                     active.begin(),
                     active.end(),
                     [labels = labels.data(),
                      pin    = partition_in.data(),
                      pout   = partition_out.data(),
                      in     = incoming.data(),
                      out    = backward_values,
                      first,
                      n,
                      seed] __device__(vertex_t v) {
                       auto i = v - first;
                       if (in[i] == out[i]) {
                         labels[i] = in[i] == n ? seed : in[i];
                       } else {
                         pin[i]  = in[i];
                         pout[i] = out[i];
                       }
                     });
    auto remaining = thrust::remove_if(
      policy, active.begin(), active.end(), [labels = labels.data(), first] __device__(vertex_t v) {
        return labels[v - first] != invalid_component_id<vertex_t>::value;
      });
    active.resize(static_cast<size_t>(remaining - active.begin()), stream);
  }
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> strongly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> components(graph_view.local_vertex_partition_range_size(),
                                           handle.get_stream());
  auto output = raft::device_span<vertex_t>(components.data(), components.size());
  strongly_connected_components_impl(handle, graph_view, output, do_expensive_check);
  return components;
}

}  // namespace cugraph

/// @endcond
