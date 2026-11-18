/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/reduce_v.cuh>
#include <cugraph/prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh>
#include <cugraph/prims/update_v_frontier.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
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

namespace cugraph {
namespace detail {
namespace ecl_scc {

// Gather before scattering so pointer jumping observes a consistent signature
// snapshot. All vertex signatures are local in this SG implementation.
template <bool remap = false, typename GraphViewType>
void shortcut_frontier(raft::handle_t const& handle,
                       GraphViewType const& graph_view,
                       typename GraphViewType::vertex_type const* active,
                       size_t count,
                       typename GraphViewType::vertex_type* signatures,
                       typename GraphViewType::vertex_type seed,
                       typename GraphViewType::vertex_type const* vertex_map = nullptr)
{
  using vertex_t   = typename GraphViewType::vertex_type;
  auto policy      = handle.get_thrust_policy();
  auto first       = graph_view.local_vertex_partition_range_first();
  auto n           = graph_view.number_of_vertices();
  auto query_first = thrust::make_transform_iterator(
    active, [signatures, first, n, seed, vertex_map] __device__(vertex_t v) -> vertex_t {
      if constexpr (remap) { v = vertex_map[v - first]; }
      auto signature = signatures[v - first];
      auto query     = signature > v ? signature : v;
      return query;
    });
  rmm::device_uvector<vertex_t> shortcuts(count, handle.get_stream());
  cugraph::gather(policy, query_first, query_first + count, signatures, shortcuts.begin());
  thrust::transform(policy,
                    shortcuts.begin(),
                    shortcuts.end(),
                    active,
                    shortcuts.begin(),
                    [signatures, first, n, vertex_map] __device__(vertex_t value, vertex_t v) {
                      if constexpr (remap) { v = vertex_map[v - first]; }
                      return signatures[v - first] == n ? n : value;
                    });
  auto offsets =
    thrust::make_transform_iterator(active, [first, vertex_map] __device__(vertex_t v) -> vertex_t {
      if constexpr (remap) { v = vertex_map[v - first]; }
      return v - first;
    });
  thrust::scatter(policy, shortcuts.begin(), shortcuts.end(), offsets, signatures);
}

// Graph primitives own edge traversal, mask handling, reduction, and frontier
// deduplication. The callbacks specify only ECL partition and signature rules.
template <bool remap = false, typename GraphViewType, typename FrontierType>
void propagate_frontier(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        FrontierType& frontier,
                        typename GraphViewType::vertex_type* signatures,
                        typename GraphViewType::vertex_type const* labels,
                        typename GraphViewType::vertex_type const* partition_in,
                        typename GraphViewType::vertex_type const* partition_out,
                        typename GraphViewType::vertex_type const* vertex_map = nullptr)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using property_t    = cuda::std::tuple<vertex_t, vertex_t, vertex_t>;
  using destination_t = cuda::std::tuple<vertex_t, vertex_t, vertex_t, vertex_t>;
  auto count          = graph_view.local_vertex_partition_range_size();
  auto source_values  = thrust::make_zip_iterator(
    static_cast<vertex_t const*>(signatures), partition_in, partition_out);
  auto destination_values = thrust::make_zip_iterator(
    labels, partition_in, partition_out, static_cast<vertex_t const*>(signatures));
  auto source_view = [&] {
    if constexpr (remap) {
      auto values = thrust::make_permutation_iterator(source_values, vertex_map);
      return detail::edge_endpoint_property_view_t<vertex_t, decltype(values), property_t>(
        std::vector<decltype(values)>{values}, std::vector<vertex_t>{0});
    } else {
      return make_edge_src_property_view<vertex_t, property_t>(graph_view, source_values, count);
    }
  }();
  auto destination_view = [&] {
    if constexpr (remap) {
      auto values = thrust::make_permutation_iterator(destination_values, vertex_map);
      return detail::edge_endpoint_property_view_t<vertex_t, decltype(values), destination_t>(
        values, vertex_t{0});
    } else {
      return make_edge_dst_property_view<vertex_t, destination_t>(
        graph_view, destination_values, count);
    }
  }();
  auto signature_values = [&] {
    if constexpr (remap) {
      return thrust::make_permutation_iterator(signatures, vertex_map);
    } else {
      return signatures;
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
    signature_values,
    signature_values,
    [] __device__(vertex_t, vertex_t previous, vertex_t candidate) {
      using result_t = cuda::std::tuple<cuda::std::optional<size_t>, cuda::std::optional<vertex_t>>;
      return candidate > previous ? result_t{size_t{1}, candidate}
                                  : result_t{cuda::std::nullopt, cuda::std::nullopt};
    });
  frontier.bucket(0).clear();
  frontier.swap_buckets(0, 1);
}

}  // namespace ecl_scc
}  // namespace detail

namespace {

template <typename GraphViewType>
void strongly_connected_components_ecl_impl(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type> components,
  bool do_expensive_check)
{
  // ECL-SCC: Alabandi et al., SC 2023, doi:10.1145/3581784.3607071.
  // Within each unresolved partition, propagate maximum signatures in both
  // directions. Equal signatures identify an SCC; otherwise their pair defines
  // the next partition. An SCC cannot span different signature pairs. The first
  // round gives the maximum-degree-product vertex the extra color N; convert
  // that sentinel back to its physical vertex ID when assigning components.
  using vertex_t                  = typename GraphViewType::vertex_type;
  using edge_t                    = typename GraphViewType::edge_type;
  using weight_t                  = float;    // dummy
  using edge_type_t               = int32_t;  // dummy
  constexpr bool store_transposed = GraphViewType::is_storage_transposed;
  constexpr bool multi_gpu        = GraphViewType::is_multi_gpu;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");
  static_assert(!GraphViewType::is_multi_gpu, "ECL SCC currently supports single-GPU graphs only.");

  using namespace detail::ecl_scc;
  auto stream = handle.get_stream();
  auto policy = handle.get_thrust_policy();
  auto n      = graph_view.number_of_vertices();
  auto labels = components;
  if (n == 0) { return; }

  // Check input arguments
  CUGRAPH_EXPECTS(
    !graph_view.is_symmetric(),
    "Invalid input argument>: call weakly_connnected_components instead for symmetric graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  // Construct a degree-ordered reverse graph. Its vertex IDs may differ from
  // the input graph, so keep ECL state in forward IDs and map reverse traversal.
  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> reverse(handle);
  std::optional<rmm::device_uvector<vertex_t>> reverse_to_forward{std::nullopt};
  {
    rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(), stream);
    cugraph::sequence(
      policy, vertices.begin(), vertices.end(), graph_view.local_vertex_partition_range_first());
    rmm::device_uvector<vertex_t> sources(0, stream), destinations(0, stream);
    std::tie(sources, destinations, std::ignore, std::ignore, std::ignore) =
      decompress_to_edgelist<vertex_t, edge_t, weight_t, edge_type_t, store_transposed, multi_gpu>(
        handle, graph_view, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    std::swap(sources, destinations);
    std::tie(reverse, std::ignore, reverse_to_forward) =
      create_graph_from_edgelist<vertex_t, edge_t, store_transposed, multi_gpu>(
        handle,
        std::move(vertices),
        std::move(sources),
        std::move(destinations),
        std::vector<cugraph::arithmetic_device_uvector_t>{},
        graph_properties_t{graph_view.is_symmetric(), graph_view.is_multigraph()},
        true,  // renumber
        std::nullopt,
        std::nullopt,
        do_expensive_check);
  }
  auto reverse_view = reverse.view();

  CUGRAPH_EXPECTS(reverse_to_forward && reverse_to_forward->size() == static_cast<size_t>(n),
                  "Invalid reverse vertex map size.");
  auto reverse_map = reverse_to_forward->data();
  rmm::device_uvector<vertex_t> forward_to_reverse(n, stream);
  auto vertex_first = thrust::make_counting_iterator(vertex_t{0});
  thrust::scatter(policy, vertex_first, vertex_first + n, reverse_map, forward_to_reverse.begin());

  // ECL algorithmic data
  rmm::device_uvector<vertex_t> partition_in(n, stream), partition_out(n, stream);
  rmm::device_uvector<vertex_t> incoming(static_cast<size_t>(n) + 1, stream);
  rmm::device_uvector<vertex_t> outgoing(static_cast<size_t>(n) + 1, stream);
  std::array<rmm::device_uvector<vertex_t>, 2> active{rmm::device_uvector<vertex_t>(n, stream),
                                                      rmm::device_uvector<vertex_t>(n, stream)};
  // Initialize vertex states
  cugraph::fill(policy, labels.begin(), labels.end(), invalid_component_id<vertex_t>::value);
  cugraph::fill(policy, partition_in.begin(), partition_in.end(), vertex_t{0});
  cugraph::fill(policy, partition_out.begin(), partition_out.end(), vertex_t{0});

  // Heuristic: label the degree with largest in*out degree product with a max
  // label value. This helps resolve the largest SCC on the first iteration.
  // Approximate degree products are sufficient for this heuristic. Negating the
  // product lets a lexicographic minimum break ties with the smallest vertex ID.
  auto out_degrees = graph_view.compute_out_degrees(handle);
  auto in_degrees  = reverse_view.compute_out_degrees(handle);
  // Reverse out-degrees are forward in-degrees, reordered into forward IDs.
  rmm::device_uvector<edge_t> aligned_in_degrees(n, stream);
  thrust::scatter(
    policy, in_degrees.begin(), in_degrees.end(), reverse_map, aligned_in_degrees.begin());
  in_degrees        = std::move(aligned_in_degrees);
  auto first_vertex = thrust::make_counting_iterator(vertex_t{0});
  using seed_t      = cuda::std::tuple<double, vertex_t>;
  auto candidates   = thrust::make_transform_iterator(
    first_vertex,
    [out_degrees = out_degrees.data(),
     in_degrees  = in_degrees.data()] __device__(vertex_t v) -> seed_t {
      auto in_degree = in_degrees[v];
      return seed_t{-static_cast<double>(in_degree) * static_cast<double>(out_degrees[v]), v};
    });
  auto seed = cuda::std::get<1>(cugraph::reduce_v(
    handle,
    graph_view,
    candidates,
    seed_t{std::numeric_limits<double>::max(), std::numeric_limits<vertex_t>::max()},
    reduce_op::minimum<seed_t>{}));

  // A single iteration on trimming to remove trivial SCCs. Recursive trimming
  // is generally not worth the extra cost.
  thrust::transform_if(
    policy,
    first_vertex,
    first_vertex + n,
    labels.begin(),
    [] __device__(vertex_t v) { return v; },
    [in_degrees = in_degrees.data(), out_degrees = out_degrees.data()] __device__(vertex_t v) {
      return in_degrees[v] == 0 || out_degrees[v] == 0;
    });

  // Compact active vertices to construct our initial active vertex queue
  auto active_end = thrust::copy_if(
    policy,
    first_vertex,
    first_vertex + n,
    labels.begin(),
    active[0].begin(),
    [] __device__(vertex_t label) { return label == invalid_component_id<vertex_t>::value; });
  auto initial_active_count = static_cast<size_t>(active_end - active[0].begin());

  // Cleanup the initialization-only data
  out_degrees.resize(0, stream);
  out_degrees.shrink_to_fit(stream);
  in_degrees.resize(0, stream);
  in_degrees.shrink_to_fit(stream);

  // Property wrappers may load trimmed vertices before evaluating the edge
  // predicate. Initialize every entry, including the fixed extra color N.
  cugraph::sequence(policy, incoming.begin(), incoming.end(), vertex_t{0});
  cugraph::sequence(policy, outgoing.begin(), outgoing.end(), vertex_t{0});

  // Begin ECL's primary loop
  size_t active_count = initial_active_count;
  int active_buffer{0};
  bool first_round{true};
  while (active_count > 0) {
    // Initialize this iteration's signatures
    auto active_first = active[active_buffer].begin();
    auto color_first  = thrust::make_transform_iterator(
      active_first,
      [first_round, n, seed] __device__(vertex_t v) -> cuda::std::tuple<vertex_t, vertex_t> {
        auto color = first_round && v == seed ? n : v;
        return cuda::std::make_tuple(color, color);
      });
    thrust::scatter(policy,
                    color_first,
                    color_first + active_count,
                    active_first,
                    thrust::make_zip_iterator(incoming.begin(), outgoing.begin()));

    first_round = false;
    vertex_frontier_t<vertex_t, void, false, true> forward_frontier(handle, 2);
    vertex_frontier_t<vertex_t, void, false, true> backward_frontier(handle, 2);
    forward_frontier.bucket(0).insert(active_first, active_first + active_count);
    // Reuse the inactive active-list buffer for the reverse frontier's IDs.
    auto reverse_first = active[1 - active_buffer].begin();
    cugraph::gather(
      policy, active_first, active_first + active_count, forward_to_reverse.begin(), reverse_first);
    cugraph::sort(policy, reverse_first, reverse_first + active_count);
    backward_frontier.bucket(0).insert(reverse_first, reverse_first + active_count);
    while (forward_frontier.bucket(0).size() > 0 || backward_frontier.bucket(0).size() > 0) {
      if (forward_frontier.bucket(0).size() > 0) {
        propagate_frontier(handle,
                           graph_view,
                           forward_frontier,
                           incoming.data(),
                           labels.data(),
                           partition_in.data(),
                           partition_out.data());
        // propagate_frontier swaps the newly generated queue into bucket 0.
        // Shortcut that queue now, so it is ready for its next expansion.
        if (forward_frontier.bucket(0).size() > 0) {
          auto& next = forward_frontier.bucket(0);
          shortcut_frontier(handle, graph_view, next.begin(), next.size(), incoming.data(), seed);
        }
      }
      if (backward_frontier.bucket(0).size() > 0) {
        propagate_frontier<true>(handle,
                                 reverse_view,
                                 backward_frontier,
                                 outgoing.data(),
                                 labels.data(),
                                 partition_in.data(),
                                 partition_out.data(),
                                 reverse_map);
        if (backward_frontier.bucket(0).size() > 0) {
          auto& next = backward_frontier.bucket(0);
          shortcut_frontier<true>(
            handle, reverse_view, next.begin(), next.size(), outgoing.data(), seed, reverse_map);
        }
      }
    }

    // Resolve equal signatures and retain unresolved vertices in stable order.
    thrust::for_each(policy,
                     active_first,
                     active_first + active_count,
                     [labels = labels.data(),
                      pin    = partition_in.data(),
                      pout   = partition_out.data(),
                      in     = incoming.data(),
                      out    = outgoing.data(),
                      n,
                      seed] __device__(vertex_t v) {
                       if (in[v] == out[v]) {
                         labels[v] = in[v] == n ? seed : in[v];
                       } else {
                         pin[v]  = in[v];
                         pout[v] = out[v];
                       }
                     });
    auto end      = thrust::copy_if(policy,
                               active_first,
                               active_first + active_count,
                               active[1 - active_buffer].begin(),
                               [labels = labels.data()] __device__(vertex_t v) {
                                 return labels[v] == invalid_component_id<vertex_t>::value;
                               });
    active_count  = static_cast<size_t>(end - active[1 - active_buffer].begin());
    active_buffer = 1 - active_buffer;
  }
  return;
}

}  // namespace

}  // namespace cugraph
