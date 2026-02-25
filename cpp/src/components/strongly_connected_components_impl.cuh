/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/fill_edge_property.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace {

// recursively find a set of vertices with either 0 out-degree or 0 in-degree
template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::vertex_type> find_trivial_singleton_scc_vertices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  GraphViewType const& inverse_graph_view,
  raft::device_span<typename GraphViewType::vertex_type const> inverse_renumber_map,
  raft::device_span<typename GraphViewType::vertex_type const> candidate_vertices)
{
  using vertex_t = typename GraphViewType::vertex_type;
  CUGRAPH_FAIL("unimplemented.");
  return rmm::device_uvector<vertex_t>(0, handle.get_stream());
}

// find pivots (one per unresolved component)
template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::vertex_type> find_pivots(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type const> unresolved_component_offsets,
  raft::device_span<typename GraphViewType::vertex_type const> unresolved_component_vertices,
  raft::device_span<typename GraphViewType::vertex_type const>
    excluded_vertices /* should not be selected as pivots */)
{
  using vertex_t = typename GraphViewType::vertex_type;
  CUGRAPH_FAIL("unimplemented.");
  return rmm::device_uvector<vertex_t>(0, handle.get_stream());
}

template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
reachable_sets(raft::handle_t const& handle,
               GraphViewType const& graph_view,
               raft::device_span<typename GraphViewType::vertex_type const> starting_vertices)
{
  using vertex_t = typename GraphViewType::vertex_type;
  CUGRAPH_FAIL("unimplemented.");
  return std::make_tuple(rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                         rmm::device_uvector<vertex_t>(0, handle.get_stream()));
}

// return component_ids, component_scc_flags, component_offsets, component_vertices
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<bool>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>>
intersect_reachable_sets(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>&& unresolved_component_offsets,
                         rmm::device_uvector<vertex_t>&& unresolved_component_vertices,
                         rmm::device_uvector<vertex_t>&& forward_set_offsets,
                         rmm::device_uvector<vertex_t>&& forward_set_vertices,
                         rmm::device_uvector<vertex_t>&& backward_set_offsets,
                         rmm::device_uvector<vertex_t>&& backward_set_vertices,
                         rmm::device_uvector<vertex_t>&& trivial_singleton_scc_vertices)
{
  CUGRAPH_FAIL("unimplemented.");
  return std::make_tuple(rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                         rmm::device_uvector<bool>(0, handle.get_stream()),
                         rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                         rmm::device_uvector<vertex_t>(0, handle.get_stream()));
}

// return component_ids, component_scc_flags, component_offsets, component_vertices
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<bool>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
forward_backward_intersect(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  GraphViewType const& inverse_graph_view,
  raft::device_span<typename GraphViewType::vertex_type const> inverse_renumber_map,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_offsets,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_vertices)
{
  using vertex_t           = typename GraphViewType::vertex_type;
  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;

  auto trivial_singleton_scc_vertices = find_trivial_singleton_scc_vertices(
    handle,
    graph_view,
    inverse_graph_view,
    inverse_renumber_map,
    raft::device_span<vertex_t>(unresolved_component_vertices.data(),
                                unresolved_component_vertices.size()));

  // FIXME: the current API assumes one pivot per unresolved component; we may need to select more
  // than one pivot per unresolved component to extract additional parallelism (for large diameter
  // graphs)
  auto pivots =
    find_pivots(handle,
                graph_view,
                raft::device_span<vertex_t const>(unresolved_component_offsets.data(),
                                                  unresolved_component_offsets.size()),
                raft::device_span<vertex_t const>(unresolved_component_vertices.data(),
                                                  unresolved_component_vertices.size()),
                raft::device_span<vertex_t const>(trivial_singleton_scc_vertices.data(),
                                                  trivial_singleton_scc_vertices.size()));

  auto [forward_set_offsets, forward_set_vertices] = reachable_sets(
    handle, graph_view, raft::device_span<vertex_t const>(pivots.data(), pivots.size()));

  if constexpr (GraphViewType::is_multi_gpu) {
    std::tie(pivots, std::ignore) = shuffle_ext_vertices(
      handle, std::move(pivots), std::vector<cugraph::arithmetic_device_uvector_t>{});
  }
  renumber_local_ext_vertices<vertex_t, multi_gpu>(
    handle,
    pivots.data(),
    pivots.size(),
    inverse_renumber_map.data(),
    inverse_graph_view.local_vertex_partition_range_first(),
    inverse_graph_view.local_vertex_partition_range_last());
  auto [backward_set_offsets, backward_set_vertices] = reachable_sets(
    handle, inverse_graph_view, raft::device_span<vertex_t const>(pivots.data(), pivots.size()));
  unrenumber_local_int_vertices(handle,
                                backward_set_vertices.data(),
                                backward_set_vertices.size(),
                                inverse_renumber_map.data(),
                                inverse_graph_view.local_vertex_partition_range_first(),
                                inverse_graph_view.local_vertex_partition_range_last());

  return intersect_reachable_sets(handle,
                                  std::move(unresolved_component_offsets),
                                  std::move(unresolved_component_vertices),
                                  std::move(forward_set_offsets),
                                  std::move(forward_set_vertices),
                                  std::move(backward_set_offsets),
                                  std::move(backward_set_vertices),
                                  std::move(trivial_singleton_scc_vertices));
}

template <typename GraphViewType>
void strongly_connected_components_impl(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type> components,
  bool do_expensive_check)
{
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

  auto const num_vertices = graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    !graph_view.is_symmetric(),
    "Invalid input argument>: call weakly_connnected_components instead for symmetric graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. initialize component IDs (initially, every vertex belongs to one unresolved component)

  thrust::fill(handle.get_thrust_policy(), components.begin(), components.end(), vertex_t{0});

  // 3. create an edge mask and mask out self-loops & multi-edges (except for the first one); this
  // edge mask will be used to mask out edges between different components

  auto forward_graph_view = graph_view;
  edge_property_t<edge_t, bool> edge_mask(handle, forward_graph_view);
  {
    if (forward_graph_view.has_edge_mask()) { forward_graph_view.clear_edge_mask(); }
    cugraph::fill_edge_property(handle, forward_graph_view, edge_mask.mutable_view(), false);
    edge_multi_index_property_t<edge_t, vertex_t> edge_multi_indices(handle, graph_view);
    transform_e(handle,
                graph_view,
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_multi_indices.view(),
                cuda::proclaim_return_type<bool>(
                  [] __device__(auto src, auto dst, auto, auto, auto multi_edge_index) {
                    return (src != dst) && (multi_edge_index == 0);
                  }),
                edge_mask.mutable_view());
    forward_graph_view.attach_edge_mask(edge_mask.view());
  }

  // 4. create an inverse graph

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> inverse_graph(handle);
  rmm::device_uvector<vertex_t> inverse_renumber_map(0, handle.get_stream());
  {  // FIXME: we may avoid this if we create a transpose_graph function that takes a const
     // reference to a graph_view_t object
    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore, std::ignore) =
      decompress_to_edgelist<vertex_t, edge_t, weight_t, edge_type_t, store_transposed, multi_gpu>(
        handle, forward_graph_view, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    std::swap(edgelist_srcs, edgelist_dsts);
    if constexpr (multi_gpu) {
      std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore) =
        shuffle_ext_edges(handle,
                          std::move(edgelist_srcs),
                          std::move(edgelist_dsts),
                          std::vector<cugraph::arithmetic_device_uvector_t>{},
                          store_transposed);
    }
    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
    std::tie(inverse_graph, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, store_transposed, multi_gpu>(
        handle,
        std::nullopt,
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::vector<cugraph::arithmetic_device_uvector_t>{},
        graph_properties_t{false, false},
        true);
    inverse_renumber_map = std::move(*tmp_renumber_map);
  }
  auto inverse_graph_view = inverse_graph.view();
  edge_property_t<edge_t, bool> inverse_edge_mask(handle, inverse_graph_view);
  cugraph::fill_edge_property(handle, inverse_graph_view, inverse_edge_mask.mutable_view(), true);
  inverse_graph_view.attach_edge_mask(inverse_edge_mask.view());

  // 5. prepare for recursvie forward-backward SCC: set unresolved_component_offsets and
  // unresolved_component_vertices and initialzie edge_src|dst_components

  rmm::device_uvector<vertex_t> unresolved_component_offsets(2, handle.get_stream());
  rmm::device_uvector<vertex_t> unresolved_component_vertices(
    forward_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  unresolved_component_offsets.set_element_to_zero_async(0, handle.get_stream());
  unresolved_component_offsets.set_element(
    1, forward_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  thrust::sequence(handle.get_thrust_policy(),
                   unresolved_component_vertices.begin(),
                   unresolved_component_vertices.end(),
                   forward_graph_view.local_vertex_partition_range_first());

  auto edge_src_components = multi_gpu
                               ? edge_src_property_t<vertex_t, vertex_t>(handle, forward_graph_view)
                               : edge_src_property_t<vertex_t, vertex_t>(handle);
  auto edge_dst_components = multi_gpu
                               ? edge_dst_property_t<vertex_t, vertex_t>(handle, forward_graph_view)
                               : edge_dst_property_t<vertex_t, vertex_t>(handle);
  auto inverse_edge_src_components =
    multi_gpu ? edge_src_property_t<vertex_t, vertex_t>(handle, inverse_graph_view)
              : edge_src_property_t<vertex_t, vertex_t>(handle);
  auto inverse_edge_dst_components =
    multi_gpu ? edge_dst_property_t<vertex_t, vertex_t>(handle, inverse_graph_view)
              : edge_dst_property_t<vertex_t, vertex_t>(handle);
  if constexpr (multi_gpu) {
    fill_edge_src_property(
      handle, forward_graph_view, edge_src_components.mutable_view(), vertex_t{0});
    fill_edge_dst_property(
      handle, forward_graph_view, edge_dst_components.mutable_view(), vertex_t{0});
    fill_edge_src_property(
      handle, inverse_graph_view, inverse_edge_src_components.mutable_view(), vertex_t{0});
    fill_edge_dst_property(
      handle, inverse_graph_view, inverse_edge_dst_components.mutable_view(), vertex_t{0});
  }

  // 6. recursive forward-backward SCC

  while ((unresolved_component_offsets.size() - 1) > 0) {
    // 6-1. perform forward-backward SCC

    auto [component_ids, component_scc_flags, component_offsets, component_vertices] =
      forward_backward_intersect(
        handle,
        forward_graph_view,
        inverse_graph_view,
        raft::device_span<vertex_t const>(inverse_renumber_map.data(), inverse_renumber_map.size()),
        std::move(unresolved_component_offsets),
        std::move(unresolved_component_vertices));

    // 6-2. update components

    auto component_id_first = cuda::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [component_offsets =
           raft::device_span<vertex_t const>(component_offsets.data(), component_offsets.size()),
         component_ids = raft::device_span<vertex_t const>(
           component_ids.data(), component_ids.size())] __device__(size_t i) {
          auto idx = cuda::std::distance(
            component_offsets.begin() + 1,
            thrust::upper_bound(
              thrust::seq, component_offsets.begin() + 1, component_offsets.end(), i));
          return component_ids[idx];
        }));
    auto map_first = cuda::make_transform_iterator(
      component_vertices.begin(),
      detail::shift_left_t<vertex_t>{forward_graph_view.local_vertex_partition_range_first()});
    thrust::scatter(handle.get_thrust_policy(),
                    component_id_first,
                    component_id_first + component_vertices.size(),
                    map_first,
                    components.begin());

    // 6-3. mask out edges between different components

    if constexpr (multi_gpu) {
      rmm::device_uvector<vertex_t> tmp_vertices(component_vertices.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   component_vertices.begin(),
                   component_vertices.end(),
                   tmp_vertices.begin());
      rmm::device_uvector<vertex_t> tmp_components(component_vertices.size(), handle.get_stream());
      thrust::gather(handle.get_thrust_policy(),
                     map_first,
                     map_first + component_vertices.size(),
                     components.begin(),
                     tmp_components.begin());
      thrust::sort_by_key(handle.get_thrust_policy(),
                          tmp_vertices.begin(),
                          tmp_vertices.end(),
                          tmp_components.begin());

      update_edge_src_property(handle,
                               forward_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               edge_src_components.mutable_view());
      update_edge_dst_property(handle,
                               forward_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               edge_dst_components.mutable_view());

      std::vector<arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(arithmetic_device_uvector_t{std::move(tmp_components)});
      std::tie(tmp_vertices, vertex_properties) =
        shuffle_ext_vertices(handle, std::move(tmp_vertices), std::move(vertex_properties));
      tmp_components = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      renumber_local_ext_vertices<vertex_t, multi_gpu>(
        handle,
        tmp_vertices.data(),
        tmp_vertices.size(),
        inverse_renumber_map.data(),
        inverse_graph_view.local_vertex_partition_range_first(),
        inverse_graph_view.local_vertex_partition_range_last());
      thrust::sort_by_key(handle.get_thrust_policy(),
                          tmp_vertices.begin(),
                          tmp_vertices.end(),
                          tmp_components.begin());

      update_edge_src_property(handle,
                               inverse_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               inverse_edge_src_components.mutable_view());
      update_edge_dst_property(handle,
                               inverse_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               inverse_edge_dst_components.mutable_view());
    }

    auto edge_src_component_view = multi_gpu
                                     ? edge_src_components.view()
                                     : make_edge_src_property_view<vertex_t, vertex_t>(
                                         forward_graph_view,
                                         components.begin(),
                                         forward_graph_view.local_vertex_partition_range_size());
    auto edge_dst_component_view = multi_gpu
                                     ? edge_dst_components.view()
                                     : make_edge_dst_property_view<vertex_t, vertex_t>(
                                         forward_graph_view,
                                         components.begin(),
                                         forward_graph_view.local_vertex_partition_range_size());

    // FIXME: we may add edge_property_t constructor that takes an initial value; then we can just
    // use the forward_graph_view (without calling clear_edge_maks()) and avoid passing
    // edge_mask.view()
    forward_graph_view.clear_edge_mask();
    auto new_edge_mask = edge_property_t<edge_t, bool>(handle, forward_graph_view);
    cugraph::fill_edge_property(handle, forward_graph_view, new_edge_mask.mutable_view(), false);
    transform_e(
      handle,
      forward_graph_view,
      edge_src_component_view,
      edge_dst_component_view,
      edge_mask.view(),
      cuda::proclaim_return_type<bool>(
        [] __device__(auto src, auto dst, auto src_component, auto dst_component, auto valid) {
          return valid && (src_component == dst_component);
        }),
      new_edge_mask.mutable_view());
    edge_mask = std::move(new_edge_mask);
    forward_graph_view.attach_edge_mask(edge_mask.view());

    {
      std::optional<rmm::device_uvector<vertex_t>> inverse_components{std::nullopt};
      if constexpr (!multi_gpu) {
        inverse_components = rmm::device_uvector<vertex_t>(
          inverse_graph_view.local_vertex_partition_range_size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       inverse_renumber_map.begin(),
                       inverse_renumber_map.end(),
                       components.begin(),
                       inverse_components->begin());
      }

      auto inverse_edge_src_component_view =
        multi_gpu ? inverse_edge_src_components.view()
                  : make_edge_src_property_view<vertex_t, vertex_t>(
                      inverse_graph_view,
                      inverse_components->begin(),
                      inverse_graph_view.local_vertex_partition_range_size());
      auto inverse_edge_dst_component_view =
        multi_gpu ? inverse_edge_dst_components.view()
                  : make_edge_dst_property_view<vertex_t, vertex_t>(
                      inverse_graph_view,
                      inverse_components->begin(),
                      inverse_graph_view.local_vertex_partition_range_size());

      // FIXME: we may add edge_property_t constructor that takes an initial value; then we can
      // just use the inverse_graph_view (without calling clear_edge_maks()) and avoid passing
      // inverse_edge_mask.view()
      inverse_graph_view.clear_edge_mask();
      auto new_inverse_edge_mask = edge_property_t<edge_t, bool>(handle, inverse_graph_view);
      cugraph::fill_edge_property(
        handle, inverse_graph_view, new_inverse_edge_mask.mutable_view(), false);
      transform_e(
        handle,
        inverse_graph_view,
        inverse_edge_src_component_view,
        inverse_edge_dst_component_view,
        inverse_edge_mask.view(),
        cuda::proclaim_return_type<bool>(
          [] __device__(auto src, auto dst, auto src_component, auto dst_component, auto valid) {
            return valid && (src_component == dst_component);
          }),
        new_inverse_edge_mask.mutable_view());
      inverse_edge_mask = std::move(new_inverse_edge_mask);
      inverse_graph_view.attach_edge_mask(inverse_edge_mask.view());
    }

    // 6-4. update unresolved_component_offsets and unresolved_component_vertices

    rmm::device_uvector<vertex_t> component_counts(component_ids.size(), handle.get_stream());
    thrust::adjacent_difference(handle.get_thrust_policy(),
                                component_offsets.begin() + 1,
                                component_offsets.end(),
                                component_counts.begin());
    thrust::replace_if(handle.get_thrust_policy(),
                       component_counts.begin(),
                       component_counts.end(),
                       component_scc_flags.begin(),
                       cuda::proclaim_return_type<bool>([] __device__(bool flag) { return flag; }),
                       vertex_t{0});
    component_vertices.resize(
      cuda::std::distance(
        component_vertices.begin(),
        thrust::remove_if(
          handle.get_thrust_policy(),
          component_vertices.begin(),
          component_vertices.end(),
          cuda::make_transform_iterator(
            thrust::make_counting_iterator(size_t{0}),
            cuda::proclaim_return_type<bool>(
              [component_offsets   = raft::device_span<vertex_t const>(component_offsets.data(),
                                                                     component_offsets.size()),
               component_scc_flags = raft::device_span<bool const>(
                 component_scc_flags.data(), component_scc_flags.size())] __device__(size_t i) {
                auto idx = cuda::std::distance(
                  component_offsets.begin() + 1,
                  thrust::upper_bound(
                    thrust::seq, component_offsets.begin() + 1, component_offsets.end(), i));
                return component_scc_flags[idx];
              })),
          cuda::proclaim_return_type<bool>([] __device__(bool scc_flag) { return scc_flag; }))),
      handle.get_stream());
    component_vertices.shrink_to_fit(handle.get_stream());

    component_counts.resize(
      cuda::std::distance(component_counts.begin(),
                          thrust::remove_if(handle.get_thrust_policy(),
                                            component_counts.begin(),
                                            component_counts.end(),
                                            component_scc_flags.begin(),
                                            cuda::proclaim_return_type<bool>(
                                              [] __device__(bool scc_flag) { return scc_flag; }))),
      handle.get_stream());
    component_counts.shrink_to_fit(handle.get_stream());

    component_offsets.resize(component_counts.size() + 1, handle.get_stream());
    component_offsets.shrink_to_fit(handle.get_stream());
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           component_counts.begin(),
                           component_counts.end(),
                           component_offsets.begin() + 1);

    unresolved_component_offsets  = std::move(component_offsets);
    unresolved_component_vertices = std::move(component_vertices);
  }

  return;
}

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> strongly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> components(graph_view.local_vertex_partition_range_size(),
                                           handle.get_stream());
  strongly_connected_components_impl(
    handle,
    graph_view,
    raft::device_span<vertex_t>(components.data(), components.size()),
    do_expensive_check);

  return std::move(components);
}

}  // namespace cugraph
