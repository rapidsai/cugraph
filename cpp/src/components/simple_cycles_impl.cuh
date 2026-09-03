/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "utilities/validation_checks.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/extract_transform_if_e.cuh>
#include <cugraph/prims/kv_store.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>
#include <cugraph/utilities/thrust_wrappers/unique.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {

namespace detail {

// return std::tuple of cycle_vertices and cycle_sizes
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> simple_cycles_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  vertex_t length_bound,
  bool do_expensive_check)
{
  using weight_t     = float;    // dummy
  using edge_type_t  = int32_t;  // dummy
  using time_stamp_t = int64_t;  // dummy

  rmm::device_uvector<vertex_t> cycle_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> cycle_sizes(0, handle.get_stream());

  /* 1. check input arguments */

  CUGRAPH_EXPECTS(length_bound > 0,
                  "Invalid input argument: length_bound should be a positive integer.");

  if (do_expensive_check) {
    if (seed_vertices) {
      CUGRAPH_EXPECTS(cugraph::count_invalid_vertices(handle, graph_view, *seed_vertices) == 0,
                      "Invalid input argument: seed_vertices has invalid vertex IDs.");
      CUGRAPH_EXPECTS(
        thrust::is_sorted(
          handle.get_thrust_policy(), (*seed_vertices).begin(), (*seed_vertices).end()),
        "Invalid input argument: seed_vertices should be sorted in non-descending "
        "order.");
    }
    if constexpr (multi_gpu) {
      auto num_gpus_with_seed_vertices =
        host_scalar_allreduce(handle.get_comms(),
                              seed_vertices.has_value() ? int32_t{1} : int32_t{0},
                              raft::comms::op_t::SUM,
                              handle.get_stream());
      CUGRAPH_EXPECTS(
        (num_gpus_with_seed_vertices == 0) ||
          (num_gpus_with_seed_vertices == handle.get_comms().get_size()),
        "Invalid input argument: seed_vertices.has_value() should be the same on every GPU "
        "(pass an empty span instead of std::nullopt if this GPU has no seed vertices).");
      if (seed_vertices) {
        auto aggregate_num_seed_vertices = host_scalar_allreduce(
          handle.get_comms(), seed_vertices->size(), raft::comms::op_t::SUM, handle.get_stream());
        CUGRAPH_EXPECTS(aggregate_num_seed_vertices > size_t{0},
                        "Invalid input argument: if seed_vertices is provided, the aggregate "
                        "number of seed vertices should be greater than 0.");
      }
    } else if (seed_vertices) {
      CUGRAPH_EXPECTS(seed_vertices->size() > size_t{0},
                      "Invalid input argument: if seed_vertices is provided, the number of seed "
                      "vertices should be greater than 0.");
    }
  }

  /* 2. extract self-loops (length 1 simple cycles) */

  auto length_1_cycle_vertices = extract_transform_if_e(
    handle,
    graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_dummy_property_t{}.view(),
    cuda::proclaim_return_type<vertex_t>(
      [] __device__(vertex_t src, vertex_t, auto, auto, auto) { return src; }),
    cuda::proclaim_return_type<bool>(
      [] __device__(vertex_t src, vertex_t dst, auto, auto, auto) { return src == dst; }));
  cugraph::sort(
    handle.get_thrust_policy(), length_1_cycle_vertices.begin(), length_1_cycle_vertices.end());
  length_1_cycle_vertices.resize(
    cuda::std::distance(length_1_cycle_vertices.begin(),
                        cugraph::unique(handle.get_thrust_policy(),
                                        length_1_cycle_vertices.begin(),
                                        length_1_cycle_vertices.end())),
    handle.get_stream());
  if constexpr (multi_gpu) {
    std::tie(length_1_cycle_vertices, std::ignore) =
      shuffle_int_vertices(handle,
                           std::move(length_1_cycle_vertices),
                           std::vector<cugraph::arithmetic_device_uvector_t>{},
                           graph_view.vertex_partition_range_lasts());
    cugraph::sort(
      handle.get_thrust_policy(), length_1_cycle_vertices.begin(), length_1_cycle_vertices.end());
    length_1_cycle_vertices.resize(
      cuda::std::distance(length_1_cycle_vertices.begin(),
                          cugraph::unique(handle.get_thrust_policy(),
                                          length_1_cycle_vertices.begin(),
                                          length_1_cycle_vertices.end())),
      handle.get_stream());
  }

  if (seed_vertices) {
    length_1_cycle_vertices.resize(
      cuda::std::distance(
        length_1_cycle_vertices.begin(),
        thrust::remove_if(
          handle.get_thrust_policy(),
          length_1_cycle_vertices.begin(),
          length_1_cycle_vertices.end(),
          cuda::proclaim_return_type<bool>([seeds = *seed_vertices] __device__(vertex_t v) {
            return !thrust::binary_search(thrust::seq, seeds.begin(), seeds.end(), v);
          }))),
      handle.get_stream());
  }

  if (length_1_cycle_vertices.size() > 0) {
    cycle_sizes.resize(length_1_cycle_vertices.size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), cycle_sizes.begin(), cycle_sizes.end(), vertex_t{1});
    cycle_vertices = std::move(length_1_cycle_vertices);
  }

  if (length_bound == 1) {
    return std::make_tuple(std::move(cycle_vertices), std::move(cycle_sizes));
  }

  /* 3. find SCCs and compute the size of the component each vertex belongs to */

  auto components = strongly_connected_components(handle, graph_view, do_expensive_check);
  rmm::device_uvector<vertex_t> component_sizes(
    0, handle.get_stream());  // component_sizes[] are set to 0 for components with no seed vertices
  {
    rmm::device_uvector<vertex_t> tmp_components(components.size(), handle.get_stream());
    thrust::copy(
      handle.get_thrust_policy(), components.begin(), components.end(), tmp_components.begin());
    std::optional<rmm::device_uvector<bool>> tmp_seed_vertex_flags(std::nullopt);
    if (seed_vertices) {
      tmp_seed_vertex_flags = rmm::device_uvector<bool>(components.size(), handle.get_stream());
      thrust::transform(
        handle.get_thrust_policy(),
        tmp_components.begin(),
        tmp_components.end(),
        tmp_seed_vertex_flags->begin(),
        cuda::proclaim_return_type<bool>([seeds = *seed_vertices] __device__(vertex_t v) {
          return thrust::binary_search(thrust::seq, seeds.begin(), seeds.end(), v);
        }));
      thrust::sort_by_key(handle.get_thrust_policy(),
                          tmp_components.begin(),
                          tmp_components.end(),
                          tmp_seed_vertex_flags->begin());
    } else {
      cugraph::sort(handle.get_thrust_policy(), tmp_components.begin(), tmp_components.end());
    }
    auto num_unique_components = thrust::unique_count(
      handle.get_thrust_policy(), tmp_components.begin(), tmp_components.end());
    rmm::device_uvector<vertex_t> unique_components(num_unique_components, handle.get_stream());
    rmm::device_uvector<vertex_t> unique_component_sizes(num_unique_components,
                                                         handle.get_stream());
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          tmp_components.begin(),
                          tmp_components.end(),
                          cuda::make_constant_iterator(vertex_t{1}),
                          unique_components.begin(),
                          unique_component_sizes.begin());
    std::optional<rmm::device_uvector<vertex_t>> unique_component_seed_counts(std::nullopt);
    if (seed_vertices) {
      unique_component_seed_counts =
        rmm::device_uvector<vertex_t>(num_unique_components, handle.get_stream());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            tmp_components.begin(),
                            tmp_components.end(),
                            cuda::make_transform_iterator(
                              tmp_seed_vertex_flags->begin(),
                              cuda::proclaim_return_type<vertex_t>([] __device__(bool flag) {
                                return flag ? vertex_t{1} : vertex_t{0};
                              })),
                            thrust::make_discard_iterator(),
                            unique_component_seed_counts->begin());
    }
    if constexpr (multi_gpu) {
      std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(unique_component_sizes));
      if (seed_vertices) { vertex_properties.push_back(std::move(*unique_component_seed_counts)); }
      std::tie(unique_components, vertex_properties) =
        shuffle_ext_vertices(handle, std::move(unique_components), std::move(vertex_properties));
      unique_component_sizes =
        std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      if (seed_vertices) {
        unique_component_seed_counts =
          std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[1]));
      }

      if (seed_vertices) {
        thrust::sort_by_key(handle.get_thrust_policy(),
                            unique_components.begin(),
                            unique_components.end(),
                            thrust::make_zip_iterator(unique_component_sizes.begin(),
                                                      unique_component_seed_counts->begin()));
      } else {
        thrust::sort_by_key(handle.get_thrust_policy(),
                            unique_components.begin(),
                            unique_components.end(),
                            unique_component_sizes.begin());
      }
      num_unique_components = thrust::unique_count(
        handle.get_thrust_policy(), unique_components.begin(), unique_components.end());
      rmm::device_uvector<vertex_t> tmp_unique_components(num_unique_components,
                                                          handle.get_stream());
      rmm::device_uvector<vertex_t> tmp_unique_component_sizes(num_unique_components,
                                                               handle.get_stream());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            unique_components.begin(),
                            unique_components.end(),
                            unique_component_sizes.begin(),
                            tmp_unique_components.begin(),
                            tmp_unique_component_sizes.begin());
      std::optional<rmm::device_uvector<vertex_t>> tmp_unique_component_seed_counts(std::nullopt);
      if (seed_vertices) {
        tmp_unique_component_seed_counts =
          rmm::device_uvector<vertex_t>(num_unique_components, handle.get_stream());
        thrust::reduce_by_key(handle.get_thrust_policy(),
                              unique_components.begin(),
                              unique_components.end(),
                              unique_component_seed_counts->begin(),
                              thrust::make_discard_iterator(),
                              tmp_unique_component_seed_counts->begin());
      }
      unique_components            = std::move(tmp_unique_components);
      unique_component_sizes       = std::move(tmp_unique_component_sizes);
      unique_component_seed_counts = std::move(tmp_unique_component_seed_counts);
    }
    if (seed_vertices) {
      auto num_valid_components = static_cast<size_t>(
        thrust::count_if(handle.get_thrust_policy(),
                         unique_component_seed_counts->begin(),
                         unique_component_seed_counts->end(),
                         cuda::proclaim_return_type<bool>([] __device__(vertex_t seed_count) {
                           return seed_count > vertex_t{0};
                         })));
      rmm::device_uvector<vertex_t> tmp_unique_components(num_valid_components,
                                                          handle.get_stream());
      rmm::device_uvector<vertex_t> tmp_unique_component_sizes(num_valid_components,
                                                               handle.get_stream());
      thrust::copy_if(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(unique_components.begin(), unique_component_sizes.begin()),
        thrust::make_zip_iterator(unique_components.end(), unique_component_sizes.end()),
        unique_component_seed_counts->begin(),
        thrust::make_zip_iterator(tmp_unique_components.begin(),
                                  tmp_unique_component_sizes.begin()),
        cuda::proclaim_return_type<bool>(
          [] __device__(vertex_t seed_count) { return seed_count > vertex_t{0}; }));
      unique_components      = std::move(tmp_unique_components);
      unique_component_sizes = std::move(tmp_unique_component_sizes);
    }

    kv_store_t<vertex_t, vertex_t, true /* use_binary_search */> component_size_store(
      std::move(unique_components),
      std::move(unique_component_sizes),
      vertex_t{0},  // invalid_value (components with no seed vertices will be treated as size 0
                    // components)
      true,         // unique_components is already sorted
      handle.get_stream());
    auto component_size_store_view = component_size_store.view();
    if constexpr (multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto const major_comm_size =
        handle.get_subcomm(cugraph::partition_manager::major_comm_name()).get_size();
      auto const minor_comm_size =
        handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size();
      cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> key_to_gpu_id{
        comm_size, major_comm_size, minor_comm_size};
      component_sizes = collect_values_for_keys(
        handle, component_size_store_view, components.begin(), components.end(), key_to_gpu_id);
    } else {
      component_sizes.resize(components.size(), handle.get_stream());
      component_size_store_view.find(
        components.begin(), components.end(), component_sizes.begin(), handle.get_stream());
    }
  }

  /* 4. extract size 2 SCCs (length 2 simple cycles) */

  rmm::device_uvector<vertex_t> length_2_cycle_vertices(0, handle.get_stream());
  {
    auto num_length_2_cycle_vertices = static_cast<size_t>(thrust::count(
      handle.get_thrust_policy(), component_sizes.begin(), component_sizes.end(), vertex_t{2}));
    rmm::device_uvector<vertex_t> length_2_cycle_components(num_length_2_cycle_vertices,
                                                            handle.get_stream());
    length_2_cycle_vertices.resize(num_length_2_cycle_vertices, handle.get_stream());
    auto input_pair_first = thrust::make_zip_iterator(
      components.begin(),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()));
    thrust::copy_if(
      handle.get_thrust_policy(),
      input_pair_first,
      input_pair_first + components.size(),
      component_sizes.begin(),
      thrust::make_zip_iterator(length_2_cycle_components.begin(), length_2_cycle_vertices.begin()),
      cugraph::detail::is_equal_t{vertex_t{2}});
    if constexpr (multi_gpu) {
      std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(length_2_cycle_vertices));
      std::tie(length_2_cycle_components, vertex_properties) = shuffle_ext_vertices(
        handle, std::move(length_2_cycle_components), std::move(vertex_properties));
      length_2_cycle_vertices =
        std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
    }
    auto pair_first =
      thrust::make_zip_iterator(length_2_cycle_components.begin(), length_2_cycle_vertices.begin());
    cugraph::sort(
      handle.get_thrust_policy(), pair_first, pair_first + length_2_cycle_components.size());
  }

  if (length_2_cycle_vertices.size() > 0) {
    auto old_num_cycles = cycle_sizes.size();
    cycle_sizes.resize(old_num_cycles + length_2_cycle_vertices.size() / size_t{2},
                       handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 cycle_sizes.begin() + old_num_cycles,
                 cycle_sizes.end(),
                 vertex_t{2});
    cycle_vertices.resize(old_num_cycles + length_2_cycle_vertices.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 length_2_cycle_vertices.begin(),
                 length_2_cycle_vertices.end(),
                 cycle_vertices.begin() + old_num_cycles);
  }

  if (length_bound == 2) {
    return std::make_tuple(std::move(cycle_vertices), std::move(cycle_sizes));
  }

  /* 5. enumerate intra-SCC edges for SCCs with more than 2 vertices and create a new graph */

  cugraph::graph_t<vertex_t, edge_t, false, false> scc_graph(handle);
  rmm::device_uvector<vertex_t> scc_graph_renumber_map(0, handle.get_stream());
  {
    thrust::transform_if(
      handle.get_thrust_policy(),
      component_sizes.begin(),
      component_sizes.end(),
      components.begin(),
      cuda::proclaim_return_type<vertex_t>(
        [] __device__(vertex_t) { return invalid_vertex_id_v<vertex_t>; }),
      cuda::proclaim_return_type<bool>([] __device__(vertex_t size) {
        return size <= vertex_t{2};
      }));  // components with no seed vertices and size 1 & 2 components will be excluded

    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_components(0, handle.get_stream());
    auto e_op = cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t, vertex_t>>(
      [] __device__(vertex_t src, vertex_t dst, vertex_t src_component, auto, auto) {
        return cuda::std::make_tuple(src, dst, src_component);
      });
    auto pred_op = cuda::proclaim_return_type<bool>(
      [invalid_component = invalid_vertex_id_v<vertex_t>] __device__(
        vertex_t src, vertex_t dst, vertex_t src_component, vertex_t dst_component, auto) {
        return (src != dst) && (src_component != invalid_component) &&
               (src_component == dst_component);
      });
    if constexpr (multi_gpu) {
      edge_src_property_t<vertex_t, vertex_t> src_components(handle, graph_view);
      edge_dst_property_t<vertex_t, vertex_t> dst_components(handle, graph_view);
      update_edge_src_property(
        handle, graph_view, components.begin(), src_components.mutable_view());
      update_edge_dst_property(
        handle, graph_view, components.begin(), dst_components.mutable_view());
      std::tie(edgelist_srcs, edgelist_dsts, edgelist_components) =
        extract_transform_if_e(handle,
                               graph_view,
                               src_components.view(),
                               dst_components.view(),
                               edge_dummy_property_t{}.view(),
                               e_op,
                               pred_op);
      // FIXME: shuffling by hashing component IDs (shuffle_ext_vertices) can lead to load
      // imbalance; a more sophisticated mapping from component ID to rank is necessary.
      std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(edgelist_srcs));
      vertex_properties.push_back(std::move(edgelist_dsts));
      std::tie(edgelist_components, vertex_properties) =
        shuffle_ext_vertices(handle, std::move(edgelist_components), std::move(vertex_properties));
      edgelist_srcs = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      edgelist_dsts = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[1]));
    } else {
      std::tie(edgelist_srcs, edgelist_dsts, edgelist_components) =
        extract_transform_if_e(handle,
                               graph_view,
                               make_edge_src_property_view<vertex_t, vertex_t>(
                                 graph_view, components.begin(), components.size()),
                               make_edge_dst_property_view<vertex_t, vertex_t>(
                                 graph_view, components.begin(), components.size()),
                               edge_dummy_property_t{}.view(),
                               e_op,
                               pred_op);
    }
    std::tie(edgelist_srcs,
             edgelist_dsts,
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore) =
      cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t>(
        handle,
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt);
    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
    std::tie(scc_graph, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, false, false>(
        handle,
        std::nullopt,
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::vector<cugraph::arithmetic_device_uvector_t>{},
        graph_properties_t{false, false},
        true /* renumber */);
    scc_graph_renumber_map = std::move(*tmp_renumber_map);
  }

  /* 6. forgot to shuffle seed_vertices based on components */
  /* 7. enumerate simple cycles (from SCCs with 3+ vertices) */

  while (true) {
    auto scc_graph_view = scc_graph.view();
    auto components = cugraph::strongly_connected_components(handle, scc_graph_view, do_expensive_check);

    /* 6-1. check for size 2 SCCs */

    /* 6-2. check for size 3+ SCCs, if there are no size 3+ SCCs, break the loop */

    /* 6-3. pick one vertex from each SCC */

    /* 6-4. enumerate simple cycles including the picked vertex from each SCC */

    /* 6-5. mask out the edges to/from the picked vertices */
  }

  return std::make_tuple(std::move(cycle_vertices), std::move(cycle_sizes));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<size_t>> simple_cycles(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  vertex_t length_bound,
  bool do_expensive_check)
{
  auto [cycle_vertices, cycle_sizes] =
    detail::simple_cycles_impl(handle, graph_view, seed_vertices, length_bound, do_expensive_check);
  rmm::device_uvector<size_t> cycle_offsets(size_t{1}, handle.get_stream());
  cycle_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
  thrust::inclusive_scan(
    handle.get_thrust_policy(), cycle_sizes.begin(), cycle_sizes.end(), cycle_offsets.begin() + 1);
  return std::make_tuple(std::move(cycle_vertices), std::move(cycle_offsets));
}

}  // namespace cugraph
