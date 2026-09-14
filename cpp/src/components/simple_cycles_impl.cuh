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
#include <cugraph/prims/extract_transform_if_v_frontier_incoming_outgoing_e.cuh>
#include <cugraph/prims/fill_edge_src_dst_property.cuh>
#include <cugraph/prims/kv_store.cuh>
#include <cugraph/prims/make_initialized_edge_property.cuh>
#include <cugraph/prims/make_initialized_edge_src_dst_property.cuh>
#include <cugraph/prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/transform_e.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/scatter.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>
#include <cugraph/utilities/thrust_wrappers/unique.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace cugraph {

namespace detail {

template <typename vertex_t, bool multi_gpu, typename ValueIterator, typename ReduceOp>
std::tuple<rmm::device_uvector<vertex_t>,
           dataframe_buffer_type_t<typename thrust::iterator_traits<ValueIterator>::value_type>>
reduce_by_component(raft::handle_t const& handle,
                    raft::device_span<vertex_t const> components,
                    ValueIterator value_first,
                    ReduceOp reduce_op)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;
  static_assert(std::is_arithmetic_v<value_t> || is_thrust_tuple_of_arithmetic_v<value_t>);

  rmm::device_uvector<vertex_t> tmp_components(components.size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), components.begin(), components.end(), tmp_components.begin());
  auto tmp_values = allocate_dataframe_buffer<value_t>(components.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               value_first,
               value_first + components.size(),
               get_dataframe_buffer_begin(tmp_values));
  thrust::sort_by_key(handle.get_thrust_policy(),
                      tmp_components.begin(),
                      tmp_components.end(),
                      get_dataframe_buffer_begin(tmp_values));
  auto num_unique_components =
    thrust::unique_count(handle.get_thrust_policy(), tmp_components.begin(), tmp_components.end());
  rmm::device_uvector<vertex_t> unique_components(num_unique_components, handle.get_stream());
  auto unique_values =
    allocate_dataframe_buffer<value_t>(num_unique_components, handle.get_stream());
  thrust::reduce_by_key(handle.get_thrust_policy(),
                        tmp_components.begin(),
                        tmp_components.end(),
                        get_dataframe_buffer_begin(tmp_values),
                        unique_components.begin(),
                        get_dataframe_buffer_begin(unique_values),
                        cuda::std::equal_to<vertex_t>{},
                        reduce_op);
  if constexpr (multi_gpu) {
    std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
    if constexpr (std::is_arithmetic_v<value_t>) {
      vertex_properties.push_back(std::move(unique_values));
    } else {
      std::apply([&vertex_properties](
                   auto&&... args) { (vertex_properties.push_back(std::move(args)), ...); },
                 unique_values);
    }
    std::tie(unique_components, vertex_properties) =
      shuffle_ext_vertices(handle, std::move(unique_components), std::move(vertex_properties));
    if constexpr (std::is_arithmetic_v<value_t>) {
      unique_values = std::move(std::get<rmm::device_uvector<value_t>>(vertex_properties[0]));
    } else {
      size_t i = 0;
      std::apply(
        [&vertex_properties, &i](auto&... args) {
          ((args = std::move(std::get<std::decay_t<decltype(args)>>(vertex_properties[i++])), ...));
        },
        unique_values);
    }
    thrust::sort_by_key(handle.get_thrust_policy(),
                        unique_components.begin(),
                        unique_components.end(),
                        get_dataframe_buffer_begin(unique_values));
    num_unique_components = thrust::unique_count(
      handle.get_thrust_policy(), unique_components.begin(), unique_components.end());
    rmm::device_uvector<vertex_t> tmp_unique_components(num_unique_components, handle.get_stream());
    auto tmp_unique_values =
      allocate_dataframe_buffer<value_t>(num_unique_components, handle.get_stream());
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          unique_components.begin(),
                          unique_components.end(),
                          get_dataframe_buffer_begin(unique_values),
                          tmp_unique_components.begin(),
                          get_dataframe_buffer_begin(tmp_unique_values),
                          cuda::std::equal_to<vertex_t>{},
                          reduce_op);
    unique_components = std::move(tmp_unique_components);
    unique_values     = std::move(tmp_unique_values);
  }
  return std::make_tuple(std::move(unique_components), std::move(unique_values));
}

// Per-vertex global SCC sizes. If @p seed_vertices is valid, sizes of components with no seed
// vertex are set to 0.
template <typename vertex_t, bool multi_gpu>
rmm::device_uvector<vertex_t> compute_component_sizes(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> components,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  vertex_t local_vertex_partition_range_first)
{
  rmm::device_uvector<vertex_t> unique_components(0, handle.get_stream());
  rmm::device_uvector<vertex_t> unique_component_sizes(0, handle.get_stream());
  if (seed_vertices) {
    auto value_first = cuda::make_transform_iterator(
      cuda::make_counting_iterator(local_vertex_partition_range_first),
      cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t>>(
        [seeds = *seed_vertices] __device__(vertex_t v) {
          return cuda::std::make_tuple(
            vertex_t{1},
            thrust::binary_search(thrust::seq, seeds.begin(), seeds.end(), v) ? vertex_t{1}
                                                                              : vertex_t{0});
        }));
    auto plus_pair = cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t>>(
      [] __device__(auto lhs, auto rhs) {
        return cuda::std::make_tuple(cuda::std::get<0>(lhs) + cuda::std::get<0>(rhs),
                                     cuda::std::get<1>(lhs) + cuda::std::get<1>(rhs));
      });
    auto [tmp_unique_components, tmp_unique_values] =
      reduce_by_component<vertex_t, multi_gpu>(handle, components, value_first, plus_pair);
    unique_components                 = std::move(tmp_unique_components);
    unique_component_sizes            = std::move(std::get<0>(tmp_unique_values));
    auto unique_component_seed_counts = std::move(std::get<1>(tmp_unique_values));
    auto num_valid_components         = static_cast<size_t>(
      thrust::count_if(handle.get_thrust_policy(),
                       unique_component_seed_counts.begin(),
                       unique_component_seed_counts.end(),
                       cuda::proclaim_return_type<bool>(
                         [] __device__(vertex_t seed_count) { return seed_count > vertex_t{0}; })));
    rmm::device_uvector<vertex_t> valid_unique_components(num_valid_components,
                                                          handle.get_stream());
    rmm::device_uvector<vertex_t> valid_unique_component_sizes(num_valid_components,
                                                               handle.get_stream());
    thrust::copy_if(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(unique_components.begin(), unique_component_sizes.begin()),
      thrust::make_zip_iterator(unique_components.end(), unique_component_sizes.end()),
      unique_component_seed_counts.begin(),
      thrust::make_zip_iterator(valid_unique_components.begin(),
                                valid_unique_component_sizes.begin()),
      cuda::proclaim_return_type<bool>(
        [] __device__(vertex_t seed_count) { return seed_count > vertex_t{0}; }));
    unique_components      = std::move(valid_unique_components);
    unique_component_sizes = std::move(valid_unique_component_sizes);
  } else {
    std::tie(unique_components, unique_component_sizes) = reduce_by_component<vertex_t, multi_gpu>(
      handle, components, cuda::make_constant_iterator(vertex_t{1}), thrust::plus<vertex_t>{});
  }

  kv_store_t<vertex_t, vertex_t, true /* use_binary_search */> component_size_store(
    std::move(unique_components),
    std::move(unique_component_sizes),
    vertex_t{0},  // invalid_value (components with no seed vertices will be treated as size 0
                  // components)
    true,         // unique_components is already sorted
    handle.get_stream());
  auto component_size_store_view = component_size_store.view();
  rmm::device_uvector<vertex_t> component_sizes(0, handle.get_stream());
  if constexpr (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const major_comm_size =
      handle.get_subcomm(cugraph::partition_manager::major_comm_name()).get_size();
    auto const minor_comm_size =
      handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size();
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t> key_to_gpu_id{
      comm_size, major_comm_size, minor_comm_size};
    component_sizes = cugraph::detail::collect_values_for_keys(handle.get_comms(),
                                                               component_size_store_view,
                                                               components.begin(),
                                                               components.end(),
                                                               key_to_gpu_id,
                                                               handle.get_stream());
  } else {
    component_sizes.resize(components.size(), handle.get_stream());
    component_size_store_view.find(
      components.begin(), components.end(), component_sizes.begin(), handle.get_stream());
  }

  return component_sizes;
}

// returnied vector size is # length 2 cycles * 2 (two vertices per cycle)
template <typename vertex_t, bool multi_gpu>
rmm::device_uvector<vertex_t> extract_length_2_cycle_vertices(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> components,
  raft::device_span<vertex_t const> component_sizes,
  vertex_t local_vertex_partition_range_first)
{
  auto num_length_2_cycle_vertices = static_cast<size_t>(thrust::count(
    handle.get_thrust_policy(), component_sizes.begin(), component_sizes.end(), vertex_t{2}));
  rmm::device_uvector<vertex_t> length_2_cycle_components(num_length_2_cycle_vertices,
                                                          handle.get_stream());
  rmm::device_uvector<vertex_t> length_2_cycle_vertices(num_length_2_cycle_vertices,
                                                        handle.get_stream());
  auto input_pair_first = thrust::make_zip_iterator(
    components.begin(), cuda::make_counting_iterator(local_vertex_partition_range_first));
  thrust::copy_if(
    handle.get_thrust_policy(),
    input_pair_first,
    input_pair_first + components.size(),
    component_sizes.begin(),
    thrust::make_zip_iterator(length_2_cycle_components.begin(), length_2_cycle_vertices.begin()),
    cugraph::detail::is_equal_to_t{vertex_t{2}});
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

  return length_2_cycle_vertices;
}

// Map seed vertices from the unrenumbered vertex space of @p renumber_map into the internal vertex
// space of the renumbered graph. Seeds that do not appear in @p renumber_map are dropped. The
// returned vector is sorted.
template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> renumber_seed_vertices(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> seed_vertices,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> renumber_map,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> tmp_seed_vertices(seed_vertices.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               seed_vertices.begin(),
               seed_vertices.end(),
               tmp_seed_vertices.begin());
  if constexpr (multi_gpu) {
    std::tie(tmp_seed_vertices, std::ignore) = shuffle_ext_vertices(
      handle, std::move(tmp_seed_vertices), std::vector<cugraph::arithmetic_device_uvector_t>{});
  }
  rmm::device_uvector<vertex_t> sorted_renumber_map(renumber_map.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               renumber_map.begin(),
               renumber_map.end(),
               sorted_renumber_map.begin());
  cugraph::sort(handle.get_thrust_policy(), sorted_renumber_map.begin(), sorted_renumber_map.end());
  tmp_seed_vertices.resize(
    cuda::std::distance(
      tmp_seed_vertices.begin(),
      thrust::remove_if(
        handle.get_thrust_policy(),
        tmp_seed_vertices.begin(),
        tmp_seed_vertices.end(),
        cuda::proclaim_return_type<bool>(
          [sorted_renumber_map = raft::device_span<vertex_t const>(
             sorted_renumber_map.data(), sorted_renumber_map.size())] __device__(vertex_t v) {
            return !thrust::binary_search(
              thrust::seq, sorted_renumber_map.begin(), sorted_renumber_map.end(), v);
          }))),
    handle.get_stream());
  tmp_seed_vertices.shrink_to_fit(handle.get_stream());
  cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
    handle,
    tmp_seed_vertices.data(),
    tmp_seed_vertices.size(),
    renumber_map.data(),
    graph_view.local_vertex_partition_range_first(),
    graph_view.local_vertex_partition_range_last(),
    do_expensive_check);
  cugraph::sort(handle.get_thrust_policy(), tmp_seed_vertices.begin(), tmp_seed_vertices.end());
  return tmp_seed_vertices;
}

// Per-vertex in-degree * out-degree (and the global sum of those products).
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<float>, double> compute_in_out_degree_products(
  raft::handle_t const& handle, graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view)
{
  auto in_degrees  = graph_view.compute_in_degrees(handle);
  auto out_degrees = graph_view.compute_out_degrees(handle);
  rmm::device_uvector<float> in_out_degree_products(in_degrees.size(), handle.get_stream());
  thrust::transform(
    handle.get_thrust_policy(),
    in_degrees.begin(),
    in_degrees.end(),
    out_degrees.begin(),
    in_out_degree_products.begin(),
    cuda::proclaim_return_type<float>([] __device__(edge_t in_degree, edge_t out_degree) {
      return static_cast<float>(in_degree) * static_cast<float>(out_degree);
    }));
  auto sum_in_out_degree_products =
    static_cast<double>(thrust::reduce(handle.get_thrust_policy(),
                                       in_out_degree_products.begin(),
                                       in_out_degree_products.end(),
                                       float{0.0},
                                       thrust::plus<float>{}));
  if constexpr (multi_gpu) {
    sum_in_out_degree_products = host_scalar_allreduce(
      handle.get_comms(), sum_in_out_degree_products, raft::comms::op_t::SUM, handle.get_stream());
  }
  return std::make_tuple(std::move(in_out_degree_products), sum_in_out_degree_products);
}

template <typename T>
rmm::device_uvector<T> concatenate(raft::handle_t const& handle,
                                   std::vector<rmm::device_uvector<T>>&& inputs)
{
  if (inputs.size() == 0) { return rmm::device_uvector<T>(0, handle.get_stream()); }
  if (inputs.size() == 1) { return std::move(inputs[0]); }

  size_t tot_size{0};
  for (auto const& input : inputs) {
    tot_size += input.size();
  }
  rmm::device_uvector<T> output(tot_size, handle.get_stream());
  size_t offset{0};
  for (auto& input : inputs) {
    thrust::copy(handle.get_thrust_policy(), input.begin(), input.end(), output.begin() + offset);
    offset += input.size();
  }
  return output;
}

template <typename vertex_t, bool multi_gpu>
std::tuple<std::vector<rmm::device_uvector<vertex_t>>, std::vector<rmm::device_uvector<vertex_t>>>
push_back_cycles(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<vertex_t>>&& cycle_vertex_chunks,
  std::vector<rmm::device_uvector<vertex_t>>&& cycle_length_chunks,
  rmm::device_uvector<vertex_t>&& new_cycle_vertices,
  std::variant<vertex_t, rmm::device_uvector<vertex_t>>&& new_cycle_lengths,
  std::optional<std::tuple<raft::device_span<vertex_t const>, raft::host_span<vertex_t const>>>
    renumber_info = std::nullopt)
{
  if (renumber_info) {
    auto [renumber_map, vertex_partition_range_lasts] = *renumber_info;
    unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                 new_cycle_vertices.data(),
                                                 new_cycle_vertices.size(),
                                                 renumber_map.data(),
                                                 vertex_partition_range_lasts);  // collective
  }

  if (new_cycle_vertices.size() == 0) {
    return std::make_tuple(std::move(cycle_vertex_chunks), std::move(cycle_length_chunks));
  }

  if (std::holds_alternative<vertex_t>(new_cycle_lengths)) {
    auto length = std::get<vertex_t>(new_cycle_lengths);
    rmm::device_uvector<vertex_t> lengths(new_cycle_vertices.size() / static_cast<size_t>(length),
                                          handle.get_stream());
    cugraph::fill(handle.get_thrust_policy(), lengths.begin(), lengths.end(), length);
    cycle_length_chunks.push_back(std::move(lengths));
  } else {
    cycle_length_chunks.push_back(
      std::move(std::get<rmm::device_uvector<vertex_t>>(new_cycle_lengths)));
  }
  cycle_vertex_chunks.push_back(std::move(new_cycle_vertices));
  return std::make_tuple(std::move(cycle_vertex_chunks), std::move(cycle_length_chunks));
}

// Expand @p path_vertices (packed paths of length @p start_path_length) up to @p length_bound.
// Stops early if the number of live paths exceeds @p max_aggregate_paths. Returns the cycles found,
// their lengths, the leftover path vertices, and the length of those leftover paths (std::nullopt
// if enumeration finished: length_bound reached, no remaining paths, or no remaining extensions).
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<vertex_t>>
enumerate_simple_cycles_expanding_paths(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> reverse_distances,
  edge_dst_property_t<vertex_t, vertex_t> const& dst_reverse_distances,
  rmm::device_uvector<vertex_t>&& path_vertices,
  vertex_t start_path_length,
  vertex_t length_bound,
  size_t max_aggregate_paths)
{
  std::vector<rmm::device_uvector<vertex_t>> cycle_vertex_chunks{};
  std::vector<rmm::device_uvector<vertex_t>> cycle_length_chunks{};
  std::optional<vertex_t> next_path_length{std::nullopt};

  for (vertex_t path_length = start_path_length; path_length < length_bound; ++path_length) {
    auto path_count = path_vertices.size() / static_cast<size_t>(path_length);

    std::vector<size_t> path_idx_lasts{};
    size_t path_idx_start_offset{0};
    if constexpr (multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      path_idx_lasts = host_scalar_allgather(handle.get_comms(), path_count, handle.get_stream());
      std::inclusive_scan(path_idx_lasts.begin(), path_idx_lasts.end(), path_idx_lasts.begin());
      path_idx_start_offset = path_idx_lasts[comm_rank] - path_count;
    } else {
      path_idx_lasts = {path_count};
    }
    if (path_idx_lasts.back() == 0) { break; }

    rmm::device_uvector<vertex_t> last_vs(path_count, handle.get_stream());
    rmm::device_uvector<size_t> path_idxs(path_count, handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      last_vs.begin(),
      last_vs.end(),
      cuda::proclaim_return_type<vertex_t>([path_vertices = raft::device_span<vertex_t const>(
                                              path_vertices.data(), path_vertices.size()),
                                            path_length] __device__(size_t i) {
        return path_vertices[i * static_cast<size_t>(path_length) +
                             static_cast<size_t>(path_length - vertex_t{1})];
      }));
    cugraph::sequence(
      handle.get_thrust_policy(), path_idxs.begin(), path_idxs.end(), path_idx_start_offset);
    auto key_first = thrust::make_zip_iterator(last_vs.begin(), path_idxs.begin());
    cugraph::sort(handle.get_thrust_policy(), key_first, key_first + path_count);

    auto frontier = key_bucket_view_t<vertex_t, size_t, multi_gpu, true>(
      handle,
      raft::device_span<vertex_t const>(last_vs.data(), last_vs.size()),
      raft::device_span<size_t const>(path_idxs.data(), path_idxs.size()));

    auto e_op = cuda::proclaim_return_type<cuda::std::tuple<vertex_t, size_t>>(
      [] __device__(auto tagged_src, vertex_t dst, auto, auto, auto) {
        auto path_idx = cuda::std::get<1>(tagged_src);
        return cuda::std::make_tuple(dst, path_idx);
      });
    auto pred_op =
      cuda::proclaim_return_type<bool>([path_vertices = raft::device_span<vertex_t const>(
                                          path_vertices.data(), path_vertices.size()),
                                        path_idx_start_offset,
                                        path_length,
                                        length_bound] __device__(auto tagged_src,
                                                                 vertex_t dst,
                                                                 auto,
                                                                 vertex_t dst_reverse_distance,
                                                                 auto) {
        if (dst_reverse_distance == vertex_t{0}) {
          return true;
        }  // a path never leaves the SCC of its root vertex and the root is the only vertex with
           // reverse distance 0 in the SCC, so dst is the root of this path (closing a cycle).
           // Self-loops were already removed, so this cannot be a length 1 cycle.
        if ((dst_reverse_distance == std::numeric_limits<vertex_t>::max()) ||
            (path_length + dst_reverse_distance > length_bound)) {
          return false;
        }
        if constexpr (multi_gpu) {
          // this GPU may not store the path, whether dst is already on the path is checked
          // after shuffling the results to the GPU storing the path
          return true;
        } else {
          auto local_path_idx = cuda::std::get<1>(tagged_src) - path_idx_start_offset;
          auto path = path_vertices.data() + local_path_idx * static_cast<size_t>(path_length);
          for (vertex_t i = vertex_t{1}; i < path_length; ++i) {  // path[0] (root) is checked above
            if (path[i] == dst) { return false; }
          }
          return true;
        }
      });

    rmm::device_uvector<vertex_t> nbrs(0, handle.get_stream());
    rmm::device_uvector<size_t> nbr_path_idxs(0, handle.get_stream());
    if constexpr (multi_gpu) {
      std::tie(nbrs, nbr_path_idxs) =
        extract_transform_if_v_frontier_outgoing_e(handle,
                                                   graph_view,
                                                   frontier,
                                                   edge_src_dummy_property_t{}.view(),
                                                   dst_reverse_distances.view(),
                                                   edge_dummy_property_t{}.view(),
                                                   e_op,
                                                   pred_op);
    } else {
      std::tie(nbrs, nbr_path_idxs) = extract_transform_if_v_frontier_outgoing_e(
        handle,
        graph_view,
        frontier,
        edge_src_dummy_property_t{}.view(),
        make_edge_dst_property_view<vertex_t, vertex_t>(
          graph_view, reverse_distances.begin(), reverse_distances.size()),
        edge_dummy_property_t{}.view(),
        e_op,
        pred_op);
    }

    if constexpr (multi_gpu) {
      // shuffle the results to the GPUs storing the paths

      {
        rmm::device_uvector<size_t> d_path_idx_lasts(path_idx_lasts.size(), handle.get_stream());
        raft::update_device(d_path_idx_lasts.data(),
                            path_idx_lasts.data(),
                            path_idx_lasts.size(),
                            handle.get_stream());
        auto pair_first = thrust::make_zip_iterator(nbrs.begin(), nbr_path_idxs.begin());
        std::forward_as_tuple(std::tie(nbrs, nbr_path_idxs), std::ignore) =
          groupby_gpu_id_and_shuffle_values(
            handle.get_comms(),
            pair_first,
            pair_first + nbrs.size(),
            cuda::proclaim_return_type<int>(
              [lasts = raft::device_span<size_t const>(
                 d_path_idx_lasts.data(), d_path_idx_lasts.size())] __device__(auto pair) {
                auto path_idx = cuda::std::get<1>(pair);
                return static_cast<int>(cuda::std::distance(
                  lasts.begin(),
                  thrust::upper_bound(thrust::seq, lasts.begin(), lasts.end(), path_idx)));
              }),
            handle.get_stream());
      }

      // drop the extensions to the vertices already on the path (this check is skipped in
      // pred_op as the GPUs processing path extensions may not store the paths)

      {
        auto pair_first = thrust::make_zip_iterator(nbrs.begin(), nbr_path_idxs.begin());
        auto pair_last  = thrust::remove_if(
          handle.get_thrust_policy(),
          pair_first,
          pair_first + nbrs.size(),
          cuda::proclaim_return_type<bool>(
            [path_vertices =
               raft::device_span<vertex_t const>(path_vertices.data(), path_vertices.size()),
             path_idx_start_offset,
             path_length] __device__(cuda::std::tuple<vertex_t, size_t> pair) {
              auto dst            = cuda::std::get<0>(pair);
              auto local_path_idx = cuda::std::get<1>(pair) - path_idx_start_offset;
              auto path = path_vertices.data() + local_path_idx * static_cast<size_t>(path_length);
              if (dst == path[0]) { return false; }  // dst closes a cycle
              for (vertex_t i = vertex_t{1}; i < path_length; ++i) {
                if (path[i] == dst) { return true; }
              }
              return false;
            }));
        nbrs.resize(cuda::std::distance(pair_first, pair_last), handle.get_stream());
        nbr_path_idxs.resize(nbrs.size(), handle.get_stream());
      }
    }

    auto pair_first  = thrust::make_zip_iterator(nbrs.begin(), nbr_path_idxs.begin());
    auto cycle_first = thrust::partition(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + nbrs.size(),
      cuda::proclaim_return_type<bool>(
        [path_vertices =
           raft::device_span<vertex_t const>(path_vertices.data(), path_vertices.size()),
         path_idx_start_offset,
         path_length] __device__(cuda::std::tuple<vertex_t, size_t> pair) {
          auto dst            = cuda::std::get<0>(pair);
          auto local_path_idx = cuda::std::get<1>(pair) - path_idx_start_offset;
          return dst != path_vertices[local_path_idx * static_cast<size_t>(path_length)];
        }));
    auto num_extensions = static_cast<size_t>(cuda::std::distance(pair_first, cycle_first));
    auto num_cycles     = nbrs.size() - num_extensions;

    rmm::device_uvector<vertex_t> new_cycle_vertices(num_cycles * static_cast<size_t>(path_length),
                                                     handle.get_stream());
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_cycles),
      [path_vertices =
         raft::device_span<vertex_t const>(path_vertices.data(), path_vertices.size()),
       nbr_path_idxs =
         raft::device_span<size_t const>(nbr_path_idxs.data() + num_extensions, num_cycles),
       new_cycle_vertices =
         raft::device_span<vertex_t>(new_cycle_vertices.data(), new_cycle_vertices.size()),
       path_idx_start_offset,
       path_length] __device__(size_t i) {
        auto local_path_idx = nbr_path_idxs[i] - path_idx_start_offset;
        auto input_offset   = local_path_idx * static_cast<size_t>(path_length);
        auto output_offset  = i * static_cast<size_t>(path_length);
        thrust::copy(thrust::seq,
                     path_vertices.begin() + input_offset,
                     path_vertices.begin() + input_offset + static_cast<size_t>(path_length),
                     new_cycle_vertices.begin() + output_offset);
      });
    std::tie(cycle_vertex_chunks, cycle_length_chunks) =
      push_back_cycles<vertex_t, multi_gpu>(handle,
                                            std::move(cycle_vertex_chunks),
                                            std::move(cycle_length_chunks),
                                            std::move(new_cycle_vertices),
                                            path_length);
    nbrs.resize(num_extensions, handle.get_stream());
    nbr_path_idxs.resize(num_extensions, handle.get_stream());
    nbrs.shrink_to_fit(handle.get_stream());
    nbr_path_idxs.shrink_to_fit(handle.get_stream());

    auto aggregate_num_extensions = num_extensions;
    if constexpr (multi_gpu) {
      aggregate_num_extensions = host_scalar_allreduce(
        handle.get_comms(), num_extensions, raft::comms::op_t::SUM, handle.get_stream());
    }
    if (aggregate_num_extensions == 0) {
      path_vertices.resize(0, handle.get_stream());
      path_vertices.shrink_to_fit(handle.get_stream());
      break;
    }

    // create the extended paths, an extended path should be stored in the GPU owning its last
    // vertex (the frontier keys should be local to each GPU)

    rmm::device_uvector<vertex_t> next_path_vertices(0, handle.get_stream());
    if constexpr (multi_gpu) {
      // shuffle the extensions to the GPUs owning the new last vertices

      {
        std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
        vertex_properties.push_back(std::move(nbr_path_idxs));
        std::tie(nbrs, vertex_properties) =
          shuffle_int_vertices(handle,
                               std::move(nbrs),
                               std::move(vertex_properties),
                               graph_view.vertex_partition_range_lasts());
        nbr_path_idxs = std::move(std::get<rmm::device_uvector<size_t>>(vertex_properties[0]));
      }

      // collect the paths for the received path indices from the GPUs storing the paths (a path may
      // be extended to multiple neighbors, so query the unique path indices to avoid communicating
      // the same path multiple times)

      rmm::device_uvector<size_t> unique_path_idxs(nbr_path_idxs.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> unique_path_vertices(0, handle.get_stream());
      {
        thrust::copy(handle.get_thrust_policy(),
                     nbr_path_idxs.begin(),
                     nbr_path_idxs.end(),
                     unique_path_idxs.begin());
        cugraph::sort(handle.get_thrust_policy(), unique_path_idxs.begin(), unique_path_idxs.end());
        unique_path_idxs.resize(cuda::std::distance(unique_path_idxs.begin(),
                                                    cugraph::unique(handle.get_thrust_policy(),
                                                                    unique_path_idxs.begin(),
                                                                    unique_path_idxs.end())),
                                handle.get_stream());
        unique_path_idxs.shrink_to_fit(handle.get_stream());

        // path indices are assigned to GPUs in contiguous blocks in the increasing order of GPU
        // ranks, so the sorted unique path indices are already grouped by the GPUs storing the
        // paths

        std::vector<size_t> tx_counts(path_idx_lasts.size());
        {
          rmm::device_uvector<size_t> d_path_idx_lasts(path_idx_lasts.size(), handle.get_stream());
          raft::update_device(d_path_idx_lasts.data(),
                              path_idx_lasts.data(),
                              path_idx_lasts.size(),
                              handle.get_stream());
          rmm::device_uvector<size_t> d_tx_lasts(d_path_idx_lasts.size(), handle.get_stream());
          thrust::lower_bound(handle.get_thrust_policy(),
                              unique_path_idxs.begin(),
                              unique_path_idxs.end(),
                              d_path_idx_lasts.begin(),
                              d_path_idx_lasts.end(),
                              d_tx_lasts.begin());
          std::vector<size_t> tx_lasts(d_tx_lasts.size());
          raft::update_host(
            tx_lasts.data(), d_tx_lasts.data(), d_tx_lasts.size(), handle.get_stream());
          handle.sync_stream();
          std::adjacent_difference(tx_lasts.begin(), tx_lasts.end(), tx_counts.begin());
        }

        rmm::device_uvector<size_t> rx_path_idxs(0, handle.get_stream());
        std::vector<size_t> rx_counts{};
        std::tie(rx_path_idxs, rx_counts) =
          shuffle_values(handle.get_comms(),
                         unique_path_idxs.begin(),
                         raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                         handle.get_stream());

        {
          rmm::device_uvector<vertex_t> tx_path_vertices(
            rx_path_idxs.size() * static_cast<size_t>(path_length), handle.get_stream());
          thrust::tabulate(
            handle.get_thrust_policy(),
            tx_path_vertices.begin(),
            tx_path_vertices.end(),
            cuda::proclaim_return_type<vertex_t>([path_vertices = raft::device_span<vertex_t const>(
                                                    path_vertices.data(), path_vertices.size()),
                                                  rx_path_idxs = raft::device_span<size_t const>(
                                                    rx_path_idxs.data(), rx_path_idxs.size()),
                                                  path_idx_start_offset,
                                                  path_length] __device__(size_t i) {
              auto local_path_idx =
                rx_path_idxs[i / static_cast<size_t>(path_length)] - path_idx_start_offset;
              return path_vertices[local_path_idx * static_cast<size_t>(path_length) +
                                   i % static_cast<size_t>(path_length)];
            }));
          rx_path_idxs.resize(0, handle.get_stream());
          rx_path_idxs.shrink_to_fit(handle.get_stream());
          for (auto& count : rx_counts) {  // reply with path_length vertices per path index
            count *= static_cast<size_t>(path_length);
          }
          std::tie(unique_path_vertices, std::ignore) =
            shuffle_values(handle.get_comms(),
                           tx_path_vertices.begin(),
                           raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                           handle.get_stream());
          // unique_path_vertices stores the path for unique_path_idxs[i] in
          // [i * path_length, (i + 1) * path_length)
        }
      }

      // create the extended paths

      next_path_vertices.resize(nbrs.size() * static_cast<size_t>(path_length + vertex_t{1}),
                                handle.get_stream());
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(nbrs.size()),
        [unique_path_idxs =
           raft::device_span<size_t const>(unique_path_idxs.data(), unique_path_idxs.size()),
         unique_path_vertices = raft::device_span<vertex_t const>(unique_path_vertices.data(),
                                                                  unique_path_vertices.size()),
         nbrs                 = raft::device_span<vertex_t const>(nbrs.data(), nbrs.size()),
         nbr_path_idxs =
           raft::device_span<size_t const>(nbr_path_idxs.data(), nbr_path_idxs.size()),
         next_path_vertices =
           raft::device_span<vertex_t>(next_path_vertices.data(), next_path_vertices.size()),
         path_length] __device__(size_t i) {
          auto it = thrust::lower_bound(
            thrust::seq, unique_path_idxs.begin(), unique_path_idxs.end(), nbr_path_idxs[i]);
          auto input_offset =
            static_cast<size_t>(cuda::std::distance(unique_path_idxs.begin(), it)) *
            static_cast<size_t>(path_length);
          auto output_offset = i * static_cast<size_t>(path_length + vertex_t{1});
          thrust::copy(
            thrust::seq,
            unique_path_vertices.begin() + input_offset,
            unique_path_vertices.begin() + input_offset + static_cast<size_t>(path_length),
            next_path_vertices.begin() + output_offset);
          next_path_vertices[output_offset + static_cast<size_t>(path_length)] = nbrs[i];
        });
    } else {
      next_path_vertices.resize(num_extensions * static_cast<size_t>(path_length + vertex_t{1}),
                                handle.get_stream());
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(num_extensions),
        [path_vertices =
           raft::device_span<vertex_t const>(path_vertices.data(), path_vertices.size()),
         nbr_path_idxs =
           raft::device_span<size_t const>(nbr_path_idxs.data(), nbr_path_idxs.size()),
         nbrs = raft::device_span<vertex_t const>(nbrs.data(), nbrs.size()),
         next_path_vertices =
           raft::device_span<vertex_t>(next_path_vertices.data(), next_path_vertices.size()),
         path_length] __device__(size_t i) {
          auto input_offset  = nbr_path_idxs[i] * static_cast<size_t>(path_length);
          auto output_offset = i * static_cast<size_t>(path_length + vertex_t{1});
          thrust::copy(thrust::seq,
                       path_vertices.begin() + input_offset,
                       path_vertices.begin() + input_offset + static_cast<size_t>(path_length),
                       next_path_vertices.begin() + output_offset);
          next_path_vertices[output_offset + static_cast<size_t>(path_length)] = nbrs[i];
        });
    }

    path_vertices  = std::move(next_path_vertices);
    auto num_paths = path_vertices.size() / static_cast<size_t>(path_length + vertex_t{1});
    auto aggregate_num_paths = num_paths;
    if constexpr (multi_gpu) {
      aggregate_num_paths = host_scalar_allreduce(
        handle.get_comms(), aggregate_num_paths, raft::comms::op_t::SUM, handle.get_stream());
    }
    // length_bound paths are already cycles (no further extract), so do not return them as
    // unfinished work; they are emitted below.
    if ((path_length + vertex_t{1} < length_bound) && (aggregate_num_paths > max_aggregate_paths)) {
      next_path_length = path_length + vertex_t{1};
      break;
    }
  }

  // Reverse-distance pruning only extends a path of length L to dst if
  // L + reverse_distance(dst) <= length_bound. A path that reaches length_bound therefore ends on a
  // vertex with reverse distance 1 (the unique non-root predecessor of the path's root), so the
  // path is already a simple cycle. Skip a last-hop extract and emit the leftover paths as cycles.
  if (!next_path_length && (path_vertices.size() > 0)) {
    std::tie(cycle_vertex_chunks, cycle_length_chunks) =
      push_back_cycles<vertex_t, multi_gpu>(handle,
                                            std::move(cycle_vertex_chunks),
                                            std::move(cycle_length_chunks),
                                            std::move(path_vertices),
                                            length_bound);
  }

  return std::make_tuple(concatenate(handle, std::move(cycle_vertex_chunks)),
                         concatenate(handle, std::move(cycle_length_chunks)),
                         std::move(path_vertices),
                         next_path_length);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
enumerate_simple_cycles_including_roots(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> roots,
  raft::device_span<vertex_t const> reverse_distances,
  vertex_t length_bound,
  double approx_path_expansion_factor)
{
  std::vector<rmm::device_uvector<vertex_t>> cycle_vertex_chunks{};
  std::vector<rmm::device_uvector<vertex_t>> cycle_length_chunks{};

  edge_dst_property_t<vertex_t, vertex_t> dst_reverse_distances(handle);
  if constexpr (multi_gpu) {
    dst_reverse_distances = edge_dst_property_t<vertex_t, vertex_t>(handle, graph_view);
    update_edge_dst_property(
      handle, graph_view, reverse_distances.begin(), dst_reverse_distances.mutable_view());
  }
  rmm::device_uvector<vertex_t> root_path_vertices(roots.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(), roots.begin(), roots.end(), root_path_vertices.begin());

  size_t constexpr target_edges_per_sm = size_t{1} << 18;  // tuning parameter
  auto chunk_size                      = std::max(
    size_t{1},
    static_cast<size_t>(static_cast<double>(handle.get_device_properties().multiProcessorCount) *
                        static_cast<double>(target_edges_per_sm) / approx_path_expansion_factor));
  auto max_chunk_size = chunk_size;
  if constexpr (multi_gpu) {
    max_chunk_size = host_scalar_allreduce(
      handle.get_comms(), max_chunk_size, raft::comms::op_t::MAX, handle.get_stream());
  }
  auto max_aggregate_paths = max_chunk_size * 8;  // tuning parameter, if # paths far exceeds the
                                                  // target chunk_size, execute in multiple chunks.
  if constexpr (multi_gpu) {
    max_aggregate_paths *= static_cast<size_t>(handle.get_comms().get_size());
  }

  auto [new_cycle_vertices, new_cycle_lengths, new_path_vertices, new_path_length] =
    enumerate_simple_cycles_expanding_paths<vertex_t, edge_t, multi_gpu>(
      handle,
      graph_view,
      reverse_distances,
      dst_reverse_distances,
      std::move(root_path_vertices),
      vertex_t{1},
      length_bound,
      max_aggregate_paths);
  std::tie(cycle_vertex_chunks, cycle_length_chunks) =
    push_back_cycles<vertex_t, multi_gpu>(handle,
                                          std::move(cycle_vertex_chunks),
                                          std::move(cycle_length_chunks),
                                          std::move(new_cycle_vertices),
                                          std::move(new_cycle_lengths));
  if (new_path_length) {
    // expanding a chunk may again exceed max_aggregate_paths, so keep the unfinished path sets in a
    // worklist. The worklist is traversed in the DFS order (chunks created by expanding a chunk sit
    // on top of the remaining sibling chunks). Every GPU pushes & pops in lock-step
    // (new_path_length is based on the aggregate path count and the number of chunks in a path set
    // is max-reduced).
    std::vector<std::tuple<rmm::device_uvector<vertex_t>, vertex_t>> unfinished{};
    auto push_unfinished = [&handle, &unfinished, chunk_size](
                             rmm::device_uvector<vertex_t>&& path_vertices, vertex_t path_length) {
      auto num_paths  = path_vertices.size() / static_cast<size_t>(path_length);
      auto num_chunks = (num_paths + (chunk_size - 1)) / chunk_size;
      if constexpr (multi_gpu) {
        num_chunks = host_scalar_allreduce(
          handle.get_comms(), num_chunks, raft::comms::op_t::MAX, handle.get_stream());
      }
      for (size_t i = 0; i < num_chunks; ++i) {
        auto chunk_first = std::min(i * chunk_size, num_paths);
        auto chunk_last  = std::min(chunk_first + chunk_size, num_paths);
        rmm::device_uvector<vertex_t> chunk_path_vertices(
          (chunk_last - chunk_first) * static_cast<size_t>(path_length), handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     path_vertices.begin() + chunk_first * static_cast<size_t>(path_length),
                     path_vertices.begin() + chunk_last * static_cast<size_t>(path_length),
                     chunk_path_vertices.begin());
        unfinished.emplace_back(std::move(chunk_path_vertices), path_length);
      }
      path_vertices.resize(0, handle.get_stream());
      path_vertices.shrink_to_fit(handle.get_stream());
    };

    push_unfinished(std::move(new_path_vertices), *new_path_length);
    while (unfinished.size() > 0) {
      auto [chunk_path_vertices, path_length] = std::move(unfinished.back());
      unfinished.pop_back();

      std::tie(new_cycle_vertices, new_cycle_lengths, new_path_vertices, new_path_length) =
        enumerate_simple_cycles_expanding_paths<vertex_t, edge_t, multi_gpu>(
          handle,
          graph_view,
          reverse_distances,
          dst_reverse_distances,
          std::move(chunk_path_vertices),
          path_length,
          length_bound,
          max_aggregate_paths);
      std::tie(cycle_vertex_chunks, cycle_length_chunks) =
        push_back_cycles<vertex_t, multi_gpu>(handle,
                                              std::move(cycle_vertex_chunks),
                                              std::move(cycle_length_chunks),
                                              std::move(new_cycle_vertices),
                                              std::move(new_cycle_lengths));
      if (new_path_length) { push_unfinished(std::move(new_path_vertices), *new_path_length); }
    }
  }

  return std::make_tuple(concatenate(handle, std::move(cycle_vertex_chunks)),
                         concatenate(handle, std::move(cycle_length_chunks)));
}

// return std::tuple of cycle vertex chunks and cycle length chunks
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<std::vector<rmm::device_uvector<vertex_t>>, std::vector<rmm::device_uvector<vertex_t>>>
simple_cycles_impl(raft::handle_t const& handle,
                   graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                   std::optional<raft::device_span<vertex_t const>> seed_vertices,
                   vertex_t length_bound,
                   bool do_expensive_check)
{
  using weight_t     = float;    // dummy
  using edge_type_t  = int32_t;  // dummy
  using time_stamp_t = int64_t;  // dummy

  std::vector<rmm::device_uvector<vertex_t>> cycle_vertex_chunks{};
  std::vector<rmm::device_uvector<vertex_t>> cycle_length_chunks{};

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

  std::tie(cycle_vertex_chunks, cycle_length_chunks) =
    push_back_cycles<vertex_t, multi_gpu>(handle,
                                          std::move(cycle_vertex_chunks),
                                          std::move(cycle_length_chunks),
                                          std::move(length_1_cycle_vertices),
                                          vertex_t{1});

  if (length_bound == 1) {
    return std::make_tuple(std::move(cycle_vertex_chunks), std::move(cycle_length_chunks));
  }

  /* 3. find SCCs and compute the size of the component each vertex belongs to */

  auto components      = strongly_connected_components(handle, graph_view, do_expensive_check);
  auto component_sizes = compute_component_sizes<vertex_t, multi_gpu>(
    handle,
    raft::device_span<vertex_t const>(components.data(), components.size()),
    seed_vertices,
    graph_view.local_vertex_partition_range_first());

  /* 4. extract length 2 SCCs (length 2 simple cycles) */

  auto length_2_cycle_vertices = extract_length_2_cycle_vertices<vertex_t, multi_gpu>(
    handle,
    raft::device_span<vertex_t const>(components.data(), components.size()),
    raft::device_span<vertex_t const>(component_sizes.data(), component_sizes.size()),
    graph_view.local_vertex_partition_range_first());

  std::tie(cycle_vertex_chunks, cycle_length_chunks) =
    push_back_cycles<vertex_t, multi_gpu>(handle,
                                          std::move(cycle_vertex_chunks),
                                          std::move(cycle_length_chunks),
                                          std::move(length_2_cycle_vertices),
                                          vertex_t{2});

  if (length_bound == 2) {
    return std::make_tuple(std::move(cycle_vertex_chunks), std::move(cycle_length_chunks));
  }

  /* 5. enumerate intra-SCC edges for SCCs with more than 2 vertices and create a new graph */

  cugraph::graph_t<vertex_t, edge_t, false, multi_gpu> scc_graph(handle);
  rmm::device_uvector<vertex_t> scc_graph_renumber_map(0, handle.get_stream());
  std::optional<rmm::device_uvector<vertex_t>> scc_graph_seed_vertices{std::nullopt};
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
      }));  // components with no seed vertices and length 1 & 2 components will be excluded
    component_sizes.resize(0, handle.get_stream());
    component_sizes.shrink_to_fit(handle.get_stream());

    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    auto e_op = cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t>>(
      [] __device__(vertex_t src, vertex_t dst, auto, auto, auto) {
        return cuda::std::make_tuple(src, dst);
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
      std::tie(edgelist_srcs, edgelist_dsts) =
        extract_transform_if_e(handle,
                               graph_view,
                               src_components.view(),
                               dst_components.view(),
                               edge_dummy_property_t{}.view(),
                               e_op,
                               pred_op);
      std::tie(edgelist_srcs, edgelist_dsts, std::ignore) =
        shuffle_ext_edges(handle,
                          std::move(edgelist_srcs),
                          std::move(edgelist_dsts),
                          std::vector<cugraph::arithmetic_device_uvector_t>{},
                          false);
    } else {
      std::tie(edgelist_srcs, edgelist_dsts) =
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
    components.resize(0, handle.get_stream());
    components.shrink_to_fit(handle.get_stream());
    auto aggregate_edge_count = edgelist_srcs.size();
    if constexpr (multi_gpu) {
      aggregate_edge_count = host_scalar_allreduce(
        handle.get_comms(), aggregate_edge_count, raft::comms::op_t::SUM, handle.get_stream());
    }
    if (aggregate_edge_count == 0) {
      return std::make_tuple(std::move(cycle_vertex_chunks), std::move(cycle_length_chunks));
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
      create_graph_from_edgelist<vertex_t, edge_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::vector<cugraph::arithmetic_device_uvector_t>{},
        graph_properties_t{false, false},
        true /* renumber */);
    scc_graph_renumber_map = std::move(*tmp_renumber_map);
    if (seed_vertices) {
      scc_graph_seed_vertices = renumber_seed_vertices<vertex_t, edge_t, multi_gpu>(
        handle,
        *seed_vertices,
        scc_graph.view(),
        raft::device_span<vertex_t const>(scc_graph_renumber_map.data(),
                                          scc_graph_renumber_map.size()),
        do_expensive_check);
    }
  }

  /* 6. enumerate simple cycles (from SCCs with 3+ vertices) */

  assert(length_bound >= 3);

  auto scc_graph_view = scc_graph.view();
  auto [in_out_degree_products, sum_in_out_degree_products] =
    compute_in_out_degree_products<vertex_t, edge_t, multi_gpu>(handle, scc_graph_view);
  auto approx_path_expansion_factor =
    std::max(sum_in_out_degree_products / static_cast<double>(scc_graph.number_of_edges()),
             double{1.0});  // in_degree weighted average out-degree considering that high in-degree
                            // vertices appear more often as path endpoints
  auto scc_graph_edge_mask = make_initialized_edge_property(handle, scc_graph_view, false);
  components = cugraph::strongly_connected_components(handle, scc_graph_view, do_expensive_check);

  while (true) {
    size_t max_inner_loop_iters = 1;
    size_t inner_iter           = 0;
    while (true) {
      // 6-1. Pick a root from each length 3+ SCCs (FIXME: when # SCCs is small & # vertices in the
      // scc graph is small & # expected cycles is small, we may pick multiple roots from each SCC
      // and find cycles concurrnently and remove duplicates)

      rmm::device_uvector<vertex_t> roots(0, handle.get_stream());
      {
        // if seed vertices are provided, pick a root among the seed vertices in each SCC; every
        // enumerated cycle includes the root of its SCC, so this guarantees that every enumerated
        // cycle includes at least one seed vertex. Prefer the candidate with the largest
        // in-degree * out-degree (then the smaller vertex id).
        auto root_candidate_first = cuda::make_transform_iterator(
          cuda::make_counting_iterator(scc_graph_view.local_vertex_partition_range_first()),
          cuda::proclaim_return_type<vertex_t>(
            [seed_vertices = scc_graph_seed_vertices
                               ? cuda::std::make_optional<raft::device_span<vertex_t const>>(
                                   scc_graph_seed_vertices->data(), scc_graph_seed_vertices->size())
                               : cuda::std::nullopt] __device__(vertex_t v) {
              if (seed_vertices &&
                  !thrust::binary_search(
                    thrust::seq, seed_vertices->begin(), seed_vertices->end(), v)) {
                return std::numeric_limits<vertex_t>::max();  // this can never be selected as a
                                                              // root
              }
              return v;
            }));
        auto select_root = cuda::proclaim_return_type<cuda::std::tuple<vertex_t, float>>(
          [] __device__(cuda::std::tuple<vertex_t, float> lhs,
                        cuda::std::tuple<vertex_t, float> rhs) {
            auto const lhs_v = cuda::std::get<0>(lhs);
            auto const rhs_v = cuda::std::get<0>(rhs);
            if (lhs_v == std::numeric_limits<vertex_t>::max()) { return rhs; }
            if (rhs_v == std::numeric_limits<vertex_t>::max()) { return lhs; }
            auto const lhs_p = cuda::std::get<1>(lhs);
            auto const rhs_p = cuda::std::get<1>(rhs);
            if (lhs_p > rhs_p) {
              return lhs;
            } else if (rhs_p > lhs_p) {
              return rhs;
            } else {
              return lhs_v <= rhs_v ? lhs : rhs;
            }  // equal products: keep the smaller vertex id
          });
        auto [unique_components, unique_values] = reduce_by_component<vertex_t, multi_gpu>(
          handle,
          raft::device_span<vertex_t const>(components.data(), components.size()),
          thrust::make_zip_iterator(root_candidate_first, in_out_degree_products.begin()),
          select_root);
        auto unique_component_roots = std::move(std::get<0>(unique_values));
        auto num_valid_roots        = static_cast<size_t>(thrust::count_if(
          handle.get_thrust_policy(),
          unique_components.begin(),
          unique_components.end(),
          cugraph::detail::is_not_equal_to_t<vertex_t>{invalid_vertex_id_v<vertex_t>}));
        roots.resize(num_valid_roots, handle.get_stream());
        thrust::copy_if(
          handle.get_thrust_policy(),
          unique_component_roots.begin(),
          unique_component_roots.end(),
          unique_components.begin(),
          roots.begin(),
          cugraph::detail::is_not_equal_to_t<vertex_t>{invalid_vertex_id_v<vertex_t>});
        if (inner_iter > 0) {
          // seedless leftover SCCs reduce to vertex_t::max(); drop them before shuffle
          roots.resize(static_cast<size_t>(
                         cuda::std::distance(roots.begin(),
                                             thrust::remove(handle.get_thrust_policy(),
                                                            roots.begin(),
                                                            roots.end(),
                                                            std::numeric_limits<vertex_t>::max()))),
                       handle.get_stream());
        }
        if constexpr (multi_gpu) {
          std::tie(roots, std::ignore) =
            shuffle_int_vertices(handle,
                                 std::move(roots),
                                 std::vector<cugraph::arithmetic_device_uvector_t>{},
                                 scc_graph_view.vertex_partition_range_lasts());
        }
        cugraph::sort(handle.get_thrust_policy(), roots.begin(), roots.end());
      }
      auto aggregate_num_roots = roots.size();
      if constexpr (multi_gpu) {
        aggregate_num_roots = host_scalar_allreduce(
          handle.get_comms(), aggregate_num_roots, raft::comms::op_t::SUM, handle.get_stream());
      }
      if (aggregate_num_roots == 0) { break; }

      if (inner_iter == 0) {
        // if the number of size 3+ SCCs is small, run more inner iterations (trade-off between the
        // cost of running cugraph::strongly_connected_components() vs missing an opportunity to
        // find new SCCs and mask out edges outside size 3+ SCCs)
        if (aggregate_num_roots < size_t{8}) { max_inner_loop_iters = size_t{16}; }
      }

      // 6-2. Compute the reverse distances from the picked root vertices

      rmm::device_uvector<vertex_t> reverse_distances(
        scc_graph_view.local_vertex_partition_range_size(), handle.get_stream());
      {
        cugraph::fill(handle.get_thrust_policy(),
                      reverse_distances.begin(),
                      reverse_distances.end(),
                      std::numeric_limits<vertex_t>::max());
        rmm::device_uvector<vertex_t> updated_vertices(0, handle.get_stream());
        cugraph::scatter(
          handle.get_thrust_policy(),
          cuda::make_constant_iterator(vertex_t{0}),
          cuda::make_constant_iterator(vertex_t{0}) + roots.size(),
          cuda::make_transform_iterator(roots.begin(),
                                        cugraph::detail::shift_left_t<vertex_t>{
                                          scc_graph_view.local_vertex_partition_range_first()}),
          reverse_distances.begin());
        for (vertex_t hop = vertex_t{0}; hop < length_bound; ++hop) {
          // uint8_t instead of bool as bool is not a supported raft::comms type (necessary for the
          // multi-GPU reduction in per_v_transform_reduce_outgoing_e)
          rmm::device_uvector<uint8_t> next_hop_flags(reverse_distances.size(),
                                                      handle.get_stream());
          if constexpr (multi_gpu) {
            auto dst_prev_hop_visited_flags =
              make_initialized_edge_dst_property(handle, scc_graph_view, false);
            fill_edge_dst_property(handle,
                                   scc_graph_view,
                                   hop == vertex_t{0} ? roots.begin() : updated_vertices.begin(),
                                   hop == vertex_t{0} ? roots.end() : updated_vertices.end(),
                                   dst_prev_hop_visited_flags.mutable_view(),
                                   true,
                                   do_expensive_check);
            per_v_transform_reduce_outgoing_e(
              handle,
              scc_graph_view,
              edge_src_dummy_property_t{}.view(),
              dst_prev_hop_visited_flags.view(),
              edge_dummy_property_t{}.view(),
              cuda::proclaim_return_type<uint8_t>(
                [] __device__(vertex_t, vertex_t, auto, bool dst_prev_hop_visited, auto) {
                  return dst_prev_hop_visited ? uint8_t{1} : uint8_t{0};
                }),
              uint8_t{0},
              cugraph::reduce_op::maximum<uint8_t>{},
              next_hop_flags.begin());
          } else {
            per_v_transform_reduce_outgoing_e(
              handle,
              scc_graph_view,
              edge_src_dummy_property_t{}.view(),
              make_edge_dst_property_view<vertex_t, vertex_t>(
                scc_graph_view, reverse_distances.begin(), reverse_distances.size()),
              edge_dummy_property_t{}.view(),
              cuda::proclaim_return_type<uint8_t>(
                [hop] __device__(vertex_t, vertex_t, auto, vertex_t dst_reverse_distance, auto) {
                  return (dst_reverse_distance == hop) ? uint8_t{1} : uint8_t{0};
                }),
              uint8_t{0},
              cugraph::reduce_op::maximum<uint8_t>{},
              next_hop_flags.begin());
          }
          thrust::transform(
            handle.get_thrust_policy(),
            next_hop_flags.begin(),
            next_hop_flags.end(),
            reverse_distances.begin(),
            next_hop_flags.begin(),
            cuda::proclaim_return_type<uint8_t>([] __device__(uint8_t flag, vertex_t distance) {
              if (distance == std::numeric_limits<vertex_t>::max()) {
                return flag;
              } else {  // already visited
                return uint8_t{0};
              }
            }));

          auto next_hop_size = thrust::count(
            handle.get_thrust_policy(), next_hop_flags.begin(), next_hop_flags.end(), uint8_t{1});
          auto aggregate_next_hop_size = next_hop_size;
          if constexpr (multi_gpu) {
            aggregate_next_hop_size = host_scalar_allreduce(handle.get_comms(),
                                                            aggregate_next_hop_size,
                                                            raft::comms::op_t::SUM,
                                                            handle.get_stream());
          }
          if (aggregate_next_hop_size == 0) { break; }

          updated_vertices.resize(next_hop_size, handle.get_stream());
          thrust::copy_if(
            handle.get_thrust_policy(),
            cuda::make_counting_iterator(scc_graph_view.local_vertex_partition_range_first()),
            cuda::make_counting_iterator(scc_graph_view.local_vertex_partition_range_first()) +
              scc_graph_view.local_vertex_partition_range_size(),
            next_hop_flags.begin(),
            updated_vertices.begin(),
            cuda::std::identity{});

          cugraph::scatter(
            handle.get_thrust_policy(),
            cuda::make_constant_iterator(hop + vertex_t{1}),
            cuda::make_constant_iterator(hop + vertex_t{1}) + updated_vertices.size(),
            cuda::make_transform_iterator(updated_vertices.begin(),
                                          cugraph::detail::shift_left_t<vertex_t>{
                                            scc_graph_view.local_vertex_partition_range_first()}),
            reverse_distances.begin());
        }
      }

      // 6-3. Enumerate length 2+ simple cycles including the picked root vertices

      size_t num_new_cycle_vertices{0};
      {
        auto [new_cycle_vertices, new_cycle_lengths] =
          enumerate_simple_cycles_including_roots<vertex_t, edge_t, multi_gpu>(
            handle,
            scc_graph_view,
            raft::device_span<vertex_t const>(roots.data(), roots.size()),
            raft::device_span<vertex_t const>(reverse_distances.data(), reverse_distances.size()),
            length_bound,
            approx_path_expansion_factor);
        num_new_cycle_vertices                             = new_cycle_vertices.size();
        std::tie(cycle_vertex_chunks, cycle_length_chunks) = push_back_cycles<vertex_t, multi_gpu>(
          handle,
          std::move(cycle_vertex_chunks),
          std::move(cycle_length_chunks),
          std::move(new_cycle_vertices),
          std::move(new_cycle_lengths),
          std::make_optional(
            std::make_tuple(raft::device_span<vertex_t const>(scc_graph_renumber_map.data(),
                                                              scc_graph_renumber_map.size()),
                            scc_graph_view.vertex_partition_range_lasts())));
      }

      // 6-4. Mask out the edges to/from the picked root vertices.

      {
        auto new_scc_graph_edge_mask =
          make_initialized_edge_property(handle, scc_graph_view, false, do_expensive_check);
        if constexpr (multi_gpu) {
          auto src_root_flags = make_initialized_edge_src_property(handle, scc_graph_view, false);
          auto dst_root_flags = make_initialized_edge_dst_property(handle, scc_graph_view, false);
          fill_edge_src_property(handle,
                                 scc_graph_view,
                                 roots.begin(),
                                 roots.end(),
                                 src_root_flags.mutable_view(),
                                 true,
                                 do_expensive_check);
          fill_edge_dst_property(handle,
                                 scc_graph_view,
                                 roots.begin(),
                                 roots.end(),
                                 dst_root_flags.mutable_view(),
                                 true,
                                 do_expensive_check);
          transform_e(
            handle,
            scc_graph_view,
            src_root_flags.view(),
            dst_root_flags.view(),
            edge_dummy_property_t{}.view(),
            [] __device__(vertex_t, vertex_t, bool src_root, bool dst_root, auto) {
              return !src_root && !dst_root;
            },
            new_scc_graph_edge_mask.mutable_view(),
            do_expensive_check);
        } else {
          transform_e(
            handle,
            scc_graph_view,
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            edge_dummy_property_t{}.view(),
            cuda::proclaim_return_type<bool>(
              [roots = raft::device_span<vertex_t const>(roots.data(), roots.size())] __device__(
                vertex_t src, vertex_t dst, auto, auto, auto) {
                return !thrust::binary_search(thrust::seq, roots.begin(), roots.end(), src) &&
                       !thrust::binary_search(thrust::seq, roots.begin(), roots.end(), dst);
              }),
            new_scc_graph_edge_mask.mutable_view(),
            do_expensive_check);
        }
        if (scc_graph_view.has_edge_mask()) { scc_graph_view.clear_edge_mask(); }
        scc_graph_edge_mask = std::move(new_scc_graph_edge_mask);
        scc_graph_view.attach_edge_mask(scc_graph_edge_mask.view());
      }

      ++inner_iter;

      bool continue_inner_loop = false;
      if (inner_iter < max_inner_loop_iters) {
        auto aggregate_num_new_cycle_vertices = num_new_cycle_vertices;
        if constexpr (multi_gpu) {
          aggregate_num_new_cycle_vertices = host_scalar_allreduce(handle.get_comms(),
                                                                   aggregate_num_new_cycle_vertices,
                                                                   raft::comms::op_t::SUM,
                                                                   handle.get_stream());
        }
        if (aggregate_num_new_cycle_vertices <
            std::max(size_t{4}, static_cast<size_t>(scc_graph.number_of_edges() / edge_t{1024}))) {
          continue_inner_loop = true;
        }
      }

      if (continue_inner_loop) {
        cugraph::scatter(
          handle.get_thrust_policy(),
          cuda::make_constant_iterator(invalid_vertex_id_v<vertex_t>),
          cuda::make_constant_iterator(invalid_vertex_id_v<vertex_t>) + roots.size(),
          cuda::make_transform_iterator(roots.begin(),
                                        cugraph::detail::shift_left_t<vertex_t>{
                                          scc_graph_view.local_vertex_partition_range_first()}),
          components.begin());
      } else {
        break;
      }
    }

    /* 6-5. Rebuild the graph if a big percentage of the edges are masked out */

    assert(scc_graph.number_of_edges() > 0);
    if (static_cast<double>(scc_graph_view.compute_number_of_edges(handle)) /
          static_cast<double>(scc_graph.number_of_edges()) <
        0.5 /* tuning parameter */) {
      rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
      std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore, std::ignore) =
        decompress_to_edgelist<vertex_t, edge_t, weight_t, edge_type_t, false, multi_gpu>(
          handle,
          scc_graph_view,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::make_optional<raft::device_span<vertex_t const>>(scc_graph_renumber_map.data(),
                                                                scc_graph_renumber_map.size()),
          std::nullopt,
          do_expensive_check);
      auto aggregate_edge_count = edgelist_srcs.size();
      if constexpr (multi_gpu) {
        aggregate_edge_count = host_scalar_allreduce(
          handle.get_comms(), aggregate_edge_count, raft::comms::op_t::SUM, handle.get_stream());
      }
      if (aggregate_edge_count == 0) {
        return std::make_tuple(std::move(cycle_vertex_chunks), std::move(cycle_length_chunks));
      }

      std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
      std::tie(scc_graph, std::ignore, tmp_renumber_map) =
        create_graph_from_edgelist<vertex_t, edge_t, false, multi_gpu>(
          handle,
          std::nullopt,
          std::move(edgelist_srcs),
          std::move(edgelist_dsts),
          std::vector<cugraph::arithmetic_device_uvector_t>{},
          graph_properties_t{false, false},
          true /* renumber */);
      scc_graph_renumber_map = std::move(*tmp_renumber_map);
      scc_graph_view         = scc_graph.view();
      scc_graph_edge_mask    = edge_property_t<edge_t, bool>(handle);
      if (seed_vertices) {
        scc_graph_seed_vertices = renumber_seed_vertices<vertex_t, edge_t, multi_gpu>(
          handle,
          *seed_vertices,
          scc_graph_view,
          raft::device_span<vertex_t const>(scc_graph_renumber_map.data(),
                                            scc_graph_renumber_map.size()),
          do_expensive_check);
      }
      std::tie(in_out_degree_products, sum_in_out_degree_products) =
        compute_in_out_degree_products<vertex_t, edge_t, multi_gpu>(handle, scc_graph_view);
      approx_path_expansion_factor =
        std::max(sum_in_out_degree_products / static_cast<double>(scc_graph.number_of_edges()),
                 double{1.0});  // in_degree weighted average out-degree considering that high
                                // in-degre vertices appear more often as path endpointse
    }

    // 6-6. Run strongly_connected_components()

    components = cugraph::strongly_connected_components(handle, scc_graph_view, do_expensive_check);

    // 6-7. Enumerate length 2 SCCs

    auto component_sizes = compute_component_sizes<vertex_t, multi_gpu>(
      handle,
      raft::device_span<vertex_t const>(components.data(), components.size()),
      scc_graph_seed_vertices ? std::make_optional<raft::device_span<vertex_t const>>(
                                  scc_graph_seed_vertices->data(), scc_graph_seed_vertices->size())
                              : std::nullopt,
      scc_graph_view.local_vertex_partition_range_first());

    auto length_2_cycle_vertices = extract_length_2_cycle_vertices<vertex_t, multi_gpu>(
      handle,
      raft::device_span<vertex_t const>(components.data(), components.size()),
      raft::device_span<vertex_t const>(component_sizes.data(), component_sizes.size()),
      scc_graph_view.local_vertex_partition_range_first());

    std::tie(cycle_vertex_chunks, cycle_length_chunks) = push_back_cycles<vertex_t, multi_gpu>(
      handle,
      std::move(cycle_vertex_chunks),
      std::move(cycle_length_chunks),
      std::move(length_2_cycle_vertices),
      vertex_t{2},
      std::make_optional(
        std::make_tuple(raft::device_span<vertex_t const>(scc_graph_renumber_map.data(),
                                                          scc_graph_renumber_map.size()),
                        scc_graph_view.vertex_partition_range_lasts())));

    // 6-8. Keep only the intra SCC edges for length 3+ SCCs

    {
      auto num_remaining_vertices = static_cast<size_t>(
        thrust::count_if(handle.get_thrust_policy(),
                         component_sizes.begin(),
                         component_sizes.end(),
                         cuda::proclaim_return_type<bool>(
                           [] __device__(vertex_t size) { return size > vertex_t{2}; })));
      auto aggregate_num_remaining_vertices = num_remaining_vertices;
      if constexpr (multi_gpu) {
        aggregate_num_remaining_vertices = host_scalar_allreduce(handle.get_comms(),
                                                                 aggregate_num_remaining_vertices,
                                                                 raft::comms::op_t::SUM,
                                                                 handle.get_stream());
      }
      if (aggregate_num_remaining_vertices == 0) { break; }

      thrust::transform_if(
        handle.get_thrust_policy(),
        component_sizes.begin(),
        component_sizes.end(),
        components.begin(),
        cuda::proclaim_return_type<vertex_t>(
          [] __device__(vertex_t) { return invalid_vertex_id_v<vertex_t>; }),
        cuda::proclaim_return_type<bool>([] __device__(vertex_t size) {
          return size <= vertex_t{2};
        }));  // components with no seed vertices and length 1 & 2 components will be excluded

      auto new_scc_graph_edge_mask =
        make_initialized_edge_property(handle, scc_graph_view, false, do_expensive_check);
      auto e_op = cuda::proclaim_return_type<bool>(
        [invalid_component = invalid_vertex_id_v<vertex_t>] __device__(
          vertex_t src, vertex_t dst, vertex_t src_component, vertex_t dst_component, auto) {
          assert(src != dst);
          return (src_component != invalid_component) && (src_component == dst_component);
        });
      if constexpr (multi_gpu) {
        edge_src_property_t<vertex_t, vertex_t> src_components(handle, scc_graph_view);
        edge_dst_property_t<vertex_t, vertex_t> dst_components(handle, scc_graph_view);
        update_edge_src_property(
          handle, scc_graph_view, components.begin(), src_components.mutable_view());
        update_edge_dst_property(
          handle, scc_graph_view, components.begin(), dst_components.mutable_view());
        transform_e(handle,
                    scc_graph_view,
                    src_components.view(),
                    dst_components.view(),
                    edge_dummy_property_t{}.view(),
                    e_op,
                    new_scc_graph_edge_mask.mutable_view(),
                    do_expensive_check);
      } else {
        transform_e(handle,
                    scc_graph_view,
                    make_edge_src_property_view<vertex_t, vertex_t>(
                      scc_graph_view, components.begin(), components.size()),
                    make_edge_dst_property_view<vertex_t, vertex_t>(
                      scc_graph_view, components.begin(), components.size()),
                    edge_dummy_property_t{}.view(),
                    e_op,
                    new_scc_graph_edge_mask.mutable_view(),
                    do_expensive_check);
      }
      if (scc_graph_view.has_edge_mask()) { scc_graph_view.clear_edge_mask(); }
      scc_graph_edge_mask = std::move(new_scc_graph_edge_mask);
      scc_graph_view.attach_edge_mask(scc_graph_edge_mask.view());
    }
  }

  return std::make_tuple(std::move(cycle_vertex_chunks), std::move(cycle_length_chunks));
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
  auto [cycle_vertex_chunks, cycle_length_chunks] =
    detail::simple_cycles_impl(handle, graph_view, seed_vertices, length_bound, do_expensive_check);
  auto cycle_vertices = detail::concatenate(handle, std::move(cycle_vertex_chunks));
  auto cycle_lengths  = detail::concatenate(handle, std::move(cycle_length_chunks));
  rmm::device_uvector<size_t> cycle_offsets(cycle_lengths.size() + 1, handle.get_stream());
  cycle_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
  cugraph::inclusive_scan(
    handle.get_thrust_policy(),
    cuda::make_transform_iterator(cycle_lengths.begin(), detail::typecast_t<vertex_t, size_t>{}),
    cuda::make_transform_iterator(cycle_lengths.end(), detail::typecast_t<vertex_t, size_t>{}),
    cycle_offsets.begin() + 1);
  return std::make_tuple(std::move(cycle_vertices), std::move(cycle_offsets));
}

}  // namespace cugraph
