/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/span.hpp>

#include <rmm/device_uvector.hpp>

#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace test {

template <typename vertex_t, typename edge_t, typename weight_t>
decltype(auto) make_graph(raft::handle_t const& handle,
                          std::vector<vertex_t> const& v_src,
                          std::vector<vertex_t> const& v_dst,
                          std::optional<std::vector<weight_t>> const& v_w,
                          vertex_t num_vertices,
                          edge_t num_edges)
{
  rmm::device_uvector<vertex_t> d_src(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst(num_edges, handle.get_stream());
  auto d_w = v_w ? std::make_optional<rmm::device_uvector<weight_t>>(num_edges, handle.get_stream())
                 : std::nullopt;

  raft::update_device(d_src.data(), v_src.data(), d_src.size(), handle.get_stream());
  raft::update_device(d_dst.data(), v_dst.data(), d_dst.size(), handle.get_stream());
  if (d_w) {
    raft::update_device((*d_w).data(), (*v_w).data(), (*d_w).size(), handle.get_stream());
  }

  cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
  std::optional<cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
    edge_weights{std::nullopt};
  std::tie(graph, edge_weights, std::ignore, std::ignore) =
    cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, false, false>(
      handle,
      std::nullopt,
      std::move(d_src),
      std::move(d_dst),
      std::move(d_w),
      std::nullopt,
      cugraph::graph_properties_t{false, false},
      false);

  return std::make_tuple(std::move(graph), std::move(edge_weights));
}

template <typename T>
std::vector<T> to_host(raft::handle_t const& handle, raft::device_span<T const> data)
{
  std::vector<T> h_data(data.size());
  if constexpr (std::is_same_v<T, bool>) {  // std::vector<bool> stores values in a packed format
    auto h_tmp = new bool[data.size()];
    raft::update_host(h_tmp, data.data(), data.size(), handle.get_stream());
    handle.sync_stream();
    std::transform(
      h_tmp, h_tmp + data.size(), h_data.begin(), [](uint8_t v) { return static_cast<bool>(v); });
    delete[] h_tmp;
  } else {
    raft::update_host(h_data.data(), data.data(), data.size(), handle.get_stream());
    handle.sync_stream();
  }
  return h_data;
}

template <typename T>
std::vector<T> to_host(raft::handle_t const& handle, rmm::device_uvector<T> const& data)
{
  return to_host(handle, raft::device_span<T const>(data.data(), data.size()));
}

template <typename T>
std::optional<std::vector<T>> to_host(raft::handle_t const& handle,
                                      std::optional<raft::device_span<T const>> data)
{
  std::optional<std::vector<T>> h_data{std::nullopt};
  if (data) { h_data = to_host(handle, *data); }
  return h_data;
}

template <typename T>
std::optional<std::vector<T>> to_host(raft::handle_t const& handle,
                                      std::optional<rmm::device_uvector<T>> const& data)
{
  std::optional<std::vector<T>> h_data{std::nullopt};
  if (data) {
    h_data = to_host(handle, raft::device_span<T const>((*data).data(), (*data).size()));
  }
  return h_data;
}

template <typename T>
rmm::device_uvector<T> to_device(raft::handle_t const& handle, raft::host_span<T const> data)
{
  rmm::device_uvector<T> d_data(data.size(), handle.get_stream());
  raft::update_device(d_data.data(), data.data(), data.size(), handle.get_stream());
  handle.sync_stream();
  return d_data;
}

template <typename T>
rmm::device_uvector<T> to_device(raft::handle_t const& handle, std::vector<T> const& data)
{
  rmm::device_uvector<T> d_data(data.size(), handle.get_stream());
  if constexpr (std::is_same_v<T, bool>) {  // std::vector<bool> stores values in a packed format
    auto h_tmp = new bool[data.size()];
    std::copy(data.begin(), data.end(), h_tmp);
    raft::update_device(d_data.data(), h_tmp, h_tmp + data.size(), handle.get_stream());
    handle.sync_stream();
    delete[] h_tmp;
  } else {
    raft::update_device(d_data.data(), data.data(), data.size(), handle.get_stream());
    handle.sync_stream();
  }
  return d_data;
}

template <typename T>
std::optional<rmm::device_uvector<T>> to_device(raft::handle_t const& handle,
                                                std::optional<raft::host_span<T const>> data)
{
  std::optional<rmm::device_uvector<T>> d_data{std::nullopt};
  if (data) {
    d_data = rmm::device_uvector<T>(data->size(), handle.get_stream());
    raft::update_device(d_data->data(), data->data(), data->size(), handle.get_stream());
    handle.sync_stream();
  }
  return d_data;
}

template <typename T>
std::optional<rmm::device_uvector<T>> to_device(raft::handle_t const& handle,
                                                std::optional<std::vector<T>> const& data)
{
  std::optional<rmm::device_uvector<T>> d_data{std::nullopt};
  if (data) { d_data = to_device(handle, *data); }
  return d_data;
}

// If multi-GPU, only the rank 0 GPU holds the valid data
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<vertex_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> renumber_map);

// If multi-GPU, only the rank 0 GPU holds the valid data
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> renumber_map);

// If multi-GPU, only the rank 0 GPU holds the valid data
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<edge_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> renumber_map);

// If multi-GPU, only the rank 0 GPU holds the valid data
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<edge_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> renumber_map);

// Only the rank 0 GPU holds the valid data
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
std::tuple<
  cugraph::graph_t<vertex_t, edge_t, store_transposed, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, store_transposed, false>,
                             weight_t>>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, store_transposed, false>,
                             edge_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<raft::device_span<vertex_t const>> renumber_map,
  bool renumber);

// Only the rank 0 GPU holds the valid data

template <typename vertex_t, typename value_t>
std::tuple<std::optional<rmm::device_uvector<vertex_t>>, rmm::device_uvector<value_t>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<vertex_t const>>
    mg_renumber_map,  // std::nullopt if the MG graph is not renumbered
  std::tuple<vertex_t, vertex_t> mg_vertex_partition_range,
  std::optional<raft::device_span<vertex_t const>>
    sg_renumber_map,  // std::nullopt if the SG graph is not renumbered
  std::optional<raft::device_span<vertex_t const>>
    mg_vertices,  // std::nullopt if the entire local vertex partition range is assumed
  raft::device_span<value_t const> mg_values);

}  // namespace test
}  // namespace cugraph
