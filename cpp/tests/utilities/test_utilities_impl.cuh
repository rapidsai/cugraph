/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#pragma once

#include <structure/detail/structure_utils.cuh>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/device_span.hpp>

#include <numeric>
#include <variant>

namespace cugraph {
namespace test {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<vertex_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  auto [d_src, d_dst, d_wgt] =
    cugraph::decompress_to_edgelist(handle,
                                    graph_view,
                                    edge_weight_view,
                                    std::optional<raft::device_span<vertex_t const>>{std::nullopt});

  if constexpr (is_multi_gpu) {
    d_src = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
    d_dst = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
    if (d_wgt)
      *d_wgt = cugraph::test::device_gatherv(
        handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});
    if (handle.get_comms().get_rank() != 0) {
      d_src.resize(0, handle.get_stream());
      d_src.shrink_to_fit(handle.get_stream());
      d_dst.resize(0, handle.get_stream());
      d_dst.shrink_to_fit(handle.get_stream());
      if (d_wgt) {
        (*d_wgt).resize(0, handle.get_stream());
        (*d_wgt).shrink_to_fit(handle.get_stream());
      }
    }
  }

  std::vector<vertex_t> h_src(d_src.size());
  std::vector<vertex_t> h_dst(d_dst.size());
  std::optional<std::vector<weight_t>> h_wgt(std::nullopt);

  raft::update_host(h_src.data(), d_src.data(), d_src.size(), handle.get_stream());
  raft::update_host(h_dst.data(), d_dst.data(), d_dst.size(), handle.get_stream());

  if (d_wgt) {
    h_wgt = std::make_optional<std::vector<weight_t>>(d_wgt->size());
    raft::update_host(h_wgt->data(), d_wgt->data(), d_wgt->size(), handle.get_stream());
  }

  return std::make_tuple(std::move(h_src), std::move(h_dst), std::move(h_wgt));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<edge_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  auto [d_src, d_dst, d_wgt] =
    cugraph::decompress_to_edgelist(handle,
                                    graph_view,
                                    edge_weight_view,
                                    std::optional<raft::device_span<vertex_t const>>{std::nullopt});

  if constexpr (is_multi_gpu) {
    d_src = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
    d_dst = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
    if (d_wgt)
      *d_wgt = cugraph::test::device_gatherv(
        handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});
    if (handle.get_comms().get_rank() != 0) {
      d_src.resize(0, handle.get_stream());
      d_src.shrink_to_fit(handle.get_stream());
      d_dst.resize(0, handle.get_stream());
      d_dst.shrink_to_fit(handle.get_stream());
      if (d_wgt) {
        (*d_wgt).resize(0, handle.get_stream());
        (*d_wgt).shrink_to_fit(handle.get_stream());
      }
    }
  }

  rmm::device_uvector<edge_t> d_offsets(0, handle.get_stream());

  if (d_wgt) {
    std::tie(d_offsets, d_dst, *d_wgt, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(d_src.begin(),
                                                          d_src.end(),
                                                          d_dst.begin(),
                                                          d_wgt->begin(),
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          graph_view.number_of_vertices(),
                                                          vertex_t{0},
                                                          graph_view.number_of_vertices(),
                                                          handle.get_stream());

    // segmented sort neighbors
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size()),
                                d_dst.begin(),
                                d_dst.end(),
                                d_wgt->begin());
  } else {
    std::tie(d_offsets, d_dst, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(d_src.begin(),
                                                          d_src.end(),
                                                          d_dst.begin(),
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          graph_view.number_of_vertices(),
                                                          vertex_t{0},
                                                          graph_view.number_of_vertices(),
                                                          handle.get_stream());
    // segmented sort neighbors
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size()),
                                d_dst.begin(),
                                d_dst.end());
  }

  return std::make_tuple(
    to_host(handle, raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size())),
    to_host(handle, raft::device_span<vertex_t const>(d_dst.data(), d_dst.size())),
    d_wgt ? to_host(handle, raft::device_span<weight_t const>(d_wgt->data(), d_wgt->size()))
          : std::optional<std::vector<weight_t>>(std::nullopt));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
std::tuple<
  cugraph::graph_t<vertex_t, edge_t, store_transposed, false>,
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, false>, weight_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> number_map,
  bool renumber)
{
  auto [d_src, d_dst, d_wgt] =
    cugraph::decompress_to_edgelist(handle, graph_view, edge_weight_view, number_map);

  d_src = cugraph::test::device_gatherv(
    handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
  d_dst = cugraph::test::device_gatherv(
    handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
  if (d_wgt)
    *d_wgt = cugraph::test::device_gatherv(
      handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});

  rmm::device_uvector<vertex_t> vertices(0, handle.get_stream());
  if (number_map) { vertices = cugraph::test::device_gatherv(handle, *number_map); }

  graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(handle);
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, false>, weight_t>>
    sg_edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> sg_number_map;
  if (handle.get_comms().get_rank() == 0) {
    if (!number_map) {
      vertices.resize(graph_view.number_of_vertices(), handle.get_stream());
      cugraph::detail::sequence_fill(
        handle.get_stream(), vertices.data(), vertices.size(), vertex_t{0});
    }

    std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore, sg_number_map) =
      cugraph::create_graph_from_edgelist<vertex_t,
                                          edge_t,
                                          weight_t,
                                          edge_t,
                                          int32_t,
                                          store_transposed,
                                          false>(
        handle,
        std::make_optional(std::move(vertices)),
        std::move(d_src),
        std::move(d_dst),
        std::move(d_wgt),
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{graph_view.is_symmetric(), graph_view.is_multigraph()},
        renumber);
  } else {
    d_src.resize(0, handle.get_stream());
    d_src.shrink_to_fit(handle.get_stream());
    d_dst.resize(0, handle.get_stream());
    d_dst.shrink_to_fit(handle.get_stream());
    if (d_wgt) {
      (*d_wgt).resize(0, handle.get_stream());
      (*d_wgt).shrink_to_fit(handle.get_stream());
    }
  }

  return std::make_tuple(std::move(sg_graph), std::move(sg_edge_weights), std::move(sg_number_map));
}

template <typename vertex_t, typename value_t>
rmm::device_uvector<value_t> mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<vertex_t const>>
    mg_renumber_map,  // std::nullopt if the MG graph is not renumbered.
  std::optional<raft::device_span<vertex_t const>>
    sg_renumber_map,  // std::nullopt if the SG graph is not renumbered.
  raft::device_span<value_t const> mg_values)
{
  auto& comm           = handle.get_comms();
  auto const comm_rank = comm.get_rank();

  std::variant<std::tuple<std::vector<size_t>, std::vector<int>>, rmm::device_uvector<vertex_t>>
    mg_aux_info{};  // (vertex_partition_sizes, vertex_partition_ids) pair or aggregated
                    // mg_renumber_map
  if (mg_renumber_map) {
    mg_aux_info = cugraph::test::device_gatherv(handle, *mg_renumber_map);
  } else {
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    auto vertex_partition_id =
      cugraph::partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);

    auto mg_vertex_partition_sizes =
      cugraph::host_scalar_gather(comm, mg_values.size(), int{0}, handle.get_stream());
    auto mg_vertex_partition_ids =
      cugraph::host_scalar_gather(comm, vertex_partition_id, int{0}, handle.get_stream());
    mg_aux_info =
      std::make_tuple(std::move(mg_vertex_partition_sizes), std::move(mg_vertex_partition_ids));
  }
  auto mg_aggregate_values = cugraph::test::device_gatherv(handle, mg_values);

  rmm::device_uvector<value_t> sg_values(0, handle.get_stream());
  if (comm_rank == 0) {
    if (mg_renumber_map) {
      auto& mg_aggregate_renumber_map = std::get<1>(mg_aux_info);
      std::tie(mg_aggregate_renumber_map, mg_aggregate_values) =
        cugraph::test::sort_by_key(handle, mg_aggregate_renumber_map, mg_aggregate_values);
    } else {
      auto& [mg_vertex_partition_sizes, mg_vertex_partition_ids] = std::get<0>(mg_aux_info);
      std::vector<size_t> mg_dst_vertex_partition_sizes(mg_vertex_partition_sizes.size());
      for (size_t i = 0; i < mg_vertex_partition_sizes.size(); ++i) {
        mg_dst_vertex_partition_sizes[mg_vertex_partition_ids[i]] = mg_vertex_partition_sizes[i];
      }
      std::vector<size_t> mg_dst_vertex_partition_displs(mg_dst_vertex_partition_sizes.size());
      std::exclusive_scan(mg_dst_vertex_partition_sizes.begin(),
                          mg_dst_vertex_partition_sizes.end(),
                          mg_dst_vertex_partition_displs.begin(),
                          size_t{0});

      std::vector<size_t> mg_vertex_partition_displs(mg_vertex_partition_ids.size());
      std::exclusive_scan(mg_vertex_partition_sizes.begin(),
                          mg_vertex_partition_sizes.end(),
                          mg_vertex_partition_displs.begin(),
                          size_t{0});

      rmm::device_uvector<value_t> tmp_mg_aggregate_values(mg_aggregate_values.size(),
                                                           handle.get_stream());
      for (size_t i = 0; i < mg_vertex_partition_ids.size(); ++i) {
        thrust::copy(handle.get_thrust_policy(),
                     mg_aggregate_values.begin() + mg_vertex_partition_displs[i],
                     mg_aggregate_values.begin() + mg_vertex_partition_displs[i] +
                       mg_vertex_partition_sizes[i],
                     tmp_mg_aggregate_values.begin() +
                       mg_dst_vertex_partition_displs[mg_vertex_partition_ids[i]]);
      }

      mg_aggregate_values = std::move(tmp_mg_aggregate_values);
    }

    if (sg_renumber_map) {
      std::optional<raft::device_span<vertex_t const>> mg_map{std::nullopt};
      if (mg_renumber_map) {
        auto& mg_aggregate_renumber_map = std::get<1>(mg_aux_info);
        mg_map = raft::device_span<vertex_t const>(mg_aggregate_renumber_map.data(),
                                                   mg_aggregate_renumber_map.size());
      }

      sg_values.resize(mg_aggregate_values.size(), handle.get_stream());
      thrust::transform(
        handle.get_thrust_policy(),
        (*sg_renumber_map).begin(),
        (*sg_renumber_map).end(),
        sg_values.begin(),
        [mg_aggregate_renumber_map = mg_map ? thrust::make_optional(*mg_map) : thrust::nullopt,
         mg_aggregate_values       = raft::device_span<value_t const>(
           mg_aggregate_values.data(), mg_aggregate_values.size())] __device__(auto sg_v) {
          size_t offset{0};
          if (mg_aggregate_renumber_map) {
            auto it = thrust::lower_bound(thrust::seq,
                                          (*mg_aggregate_renumber_map).begin(),
                                          (*mg_aggregate_renumber_map).end(),
                                          sg_v);
            assert(*it == sg_v);
            offset =
              static_cast<size_t>(thrust::distance((*mg_aggregate_renumber_map).begin(), it));
          } else {
            offset = static_cast<size_t>(sg_v);
          }
          return mg_aggregate_values[offset];
        });
    } else {
      sg_values = std::move(mg_aggregate_values);
    }
  }

  return sg_values;
}

}  // namespace test
}  // namespace cugraph
