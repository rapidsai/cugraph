/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/prims/transform_e.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>

#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

template <typename T, bool multi_gpu>
void run_ecl_mg_reference_case(raft::handle_t const& h, int n, bool masked)
{
  int rank, ranks;
  rank  = 0;
  ranks = 1;
  if constexpr (multi_gpu) {
    rank  = h.get_comms().get_rank();
    ranks = h.get_comms().get_size();
  }
  std::vector<T> all_src, all_dst;
  auto edge = [&](int u, int v) {
    all_src.push_back(u);
    all_dst.push_back(v);
  };
  if (n == 1) edge(0, 0);
  if (n == 2) edge(0, 1);
  if (n == 7) {
    edge(0, 1);
    edge(1, 0);
    edge(1, 2);
    edge(2, 3);
    edge(3, 2);
    edge(3, 4);
    edge(2, 2);
    edge(0, 1);
  }
  if (n == 37) {
    for (int u = 0; u < n - 1; ++u)
      edge(u, u + 1);
  }
  if (n == 193) {
    for (int u = 1; u < 100; ++u) {
      edge(0, u);
      edge(u, 0);
      edge(u, 100);
    }
    for (int u = 100; u < 180; ++u) {
      edge(u, 100 + (u - 100 + 1) % 80);
      if (u % 3 == 0) edge(u, 181);
    }
    edge(181, 182);
    edge(182, 181);
    edge(182, 183);
  }
  if (n == 257) {
    // Skewed degrees, isolated vertices, cross-owner SCCs, one-way links,
    // and duplicate edges exercise both independently renumbered orientations.
    for (int v = 1; v < 180; ++v) {
      edge(0, v);
      edge(v, 0);
      edge(v, 180);
    }
    for (int v = 180; v < 230; ++v) {
      edge(v, 180 + (v - 180 + 1) % 50);
      edge(v, 240);
    }
    edge(240, 241);
    edge(241, 240);
    edge(241, 242);
    edge(0, 1);
    edge(0, 1);
  }
  if (n == 73) {
    std::mt19937 random(1729);
    for (int u = 0; u < 65; ++u) {
      int first = (u / 5) * 5;
      edge(u, first + (u - first + 1) % 5);
      edge(u, u);
      for (int j = 0; j < 3; ++j) {
        int v = first + random() % (n - first);
        edge(u, v);
      }
    }
    edge(0, 1);
    edge(0, 1);
  }
  std::vector<std::vector<bool>> reachable(n, std::vector<bool>(n, false));
  for (int v = 0; v < n; ++v)
    reachable[v][v] = true;
  for (size_t e = 0; e < all_src.size(); ++e)
    if (!masked || (all_src[e] * 17 + all_dst[e] * 13) % 5 != 0)
      reachable[all_src[e]][all_dst[e]] = true;
  for (int k = 0; k < n; ++k)
    for (int u = 0; u < n; ++u)
      for (int v = 0; v < n; ++v)
        reachable[u][v] = reachable[u][v] || (reachable[u][k] && reachable[k][v]);
  std::vector<T> vertices, src, dst;
  for (int v = 0; v < n; ++v)
    if (v % ranks == rank) vertices.push_back(v);
  for (size_t e = 0; e < all_src.size(); ++e)
    if (e % ranks == size_t(rank)) {
      src.push_back(all_src[e]);
      dst.push_back(all_dst[e]);
    }
  auto stream = h.get_stream();
  rmm::device_uvector<T> dv(vertices.size(), stream), ds(src.size(), stream),
    dd(dst.size(), stream);
  raft::update_device(dv.data(), vertices.data(), vertices.size(), stream);
  raft::update_device(ds.data(), src.data(), src.size(), stream);
  raft::update_device(dd.data(), dst.data(), dst.size(), stream);
  if constexpr (multi_gpu) {
    std::tie(dv, std::ignore) = cugraph::shuffle_ext_vertices(
      h, std::move(dv), std::vector<cugraph::arithmetic_device_uvector_t>{});
    std::tie(ds, dd, std::ignore) = cugraph::shuffle_ext_edges(
      h, std::move(ds), std::move(dd), std::vector<cugraph::arithmetic_device_uvector_t>{}, false);
  }
  auto [graph, properties, map] = cugraph::create_graph_from_edgelist<T, T, false, multi_gpu>(
    h,
    std::move(dv),
    std::move(ds),
    std::move(dd),
    std::vector<cugraph::arithmetic_device_uvector_t>{},
    cugraph::graph_properties_t{false, true},
    true,
    std::nullopt,
    std::nullopt,
    true);
  auto view = graph.view();
  std::optional<cugraph::edge_property_t<T, bool>> mask;
  if (masked) {
    cugraph::edge_src_property_t<T, T> source(h, view);
    cugraph::edge_dst_property_t<T, T> destination(h, view);
    cugraph::update_edge_src_property(h, view, map->begin(), source.mutable_view());
    cugraph::update_edge_dst_property(h, view, map->begin(), destination.mutable_view());
    mask.emplace(h, view);
    cugraph::transform_e(
      h,
      view,
      source.view(),
      destination.view(),
      cugraph::edge_dummy_property_t{}.view(),
      [] __device__(T, T, T src, T dst, cuda::std::nullopt_t) -> bool {
        return (src * 17 + dst * 13) % 5 != 0;
      },
      mask->mutable_view());
    view.attach_edge_mask(mask->view());
  }
  auto labels = cugraph::strongly_connected_components(h, view, true);
  std::vector<T> global_vertices(n), global_labels(n);
  if constexpr (multi_gpu) {
    rmm::device_uvector<T> queries(n, stream);
    cugraph::sequence(h.get_thrust_policy(), queries.begin(), queries.end(), T{0});
    auto vertices =
      cugraph::collect_values_for_int_vertices(h,
                                               queries.begin(),
                                               queries.end(),
                                               map->data(),
                                               view.vertex_partition_range_lasts(),
                                               view.local_vertex_partition_range_first());
    auto values =
      cugraph::collect_values_for_int_vertices(h,
                                               queries.begin(),
                                               queries.end(),
                                               labels.data(),
                                               view.vertex_partition_range_lasts(),
                                               view.local_vertex_partition_range_first());
    raft::update_host(global_vertices.data(), vertices.data(), n, stream);
    raft::update_host(global_labels.data(), values.data(), n, stream);
    h.sync_stream();
  } else {
    raft::update_host(global_vertices.data(), map->data(), n, stream);
    raft::update_host(global_labels.data(), labels.data(), n, stream);
    h.sync_stream();
  }
  std::vector<T> ordered(n, T{-1});
  for (int i = 0; i < n; ++i) {
    if (global_vertices[i] < 0 || global_vertices[i] >= n)
      throw std::runtime_error("invalid vertex");
    ordered[global_vertices[i]] = global_labels[i];
  }
  for (int u = 0; u < n; ++u)
    for (int v = 0; v < n; ++v)
      if (ordered[u] < 0 || ordered[u] >= n ||
          ((ordered[u] == ordered[v]) != (reachable[u][v] && reachable[v][u])))
        throw std::runtime_error("serial partition mismatch n=" + std::to_string(n) +
                                 " u=" + std::to_string(u) + " v=" + std::to_string(v));
  if (rank == 0)
    std::cout << "PASS width=" << sizeof(T) * 8 << " n=" << n << " mask=" << masked << std::endl;
}
}  // namespace

void run_ecl_mg_reference_cases(raft::handle_t const& handle)
{
  for (int n : {0, 1, 2, 7, 37, 73, 193, 257}) {
    for (bool masked : {false, true}) {
      run_ecl_mg_reference_case<int32_t, true>(handle, n, masked);
      run_ecl_mg_reference_case<int64_t, true>(handle, n, masked);
    }
  }
}
