/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "structure/detail/structure_utils.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/device_comm_wrapper.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/edge_bucket.cuh>
#include <cugraph/prims/extract_transform_if_v_frontier_incoming_outgoing_e.cuh>
#include <cugraph/prims/transform_gather_e.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {

namespace detail {

template <typename vertex_t>
struct induced_subgraph_edge_op {
  cuda::std::tuple<vertex_t, vertex_t, size_t> __device__
  operator()(cuda::std::tuple<vertex_t, size_t> tagged_src,
             vertex_t dst,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t) const
  {
    return cuda::std::make_tuple(cuda::std::get<0>(tagged_src), dst, cuda::std::get<1>(tagged_src));
  }
};

template <typename vertex_t, typename edge_t>
struct induced_subgraph_multi_edge_index_edge_op {
  cuda::std::tuple<vertex_t, vertex_t, edge_t, size_t> __device__
  operator()(cuda::std::tuple<vertex_t, size_t> tagged_src,
             vertex_t dst,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t,
             edge_t multi_edge_index) const
  {
    return cuda::std::make_tuple(
      cuda::std::get<0>(tagged_src), dst, multi_edge_index, cuda::std::get<1>(tagged_src));
  }
};

template <typename vertex_t>
struct induced_subgraph_pred_op {
  raft::device_span<size_t const> dst_subgraph_offsets;
  raft::device_span<vertex_t const> dst_subgraph_vertices;

  template <typename edge_value_t>
  bool __device__ operator()(cuda::std::tuple<vertex_t, size_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             edge_value_t) const
  {
    size_t subgraph = cuda::std::get<1>(tagged_src);
    return thrust::binary_search(thrust::seq,
                                 dst_subgraph_vertices.data() + dst_subgraph_offsets[subgraph],
                                 dst_subgraph_vertices.data() + dst_subgraph_offsets[subgraph + 1],
                                 dst);
  }
};

struct induced_subgraph_return_edge_property_op {
  template <typename vertex_t, typename T>
  T __device__
  operator()(vertex_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, T edge_property) const
  {
    return edge_property;
  }
};

}  // namespace detail

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           rmm::device_uvector<size_t>>
extract_induced_subgraphs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::vector<edge_arithmetic_property_view_t<edge_t>> const& edge_property_views,
  raft::device_span<size_t const> subgraph_offsets,
  raft::device_span<vertex_t const> subgraph_vertices,
  bool do_expensive_check)
{
  // 1. check input arguments

  if (do_expensive_check) {
    size_t should_be_zero{std::numeric_limits<size_t>::max()};
    raft::update_host(&should_be_zero, subgraph_offsets.data(), 1, handle.get_stream());
    handle.sync_stream();
    CUGRAPH_EXPECTS(should_be_zero == 0,
                    "Invalid input argument: subgraph_offsets[0] should be 0.");

    CUGRAPH_EXPECTS(thrust::is_sorted(
                      handle.get_thrust_policy(), subgraph_offsets.begin(), subgraph_offsets.end()),
                    "Invalid input argument: subgraph_offsets is not sorted.");
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());

    CUGRAPH_EXPECTS(
      thrust::count_if(handle.get_thrust_policy(),
                       subgraph_vertices.begin(),
                       subgraph_vertices.end(),
                       [vertex_partition] __device__(auto v) {
                         return !vertex_partition.is_valid_vertex(v) ||
                                !vertex_partition.in_local_vertex_partition_range_nocheck(v);
                       }) == 0,
      "Invalid input argument: subgraph_vertices has invalid vertex IDs.");
  }

  // 2. Need to create list of src_subgraph_vertices and
  //    dst_subgraph_vertices and corresponding src_subgraph_offsets
  //    and dst_subgraph_offsets to use for checking endpoints
  //    of edges.
  //
  rmm::device_uvector<vertex_t> dst_subgraph_vertices_v(0, handle.get_stream());
  rmm::device_uvector<size_t> dst_subgraph_offsets_v(0, handle.get_stream());

  raft::device_span<size_t const> dst_subgraph_offsets{subgraph_offsets};
  raft::device_span<vertex_t const> dst_subgraph_vertices{subgraph_vertices};

  auto graph_ids_v =
    detail::expand_sparse_offsets(subgraph_offsets, size_t{0}, handle.get_stream());

  if constexpr (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();
    auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());

    dst_subgraph_vertices_v =
      cugraph::detail::device_allgatherv(handle, major_comm, subgraph_vertices);

    graph_ids_v = cugraph::detail::device_allgatherv(
      handle, major_comm, raft::device_span<size_t const>(graph_ids_v.data(), graph_ids_v.size()));

    cugraph::sort(handle.get_thrust_policy(),
                  thrust::make_zip_iterator(graph_ids_v.begin(), dst_subgraph_vertices_v.begin()),
                  thrust::make_zip_iterator(graph_ids_v.end(), dst_subgraph_vertices_v.end()));

    dst_subgraph_offsets_v =
      detail::compute_sparse_offsets<size_t>(handle,
                                             graph_ids_v.begin(),
                                             graph_ids_v.end(),
                                             size_t{0},
                                             size_t{subgraph_offsets.size() - 1},
                                             true);

    dst_subgraph_offsets =
      raft::device_span<size_t const>(dst_subgraph_offsets_v.data(), dst_subgraph_offsets_v.size());
  } else {
    dst_subgraph_vertices_v.resize(graph_ids_v.size(), handle.get_stream());
    raft::copy(dst_subgraph_vertices_v.data(),
               subgraph_vertices.data(),
               subgraph_vertices.size(),
               handle.get_stream());
    cugraph::sort(handle.get_thrust_policy(),
                  thrust::make_zip_iterator(graph_ids_v.begin(), dst_subgraph_vertices_v.begin()),
                  thrust::make_zip_iterator(graph_ids_v.end(), dst_subgraph_vertices_v.end()));
  }

  graph_ids_v.resize(0, handle.get_stream());
  graph_ids_v.shrink_to_fit(handle.get_stream());

  dst_subgraph_vertices = raft::device_span<vertex_t const>(dst_subgraph_vertices_v.data(),
                                                            dst_subgraph_vertices_v.size());

  // 3. Call extract_transform_v_frontier_outgoing_e with a functor that returns cuda::std::nullopt
  // if the destination vertex has a property of 0, return the edge if the destination vertex has a
  // property of 1
  vertex_frontier_t<vertex_t, size_t, multi_gpu, false> vertex_frontier(handle, 1);

  graph_ids_v = detail::expand_sparse_offsets(subgraph_offsets, size_t{0}, handle.get_stream());

  vertex_frontier.bucket(0).insert(
    thrust::make_zip_iterator(subgraph_vertices.begin(), graph_ids_v.begin()),
    thrust::make_zip_iterator(subgraph_vertices.end(), graph_ids_v.end()));

  graph_ids_v.resize(0, handle.get_stream());
  graph_ids_v.shrink_to_fit(handle.get_stream());

  rmm::device_uvector<vertex_t> edge_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edge_dsts(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_t>> edge_multi_indices{std::nullopt};
  rmm::device_uvector<size_t> subgraph_edge_graph_ids(0, handle.get_stream());

  if ((edge_property_views.size() > 0) && graph_view.is_multigraph()) {
    edge_multi_indices = rmm::device_uvector<edge_t>(0, handle.get_stream());
    std::tie(edge_srcs, edge_dsts, *edge_multi_indices, subgraph_edge_graph_ids) =
      extract_transform_if_v_frontier_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_multi_index_property_t<edge_t, vertex_t>(handle, graph_view).view(),
        detail::induced_subgraph_multi_edge_index_edge_op<vertex_t, edge_t>{},
        detail::induced_subgraph_pred_op<vertex_t>{dst_subgraph_offsets, dst_subgraph_vertices},
        do_expensive_check);

    thrust::sort_by_key(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(
        subgraph_edge_graph_ids.begin(), edge_srcs.begin(), edge_dsts.begin()),
      thrust::make_zip_iterator(subgraph_edge_graph_ids.end(), edge_srcs.end(), edge_dsts.end()),
      edge_multi_indices->begin());
  } else {
    std::tie(edge_srcs, edge_dsts, subgraph_edge_graph_ids) =
      extract_transform_if_v_frontier_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        detail::induced_subgraph_edge_op<vertex_t>{},
        detail::induced_subgraph_pred_op<vertex_t>{dst_subgraph_offsets, dst_subgraph_vertices},
        do_expensive_check);

    cugraph::sort(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(
        subgraph_edge_graph_ids.begin(), edge_srcs.begin(), edge_dsts.begin()),
      thrust::make_zip_iterator(subgraph_edge_graph_ids.end(), edge_srcs.end(), edge_dsts.end()));
  }

  auto subgraph_edge_offsets =
    detail::compute_sparse_offsets<size_t>(handle,
                                           subgraph_edge_graph_ids.begin(),
                                           subgraph_edge_graph_ids.end(),
                                           size_t{0},
                                           size_t{subgraph_offsets.size() - 1},
                                           true);
  subgraph_edge_graph_ids.resize(0, handle.get_stream());
  subgraph_edge_graph_ids.shrink_to_fit(handle.get_stream());

  std::vector<arithmetic_device_uvector_t> edge_properties{};
  if (edge_property_views.size() > 0) {
    // the same edge can appear in multiple subgraphs, so the edge list is not unique
    edge_bucket_t<vertex_t, edge_t, true /* src_major */, multi_gpu, false /* sorted_unique */>
      edge_list(handle, std::move(edge_srcs), std::move(edge_dsts), std::move(edge_multi_indices));
    for (auto const& edge_property_view : edge_property_views) {
      variant_type_dispatch(edge_property_view,
                            [&handle, &graph_view, &edge_list, &edge_properties](auto view) {
                              using T = typename decltype(view)::value_type;
                              rmm::device_uvector<T> values(edge_list.size(), handle.get_stream());
                              transform_gather_e(handle,
                                                 graph_view,
                                                 edge_list,
                                                 edge_src_dummy_property_t{}.view(),
                                                 edge_dst_dummy_property_t{}.view(),
                                                 view,
                                                 detail::induced_subgraph_return_edge_property_op{},
                                                 values.begin());
                              edge_properties.push_back(std::move(values));
                            });
    }
    std::tie(edge_srcs, edge_dsts, std::ignore) = edge_list.take_edgelist();
  }

  return std::make_tuple(std::move(edge_srcs),
                         std::move(edge_dsts),
                         std::move(edge_properties),
                         std::move(subgraph_edge_offsets));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract_induced_subgraphs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<size_t const> subgraph_offsets,
  raft::device_span<vertex_t const> subgraph_vertices,
  bool do_expensive_check)
{
  auto [edge_srcs, edge_dsts, edge_properties, subgraph_edge_offsets] = extract_induced_subgraphs(
    handle,
    graph_view,
    edge_weight_view ? std::vector<edge_arithmetic_property_view_t<edge_t>>{*edge_weight_view}
                     : std::vector<edge_arithmetic_property_view_t<edge_t>>{},
    subgraph_offsets,
    subgraph_vertices,
    do_expensive_check);

  return std::make_tuple(
    std::move(edge_srcs),
    std::move(edge_dsts),
    edge_weight_view
      ? std::make_optional(std::move(std::get<rmm::device_uvector<weight_t>>(edge_properties[0])))
      : std::nullopt,
    std::move(subgraph_edge_offsets));
}

}  // namespace cugraph
