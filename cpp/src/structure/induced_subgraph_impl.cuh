/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

//#define TIMING

#include <prims/extract_transform_v_frontier_outgoing_e.cuh>
#include <prims/vertex_frontier.cuh>
#include <structure/detail/structure_utils.cuh>
#include <utilities/collect_comm.cuh>

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>
#ifdef TIMING
#include <cugraph/utilities/high_res_timer.hpp>
#endif

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename weight_t, typename property_t>
struct induced_subgraph_weighted_edge_op {
  using return_type = thrust::optional<thrust::tuple<vertex_t, vertex_t, weight_t, size_t>>;

  raft::device_span<size_t const> dst_subgraph_offsets;
  raft::device_span<vertex_t const> dst_subgraph_vertices;

  return_type __device__ operator()(thrust::tuple<vertex_t, size_t> tagged_src,
                                    vertex_t dst,
                                    property_t sv,
                                    property_t dv,
                                    weight_t wgt)
  {
    size_t subgraph = thrust::get<1>(tagged_src);
    return thrust::binary_search(thrust::seq,
                                 dst_subgraph_vertices.data() + dst_subgraph_offsets[subgraph],
                                 dst_subgraph_vertices.data() + dst_subgraph_offsets[subgraph + 1],
                                 dst)
             ? thrust::make_optional(
                 thrust::make_tuple(thrust::get<0>(tagged_src), dst, wgt, subgraph))
             : thrust::nullopt;
  }
};

template <typename vertex_t, typename property_t>
struct induced_subgraph_unweighted_edge_op {
  using return_type = thrust::optional<thrust::tuple<vertex_t, vertex_t, size_t>>;

  raft::device_span<size_t const> dst_subgraph_offsets;
  raft::device_span<vertex_t const> dst_subgraph_vertices;

  return_type __device__ operator()(thrust::tuple<vertex_t, size_t> tagged_src,
                                    vertex_t dst,
                                    property_t sv,
                                    property_t dv,
                                    thrust::nullopt_t)
  {
    size_t subgraph = thrust::get<1>(tagged_src);
    return thrust::binary_search(thrust::seq,
                                 dst_subgraph_vertices.data() + dst_subgraph_offsets[subgraph],
                                 dst_subgraph_vertices.data() + dst_subgraph_offsets[subgraph + 1],
                                 dst)
             ? thrust::make_optional(thrust::make_tuple(thrust::get<0>(tagged_src), dst, subgraph))
             : thrust::nullopt;
  }
};

}  // namespace detail

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
#ifdef TIMING
  HighResTimer hr_timer;
  hr_timer.start("extract_induced_subgraphs");
#endif
  // 1. check input arguments

  if (do_expensive_check) {
    size_t should_be_zero{std::numeric_limits<size_t>::max()};
    size_t num_aggregate_subgraph_vertices{};
    raft::update_host(&should_be_zero, subgraph_offsets.data(), 1, handle.get_stream());
    raft::update_host(&num_aggregate_subgraph_vertices,
                      subgraph_offsets.data() + subgraph_offsets.size() - 1,
                      1,
                      handle.get_stream());
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
    auto& comm               = handle.get_comms();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    dst_subgraph_vertices_v = cugraph::device_allgatherv(handle, row_comm, subgraph_vertices);

    graph_ids_v = cugraph::device_allgatherv(
      handle, row_comm, raft::device_span<size_t const>(graph_ids_v.data(), graph_ids_v.size()));

    thrust::sort(handle.get_thrust_policy(),
                 thrust::make_zip_iterator(graph_ids_v.begin(), dst_subgraph_vertices_v.begin()),
                 thrust::make_zip_iterator(graph_ids_v.end(), dst_subgraph_vertices_v.end()));

    dst_subgraph_offsets_v =
      detail::compute_sparse_offsets<size_t>(graph_ids_v.begin(),
                                             graph_ids_v.end(),
                                             size_t{0},
                                             size_t{subgraph_offsets.size() - 1},
                                             handle.get_stream());

    dst_subgraph_offsets =
      raft::device_span<size_t const>(dst_subgraph_offsets_v.data(), dst_subgraph_offsets_v.size());
  } else {
    dst_subgraph_vertices_v.resize(graph_ids_v.size(), handle.get_stream());
    raft::copy(dst_subgraph_vertices_v.data(),
               subgraph_vertices.data(),
               subgraph_vertices.size(),
               handle.get_stream());
    thrust::sort(handle.get_thrust_policy(),
                 thrust::make_zip_iterator(graph_ids_v.begin(), dst_subgraph_vertices_v.begin()),
                 thrust::make_zip_iterator(graph_ids_v.end(), dst_subgraph_vertices_v.end()));
  }

  graph_ids_v.resize(0, handle.get_stream());
  graph_ids_v.shrink_to_fit(handle.get_stream());

  dst_subgraph_vertices = raft::device_span<vertex_t const>(dst_subgraph_vertices_v.data(),
                                                            dst_subgraph_vertices_v.size());

  // 3. Call extract_transform_v_frontier_e with a functor that
  //    returns thrust::nullopt if the destination vertex has
  //    a property of 0, return the edge if the destination
  //    vertex has a property of 1
  vertex_frontier_t<vertex_t, size_t, multi_gpu, false> vertex_frontier(handle, 1);

  std::vector<size_t> h_subgraph_offsets(subgraph_offsets.size());
  raft::update_host(h_subgraph_offsets.data(),
                    subgraph_offsets.data(),
                    subgraph_offsets.size(),
                    handle.get_stream());

  graph_ids_v = detail::expand_sparse_offsets(subgraph_offsets, size_t{0}, handle.get_stream());

  vertex_frontier.bucket(0).insert(
    thrust::make_zip_iterator(subgraph_vertices.begin(), graph_ids_v.begin()),
    thrust::make_zip_iterator(subgraph_vertices.end(), graph_ids_v.end()));

  graph_ids_v.resize(0, handle.get_stream());
  graph_ids_v.shrink_to_fit(handle.get_stream());

  rmm::device_uvector<vertex_t> edge_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edge_minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> edge_weights{std::nullopt};
  rmm::device_uvector<size_t> subgraph_edge_graph_ids(0, handle.get_stream());

  if (edge_weight_view) {
    edge_weights = std::make_optional(rmm::device_uvector<weight_t>(0, handle.get_stream()));

    std::tie(edge_majors, edge_minors, *edge_weights, subgraph_edge_graph_ids) =
      extract_transform_v_frontier_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        *edge_weight_view,
        detail::induced_subgraph_weighted_edge_op<vertex_t, weight_t, thrust::nullopt_t>{
          dst_subgraph_offsets, dst_subgraph_vertices},
        do_expensive_check);

    thrust::sort_by_key(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(
        subgraph_edge_graph_ids.begin(), edge_majors.begin(), edge_minors.begin()),
      thrust::make_zip_iterator(
        subgraph_edge_graph_ids.end(), edge_majors.end(), edge_minors.end()),
      edge_weights->begin());
  } else {
    std::tie(edge_majors, edge_minors, subgraph_edge_graph_ids) =
      extract_transform_v_frontier_outgoing_e(
        handle,
        graph_view,
        vertex_frontier.bucket(0),
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        detail::induced_subgraph_unweighted_edge_op<vertex_t, thrust::nullopt_t>{
          dst_subgraph_offsets, dst_subgraph_vertices},
        do_expensive_check);

    thrust::sort(handle.get_thrust_policy(),
                 thrust::make_zip_iterator(
                   subgraph_edge_graph_ids.begin(), edge_majors.begin(), edge_minors.begin()),
                 thrust::make_zip_iterator(
                   subgraph_edge_graph_ids.end(), edge_majors.end(), edge_minors.end()));
  }

  auto subgraph_edge_offsets =
    detail::compute_sparse_offsets<size_t>(subgraph_edge_graph_ids.begin(),
                                           subgraph_edge_graph_ids.end(),
                                           size_t{0},
                                           size_t{subgraph_offsets.size() - 1},
                                           handle.get_stream());

#ifdef TIMING
  hr_timer.stop();
  hr_timer.display_and_clear(std::cout);
#endif
  return std::make_tuple(std::move(edge_majors),
                         std::move(edge_minors),
                         std::move(edge_weights),
                         std::move(subgraph_edge_offsets));
}

}  // namespace cugraph
