/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <prims/reduce_op.cuh>
#include <prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh>
#include <prims/vertex_frontier.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>

namespace cugraph {

namespace {

template <typename vertex_t>
struct e_op_t {
  __device__ thrust::optional<size_t> operator()(thrust::tuple<vertex_t, size_t> tagged_src,
                                                 vertex_t,
                                                 thrust::nullopt_t,
                                                 thrust::nullopt_t,
                                                 thrust::nullopt_t) const
  {
    return thrust::get<1>(tagged_src);
  }
};

struct compute_gpu_id_t {
  raft::device_span<size_t> lasts{};

  __device__ int operator()(size_t i) const
  {
    return static_cast<int>(thrust::distance(
      lasts.begin(), thrust::upper_bound(thrust::seq, lasts.begin(), lasts.end(), i)));
  }
};

}  // namespace

namespace detail {

template <typename GraphViewType>
std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<typename GraphViewType::vertex_type>>
k_hop_nbrs(raft::handle_t const& handle,
           GraphViewType const& push_graph_view,
           raft::device_span<typename GraphViewType::vertex_type const> start_vertices,
           size_t k,
           bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;

  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  // 1. check input arguments

  std::vector<size_t> start_vertex_counts{};
  if constexpr (GraphViewType::is_multi_gpu) {
    start_vertex_counts =
      host_scalar_allgather(handle.get_comms(), start_vertices.size(), handle.get_stream());
  } else {
    start_vertex_counts = std::vector<size_t>{start_vertices.size()};
  }
  std::vector<size_t> start_vertex_displacements(start_vertex_counts.size());
  if constexpr (GraphViewType::is_multi_gpu) {
    std::exclusive_scan(start_vertex_counts.begin(),
                        start_vertex_counts.end(),
                        start_vertex_displacements.begin(),
                        size_t{0});
  } else {
    start_vertex_displacements[0] = 0;
  }
  CUGRAPH_EXPECTS(start_vertex_displacements.back() + start_vertex_counts.back() > 0,
                  "Invalid input argument: input should have at least one starting vertex.");

  CUGRAPH_EXPECTS(k > 0, "Invalid input argument: k should be a positive integer.");

  if (do_expensive_check) {
    auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
      push_graph_view.local_vertex_partition_view());
    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       start_vertices.begin(),
                       start_vertices.end(),
                       [vertex_partition] __device__(auto val) {
                         return !(vertex_partition.is_valid_vertex(val) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(val));
                       });
    if constexpr (GraphViewType::is_multi_gpu) {
      num_invalid_vertices = host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: start_vertices have invalid vertex IDs.");
  }

  // 2. initialize the frontier

  constexpr size_t bucket_idx_cur = 0;
  constexpr size_t num_buckets    = 1;

  vertex_frontier_t<vertex_t, size_t, GraphViewType::is_multi_gpu, false> frontier(handle,
                                                                                   num_buckets);

  auto key_first = thrust::make_zip_iterator(
    start_vertices.begin(),
    thrust::make_counting_iterator(
      start_vertex_displacements[GraphViewType::is_multi_gpu ? handle.get_comms().get_rank() : 0]));
  frontier.bucket(bucket_idx_cur).insert(key_first, key_first + start_vertices.size());

  // 3. K-hop nbrs iteration

  rmm::device_uvector<size_t> start_vertex_indices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> nbrs(0, handle.get_stream());
  for (size_t iter = 0; iter < k; ++iter) {
    auto new_frontier_key_buffer =
      transform_reduce_v_frontier_outgoing_e_by_dst(handle,
                                                    push_graph_view,
                                                    frontier.bucket(bucket_idx_cur),
                                                    edge_src_dummy_property_t{}.view(),
                                                    edge_dst_dummy_property_t{}.view(),
                                                    edge_dummy_property_t{}.view(),
                                                    e_op_t<vertex_t>{},
                                                    reduce_op::null{},
                                                    do_expensive_check);
    if (iter < (k - 1)) {
      frontier.bucket(bucket_idx_cur).clear();
      frontier.bucket(bucket_idx_cur)
        .insert(get_dataframe_buffer_begin(new_frontier_key_buffer),
                get_dataframe_buffer_end(new_frontier_key_buffer));
      frontier.bucket(bucket_idx_cur).shrink_to_fit();
    } else {
      start_vertex_indices = std::move(std::get<1>(new_frontier_key_buffer));
      nbrs                 = std::move(std::get<0>(new_frontier_key_buffer));
    }
  }

  // 4. update offsets (and sort nbrs accordingly)

  if (GraphViewType::is_multi_gpu && (handle.get_comms().get_size() > 1)) {
    rmm::device_uvector<size_t> lasts(handle.get_comms().get_size(), handle.get_stream());
    raft::update_device(lasts.data(),
                        start_vertex_displacements.data() + 1,
                        start_vertex_displacements.size() - 1,
                        handle.get_stream());
    auto num_indices = start_vertex_displacements.back() + start_vertex_counts.back();
    lasts.set_element_async(lasts.size() - 1, num_indices, handle.get_stream());
    std::tie(start_vertex_indices, nbrs, std::ignore) = groupby_gpu_id_and_shuffle_kv_pairs(
      handle.get_comms(),
      start_vertex_indices.begin(),
      start_vertex_indices.end(),
      nbrs.begin(),
      compute_gpu_id_t{raft::device_span<size_t>(lasts.data(), lasts.size())},
      handle.get_stream());
  }
  thrust::sort_by_key(handle.get_thrust_policy(),
                      start_vertex_indices.begin(),
                      start_vertex_indices.end(),
                      nbrs.begin());

  auto num_unique_indices =
    thrust::count_if(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(start_vertex_indices.size()),
                     is_first_in_run_t<size_t const*>{start_vertex_indices.data()});
  rmm::device_uvector<size_t> tmp_indices(num_unique_indices, handle.get_stream());
  rmm::device_uvector<size_t> tmp_counts(num_unique_indices, handle.get_stream());
  thrust::reduce_by_key(handle.get_thrust_policy(),
                        start_vertex_indices.begin(),
                        start_vertex_indices.end(),
                        thrust::make_constant_iterator(size_t{1}),
                        tmp_indices.begin(),
                        tmp_counts.begin());

  rmm::device_uvector<size_t> offsets(start_vertices.size() + size_t{1}, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), offsets.begin(), offsets.end(), size_t{0});
  thrust::scatter(
    handle.get_thrust_policy(),
    tmp_counts.begin(),
    tmp_counts.end(),
    thrust::make_transform_iterator(
      tmp_indices.begin(),
      shift_left_t<size_t>{
        start_vertex_displacements[GraphViewType::is_multi_gpu ? handle.get_comms().get_rank()
                                                               : int{0}]}),
    offsets.begin());
  thrust::exclusive_scan(
    handle.get_thrust_policy(), offsets.begin(), offsets.end(), offsets.begin(), size_t{0});

  return std::make_tuple(std::move(offsets), std::move(nbrs));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<vertex_t>> k_hop_nbrs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> start_vertices,
  size_t k,
  bool do_expensive_check)
{
  return detail::k_hop_nbrs(handle, graph_view, start_vertices, k, do_expensive_check);
}

}  // namespace cugraph
