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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <detail/graph_utils.cuh>
#include <prims/detail/nbr_intersection.cuh>
#include <prims/property_op_utils.cuh>
#include <utilities/collect_comm.cuh>

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace cugraph {

namespace detail {

template <typename VertexPairIterator>
struct compute_local_edge_partition_id_t {
  using vertex_t = typename thrust::
    tuple_element<0, typename thrust::iterator_traits<VertexPairIterator>::value_type>::type;

  VertexPairIterator vertex_pair_first{};
  size_t num_local_edge_partitions{};
  raft::device_span<vertex_t const> edge_partition_major_range_lasts{};

  __device__ int operator()(size_t i) const
  {
    auto major = thrust::get<0>(*(vertex_pair_first + i));
    return static_cast<int>(
      thrust::distance(edge_partition_major_range_lasts.begin(),
                       thrust::upper_bound(thrust::seq,
                                           edge_partition_major_range_lasts.begin(),
                                           edge_partition_major_range_lasts.end(),
                                           major)));
  }
};

template <typename VertexPairIterator>
struct compute_chunk_id_t {
  VertexPairIterator vertex_pair_first{};
  size_t num_chunks{};

  __device__ int operator()(size_t i) const
  {
    return static_cast<int>(thrust::get<1>(*(vertex_pair_first + i)) % num_chunks);
  }
};

template <typename VertexPairIterator>
struct indirection_compare_less_t {
  VertexPairIterator vertex_pair_first{};

  __device__ bool operator()(size_t i, size_t j) const
  {
    return *(vertex_pair_first + i) < *(vertex_pair_first + j);
  }
};

template <typename GraphViewType,
          typename VertexValueInputIterator,
          typename IntersectionOp,
          typename VertexPairIndexIterator,
          typename VertexPairIterator,
          typename VertexPairValueOutputIterator>
struct call_intersection_op_t {
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu>
    edge_partition{};
  thrust::optional<raft::device_span<typename GraphViewType::vertex_type const>> unique_vertices;
  VertexValueInputIterator vertex_property_first;
  IntersectionOp intersection_op{};
  size_t const* nbr_offsets{nullptr};
  typename GraphViewType::vertex_type const* nbr_indices{nullptr};
  VertexPairIndexIterator major_minor_pair_index_first{};
  VertexPairIterator major_minor_pair_first{};
  VertexPairValueOutputIterator major_minor_pair_value_output_first{};

  __device__ void operator()(size_t i) const
  {
    using property_t = typename thrust::iterator_traits<VertexValueInputIterator>::value_type;

    auto index        = *(major_minor_pair_index_first + i);
    auto pair         = *(major_minor_pair_first + index);
    auto major        = thrust::get<0>(pair);
    auto minor        = thrust::get<1>(pair);
    auto src          = GraphViewType::is_storage_transposed ? minor : major;
    auto dst          = GraphViewType::is_storage_transposed ? major : minor;
    auto intersection = raft::device_span<typename GraphViewType::vertex_type const>(
      nbr_indices + nbr_offsets[i], nbr_indices + nbr_offsets[i + 1]);

    property_t src_prop{};
    property_t dst_prop{};
    if (unique_vertices) {
      src_prop = *(vertex_property_first +
                   thrust::distance(
                     (*unique_vertices).begin(),
                     thrust::lower_bound(
                       thrust::seq, (*unique_vertices).begin(), (*unique_vertices).end(), src)));
      dst_prop = *(vertex_property_first +
                   thrust::distance(
                     (*unique_vertices).begin(),
                     thrust::lower_bound(
                       thrust::seq, (*unique_vertices).begin(), (*unique_vertices).end(), dst)));
    } else {
      auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
      auto src_offset   = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
      auto dst_offset   = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
      src_prop          = *(vertex_property_first + src_offset);
      dst_prop          = *(vertex_property_first + dst_offset);
    }
    *(major_minor_pair_value_output_first + index) =
      intersection_op(src, dst, src_prop, dst_prop, intersection);
  }
};

}  // namespace detail

/**
 * @brief Iterate over each input vertex pair and apply a functor to the common destination neighbor
 * list of the pair.
 *
 * Iterate over every vertex pair; intersect destination neighbor lists of the two vertices in the
 * pair; invoke a user-provided functor, and store the functor output.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexPairIterator Type of the iterator for input vertex pairs.
 * @tparam VertexValueInputWrapper Type of the wrapper for vertex property values.
 * @tparam IntersectionOp Type of the quinary per intersection operator.
 * @tparam VertexPairValueOutputIterator Type of the iterator for vertex pair output property
 * variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_pair_first Iterator pointing to the first (inclusive) input vertex pair.
 * @param vertex_pair_last Iterator pointing to the last (exclusive) input vertex pair.
 * @param vertex_src_value_input Wrapper used to access vertex input property values (for the
 * vertices assigned to this process in multi-GPU).
 * @param intersection_op quinary operator takes first vertex of the pair, second vertex of the
 * pair, property values for the first vertex, property values for the second vertex, and a list of
 * vertices in the intersection of the first & second vertices' destination neighbors and returns an
 * output value for the input pair.
 * @param vertex_pair_value_output_first Iterator pointing to the vertex pair property variables for
 * the first vertex pair (inclusive). `vertex_pair_value_output_last` (exclusive) is deduced as @p
 * vertex_pair_value_output_first + @p thrust::distance(vertex_pair_first, vertex_pair_last).
 * @param A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexPairIterator,
          typename VertexValueInputIterator,
          typename IntersectionOp,
          typename VertexPairValueOutputIterator>
void per_v_pair_transform_dst_nbr_intersection(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPairIterator vertex_pair_first,
  VertexPairIterator vertex_pair_last,
  VertexValueInputIterator vertex_value_input_first,
  IntersectionOp intersection_op,
  VertexPairValueOutputIterator vertex_pair_value_output_first,
  bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed);

  using vertex_t   = typename GraphViewType::vertex_type;
  using edge_t     = typename GraphViewType::edge_type;
  using property_t = typename thrust::iterator_traits<VertexValueInputIterator>::value_type;
  using result_t   = typename thrust::iterator_traits<VertexPairValueOutputIterator>::value_type;

  if (do_expensive_check) {
    auto num_invalids =
      detail::count_invalid_vertex_pairs(handle, graph_view, vertex_pair_first, vertex_pair_last);
    CUGRAPH_EXPECTS(num_invalids == 0,
                    "Invalid input arguments: there are invalid input vertex pairs.");
  }

  auto num_input_pairs = static_cast<size_t>(thrust::distance(vertex_pair_first, vertex_pair_last));
  std::optional<rmm::device_uvector<vertex_t>> unique_vertices{std::nullopt};
  std::optional<decltype(allocate_dataframe_buffer<property_t>(size_t{0}, rmm::cuda_stream_view{}))>
    property_buffer_for_unique_vertices{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    unique_vertices  = rmm::device_uvector<vertex_t>(num_input_pairs * 2, handle.get_stream());
    auto elem0_first = thrust::make_transform_iterator(
      vertex_pair_first,
      cugraph::thrust_tuple_get<typename thrust::iterator_traits<VertexPairIterator>::value_type,
                                0>{});
    thrust::copy(handle.get_thrust_policy(),
                 elem0_first,
                 elem0_first + num_input_pairs,
                 (*unique_vertices).begin());
    auto elem1_first = thrust::make_transform_iterator(
      vertex_pair_first,
      cugraph::thrust_tuple_get<typename thrust::iterator_traits<VertexPairIterator>::value_type,
                                1>{});
    thrust::copy(handle.get_thrust_policy(),
                 elem1_first,
                 elem1_first + num_input_pairs,
                 (*unique_vertices).begin() + num_input_pairs);
    thrust::sort(handle.get_thrust_policy(), (*unique_vertices).begin(), (*unique_vertices).end());
    (*unique_vertices)
      .resize(thrust::distance((*unique_vertices).begin(),
                               thrust::unique(handle.get_thrust_policy(),
                                              (*unique_vertices).begin(),
                                              (*unique_vertices).end())),
              handle.get_stream());

    property_buffer_for_unique_vertices =
      collect_values_for_sorted_unique_int_vertices(handle.get_comms(),
                                                    (*unique_vertices).begin(),
                                                    (*unique_vertices).end(),
                                                    vertex_value_input_first,
                                                    graph_view.vertex_partition_range_lasts(),
                                                    handle.get_stream());
  }

  rmm::device_uvector<size_t> vertex_pair_indices(num_input_pairs, handle.get_stream());
  thrust::sequence(
    handle.get_thrust_policy(), vertex_pair_indices.begin(), vertex_pair_indices.end(), size_t{0});

  std::vector<vertex_t> h_edge_partition_major_range_lasts(
    graph_view.number_of_local_edge_partitions());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    h_edge_partition_major_range_lasts[i] = graph_view.local_edge_partition_src_range_last(i);
  }
  rmm::device_uvector<vertex_t> d_edge_partition_major_range_lasts(
    h_edge_partition_major_range_lasts.size(), handle.get_stream());
  raft::update_device(d_edge_partition_major_range_lasts.data(),
                      h_edge_partition_major_range_lasts.data(),
                      h_edge_partition_major_range_lasts.size(),
                      handle.get_stream());
  auto d_edge_partition_group_sizes = groupby_and_count(
    vertex_pair_indices.begin(),
    vertex_pair_indices.end(),
    detail::compute_local_edge_partition_id_t<VertexPairIterator>{
      vertex_pair_first,
      graph_view.number_of_local_edge_partitions(),
      raft::device_span<vertex_t const>(d_edge_partition_major_range_lasts.data(),
                                        d_edge_partition_major_range_lasts.size())},
    static_cast<int>(graph_view.number_of_local_edge_partitions()),
    std::numeric_limits<size_t>::max(),
    handle.get_stream());
  std::vector<size_t> h_edge_partition_group_sizes(d_edge_partition_group_sizes.size());
  raft::update_host(h_edge_partition_group_sizes.data(),
                    d_edge_partition_group_sizes.data(),
                    d_edge_partition_group_sizes.size(),
                    handle.get_stream());
  handle.sync_stream();
  std::vector<size_t> h_edge_partition_group_displacements(h_edge_partition_group_sizes.size());
  std::exclusive_scan(h_edge_partition_group_sizes.begin(),
                      h_edge_partition_group_sizes.end(),
                      h_edge_partition_group_displacements.begin(),
                      size_t{0});

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    auto edge_partition_vertex_pair_index_first =
      vertex_pair_indices.begin() + h_edge_partition_group_displacements[i];

    // FIXME: Peak memory requirement is also dependent on the average minimum degree of the input
    // vertex pairs. We may need a more sophisticated mechanism to set max_chunk_size considering
    // vertex degrees. to limit memory footprint ((1 << 15) is a tuning parameter)
    auto max_chunk_size =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 15);
    auto max_num_chunks = (h_edge_partition_group_sizes[i] + max_chunk_size - 1) / max_chunk_size;
    if constexpr (GraphViewType::is_multi_gpu) {
      max_num_chunks = host_scalar_allreduce(
        handle.get_comms(), max_num_chunks, raft::comms::op_t::MAX, handle.get_stream());
    }

    std::vector<size_t> h_chunk_sizes(max_num_chunks);
    if (h_chunk_sizes.size() > size_t{1}) {
      auto d_chunk_sizes = groupby_and_count(
        edge_partition_vertex_pair_index_first,
        edge_partition_vertex_pair_index_first + h_edge_partition_group_sizes[i],
        detail::compute_chunk_id_t<VertexPairIterator>{vertex_pair_first, max_num_chunks},
        static_cast<int>(max_num_chunks),
        std::numeric_limits<size_t>::max(),
        handle.get_stream());
      raft::update_host(
        h_chunk_sizes.data(), d_chunk_sizes.data(), d_chunk_sizes.size(), handle.get_stream());
      handle.sync_stream();
    } else if (h_chunk_sizes.size() == size_t{1}) {
      h_chunk_sizes[0] = h_edge_partition_group_sizes[i];
    }

    auto chunk_vertex_pair_index_first = edge_partition_vertex_pair_index_first;
    for (size_t j = 0; j < h_chunk_sizes.size(); ++j) {
      auto this_chunk_size = h_chunk_sizes[j];

      thrust::sort(handle.get_thrust_policy(),
                   chunk_vertex_pair_index_first,
                   chunk_vertex_pair_index_first + this_chunk_size,
                   detail::indirection_compare_less_t<VertexPairIterator>{
                     vertex_pair_first});  // detail::nbr_intersection() requires the input vertex
                                           // pairs to be sorted.

      // FIXME: better restrict detail::nbr_intersection input vertex pairs to a single edge
      // partition? This may provide additional performance improvement opportunities???
      auto chunk_vertex_pair_first = thrust::make_transform_iterator(
        chunk_vertex_pair_index_first,
        detail::indirection_t<VertexPairIterator>{vertex_pair_first});
      auto [intersection_offsets, intersection_indices] =
        detail::nbr_intersection(handle,
                                 graph_view,
                                 chunk_vertex_pair_first,
                                 chunk_vertex_pair_first + this_chunk_size,
                                 std::array<bool, 2>{true, true},
                                 do_expensive_check);

      if (unique_vertices) {
        auto vertex_value_input_for_unique_vertices_first =
          get_dataframe_buffer_begin(*property_buffer_for_unique_vertices);
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(this_chunk_size),
          detail::call_intersection_op_t<GraphViewType,
                                         decltype(vertex_value_input_for_unique_vertices_first),
                                         IntersectionOp,
                                         decltype(chunk_vertex_pair_index_first),
                                         VertexPairIterator,
                                         VertexPairValueOutputIterator>{
            edge_partition,
            thrust::make_optional<raft::device_span<vertex_t const>>((*unique_vertices).data(),
                                                                     (*unique_vertices).size()),
            vertex_value_input_for_unique_vertices_first,
            intersection_op,
            intersection_offsets.data(),
            intersection_indices.data(),
            chunk_vertex_pair_index_first,
            vertex_pair_first,
            vertex_pair_value_output_first});
      } else {
        thrust::for_each(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(this_chunk_size),
                         detail::call_intersection_op_t<GraphViewType,
                                                        VertexValueInputIterator,
                                                        IntersectionOp,
                                                        decltype(chunk_vertex_pair_index_first),
                                                        VertexPairIterator,
                                                        VertexPairValueOutputIterator>{
                           edge_partition,
                           thrust::optional<raft::device_span<vertex_t const>>{thrust::nullopt},
                           vertex_value_input_first,
                           intersection_op,
                           intersection_offsets.data(),
                           intersection_indices.data(),
                           chunk_vertex_pair_index_first,
                           vertex_pair_first,
                           vertex_pair_value_output_first});
      }

      chunk_vertex_pair_index_first += this_chunk_size;
    }
  }
}

}  // namespace cugraph
