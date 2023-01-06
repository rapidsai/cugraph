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

template <typename vertex_t>
struct compute_chunk_id_t {
  size_t num_chunks{};

  __device__ int operator()(thrust::tuple<vertex_t, vertex_t> tup) const
  {
    return static_cast<int>(thrust::get<1>(tup) % num_chunks);
  }
};

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename IntersectionOp,
          typename VertexPairIterator>
struct call_intersection_op_t {
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu>
    edge_partition{};
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input{};
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input{};
  IntersectionOp intersection_op{};
  size_t const* nbr_offsets{nullptr};
  typename GraphViewType::vertex_type const* nbr_indices{nullptr};
  VertexPairIterator major_minor_pair_first{};

  __device__ auto operator()(size_t i) const
  {
    auto pair         = *(major_minor_pair_first + i);
    auto major        = thrust::get<0>(pair);
    auto minor        = thrust::get<1>(pair);
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
    auto src          = GraphViewType::is_storage_transposed ? minor : major;
    auto dst          = GraphViewType::is_storage_transposed ? major : minor;
    auto src_offset   = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
    auto dst_offset   = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
    auto intersection = raft::device_span<typename GraphViewType::vertex_type const>(
      nbr_indices + nbr_offsets[i], nbr_indices + nbr_offsets[i + 1]);
    return intersection_op(src,
                           dst,
                           edge_partition_src_value_input.get(src_offset),
                           edge_partition_dst_value_input.get(dst_offset),
                           intersection);
  }
};

// FIXME: better move this elsewhere for reuse
template <typename vertex_t, typename ValueBuffer>
std::tuple<rmm::device_uvector<vertex_t>, ValueBuffer> sort_and_reduce_by_vertices(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  ValueBuffer&& value_buffer)
{
  using value_t = typename thrust::iterator_traits<decltype(
    get_dataframe_buffer_begin(value_buffer))>::value_type;

  thrust::sort_by_key(handle.get_thrust_policy(),
                      vertices.begin(),
                      vertices.end(),
                      get_dataframe_buffer_begin(value_buffer));
  auto num_uniques = thrust::count_if(handle.get_thrust_policy(),
                                      thrust::make_counting_iterator(size_t{0}),
                                      thrust::make_counting_iterator(vertices.size()),
                                      detail::is_first_in_run_t<vertex_t const*>{vertices.data()});
  rmm::device_uvector<vertex_t> reduced_vertices(num_uniques, handle.get_stream());
  auto reduced_value_buffer = allocate_dataframe_buffer<value_t>(num_uniques, handle.get_stream());
  thrust::reduce_by_key(handle.get_thrust_policy(),
                        vertices.begin(),
                        vertices.end(),
                        get_dataframe_buffer_begin(value_buffer),
                        reduced_vertices.begin(),
                        get_dataframe_buffer_begin(reduced_value_buffer));

  vertices.resize(size_t{0}, handle.get_stream());
  resize_dataframe_buffer(value_buffer, size_t{0}, handle.get_stream());
  vertices.shrink_to_fit(handle.get_stream());
  shrink_to_fit_dataframe_buffer(value_buffer, handle.get_stream());

  return std::make_tuple(std::move(reduced_vertices), std::move(reduced_value_buffer));
}

template <typename vertex_t, typename ValueIterator>
struct segmented_fill_t {
  size_t const* segment_offsets{nullptr};
  ValueIterator fill_value_first{};
  ValueIterator output_value_first{};

  __device__ void operator()(size_t i) const
  {
    auto value = *(fill_value_first + i);
    // FIXME: this can lead to thread-divergence with a mix of segment sizes (better optimize if
    // this becomes a performance bottleneck)
    thrust::fill(thrust::seq,
                 output_value_first + segment_offsets[i],
                 output_value_first + segment_offsets[i + 1],
                 value);
  }
};

template <typename vertex_t, typename VertexValueOutputIterator>
struct accumulate_vertex_property_t {
  using value_type = typename thrust::iterator_traits<VertexValueOutputIterator>::value_type;

  vertex_t local_vertex_partition_range_first{};
  VertexValueOutputIterator vertex_value_output_first{};
  property_op<value_type, thrust::plus> vertex_property_add{};

  __device__ void operator()(thrust::tuple<vertex_t, value_type> pair) const
  {
    auto v        = thrust::get<0>(pair);
    auto val      = thrust::get<1>(pair);
    auto v_offset = v - local_vertex_partition_range_first;
    *(vertex_value_output_first + v_offset) =
      vertex_property_add(*(vertex_value_output_first + v_offset), val);
  }
};

}  // namespace detail

/**
 * @brief Iterate over each edge and apply a functor to the common destination neighbor list of the
 * edge endpoints, reduce the functor output values per-vertex.
 *
 * Iterate over every edge; intersect destination neighbor lists of source vertex & destination
 * vertex; invoke a user-provided functor per intersection, and reduce the functor output
 * values (thrust::tuple of three values having the same type: one for source, one for destination,
 * and one for every vertex in the intersection) per-vertex. We may add
 * transform_reduce_triplet_of_dst_nbr_intersection_of_e_endpoints_by_v in the future to allow
 * emitting different values for different vertices in the intersection of edge endpoints. This
 * function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam IntersectionOp Type of the quinary per intersection operator.
 * @tparam T Type of the initial value for per-vertex reduction.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param intersection_op quinary operator takes edge source, edge destination, property values for
 * the source, property values for the destination, and a list of vertices in the intersection of
 * edge source & destination vertices' destination neighbors and returns a thrust::tuple of three
 * values: one value per source vertex, one value for destination vertex, and one value for every
 * vertex in the intersection.
 * @param init Initial value to be added to the reduced @p intersection_op return values for each
 * vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the
 * first (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename IntersectionOp,
          typename T,
          typename VertexValueOutputIterator>
void transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  IntersectionOp intersection_op,
  T init,
  VertexValueOutputIterator vertex_value_output_first,
  bool do_expensive_check = false)
{
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<VertexValueOutputIterator>::value_type, T>);

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = float;  // dummy

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    std::conditional_t<GraphViewType::is_storage_transposed,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcValueInputWrapper::value_iterator>,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcValueInputWrapper::value_iterator>>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    std::conditional_t<GraphViewType::is_storage_transposed,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstValueInputWrapper::value_iterator>,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstValueInputWrapper::value_iterator>>>;

  if (do_expensive_check) {
    // currently, nothing to do.
  }

  thrust::fill(handle.get_thrust_policy(),
               vertex_value_output_first,
               vertex_value_output_first + graph_view.local_vertex_partition_range_size(),
               init);

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    edge_partition_src_input_device_view_t edge_partition_src_value_input{};
    edge_partition_dst_input_device_view_t edge_partition_dst_value_input{};
    if constexpr (GraphViewType::is_storage_transposed) {
      edge_partition_src_value_input = edge_partition_src_input_device_view_t(edge_src_value_input);
      edge_partition_dst_value_input =
        edge_partition_dst_input_device_view_t(edge_dst_value_input, i);
    } else {
      edge_partition_src_value_input =
        edge_partition_src_input_device_view_t(edge_src_value_input, i);
      edge_partition_dst_value_input = edge_partition_dst_input_device_view_t(edge_dst_value_input);
    }

    rmm::device_uvector<vertex_t> majors(edge_partition.number_of_edges(), handle.get_stream());
    rmm::device_uvector<vertex_t> minors(majors.size(), handle.get_stream());

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    detail::decompress_edge_partition_to_edgelist<vertex_t,
                                                  edge_t,
                                                  weight_t,
                                                  GraphViewType::is_multi_gpu>(handle,
                                                                               edge_partition,
                                                                               std::nullopt,
                                                                               majors.data(),
                                                                               minors.data(),
                                                                               std::nullopt,
                                                                               segment_offsets);

    auto vertex_pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(majors.begin(), minors.begin()));

    // FIXME: Peak memory requirement is also dependent on the average minimum degree of the input
    // vertex pairs. We may need a more sophisticated mechanism to set max_chunk_size considering
    // vertex degrees. to limit memory footprint ((1 << 15) is a tuning parameter)
    auto max_chunk_size =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 15);
    auto max_num_chunks = (majors.size() + max_chunk_size - 1) / max_chunk_size;
    if constexpr (GraphViewType::is_multi_gpu) {
      max_num_chunks = host_scalar_allreduce(
        handle.get_comms(), max_num_chunks, raft::comms::op_t::MAX, handle.get_stream());
    }

    std::vector<size_t> h_chunk_sizes(max_num_chunks);
    if (h_chunk_sizes.size() > size_t{1}) {
      auto d_chunk_sizes = groupby_and_count(vertex_pair_first,
                                             vertex_pair_first + majors.size(),
                                             detail::compute_chunk_id_t<vertex_t>{max_num_chunks},
                                             static_cast<int>(max_num_chunks),
                                             std::numeric_limits<size_t>::max(),
                                             handle.get_stream());
      raft::update_host(
        h_chunk_sizes.data(), d_chunk_sizes.data(), d_chunk_sizes.size(), handle.get_stream());
      handle.sync_stream();
    } else if (h_chunk_sizes.size() == size_t{1}) {
      h_chunk_sizes[0] = majors.size();
    }

    auto chunk_vertex_pair_first = vertex_pair_first;
    for (size_t j = 0; j < h_chunk_sizes.size(); ++j) {
      auto this_chunk_size = h_chunk_sizes[j];

      thrust::sort(
        handle.get_thrust_policy(),
        chunk_vertex_pair_first,
        chunk_vertex_pair_first + this_chunk_size);  // detail::nbr_intersection() requires the
                                                     // input vertex pairs to be sorted.

      auto [intersection_offsets, intersection_indices] =
        detail::nbr_intersection(handle,
                                 graph_view,
                                 chunk_vertex_pair_first,
                                 chunk_vertex_pair_first + this_chunk_size,
                                 std::array<bool, 2>{true, true},
                                 do_expensive_check);

      auto src_value_buffer = allocate_dataframe_buffer<T>(this_chunk_size, handle.get_stream());
      auto dst_value_buffer = allocate_dataframe_buffer<T>(this_chunk_size, handle.get_stream());
      auto intersection_value_buffer =
        allocate_dataframe_buffer<T>(this_chunk_size, handle.get_stream());

      auto triplet_first = thrust::make_zip_iterator(
        thrust::make_tuple(get_dataframe_buffer_begin(src_value_buffer),
                           get_dataframe_buffer_begin(dst_value_buffer),
                           get_dataframe_buffer_begin(intersection_value_buffer)));
      thrust::tabulate(handle.get_thrust_policy(),
                       triplet_first,
                       triplet_first + this_chunk_size,
                       detail::call_intersection_op_t<GraphViewType,
                                                      edge_partition_src_input_device_view_t,
                                                      edge_partition_dst_input_device_view_t,
                                                      IntersectionOp,
                                                      decltype(chunk_vertex_pair_first)>{
                         edge_partition,
                         edge_partition_src_value_input,
                         edge_partition_dst_value_input,
                         intersection_op,
                         intersection_offsets.data(),
                         intersection_indices.data(),
                         chunk_vertex_pair_first});

      rmm::device_uvector<vertex_t> endpoint_vertices(size_t{0}, handle.get_stream());
      auto endpoint_value_buffer = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());
      {
        rmm::device_uvector<vertex_t> chunk_majors(this_chunk_size, handle.get_stream());
        rmm::device_uvector<vertex_t> chunk_minors(this_chunk_size, handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     chunk_vertex_pair_first,
                     chunk_vertex_pair_first + this_chunk_size,
                     thrust::make_zip_iterator(
                       thrust::make_tuple(chunk_majors.begin(), chunk_minors.begin())));

        auto [reduced_src_vertices, reduced_src_value_buffer] = detail::sort_and_reduce_by_vertices(
          handle,
          GraphViewType::is_storage_transposed ? std::move(chunk_minors) : std::move(chunk_majors),
          std::move(src_value_buffer));
        auto [reduced_dst_vertices, reduced_dst_value_buffer] = detail::sort_and_reduce_by_vertices(
          handle,
          GraphViewType::is_storage_transposed ? std::move(chunk_majors) : std::move(chunk_minors),
          std::move(dst_value_buffer));

        endpoint_vertices.resize(reduced_src_vertices.size() + reduced_dst_vertices.size(),
                                 handle.get_stream());
        resize_dataframe_buffer(
          endpoint_value_buffer, endpoint_vertices.size(), handle.get_stream());

        thrust::merge_by_key(handle.get_thrust_policy(),
                             reduced_src_vertices.begin(),
                             reduced_src_vertices.end(),
                             reduced_dst_vertices.begin(),
                             reduced_dst_vertices.end(),
                             get_dataframe_buffer_begin(reduced_src_value_buffer),
                             get_dataframe_buffer_begin(reduced_dst_value_buffer),
                             endpoint_vertices.begin(),
                             get_dataframe_buffer_begin(endpoint_value_buffer));
      }

      auto tmp_intersection_value_buffer =
        allocate_dataframe_buffer<T>(intersection_indices.size(), handle.get_stream());
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(size_dataframe_buffer(intersection_value_buffer)),
        detail::segmented_fill_t<vertex_t,
                                 decltype(get_dataframe_buffer_begin(intersection_value_buffer))>{
          intersection_offsets.data(),
          get_dataframe_buffer_begin(intersection_value_buffer),
          get_dataframe_buffer_begin(tmp_intersection_value_buffer)});
      resize_dataframe_buffer(intersection_value_buffer, size_t{0}, handle.get_stream());
      shrink_to_fit_dataframe_buffer(intersection_value_buffer, handle.get_stream());

      auto [reduced_intersection_indices, reduced_intersection_value_buffer] =
        detail::sort_and_reduce_by_vertices(
          handle, std::move(intersection_indices), std::move(tmp_intersection_value_buffer));

      rmm::device_uvector<vertex_t> merged_vertices(
        endpoint_vertices.size() + reduced_intersection_indices.size(), handle.get_stream());
      auto merged_value_buffer =
        allocate_dataframe_buffer<T>(merged_vertices.size(), handle.get_stream());
      thrust::merge_by_key(handle.get_thrust_policy(),
                           endpoint_vertices.begin(),
                           endpoint_vertices.end(),
                           reduced_intersection_indices.begin(),
                           reduced_intersection_indices.end(),
                           get_dataframe_buffer_begin(endpoint_value_buffer),
                           get_dataframe_buffer_begin(reduced_intersection_value_buffer),
                           merged_vertices.begin(),
                           get_dataframe_buffer_begin(merged_value_buffer));

      endpoint_vertices.resize(size_t{0}, handle.get_stream());
      endpoint_vertices.shrink_to_fit(handle.get_stream());
      resize_dataframe_buffer(endpoint_value_buffer, size_t{0}, handle.get_stream());
      shrink_to_fit_dataframe_buffer(endpoint_value_buffer, handle.get_stream());
      reduced_intersection_indices.resize(size_t{0}, handle.get_stream());
      reduced_intersection_indices.shrink_to_fit(handle.get_stream());
      resize_dataframe_buffer(reduced_intersection_value_buffer, size_t{0}, handle.get_stream());
      shrink_to_fit_dataframe_buffer(reduced_intersection_value_buffer, handle.get_stream());

      auto num_uniques =
        thrust::count_if(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(merged_vertices.size()),
                         detail::is_first_in_run_t<vertex_t const*>{merged_vertices.data()});
      rmm::device_uvector<vertex_t> reduced_vertices(num_uniques, handle.get_stream());
      auto reduced_value_buffer = allocate_dataframe_buffer<T>(num_uniques, handle.get_stream());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            merged_vertices.begin(),
                            merged_vertices.end(),
                            get_dataframe_buffer_begin(merged_value_buffer),
                            reduced_vertices.begin(),
                            get_dataframe_buffer_begin(reduced_value_buffer));
      merged_vertices.resize(size_t{0}, handle.get_stream());
      merged_vertices.shrink_to_fit(handle.get_stream());
      resize_dataframe_buffer(merged_value_buffer, size_t{0}, handle.get_stream());
      shrink_to_fit_dataframe_buffer(merged_value_buffer, handle.get_stream());

      if constexpr (GraphViewType::is_multi_gpu) {
        // FIXME: better refactor this shuffle code for reuse
        auto& comm = handle.get_comms();

        auto h_vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
        rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
          h_vertex_partition_range_lasts.size(), handle.get_stream());
        raft::update_device(d_vertex_partition_range_lasts.data(),
                            h_vertex_partition_range_lasts.data(),
                            h_vertex_partition_range_lasts.size(),
                            handle.get_stream());
        rmm::device_uvector<size_t> d_lasts(d_vertex_partition_range_lasts.size(),
                                            handle.get_stream());
        thrust::lower_bound(handle.get_thrust_policy(),
                            reduced_vertices.begin(),
                            reduced_vertices.end(),
                            d_vertex_partition_range_lasts.begin(),
                            d_vertex_partition_range_lasts.end(),
                            d_lasts.begin());
        std::vector<size_t> h_lasts(d_lasts.size());
        raft::update_host(h_lasts.data(), d_lasts.data(), d_lasts.size(), handle.get_stream());
        handle.sync_stream();

        std::vector<size_t> tx_counts(h_lasts.size());
        std::adjacent_difference(h_lasts.begin(), h_lasts.end(), tx_counts.begin());

        rmm::device_uvector<vertex_t> rx_reduced_vertices(size_t{0}, handle.get_stream());
        auto rx_reduced_value_buffer = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());
        std::tie(rx_reduced_vertices, std::ignore) =
          shuffle_values(comm, reduced_vertices.begin(), tx_counts, handle.get_stream());
        std::tie(rx_reduced_value_buffer, std::ignore) = shuffle_values(
          comm, get_dataframe_buffer_begin(reduced_value_buffer), tx_counts, handle.get_stream());

        std::tie(reduced_vertices, reduced_value_buffer) = detail::sort_and_reduce_by_vertices(
          handle, std::move(rx_reduced_vertices), std::move(rx_reduced_value_buffer));
      }

      auto vertex_value_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        reduced_vertices.begin(), get_dataframe_buffer_begin(reduced_value_buffer)));
      thrust::for_each(
        handle.get_thrust_policy(),
        vertex_value_pair_first,
        vertex_value_pair_first + reduced_vertices.size(),
        detail::accumulate_vertex_property_t<vertex_t, VertexValueOutputIterator>{
          graph_view.local_vertex_partition_range_first(), vertex_value_output_first});

      chunk_vertex_pair_first += this_chunk_size;
    }
  }
}

}  // namespace cugraph
