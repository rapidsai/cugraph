/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <prims/extract_if_e.cuh>
#include <prims/property_op_utils.cuh>

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgeOp>
struct call_e_op_t {
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu>
    edge_partition{};
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input{};
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input{};
  EdgeOp e_op{};

  template <typename Edge>
  __device__ bool operator()(Edge e) const
  {
    static_assert(thrust::tuple_size<Edge>::value == 2);

    using vertex_t = typename GraphViewType::vertex_type;
    using edge_t   = typename GraphViewType::edge_type;

    auto major        = thrust::get<0>(e);
    auto minor        = thrust::get<1>(e);
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
    auto src          = GraphViewType::is_storage_transposed ? minor : major;
    auto dst          = GraphViewType::is_storage_transposed ? major : minor;
    auto src_offset   = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
    auto dst_offset   = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
    return !e_op(src,
                 dst,
                 edge_partition_src_value_input.get(src_offset),
                 edge_partition_dst_value_input.get(dst_offset),
                 thrust::nullopt);
  }
};

}  // namespace detail

// FIXME: better rename this primitive to extract_transform_e and update to return a vector of @p
// e_op outputs (matching extract_transform_v_frontier_outgoing_e).
/**
 * @brief Iterate over the entire set of edges and return an edge list with the edges with @p
 * edge_op evaluated to be true.
 *
 * This function is inspired by thrust::copy_if & thrust::remove_if().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
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
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), property values for the source, and property values for the destination and returns a
 * boolean value to designate whether to include this edge in the returned edge list (if true is
 * returned) or not (if false is returned).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
 * rmm::device_uvector<typename GraphViewType::vertex_type>,
 * std::optional<rmm::device_uvector<typename GraphViewType::weight_type>>> Tuple storing an
 * extracted edge list (sources, destinations, and optional weights).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeOp>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
extract_if_e(raft::handle_t const& handle,
             GraphViewType const& graph_view,
             EdgeSrcValueInputWrapper edge_src_value_input,
             EdgeDstValueInputWrapper edge_dst_value_input,
             EdgeOp e_op,
             bool do_expensive_check = false)
{
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
    // currently, nothing to do
  }

  std::vector<size_t> edgelist_edge_counts(graph_view.number_of_local_edge_partitions(), size_t{0});
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    edgelist_edge_counts[i] =
      static_cast<size_t>(graph_view.number_of_local_edge_partition_edges(i));
  }
  auto number_of_local_edges =
    std::reduce(edgelist_edge_counts.begin(), edgelist_edge_counts.end());

  rmm::device_uvector<vertex_t> edgelist_majors(number_of_local_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());

  size_t cur_size{0};
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
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

    detail::decompress_edge_partition_to_edgelist<vertex_t,
                                                  edge_t,
                                                  weight_t,
                                                  GraphViewType::is_multi_gpu>(
      handle,
      edge_partition,
      std::nullopt,
      edgelist_majors.data() + cur_size,
      edgelist_minors.data() + cur_size,
      std::nullopt,
      graph_view.local_edge_partition_segment_offsets(i));
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
    cur_size += static_cast<size_t>(thrust::distance(
      edge_first + cur_size,
      thrust::remove_if(
        handle.get_thrust_policy(),
        edge_first + cur_size,
        edge_first + cur_size + edgelist_edge_counts[i],
        detail::call_e_op_t<GraphViewType,
                            edge_partition_src_input_device_view_t,
                            edge_partition_dst_input_device_view_t,
                            EdgeOp>{
          edge_partition, edge_partition_src_value_input, edge_partition_dst_value_input, e_op})));
  }

  edgelist_majors.resize(cur_size, handle.get_stream());
  edgelist_minors.resize(edgelist_majors.size(), handle.get_stream());
  edgelist_majors.shrink_to_fit(handle.get_stream());
  edgelist_minors.shrink_to_fit(handle.get_stream());

  return std::make_tuple(
    std::move(GraphViewType::is_storage_transposed ? edgelist_minors : edgelist_majors),
    std::move(GraphViewType::is_storage_transposed ? edgelist_majors : edgelist_minors));
}

}  // namespace cugraph
