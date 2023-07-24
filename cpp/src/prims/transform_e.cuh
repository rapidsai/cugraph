/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include <type_traits>
#include <vector>

namespace cugraph {

/**
 * @brief Iterate over the entire set of edges and update edge property values.
 *
 * This function is inspired by thrust::transform().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for input edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for input edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for input edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam EdgeValueOutputWrapper Type of the wrapper for output edge property values.
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
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param edge_value_output Wrapper used to store edge output property values (for the edges
 * assigned to this process in multi-GPU). Use cugraph::edge_property_t::mutable_view().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename EdgeValueOutputWrapper>
void transform_e(raft::handle_t const& handle,
                 GraphViewType const& graph_view,
                 EdgeSrcValueInputWrapper edge_src_value_input,
                 EdgeDstValueInputWrapper edge_dst_value_input,
                 EdgeValueInputWrapper edge_value_input,
                 EdgeOp e_op,
                 EdgeValueOutputWrapper edge_value_output,
                 bool do_expensive_check = false)
{
  // CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  CUGRAPH_FAIL("unimplemented.");
}

/**
 * @brief Iterate over the edges in the input edge list and update edge property values.
 *
 * This function is inspired by thrust::transform().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeBucketType Type of the edge bucket class which stores the edge list.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for input edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for input edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for input edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam EdgeValueOutputWrapper Type of the wrapper for output edge property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_list EdgeBucketType class object storing the edge list to update edge property
 * values.
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
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param edge_value_output Wrapper used to store edge output property values (for the edges
 * assigned to this process in multi-GPU). Use cugraph::edge_property_t::mutable_view().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename EdgeValueOutputWrapper>
void transform_e(raft::handle_t const& handle,
                 GraphViewType const& graph_view,
                 EdgeBucketType const& edge_list,
                 EdgeSrcValueInputWrapper edge_src_value_input,
                 EdgeDstValueInputWrapper edge_dst_value_input,
                 EdgeValueInputWrapper edge_value_input,
                 EdgeOp e_op,
                 EdgeValueOutputWrapper edge_value_output,
                 bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(GraphViewType::is_storage_transposed != EdgeBucketType::is_src_major);
  static_assert(EdgeBucketType::is_sorted_unique);
  static_assert(
    std::is_same_v<typename EdgeBucketType::key_type, thrust::tuple<vertex_t, vertex_t>>);

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeSrcValueInputWrapper::value_iterator,
      typename EdgeSrcValueInputWrapper::value_type>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeDstValueInputWrapper::value_iterator,
      typename EdgeDstValueInputWrapper::value_type>>;
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;
  using edge_partition_e_output_device_view_t = detail::edge_partition_edge_property_device_view_t<
    edge_t,
    typename EdgeValueOutputWrapper::value_iterator,
    typename EdgeValueOutputWrapper::value_type>;

  // CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  auto major_first =
    GraphViewType::is_storage_transposed ? edge_list.dst_begin() : edge_list.src_begin();
  auto minor_first =
    GraphViewType::is_storage_transposed ? edge_list.src_begin() : edge_list.dst_begin();

  auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(major_first, minor_first));

  if (do_expensive_check) {
    CUGRAPH_EXPECTS(
      thrust::is_sorted(handle.get_thrust_policy(), edge_first, edge_first + edge_list.size()),
      "Invalid input arguments: edge_list is not sorted.");
  }

  std::vector<size_t> edge_partition_offsets(graph_view.number_of_local_edge_partitions() + 1, 0);
  if constexpr (GraphViewType::is_multi_gpu) {
    std::vector<vertex_t> h_major_range_lasts(graph_view.number_of_local_edge_partitions());
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(i));
      h_major_range_lasts[i] = edge_partition.major_range_last();
    }
    rmm::device_uvector<vertex_t> d_major_range_lasts(h_major_range_lasts.size(),
                                                      handle.get_stream());
    raft::update_device(d_major_range_lasts.data(),
                        h_major_range_lasts.data(),
                        h_major_range_lasts.size(),
                        handle.get_stream());
    rmm::device_uvector<size_t> d_lower_bounds(d_major_range_lasts.size(), handle.get_stream());
    thrust::lower_bound(handle.get_thrust_policy(),
                        major_first,
                        major_first + edge_list.size(),
                        d_major_range_lasts.begin(),
                        d_major_range_lasts.end(),
                        d_lower_bounds.begin());
    raft::update_host(edge_partition_offsets.data() + 1,
                      d_lower_bounds.data(),
                      d_lower_bounds.size(),
                      handle.get_stream());
    handle.sync_stream();
  } else {
    edge_partition_offsets.back() = edge_list.size();
  }

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    if (do_expensive_check) {
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edge_first + edge_partition_offsets[i],
          edge_first + edge_partition_offsets[i + 1],
          [edge_partition] __device__(thrust::tuple<vertex_t, vertex_t> edge) {
            auto major = thrust::get<0>(edge);
            auto minor = thrust::get<1>(edge);
            vertex_t major_idx{};
            auto major_hypersparse_first = edge_partition.major_hypersparse_first();
            if (major_hypersparse_first) {
              if (major < *major_hypersparse_first) {
                major_idx = edge_partition.major_offset_from_major_nocheck(major);
              } else {
                auto major_hypersparse_idx =
                  edge_partition.major_hypersparse_idx_from_major_nocheck(major);
                if (!major_hypersparse_idx) { return true; }
                major_idx =
                  edge_partition.major_offset_from_major_nocheck(*major_hypersparse_first) +
                  *major_hypersparse_idx;
              }
            } else {
              major_idx = edge_partition.major_offset_from_major_nocheck(major);
            }
            vertex_t const* indices{nullptr};
            edge_t edge_offset{};
            edge_t local_degree{};
            thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_idx);
            auto it = thrust::lower_bound(thrust::seq, indices, indices + local_degree, minor);
            return *it != minor;
          }) == 0,
        "Invalid input arguments: edge_list contains edges that do not exist in the input graph.");
    }

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
    auto edge_partition_e_value_input = edge_partition_e_input_device_view_t(edge_value_input, i);
    auto edge_partition_e_value_output =
      edge_partition_e_output_device_view_t(edge_value_output, i);

    thrust::for_each(
      handle.get_thrust_policy(),
      edge_first + edge_partition_offsets[i],
      edge_first + edge_partition_offsets[i + 1],
      [e_op,
       edge_partition,
       edge_partition_src_value_input,
       edge_partition_dst_value_input,
       edge_partition_e_value_input,
       edge_partition_e_value_output] __device__(thrust::tuple<vertex_t, vertex_t> edge) {
        auto major = thrust::get<0>(edge);
        auto minor = thrust::get<1>(edge);

        auto major_hypersparse_first = edge_partition.major_hypersparse_first();
        auto major_offset            = edge_partition.major_offset_from_major_nocheck(major);
        vertex_t major_idx{major_offset};

        if ((major_hypersparse_first) && (major >= *major_hypersparse_first)) {
          auto major_hypersparse_idx =
            edge_partition.major_hypersparse_idx_from_major_nocheck(major);
          assert(major_hypersparse_idx);
          major_idx = edge_partition.major_offset_from_major_nocheck(*major_hypersparse_first) +
                      *major_hypersparse_idx;
        }

        auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);

        vertex_t const* indices{nullptr};
        edge_t edge_offset{};
        edge_t local_degree{};
        thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_idx);
        auto lower_it = thrust::lower_bound(thrust::seq, indices, indices + local_degree, minor);
        auto upper_it = thrust::upper_bound(thrust::seq, indices, indices + local_degree, minor);

        auto src        = GraphViewType::is_storage_transposed ? minor : major;
        auto dst        = GraphViewType::is_storage_transposed ? major : minor;
        auto src_offset = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
        auto dst_offset = GraphViewType::is_storage_transposed ? major_offset : minor_offset;

        for (auto it = lower_it; it != upper_it; ++it) {
          assert(*it == minor);
          auto e_op_result =
            e_op(src,
                 dst,
                 edge_partition_src_value_input.get(src_offset),
                 edge_partition_dst_value_input.get(dst_offset),
                 edge_partition_e_value_input.get(edge_offset + thrust::distance(indices, it)));
          edge_partition_e_value_output.set(edge_offset + thrust::distance(indices, it),
                                            e_op_result);
        }
      });
  }
}

}  // namespace cugraph
