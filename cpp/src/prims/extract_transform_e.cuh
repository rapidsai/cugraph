/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "prims/detail/extract_transform_if_v_frontier_e.cuh"
#include "prims/property_op_utils.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/iterator>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {

/**
 * @brief Iterate over the entire set of edges and extract all edge functor outputs.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
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
 * property values for the destination, and property values for the edge and returns
 * cuda::std::nullopt (if the return value is to be discarded) or a valid @p e_op output to be
 * extracted and accumulated.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Dataframe buffer object storing extracted and accumulated valid @p e_op return values.
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp>
dataframe_buffer_type_t<
  typename detail::edge_op_result_type<typename GraphViewType::vertex_type,
                                       typename GraphViewType::vertex_type,
                                       typename EdgeSrcValueInputWrapper::value_type,
                                       typename EdgeDstValueInputWrapper::value_type,
                                       typename EdgeValueInputWrapper::value_type,
                                       EdgeOp>::type>
extract_transform_e(raft::handle_t const& handle,
                    GraphViewType const& graph_view,
                    EdgeSrcValueInputWrapper edge_src_value_input,
                    EdgeDstValueInputWrapper edge_dst_value_input,
                    EdgeValueInputWrapper edge_value_input,
                    EdgeOp e_op,
                    bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using e_op_result_t =
    typename detail::edge_op_result_type<typename GraphViewType::vertex_type,
                                         typename GraphViewType::vertex_type,
                                         typename EdgeSrcValueInputWrapper::value_type,
                                         typename EdgeDstValueInputWrapper::value_type,
                                         typename EdgeValueInputWrapper::value_type,
                                         EdgeOp>::type;
  static_assert(!std::is_same_v<e_op_result_t, void>);

  // FIXME: Consider updating detail::extract_transform_if_v_forntier_e to take std::nullopt to as a
  // frontier or create a new key bucket type that just stores [vertex_first, vertex_last) for
  // further optimization. Better revisit this once this becomes a performance bottleneck and after
  // updating primitives to support masking & graph updates.
  key_bucket_t<vertex_t, void, GraphViewType::is_multi_gpu, true> frontier(handle);
  frontier.insert(thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
                  thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()));

  auto value_buffer = allocate_dataframe_buffer<e_op_result_t>(size_t{0}, handle.get_stream());
  std::tie(std::ignore, value_buffer) = detail::
    extract_transform_if_v_frontier_e<GraphViewType::is_storage_transposed, void, e_op_result_t>(
      handle,
      graph_view,
      frontier,
      edge_src_value_input,
      edge_dst_value_input,
      edge_value_input,
      e_op,
      detail::const_true_e_op_t<vertex_t,
                                vertex_t,
                                typename EdgeSrcValueInputWrapper::value_type,
                                typename EdgeDstValueInputWrapper::value_type,
                                typename EdgeValueInputWrapper::value_type,
                                GraphViewType::is_storage_transposed>{},
      do_expensive_check);

  return value_buffer;
}

}  // namespace cugraph
