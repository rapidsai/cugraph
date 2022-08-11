/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <prims/edge_partition_src_dst_property.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh>
#include <prims/update_edge_partition_src_dst_property.cuh>
#include <prims/update_v_frontier.cuh>
#include <prims/vertex_frontier.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>

namespace cugraph {

namespace detail {
template <typename GraphViewType>
std::pair<std::unique_ptr<Dendrogram<typename GraphViewType::vertex_type>>,
          typename GraphViewType::weight_type>
leiden(raft::handle_t const& handle,
       GraphViewType const& graph_view,
       size_t max_level,
       typename GraphViewType::weight_type resolution)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using weight_t = typename GraphViewType::weight_type;

  // TODO: everything
  CUGRAPH_FAIL("unimplemented");
  return std::make_pair(std::make_unique<Dendrogram<vertex_t>>(), (weight_t)0.0);
}

}  // namespace detail

template <typename graph_view_t>
std::pair<std::unique_ptr<Dendrogram<typename graph_view_t::vertex_type>>,
          typename graph_view_t::weight_type>
leiden(raft::handle_t const& handle,
       graph_view_t const& graph_view,
       size_t max_level,
       typename graph_view_t::weight_type resolution)
{
  return detail::leiden(handle, graph_view, max_level, resolution);
}
}  // namespace cugraph
