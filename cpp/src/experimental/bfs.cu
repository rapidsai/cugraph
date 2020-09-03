/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <algorithms.hpp>
#include <experimental/graph.hpp>
#include <graph_device_view.cuh>
#include <patterns/reduce_op.cuh>
#include <patterns/update_frontier_v_push_if_out_nbr.cuh>
#include <patterns/vertex_frontier.cuh>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>

namespace cugraph {
namespace experimental {
namespace detail {

template <typename GraphViewType, typename PredecessorIterator>
void bfs(raft::handle_t &handle,
         GraphViewType const &push_graph_view,
         typename GraphViewType::vertex_type *distances,
         PredecessorIterator predecessor_first,
         typename GraphViewType::vertex_type source_vertex,
         bool direction_optimizing,
         typename GraphViewType::vertex_type depth_limit,
         bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;

  static_assert(std::is_integral<vertex_t>::value, "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_adj_matrix_transposed, "GraphViewType should support the push model.");

  auto p_graph_device_view     = graph_device_view_t<GraphViewType>::create(push_graph_view);
  auto const& graph_device_view = *p_graph_device_view;

  auto const num_vertices = graph_device_view.get_number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    graph_device_view.is_symmetric() || !direction_optimizing,
    "Invalid input argument: input graph should be symmetric for direction optimizing BFS.");
  CUGRAPH_EXPECTS(graph_device_view.is_valid_vertex(source_vertex),
                  "Invalid input argument: source vertex out-of-range.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. initialize distances and predecessors

  auto constexpr invalid_distance = std::numeric_limits<vertex_t>::max();
  auto constexpr invalid_vertex   = invalid_vertex_id<vertex_t>::value;

  auto val_first = thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first));
  thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    graph_device_view.local_vertex_begin(),
                    graph_device_view.local_vertex_end(),
                    val_first,
                    [graph_device_view, source_vertex] __device__(auto val) {
                      auto distance = invalid_distance;
                      if (val == source_vertex) { distance = vertex_t{0}; }
                      return thrust::make_tuple(distance, invalid_vertex);
                    });

  // 3. initialize BFS frontier

  enum class Bucket { cur, num_buckets };
  std::vector<size_t> bucket_sizes(static_cast<size_t>(Bucket::num_buckets),
                                   graph_device_view.get_number_of_local_vertices());
  VertexFrontier<raft::handle_t,
                 thrust::tuple<vertex_t>,
                 vertex_t,
                 false,
                 static_cast<size_t>(Bucket::num_buckets)>
    vertex_frontier(handle, bucket_sizes);

  if (graph_device_view.is_local_vertex_nocheck(source_vertex)) {
    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).insert(source_vertex);
  }

  // 4. BFS iteration

  vertex_t depth{0};
  auto cur_local_vertex_frontier_first =
    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).begin();
  auto cur_vertex_frontier_aggregate_size =
    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).aggregate_size();
  while (true) {
    if (direction_optimizing) {
      CUGRAPH_FAIL("unimplemented.");
    } else {
      auto cur_local_vertex_frontier_last =
        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).end();
      update_frontier_v_push_if_out_nbr(
        handle,
        graph_device_view,
        cur_local_vertex_frontier_first,
        cur_local_vertex_frontier_last,
        graph_device_view.adj_matrix_local_row_begin(),
        graph_device_view.adj_matrix_local_col_begin(),
        [graph_device_view, distances] __device__(auto src_val, auto dst_val) {
          uint32_t push = true;
          if (graph_device_view.is_local_vertex_nocheck(dst_val)) {
            auto distance =
              *(distances + graph_device_view.get_local_vertex_offset_from_vertex_nocheck(dst_val));
            if (distance != invalid_distance) { push = false; }
          }
          // FIXME: need to test this works properly if payload size is 0 (returns a tuple of size
          // 1)
          return thrust::make_tuple(push, src_val);
        },
        reduce_op::any<thrust::tuple<vertex_t>>(),
        distances,
        thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first)),
        vertex_frontier,
        [depth] __device__(auto v_val, auto pushed_val) {
          auto idx = (v_val == invalid_distance) ? static_cast<size_t>(Bucket::cur)
                                                 : VertexFrontier<raft::handle_t,
                                                                  thrust::tuple<vertex_t>,
                                                                  vertex_t>::kInvalidBucketIdx;
          return thrust::make_tuple(idx, depth + 1, thrust::get<0>(pushed_val));
        });

      auto new_vertex_frontier_aggregate_size =
        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).aggregate_size() -
        cur_vertex_frontier_aggregate_size;
      if (new_vertex_frontier_aggregate_size == 0) { break; }

      cur_local_vertex_frontier_first = cur_local_vertex_frontier_last;
      cur_vertex_frontier_aggregate_size += new_vertex_frontier_aggregate_size;
    }

    depth++;
    if (depth >= depth_limit) { break; }
  }

  return;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void bfs(raft::handle_t &handle,
         graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const &graph_view,
         vertex_t *distances,
         vertex_t *predecessors,
         vertex_t source_vertex,
         bool direction_optimizing,
         vertex_t depth_limit,
         bool do_expensive_check)
{
  if (predecessors != nullptr) {
    detail::bfs(handle,
                graph_view,
                distances,
                predecessors,
                source_vertex,
                direction_optimizing,
                depth_limit,
                do_expensive_check);
  } else {
    detail::bfs(handle,
                graph_view,
                distances,
                thrust::make_discard_iterator(),
                source_vertex,
                direction_optimizing,
                depth_limit,
                do_expensive_check);
  }
}

// explicit instantiation

template void bfs(raft::handle_t &handle,
                  graph_view_t<uint32_t, uint32_t, float, false, false> const &graph_view,
                  uint32_t *distances,
                  uint32_t *predecessors,
                  uint32_t source_vertex,
                  bool direction_optimizing,
                  uint32_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t &handle,
                  graph_view_t<uint32_t, uint32_t, float, false, true> const &graph_view,
                  uint32_t *distances,
                  uint32_t *predecessors,
                  uint32_t source_vertex,
                  bool direction_optimizing,
                  uint32_t depth_limit,
                  bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
