/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <experimental/graph_view.hpp>
#include <iostream>
#include <patterns/count_if_v.cuh>
#include <patterns/reduce_op.cuh>
#include <patterns/update_frontier_v_push_if_out_nbr.cuh>
#include <patterns/vertex_frontier.cuh>

#include <utilities/error.hpp>
#include <vertex_partition_device.cuh>

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
void bfs(raft::handle_t const &handle,
         GraphViewType const &push_graph_view,
         typename GraphViewType::vertex_type *distances,
         PredecessorIterator predecessor_first,
         typename GraphViewType::vertex_type *sources,
         size_t n_sources,
         bool direction_optimizing,
         typename GraphViewType::vertex_type depth_limit,
         bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  auto const num_vertices = push_graph_view.get_number_of_vertices();
  if (num_vertices == 0) { return; }
  CUGRAPH_EXPECTS(sources != nullptr, "Invalid input argument: sources cannot be null");

  auto aggregate_n_sources =
    GraphViewType::is_multi_gpu
      ? host_scalar_allreduce(handle.get_comms(), n_source, handle.get_stream())
      : n_source;
  CUGRAPH_EXPECTS(aggregate_n_sources > 0,
                  "Invalid input argument: input should have more than one source");

  // 1. check input arguments
  CUGRAPH_EXPECTS(
    push_graph_view.is_symmetric() || !direction_optimizing,
    "Invalid input argument: input graph should be symmetric for direction optimizing BFS.");

  // Transfer single source to the device for single source case
  vertex_t *d_sources;
  rmm::device_uvector<vertex_t> d_sources_v(0, handle.get_stream());
  if (aggregate_n_sources == 1 && n_source) {
    cudaPointerAttributes s_att;
    CUDA_CHECK(cudaPointerGetAttributes(&s_att, sources));
    if (s_att.devicePointer == nullptr) {
      d_sources_v.resize(n_sources, handle.get_stream());
      d_sources = d_sources_v.data();
      raft::copy(d_sources, sources, n_sources, handle.get_stream());
    }
  } else {
    d_sources = sources;
  }

  if (do_expensive_check) {
    vertex_partition_device_t<GraphViewType> vertex_partition(push_graph_view);
    auto num_invalid_vertices =
      count_if_v(handle,
                 push_graph_view,
                 d_sources,
                 d_sources + n_sources,
                 [vertex_partition] __device__(auto val) {
                   return !(vertex_partition.is_valid_vertex(val) &&
                            vertex_partition.is_local_vertex_nocheck(val));
                 });
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: sources have invalid vertex IDs.");
  }

  // 2. initialize distances and predecessors

  auto constexpr invalid_distance = std::numeric_limits<vertex_t>::max();
  auto constexpr invalid_vertex   = invalid_vertex_id<vertex_t>::value;

  thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               distances,
               distances + push_graph_view.get_number_of_local_vertices(),
               invalid_distance);
  thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               predecessor_first,
               predecessor_first + push_graph_view.get_number_of_local_vertices(),
               invalid_vertex);
  vertex_partition_device_t<GraphViewType> vertex_partition(push_graph_view);
  if (n_sources)
    thrust::for_each(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      d_sources,
      d_sources + n_sources,
      [vertex_partition, distances, predecessor_first] __device__(auto v) {
        *(distances + vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v)) =
          vertex_t{0};
      });

  // 3. initialize BFS frontier
  enum class Bucket { cur, next, num_buckets };
  VertexFrontier<vertex_t, GraphViewType::is_multi_gpu, static_cast<size_t>(Bucket::num_buckets)>
    vertex_frontier(handle);

  // insert local source(s) in the bucket
  if (aggregate_n_sources == 1) {
    vertex_t src;
    // Fixme: this (cheap) transfer could be skiped when is_local_vertex_nocheck accpets device mem
    raft::copy(&src, sources, n_sources, handle.get_stream());
    if (push_graph_view.is_local_vertex_nocheck(src))
      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).insert(d_sources, n_sources);
  } else {
    // pre-shuffled
    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).insert(d_sources, n_sources);
  }
  // 4. BFS iteration
  vertex_t depth{0};
  while (true) {
    if (direction_optimizing) {
      CUGRAPH_FAIL("unimplemented.");
    } else {
      vertex_partition_device_t<GraphViewType> vertex_partition(push_graph_view);

      update_frontier_v_push_if_out_nbr(
        handle,
        push_graph_view,
        vertex_frontier,
        static_cast<size_t>(Bucket::cur),
        std::vector<size_t>{static_cast<size_t>(Bucket::next)},
        thrust::make_constant_iterator(0) /* dummy */,
        thrust::make_constant_iterator(0) /* dummy */,
        [vertex_partition, distances] __device__(
          vertex_t src, vertex_t dst, auto src_val, auto dst_val) {
          auto push = true;
          if (vertex_partition.is_local_vertex_nocheck(dst)) {
            auto distance =
              *(distances + vertex_partition.get_local_vertex_offset_from_vertex_nocheck(dst));
            if (distance != invalid_distance) { push = false; }
          }
          return thrust::make_tuple(push, src);
        },
        reduce_op::any<vertex_t>(),
        distances,
        thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first)),
        [depth] __device__(auto v_val, auto pushed_val) {
          auto idx = (v_val == invalid_distance) ? static_cast<size_t>(Bucket::next)
                                                 : VertexFrontier<vertex_t>::kInvalidBucketIdx;
          return thrust::make_tuple(idx, thrust::make_tuple(depth + 1, pushed_val));
        });

      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).clear();
      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).shrink_to_fit();
      vertex_frontier.swap_buckets(static_cast<size_t>(Bucket::cur),
                                   static_cast<size_t>(Bucket::next));
      if (vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).aggregate_size() == 0) {
        break;
      }
    }

    depth++;
    if (depth >= depth_limit) { break; }
  }

  CUDA_TRY(cudaStreamSynchronize(
    handle.get_stream()));  // this is as necessary vertex_frontier will become out-of-scope once
                            // this function returns (FIXME: should I stream sync in VertexFrontier
                            // destructor?)
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void bfs(raft::handle_t const &handle,
         graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const &graph_view,
         vertex_t *distances,
         vertex_t *predecessors,
         vertex_t *sources,
         size_t n_sources,
         bool direction_optimizing,
         vertex_t depth_limit,
         bool do_expensive_check)
{
  if (predecessors != nullptr) {
    detail::bfs(handle,
                graph_view,
                distances,
                predecessors,
                sources,
                n_sources,
                direction_optimizing,
                depth_limit,
                do_expensive_check);
  } else {
    detail::bfs(handle,
                graph_view,
                distances,
                thrust::make_discard_iterator(),
                sources,
                n_sources,
                direction_optimizing,
                depth_limit,
                do_expensive_check);
  }
}

// explicit instantiation

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int32_t, int32_t, float, false, true> const &graph_view,
                  int32_t *distances,
                  int32_t *predecessors,
                  int32_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int32_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int32_t, int32_t, double, false, true> const &graph_view,
                  int32_t *distances,
                  int32_t *predecessors,
                  int32_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int32_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int32_t, int64_t, float, false, true> const &graph_view,
                  int32_t *distances,
                  int32_t *predecessors,
                  int32_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int32_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int32_t, int64_t, double, false, true> const &graph_view,
                  int32_t *distances,
                  int32_t *predecessors,
                  int32_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int32_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int64_t, int64_t, float, false, true> const &graph_view,
                  int64_t *distances,
                  int64_t *predecessors,
                  int64_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int64_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int64_t, int64_t, double, false, true> const &graph_view,
                  int64_t *distances,
                  int64_t *predecessors,
                  int64_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int64_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int32_t, int32_t, float, false, false> const &graph_view,
                  int32_t *distances,
                  int32_t *predecessors,
                  int32_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int32_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int32_t, int32_t, double, false, false> const &graph_view,
                  int32_t *distances,
                  int32_t *predecessors,
                  int32_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int32_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int32_t, int64_t, float, false, false> const &graph_view,
                  int32_t *distances,
                  int32_t *predecessors,
                  int32_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int32_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int32_t, int64_t, double, false, false> const &graph_view,
                  int32_t *distances,
                  int32_t *predecessors,
                  int32_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int32_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int64_t, int64_t, float, false, false> const &graph_view,
                  int64_t *distances,
                  int64_t *predecessors,
                  int64_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int64_t depth_limit,
                  bool do_expensive_check);

template void bfs(raft::handle_t const &handle,
                  graph_view_t<int64_t, int64_t, double, false, false> const &graph_view,
                  int64_t *distances,
                  int64_t *predecessors,
                  int64_t *sources,
                  size_t n_sources,
                  bool direction_optimizing,
                  int64_t depth_limit,
                  bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
