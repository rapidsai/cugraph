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
#include <experimental/graph_view.hpp>
#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <patterns/count_if_e.cuh>
#include <patterns/reduce_op.cuh>
#include <patterns/transform_reduce_e.cuh>
#include <patterns/update_frontier_v_push_if_out_nbr.cuh>
#include <patterns/vertex_frontier.cuh>
#include <utilities/error.hpp>
#include <vertex_partition_device.cuh>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>

namespace cugraph {
namespace experimental {
namespace detail {

template <typename GraphViewType, typename PredecessorIterator>
void sssp(raft::handle_t const &handle,
          GraphViewType const &push_graph_view,
          typename GraphViewType::weight_type *distances,
          PredecessorIterator predecessor_first,
          typename GraphViewType::vertex_type source_vertex,
          typename GraphViewType::weight_type cutoff,
          bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using weight_t = typename GraphViewType::weight_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  auto const num_vertices = push_graph_view.get_number_of_vertices();
  auto const num_edges    = push_graph_view.get_number_of_edges();
  if (num_vertices == 0) { return; }

  // implements the Near-Far Pile method in
  // A. Davidson, S. Baxter, M. Garland, and J. D. Owens, "Work-efficient parallel GPU methods for
  // single-source shortest paths," 2014.

  // 1. check input arguments

  CUGRAPH_EXPECTS(push_graph_view.is_valid_vertex(source_vertex),
                  "Invalid input argument: source vertex out-of-range.");

  if (do_expensive_check) {
    auto num_negative_edge_weights =
      count_if_e(handle,
                 push_graph_view,
                 thrust::make_constant_iterator(0) /* dummy */,
                 thrust::make_constant_iterator(0) /* dummy */,
                 [] __device__(vertex_t src, vertex_t dst, weight_t w, auto src_val, auto dst_val) {
                   return w < 0.0;
                 });
    CUGRAPH_EXPECTS(num_negative_edge_weights == 0,
                    "Invalid input argument: input graph should have non-negative edge weights.");
  }

  // 2. initialize distances and predecessors

  auto constexpr invalid_distance = std::numeric_limits<weight_t>::max();
  auto constexpr invalid_vertex   = invalid_vertex_id<vertex_t>::value;

  auto val_first = thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first));
  thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    thrust::make_counting_iterator(push_graph_view.get_local_vertex_first()),
                    thrust::make_counting_iterator(push_graph_view.get_local_vertex_last()),
                    val_first,
                    [source_vertex] __device__(auto val) {
                      auto distance = invalid_distance;
                      if (val == source_vertex) { distance = weight_t{0.0}; }
                      return thrust::make_tuple(distance, invalid_vertex);
                    });

  if (num_edges == 0) { return; }

  // 3. update delta

  weight_t average_vertex_degree{0.0};
  weight_t average_edge_weight{0.0};
  thrust::tie(average_vertex_degree, average_edge_weight) = transform_reduce_e(
    handle,
    push_graph_view,
    thrust::make_constant_iterator(0) /* dummy */,
    thrust::make_constant_iterator(0) /* dummy */,
    [] __device__(vertex_t row, vertex_t col, weight_t w, auto row_val, auto col_val) {
      return thrust::make_tuple(weight_t{1.0}, w);
    },
    thrust::make_tuple(weight_t{0.0}, weight_t{0.0}));
  average_vertex_degree /= static_cast<weight_t>(num_vertices);
  average_edge_weight /= static_cast<weight_t>(num_edges);
  auto delta =
    (static_cast<weight_t>(raft::warp_size()) * average_edge_weight) / average_vertex_degree;

  // 4. initialize SSSP frontier

  enum class Bucket { cur_near, new_near, far, num_buckets };
  // FIXME: need to double check the bucket sizes are sufficient
  std::vector<size_t> bucket_sizes(static_cast<size_t>(Bucket::num_buckets),
                                   push_graph_view.get_number_of_local_vertices());
  VertexFrontier<thrust::tuple<weight_t, vertex_t>,
                 vertex_t,
                 GraphViewType::is_multi_gpu,
                 static_cast<size_t>(Bucket::num_buckets)>
    vertex_frontier(handle, bucket_sizes);

  // 5. SSSP iteration

  bool vertex_and_adj_matrix_row_ranges_coincide =
    push_graph_view.get_number_of_local_vertices() ==
        push_graph_view.get_number_of_local_adj_matrix_partition_rows()
      ? true
      : false;
  rmm::device_uvector<weight_t> adj_matrix_row_distances(0, handle.get_stream());
  if (!vertex_and_adj_matrix_row_ranges_coincide) {
    adj_matrix_row_distances.resize(push_graph_view.get_number_of_local_adj_matrix_partition_rows(),
                                    handle.get_stream());
    thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 adj_matrix_row_distances.begin(),
                 adj_matrix_row_distances.end(),
                 std::numeric_limits<weight_t>::max());
  }
  auto row_distances =
    !vertex_and_adj_matrix_row_ranges_coincide ? adj_matrix_row_distances.data() : distances;

  if (push_graph_view.is_local_vertex_nocheck(source_vertex)) {
    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).insert(source_vertex);
  }

  auto near_far_threshold = delta;
  while (true) {
    if (!vertex_and_adj_matrix_row_ranges_coincide) {
      copy_to_adj_matrix_row(
        handle,
        push_graph_view,
        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).begin(),
        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).end(),
        distances,
        row_distances);
    }

    vertex_partition_device_t<GraphViewType> vertex_partition(push_graph_view);

    update_frontier_v_push_if_out_nbr(
      handle,
      push_graph_view,
      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).begin(),
      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).end(),
      row_distances,
      thrust::make_constant_iterator(0) /* dummy */,
      [vertex_partition, distances, cutoff] __device__(
        vertex_t src, vertex_t dst, weight_t w, auto src_val, auto dst_val) {
        auto push         = true;
        auto new_distance = src_val + w;
        auto threshold    = cutoff;
        if (vertex_partition.is_local_vertex_nocheck(dst)) {
          auto local_vertex_offset =
            vertex_partition.get_local_vertex_offset_from_vertex_nocheck(dst);
          auto old_distance = *(distances + local_vertex_offset);
          threshold         = old_distance < threshold ? old_distance : threshold;
        }
        if (new_distance >= threshold) { push = false; }
        return thrust::make_tuple(push, new_distance, src);
      },
      reduce_op::min<thrust::tuple<weight_t, vertex_t>>(),
      distances,
      thrust::make_zip_iterator(thrust::make_tuple(distances, predecessor_first)),
      vertex_frontier,
      [near_far_threshold] __device__(auto v_val, auto pushed_val) {
        auto new_dist = thrust::get<0>(pushed_val);
        auto idx      = new_dist < v_val
                     ? (new_dist < near_far_threshold ? static_cast<size_t>(Bucket::new_near)
                                                      : static_cast<size_t>(Bucket::far))
                     : VertexFrontier<thrust::tuple<vertex_t>, vertex_t>::kInvalidBucketIdx;
        return thrust::make_tuple(idx, thrust::get<0>(pushed_val), thrust::get<1>(pushed_val));
      });

    vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).clear();
    if (vertex_frontier.get_bucket(static_cast<size_t>(Bucket::new_near)).aggregate_size() > 0) {
      vertex_frontier.swap_buckets(static_cast<size_t>(Bucket::cur_near),
                                   static_cast<size_t>(Bucket::new_near));
    } else if (vertex_frontier.get_bucket(static_cast<size_t>(Bucket::far)).aggregate_size() >
               0) {  // near queue is empty, split the far queue
      auto old_near_far_threshold = near_far_threshold;
      near_far_threshold += delta;

      size_t new_near_size{0};
      size_t new_far_size{0};
      while (true) {
        vertex_frontier.split_bucket(
          static_cast<size_t>(Bucket::far),
          [vertex_partition, distances, old_near_far_threshold, near_far_threshold] __device__(
            auto v) {
            auto dist =
              *(distances + vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v));
            if (dist < old_near_far_threshold) {
              return VertexFrontier<thrust::tuple<vertex_t>, vertex_t>::kInvalidBucketIdx;
            } else if (dist < near_far_threshold) {
              return static_cast<size_t>(Bucket::cur_near);
            } else {
              return static_cast<size_t>(Bucket::far);
            }
          });
        new_near_size =
          vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).aggregate_size();
        new_far_size =
          vertex_frontier.get_bucket(static_cast<size_t>(Bucket::far)).aggregate_size();
        if ((new_near_size > 0) || (new_far_size == 0)) {
          break;
        } else {
          near_far_threshold += delta;
        }
      }
      if ((new_near_size == 0) && (new_far_size == 0)) { break; }
    } else {
      break;
    }
  }

  CUDA_TRY(cudaStreamSynchronize(
    handle.get_stream()));  // this is as necessary vertex_frontier will become out-of-scope once
                            // this function returns (FIXME: should I stream sync in VertexFrontier
                            // destructor?)

  return;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void sssp(raft::handle_t const &handle,
          graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const &graph_view,
          weight_t *distances,
          vertex_t *predecessors,
          vertex_t source_vertex,
          weight_t cutoff,
          bool do_expensive_check)
{
  if (predecessors != nullptr) {
    detail::sssp(
      handle, graph_view, distances, predecessors, source_vertex, cutoff, do_expensive_check);
  } else {
    detail::sssp(handle,
                 graph_view,
                 distances,
                 thrust::make_discard_iterator(),
                 source_vertex,
                 cutoff,
                 do_expensive_check);
  }
}

// explicit instantiation

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int32_t, int32_t, float, false, true> const &graph_view,
                   float *distances,
                   int32_t *predecessors,
                   int32_t source_vertex,
                   float cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int32_t, int32_t, double, false, true> const &graph_view,
                   double *distances,
                   int32_t *predecessors,
                   int32_t source_vertex,
                   double cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int32_t, int64_t, float, false, true> const &graph_view,
                   float *distances,
                   int32_t *predecessors,
                   int32_t source_vertex,
                   float cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int32_t, int64_t, double, false, true> const &graph_view,
                   double *distances,
                   int32_t *predecessors,
                   int32_t source_vertex,
                   double cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int64_t, int64_t, float, false, true> const &graph_view,
                   float *distances,
                   int64_t *predecessors,
                   int64_t source_vertex,
                   float cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int64_t, int64_t, double, false, true> const &graph_view,
                   double *distances,
                   int64_t *predecessors,
                   int64_t source_vertex,
                   double cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int32_t, int32_t, float, false, false> const &graph_view,
                   float *distances,
                   int32_t *predecessors,
                   int32_t source_vertex,
                   float cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int32_t, int32_t, double, false, false> const &graph_view,
                   double *distances,
                   int32_t *predecessors,
                   int32_t source_vertex,
                   double cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int32_t, int64_t, float, false, false> const &graph_view,
                   float *distances,
                   int32_t *predecessors,
                   int32_t source_vertex,
                   float cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int32_t, int64_t, double, false, false> const &graph_view,
                   double *distances,
                   int32_t *predecessors,
                   int32_t source_vertex,
                   double cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int64_t, int64_t, float, false, false> const &graph_view,
                   float *distances,
                   int64_t *predecessors,
                   int64_t source_vertex,
                   float cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const &handle,
                   graph_view_t<int64_t, int64_t, double, false, false> const &graph_view,
                   double *distances,
                   int64_t *predecessors,
                   int64_t source_vertex,
                   double cutoff,
                   bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
