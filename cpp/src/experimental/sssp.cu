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
#include <utilities/error_utils.h>

#include <detail/graph_device_view.cuh>
#include <detail/patterns/adj_matrix_row_frontier.cuh>
#include <detail/patterns/expand_row_and_transform_if_e.cuh>
#include <detail/patterns/reduce_op.cuh>
#include <detail/patterns/transform_reduce_e.cuh>
#include <detail/utilities/cuda.cuh>
#include <graph.hpp>

#include <rmm/rmm.h>

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

template <typename GraphType, typename VertexIterator, typename WeightIterator>
void sssp_this_partition(
    raft::Handle handle, GraphType const& csr_graph,
    WeightIterator distance_first, VertexIterator predecessor_first,
    typename GraphType::vertex_type starting_vertex,
    size_t depth_limit = std::numeric_limits<size_t>::max(), bool do_expensive_check = false) {
  using vertex_t = typename GraphType::vertex_type;
  using weight_t = typename GraphType::weight_type;
  
  static_assert(
    std::is_integral<vertex_t>::value,
    "GraphType::vertex_type should be integral.");
  static_assert(
    std::is_same<typename std::iterator_traits<VertexIterator>::value_type, vertex_t>::value,
    "GraphType::vertex_type and VertexIterator mismatch.");
  static_assert(
    std::is_same<typename std::iterator_traits<WeightIterator>::value_type, weight_t>::value,
    "GraphType::weight_type and WeightIterator mismatch.");
  static_assert(GraphType::is_row_major, "GraphType should be CSR.");
  
  auto p_graph_device_view =
    graph_compressed_sparse_device_view_t<GraphType>::create(csr_graph);
  auto const graph_device_view = *p_graph_device_view;

  auto const num_vertices = graph_device_view.get_number_of_vertices();
  auto const num_edges = graph_device_view.get_number_of_edges();
  if (num_vertices == 0) {
    return;
  }

  // implements the Near-Far Pile method in
  // A. Davidson, S. Baxter, M. Garland, and J. D. Owens, "Work-efficient parallel GPU methods for
  // single-source shortest paths," 2014.

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    graph_device_view.in_vertex_range(starting_vertex),
    "Invalid input argument: starting vertex out-of-range.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. update delta

  weight_t average_vertex_degree{0.0};
  weight_t average_edge_weight{0.0};
  thrust::tie(average_vertex_degree, average_edge_weight) =
    transform_reduce_e(
      handle, graph_device_view,
      thrust::make_constant_iterator(0)/* dummy */, thrust::make_constant_iterator(0)/* dummy */,
      [] __device__ (auto row_val, auto col_val, weight_t w) {
        return thrust::make_tuple(static_cast<weight_t>(1.0), w);
      },
      thrust::make_tuple(static_cast<weight_t>(0.0), static_cast<weight_t>(0.0)));
  average_vertex_degree /= static_cast<weight_t>(num_vertices);
  average_edge_weight /=
    num_edges > 0 ? static_cast<weight_t>(num_edges) : static_cast<weight_t>(1.0);
  auto delta =
    (static_cast<weight_t>(warp_size) * average_edge_weight) / average_vertex_degree;

  // 3. initialize distances and predecessors

  auto constexpr invalid_distance = std::numeric_limits<weight_t>::max();
  auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

  auto val_first =
    thrust::make_zip_iterator(thrust::make_tuple(distance_first, predecessor_first));
  thrust::transform(
    thrust::cuda::par.on(handle.get_default_stream()),
    graph_device_view.this_partition_vertex_begin(),
    graph_device_view.this_partition_vertex_end(),
    val_first,
    [graph_device_view, starting_vertex] __device__ (auto val) {
      auto distance = invalid_distance;
      auto v = graph_device_view.get_vertex_from_this_partition_vertex_offset_nocheck(val);
      if (v == starting_vertex) {
        distance = static_cast<weight_t>(0.0);
      }
      return thrust::make_tuple(distance, invalid_vertex);
    });

  // 4. initialize SSSP frontier

  enum class Bucket { cur_near, new_near, far, num_buckets };
  std::vector<size_t> bucket_sizes(
    static_cast<size_t>(Bucket::num_buckets),
    graph_device_view.get_number_of_this_partition_adj_matrix_rows());
  AdjMatrixRowFrontier<
    raft::Handle, thrust::tuple<weight_t, vertex_t>, vertex_t, static_cast<size_t>(Bucket::num_buckets)
  > adj_matrix_row_frontier(handle, bucket_sizes);
  //adj_matrix_row_frontier.track_updated_vertices_in_this_partition();

  // 5. SSSP iteration

  rmm::device_vector<weight_t> adj_matrix_row_distances{};
  if (raft::Handle::is_opg) {
    adj_matrix_row_distances.assign(
      graph_device_view.get_number_of_this_partition_adj_matrix_rows(),
      std::numeric_limits<weight_t>::max());
  }

  if (graph_device_view.in_this_partition_adj_matrix_row_range_nocheck(starting_vertex)) {
    adj_matrix_row_frontier.get_bucket(
      static_cast<size_t>(Bucket::cur_near)
    ).insert(starting_vertex);
    if (adj_matrix_row_distances.size() > 0) {
      adj_matrix_row_distances[
        graph_device_view.get_this_partition_row_offset_from_row_nocheck(starting_vertex)
      ] = static_cast<weight_t>(0.0);
    }
  }

  auto near_far_threshold = delta;
  while (true) {
    auto v_op =
      [near_far_threshold] __device__ (auto v_val, auto pushed_val) {
        auto new_dist = thrust::get<0>(pushed_val);
        auto idx =
          new_dist < v_val
          ? (new_dist < near_far_threshold
            ? static_cast<size_t>(Bucket::new_near) : static_cast<size_t>(Bucket::far))
          : AdjMatrixRowFrontier<
              raft::Handle, thrust::tuple<vertex_t>, vertex_t
            >::kInvalidBucketIdx;
        return thrust::make_tuple(idx, thrust::get<0>(pushed_val), thrust::get<1>(pushed_val));
      };

    if (adj_matrix_row_distances.size() > 0) {
      expand_row_and_transform_if_v_push_if_e(
        handle, graph_device_view,
        adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).begin(),
        adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).end(),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            adj_matrix_row_distances.begin(),
            graph_device_view.this_partition_adj_matrix_row_begin())),
        graph_device_view.this_partition_adj_matrix_col_begin(),
        distance_first,
        thrust::make_zip_iterator(thrust::make_tuple(distance_first, predecessor_first)),
        thrust::make_zip_iterator(
          thrust::make_tuple(adj_matrix_row_distances.begin(), thrust::make_discard_iterator())),
        thrust::make_discard_iterator(),
        adj_matrix_row_frontier,
        [graph_device_view, distance_first] __device__ (
            auto row_val, auto col_val, weight_t w) {
          auto push = true;
          auto new_distance = thrust::get<0>(row_val) + w;
          bool local =
            graph_device_view.in_this_partition_vertex_range_nocheck(thrust::get<1>(row_val));
          if (local) {
            auto this_partition_vertex_offset =
              graph_device_view.get_this_partition_vertex_offset_from_vertex_nocheck(col_val);
            auto old_distance = *(distance_first + this_partition_vertex_offset);
            if (new_distance >= old_distance) {
              push = false;
            }
          }
          return thrust::make_tuple(push, new_distance, thrust::get<1>(row_val));
        },
        reduce_op::min<thrust::tuple<weight_t, vertex_t>>(), v_op);
    }
    else {
      expand_row_and_transform_if_v_push_if_e(
        handle, graph_device_view,
        adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).begin(),
        adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).end(),
        graph_device_view.this_partition_adj_matrix_row_begin(),
        graph_device_view.this_partition_adj_matrix_col_begin(),
        distance_first,
        thrust::make_zip_iterator(thrust::make_tuple(distance_first, predecessor_first)),
        thrust::make_discard_iterator(), thrust::make_discard_iterator(),
        adj_matrix_row_frontier,
        [graph_device_view, distance_first] __device__ (
            auto row_val, auto col_val, weight_t w) {
          auto push = true;
          bool local =
            graph_device_view.in_this_partition_vertex_range_nocheck(row_val);
          if (local) {
            auto row_this_partition_vertex_offset =
              graph_device_view.get_this_partition_vertex_offset_from_vertex_nocheck(row_val);
            auto col_this_partition_vertex_offset =
              graph_device_view.get_this_partition_vertex_offset_from_vertex_nocheck(col_val);
            auto old_distance = *(distance_first + col_this_partition_vertex_offset);
            auto new_distance = *(distance_first + row_this_partition_vertex_offset) + w;
            if (new_distance >= old_distance) {
              push = false;
            }
            return thrust::make_tuple(push, new_distance, row_val);
          }
          else {
            assert(0);
            return thrust::make_tuple(false, invalid_distance, invalid_vertex);
          }
        },
        reduce_op::min<thrust::tuple<weight_t, vertex_t>>(), v_op);
    }

    adj_matrix_row_frontier.get_bucket(static_cast<size_t>(Bucket::cur_near)).clear();
    if (adj_matrix_row_frontier.get_bucket(
          static_cast<size_t>(Bucket::new_near)
        ).aggregate_size() > 0) {
      adj_matrix_row_frontier.swap_buckets(
        static_cast<size_t>(Bucket::cur_near), static_cast<size_t>(Bucket::new_near));
    }
    else {  // near queue is empty, split the far queue
      auto old_near_far_threshold = near_far_threshold;
      near_far_threshold += delta;
      
      while (true) {
        if (adj_matrix_row_distances.size() > 0) {
          auto adj_matrix_row_distance_first = adj_matrix_row_distances.begin();
          adj_matrix_row_frontier.split_bucket(
            static_cast<size_t>(Bucket::far),
            [graph_device_view, adj_matrix_row_distance_first, old_near_far_threshold,
             near_far_threshold] __device__ (auto v) {
              auto dist =
                *(adj_matrix_row_distance_first +
                  graph_device_view.get_this_partition_row_offset_from_row_nocheck(v));
              if (dist < old_near_far_threshold) {
                return AdjMatrixRowFrontier<
                         raft::Handle, thrust::tuple<vertex_t>, vertex_t
                       >::kInvalidBucketIdx;
              }
              else if (dist < near_far_threshold) {
                return static_cast<size_t>(Bucket::cur_near);
              }
              else {
                return static_cast<size_t>(Bucket::far);
              }
            });
        }
        else {
          adj_matrix_row_frontier.split_bucket(
            static_cast<size_t>(Bucket::far),
            [graph_device_view, distance_first, old_near_far_threshold,
             near_far_threshold] __device__ (auto v) {
              auto dist =
                *(distance_first +
                  graph_device_view.get_this_partition_vertex_offset_from_vertex_nocheck(v));
              if (dist < old_near_far_threshold) {
                return AdjMatrixRowFrontier<
                         raft::Handle, thrust::tuple<vertex_t>, vertex_t
                       >::kInvalidBucketIdx;
              }
              else if (dist < near_far_threshold) {
                return static_cast<size_t>(Bucket::cur_near);
              }
              else {
                return static_cast<size_t>(Bucket::far);
              }
            });
        }
        if (adj_matrix_row_frontier.get_bucket(
              static_cast<size_t>(Bucket::cur_near)
            ).aggregate_size() > 0) {
          break;
        }
        else if (adj_matrix_row_frontier.get_bucket(
                   static_cast<size_t>(Bucket::far)
                 ).aggregate_size() > 0) {
          near_far_threshold += delta;
        }
        else {
          return;
        }
      }
    }
  }

  return;
}

// explicit instantiation

template void sssp_this_partition(
    raft::Handle handle, GraphCSRView<uint32_t, uint32_t, float> const& csr_graph,
    float* distance_first, uint32_t* predecessor_first, uint32_t starting_vertex,
    size_t depth_limit, bool do_expensive_check);

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
