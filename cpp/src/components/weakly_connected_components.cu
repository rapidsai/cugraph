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
#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <patterns/update_frontier_v_push_if_out_nbr.cuh>
#include <patterns/vertex_frontier.cuh>
#include <utilities/error.hpp>
#include <vertex_partition_device.cuh>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/tuple.h>

#include <cuda/atomic>

#include <limits>
#include <type_traits>

namespace cugraph {
namespace experimental {

namespace {

// FIXME: this function (after modification) may be useful for SSSP with the near-far method to
// determine the near-far threshold.
// add new roots till the sum of the degrees first becomes no smaller than degree_sum_threshold and
// returns a triplet of (new roots, number of scanned vertices, sum of the degrees of the new roots)
template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<vertex_t>, vertex_t, edge_t> accumulate_new_roots(
  raft::handle_t const &handle,
  vertex_t const *components,
  edge_t const *degrees,
  vertex_t vertex_first,
  vertex_t vertex_last,
  edge_t degree_sum_threshold)
{
  // FIXME: tuning parameter (time to scan max_scan_size elements should not take significantly
  // longer than scanning a single element)
  vertex_t max_scan_size =
    static_cast<vertex_t>(handle.get_device_properties().multiProcessorCount) * vertex_t{1024};

  rmm::device_uvector<vertex_t> new_roots(0, handle.get_stream_view());
  vertex_t num_new_roots{0};
  vertex_t num_scanned{0};
  edge_t degree_sum{0};
  while (vertex_first + num_scanned < vertex_last) {
    auto scan_size = std::min(max_scan_size, vertex_last - (vertex_first + num_scanned));

    rmm::device_uvector<edge_t> tmp_cumulative_degrees(scan_size, handle.get_stream_view());
    auto component_degree_pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(components, degrees)) + num_scanned;
    thrust::transform(rmm::exec_policy(handle.get_stream_view()),
                      component_degree_pair_first,
                      component_degree_pair_first + scan_size,
                      tmp_cumulative_degrees.begin(),
                      [] __device__(auto pair) {
                        auto c = thrust::get<0>(pair);
                        auto d = thrust::get<1>(pair);
                        return c == invalid_component_id<vertex_t>::value ? d : edge_t{0};
                      });
    thrust::inclusive_scan(rmm::exec_policy(handle.get_stream_view()),
                           tmp_cumulative_degrees.begin(),
                           tmp_cumulative_degrees.end(),
                           tmp_cumulative_degrees.begin());

    auto component_degree_last = thrust::lower_bound(rmm::exec_policy(handle.get_stream_view()),
                                                     tmp_cumulative_degrees.begin(),
                                                     tmp_cumulative_degrees.end(),
                                                     degree_sum_threshold - degree_sum);

    auto tmp_num_scanned = component_degree_last == tmp_cumulative_degrees.end()
                             ? scan_size
                             : static_cast<vertex_t>(thrust::distance(
                                 tmp_cumulative_degrees.begin(), component_degree_last)) +
                                 vertex_t{1};
    new_roots.resize(num_new_roots + tmp_num_scanned, handle.get_stream_view());
    auto component_vertex_pair_input_first =
      thrust::make_zip_iterator(
        thrust::make_tuple(components, thrust::make_counting_iterator(vertex_first))) +
      num_scanned;
    auto component_vertex_pair_output_first =
      thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_discard_iterator(), new_roots.begin())) +
      num_new_roots;
    auto component_vertex_pair_output_last =
      thrust::copy_if(rmm::exec_policy(handle.get_stream_view()),
                      component_vertex_pair_input_first,
                      component_vertex_pair_input_first + tmp_num_scanned,
                      component_vertex_pair_output_first,
                      [] __device__(auto pair) {
                        auto c = thrust::get<0>(pair);
                        return c == invalid_component_id<vertex_t>::value;
                      });

    num_new_roots += static_cast<vertex_t>(
      thrust::distance(component_vertex_pair_output_first, component_vertex_pair_output_last));
    num_scanned += tmp_num_scanned;
    edge_t tmp_degree_sum{0};
    raft::update_host(&tmp_degree_sum,
                      tmp_cumulative_degrees.data() + (tmp_num_scanned - vertex_t{1}),
                      size_t{1},
                      handle.get_stream());
    handle.get_stream_view().synchronize();
    degree_sum += tmp_degree_sum;

    if (degree_sum >= degree_sum_threshold) { break; }
  }

  new_roots.resize(num_new_roots, handle.get_stream_view());
  new_roots.shrink_to_fit(handle.get_stream_view());

  return std::make_tuple(std::move(new_roots), num_scanned, degree_sum);
}

template <typename GraphViewType>
void weakly_connected_components_impl(raft::handle_t const &handle,
                                      GraphViewType const &push_graph_view,
                                      typename GraphViewType::vertex_type *components,
                                      bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  auto const num_vertices = push_graph_view.get_number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    push_graph_view.is_symmetric(),
    "Invalid input argument: input graph should be symmetric for weakly connected components.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. recursively run multi-root frontier expansion

  enum class Bucket { cur, next, num_buckets };
  // tuning parameter to balance work per iteration (should be large enough to be throughput
  // bounded) vs # conflicts between frontiers with different roots (# conflicts == # edges for the
  // next level)
  auto degree_sum_threshold =
    static_cast<edge_t>(handle.get_device_properties().multiProcessorCount) * edge_t{1024};

  size_t level{0};
  graph_t<vertex_t,
          edge_t,
          typename GraphViewType::weight_type,
          GraphViewType::is_adj_matrix_transposed,
          GraphViewType::is_multi_gpu>
    level_graph(handle);
  std::vector<rmm::device_uvector<vertex_t>> level_component_vectors{};
  while (true) {
    auto level_graph_view = level == 0 ? push_graph_view : level_graph.view();
    vertex_partition_device_t<GraphViewType> vertex_partition(level_graph_view);
    level_component_vectors.push_back(rmm::device_uvector<vertex_t>(
      level == 0 ? vertex_t{0} : level_graph_view.get_number_of_local_vertices(),
      handle.get_stream_view()));
    auto level_components = level == 0 ? components : level_component_vectors[level].data();
    auto degrees          = level_graph_view.compute_out_degrees(handle);
    auto local_vertex_in_degree_sum =
      thrust::reduce(rmm::exec_policy(handle.get_stream_view()), degrees.begin(), degrees.end());

    // 2-1. filter out isolated vertices

    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
      thrust::make_counting_iterator(level_graph_view.get_local_vertex_first()), degrees.begin()));
    thrust::transform(rmm::exec_policy(handle.get_stream_view()),
                      pair_first,
                      pair_first + level_graph_view.get_number_of_local_vertices(),
                      level_components,
                      [] __device__(auto pair) {
                        auto v      = thrust::get<0>(pair);
                        auto degree = thrust::get<1>(pair);
                        return degree > 0 ? invalid_component_id<vertex_t>::value : v;
                      });

    // 2-2. initialize vertex frontier

    VertexFrontier<vertex_t,
                   vertex_t,
                   GraphViewType::is_multi_gpu,
                   static_cast<size_t>(Bucket::num_buckets)>
      vertex_frontier(handle);
    vertex_t next_local_vertex_offset{0};
    edge_t edge_count{0};

    {
      auto [new_roots, num_scanned, degree_sum] =
        accumulate_new_roots(handle,
                             level_components + next_local_vertex_offset,
                             degrees.begin() + next_local_vertex_offset,
                             level_graph_view.get_local_vertex_first() + next_local_vertex_offset,
                             level_graph_view.get_local_vertex_last(),
                             degree_sum_threshold);
      next_local_vertex_offset += num_scanned;
      edge_count = degree_sum;

      auto pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(new_roots.begin(), new_roots.begin()));
      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur))
        .insert(pair_first, pair_first + new_roots.size());

      thrust::for_each(
        rmm::exec_policy(handle.get_stream_view()),
        new_roots.begin(),
        new_roots.end(),
        [vertex_partition, components = level_components] __device__(auto c) {
          components[vertex_partition.get_local_vertex_offset_from_vertex_nocheck(c)] = c;
        });
    }

    // FIXME: if we use cuco::static_map (no duplicates, ideally we need static_set), edge_buffer
    // size cannot exceed (# local roots * # aggregate roots)
    auto edge_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
      edge_count + (GraphViewType::is_multi_gpu ? local_vertex_in_degree_sum : edge_t{0}),
      handle.get_stream());
    cuda::atomic<edge_t, cuda::thread_scope_device> num_edge_inserts{
      0};  // FIXME: need to check I am using this properly.

    rmm::device_uvector<vertex_t> col_components(
      GraphViewType::is_multi_gpu ? level_graph_view.get_number_of_local_adj_matrix_partition_cols()
                                  : vertex_t{0},
      handle.get_stream_view());
    if (GraphViewType::is_multi_gpu) {
      thrust::fill(rmm::exec_policy(handle.get_stream_view()),
                   col_components.begin(),
                   col_components.end(),
                   invalid_component_id<vertex_t>::value);

      copy_to_adj_matrix_col(
        handle,
        level_graph_view,
        thrust::get<0>(vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur))
                         .begin()
                         .get_iterator_tuple()),
        thrust::get<0>(
          vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).end().get_iterator_tuple()),
        level_components,
        col_components.begin());
    }

    // 2.3 iterate till every vertex gets visited

    while (true) {
      update_frontier_v_push_if_out_nbr(
        handle,
        push_graph_view,
        vertex_frontier,
        static_cast<size_t>(Bucket::cur),
        std::vector<size_t>{static_cast<size_t>(Bucket::next)},
        thrust::make_counting_iterator(0) /* dummy */,
        thrust::make_counting_iterator(0) /* dummy */,
        [col_components = GraphViewType::is_multi_gpu ? col_components.data() : level_components,
         col_first      = level_graph_view.get_local_adj_matrix_partition_col_first(),
         edge_buffer_first =
           get_dataframe_buffer_begin<thrust::tuple<vertex_t, vertex_t>>(edge_buffer),
         num_edge_inserts = &num_edge_inserts] __device__(auto tagged_src,
                                                          vertex_t dst,
                                                          auto src_val,
                                                          auto dst_val) {
          auto tag        = thrust::get<1>(tagged_src);
          auto col_offset = dst - col_first;
          // FIXME: better switch to atomic_ref after
          // https://github.com/nvidia/libcudacxx/milestone/2
          auto old =
            atomicCAS(col_components + col_offset, invalid_component_id<vertex_t>::value, tag);
          if (old != invalid_component_id<vertex_t>::value && old != tag) {  // conflict
            *(edge_buffer_first + (*num_edge_inserts)++) = thrust::make_tuple(tag, old);
          }
          return (old == invalid_component_id<vertex_t>::value) ? thrust::optional<vertex_t>{tag}
                                                                : thrust::nullopt;
        },
        reduce_op::null(),
        thrust::make_constant_iterator(0) /* dummy */,
        thrust::make_discard_iterator() /* dummy */,
        [vertex_partition,
         level_components,
         edge_buffer_first =
           get_dataframe_buffer_begin<thrust::tuple<vertex_t, vertex_t>>(edge_buffer),
         num_edge_inserts =
           &num_edge_inserts] __device__(auto tagged_v, auto v_val) {
          if (GraphViewType::is_multi_gpu) {
            auto tag      = thrust::get<1>(tagged_v);
            auto v_offset = vertex_partition.get_local_vertex_offset_from_vertex_nocheck(
              thrust::get<0>(tagged_v));
            // FIXME: better switch to atomic_ref after
            // https://github.com/nvidia/libcudacxx/milestone/2
            auto old =
              atomicCAS(level_components + v_offset, invalid_component_id<vertex_t>::value, tag);
            if (old != invalid_component_id<vertex_t>::value && old != tag) {  // conflict
              // FIXME: potential overflow? we need to count dst for square root P GPUs then do
              // reduction to figure out actual maximum... this is too pessimistic and also
              // expensive, actually, we know the graph is symmetric and we alredy have in==out
              // dgree, so just need to sum the degree for the vertices in this GPU... yeah... but
              // that's 1 over square root P of the total edges so not small... (but still not
              // prohibitive especially for large P so not a bad thing....)
              *(edge_buffer_first + (*num_edge_inserts)++) = thrust::make_tuple(tag, old);
            }
            return (old == invalid_component_id<vertex_t>::value)
                     ? thrust::optional<thrust::tuple<size_t, std::byte>>{thrust::make_tuple(static_cast<size_t>(Bucket::next), std::byte{0}/* dummy */)}
                     : thrust::nullopt;
          } else {
            return thrust::optional<thrust::tuple<size_t, std::byte>>{thrust::make_tuple(static_cast<size_t>(Bucket::next), std::byte{0}/* dummy */)};
          }
        });

      // FIXME: if we maintain sorted & unique edge_buffer elements, we can run sort & unique to the
      // newly added edges and run merge & unique (this is unnecessary if we use cuco::static_map
      // (no duplicates, ideally we need static_set)

      // FIXME: what if I run pointer-jumping here to reduce # tags?

      if (GraphViewType::is_multi_gpu) {
        copy_to_adj_matrix_col(
          handle,
          level_graph_view,
          thrust::get<0>(vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next))
                           .begin()
                           .get_iterator_tuple()),
          thrust::get<0>(vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next))
                           .end()
                           .get_iterator_tuple()),
          level_components,
          col_components.begin());
      }

      edge_count = thrust::transform_reduce(
        rmm::exec_policy(handle.get_stream_view()),
        thrust::get<0>(vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next))
                         .begin()
                         .get_iterator_tuple()),
        thrust::get<0>(
          vertex_frontier.get_bucket(static_cast<size_t>(Bucket::next)).end().get_iterator_tuple()),
        [vertex_partition, degrees = degrees.data()] __device__(auto v) {
          return degrees[vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v)];
        },
        edge_t{0},
        thrust::plus<edge_t>());

      // FIXME: if the total number of edges from the frontier is too large, we really don't need to
      // expand the entire frontier.
      if (edge_count < degree_sum_threshold) {
        auto [new_roots, num_scanned, degree_sum] =
          accumulate_new_roots(handle,
                               level_components + next_local_vertex_offset,
                               degrees.begin() + next_local_vertex_offset,
                               level_graph_view.get_local_vertex_first() + next_local_vertex_offset,
                               level_graph_view.get_local_vertex_last(),
                               degree_sum_threshold - edge_count);
        next_local_vertex_offset += num_scanned;
        edge_count += degree_sum;
      }

      // FIXME: no sync necessary before accessing num_edge_inserts?

      // FIXME: if we use cuco::static_map (no duplicates, ideally we need static_set), edge_buffer
      // size cannot exceed (# local roots * # aggregate roots)
      resize_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        edge_buffer,
        num_edge_inserts.load(cuda::std::memory_order_relaxed) + edge_count +
          (GraphViewType::is_multi_gpu ? local_vertex_in_degree_sum : edge_t{0}),
        handle.get_stream());

      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).clear();
      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).shrink_to_fit();
      vertex_frontier.swap_buckets(static_cast<size_t>(Bucket::cur),
                                   static_cast<size_t>(Bucket::next));
      if (vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).aggregate_size() == 0) {
        break;
      }
    }

#if 0
    // FIXME: may need some sync to ensure that device operations on num_edge_inserts are finished.

    if (num_edge_inserts.load(cuda::std::memory_order_relaxed) > 0) {
      // FIXME: remove duplicates from edge list (we may use cuco::static_map to avoid this)
      // FIXME: move generate_graph_from_edge_list from test/utilities
      std::tie(level_graph, ) = generate_graph_from_edgelist();
      if (graph is small enough) {  // don't care about work optimality if the graph is small
        label_prop + pointer jumping
        break;
      }
    }
#endif

    ++level;
  }

#if 0
  relabel<vertex_t, GraphViewType::is_multi_gpu>(handle, );
#endif
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void weakly_connected_components(
  raft::handle_t const &handle,
  graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const &graph_view,
  vertex_t *components,
  bool do_expensive_check)
{
  weakly_connected_components_impl(handle, graph_view, components, do_expensive_check);
}

// explicit instantiation

template void weakly_connected_components(
  raft::handle_t const &handle,
  graph_view_t<int32_t, int32_t, float, false, false> const &graph_view,
  int32_t *components,
  bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
