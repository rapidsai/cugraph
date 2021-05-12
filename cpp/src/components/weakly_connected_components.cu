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
#include <experimental/graph_functions.hpp>
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
#include <thrust/shuffle.h>
#include <thrust/tuple.h>

#include <limits>
#include <type_traits>

namespace cugraph {
namespace experimental {

namespace {

// FIXME: this function (after modification) may be useful for SSSP with the near-far method to
// determine the near-far threshold.
// add new roots till the sum of the degrees first becomes no smaller than degree_sum_threshold and
// returns a triplet of (new roots, number of scanned candidates, sum of the degrees of the new
// roots)
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           typename GraphViewType::vertex_type,
           typename GraphViewType::edge_type>
accumulate_new_roots(raft::handle_t const &handle,
                     vertex_partition_device_t<GraphViewType> vertex_partition,
                     typename GraphViewType::vertex_type const *components,
                     typename GraphViewType::edge_type const *degrees,
                     typename GraphViewType::vertex_type const *candidate_first,
                     typename GraphViewType::vertex_type const *candidate_last,
                     typename GraphViewType::vertex_type max_new_roots,
                     typename GraphViewType::edge_type degree_sum_threshold)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  // FIXME: tuning parameter (time to scan max_scan_size elements should not take significantly
  // longer than scanning a single element)
  vertex_t max_scan_size =
    static_cast<vertex_t>(handle.get_device_properties().multiProcessorCount) * vertex_t{1024};

  rmm::device_uvector<vertex_t> new_roots(0, handle.get_stream_view());
  vertex_t num_scanned{0};
  edge_t degree_sum{0};
  while (candidate_first + num_scanned < candidate_last) {
    auto scan_size = std::min(
      max_scan_size,
      static_cast<vertex_t>(thrust::distance(candidate_first + num_scanned, candidate_last)));

    auto num_new_roots = static_cast<vertex_t>(new_roots.size());
    new_roots.resize(num_new_roots + scan_size, handle.get_stream_view());
    rmm::device_uvector<vertex_t> scan_counts(0, handle.get_stream_view());
    {
      rmm::device_uvector<vertex_t> indices(scan_size, handle.get_stream_view());
      auto input_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
                                candidate_first, thrust::make_counting_iterator(vertex_t{0}))) +
                              num_scanned;
      auto output_pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(new_roots.begin() + num_new_roots, indices.begin()));
      new_roots.resize(
        static_cast<vertex_t>(thrust::distance(
          output_pair_first,
          thrust::copy_if(
            rmm::exec_policy(handle.get_stream_view()),
            input_pair_first,
            input_pair_first + scan_size,
            output_pair_first,
            [vertex_partition, components] __device__(auto pair) {
              auto v = thrust::get<0>(pair);
              return (components[vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v)] ==
                      invalid_component_id<vertex_t>::value);
            }))),
        handle.get_stream_view());
      indices.resize(new_roots.size(), handle.get_stream_view());

      scan_counts.resize(new_roots.size() - num_new_roots, handle.get_stream_view());
      thrust::tabulate(
        rmm::exec_policy(handle.get_stream_view()),
        scan_counts.begin(),
        scan_counts.end(),
        [scan_size, indices = indices.data(), num_indices = indices.size()] __device__(auto i) {
          return i < num_indices - 1 ? indices[i + 1] : scan_size;
        });

      if (static_cast<vertex_t>(new_roots.size()) > max_new_roots) {
        new_roots.resize(max_new_roots, handle.get_stream_view());
        scan_counts.resize(new_roots.size() - num_new_roots, handle.get_stream_view());
      }
    }

    if (static_cast<vertex_t>(new_roots.size()) > num_new_roots) {
      rmm::device_uvector<edge_t> cumulative_degrees(new_roots.size() - num_new_roots,
                                                     handle.get_stream_view());
      thrust::transform(
        rmm::exec_policy(handle.get_stream_view()),
        new_roots.begin() + num_new_roots,
        new_roots.end(),
        cumulative_degrees.begin(),
        [vertex_partition, degrees] __device__(auto v) {
          return degrees[vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v)];
        });
      thrust::inclusive_scan(rmm::exec_policy(handle.get_stream_view()),
                             cumulative_degrees.begin(),
                             cumulative_degrees.end(),
                             cumulative_degrees.begin());

      auto last = thrust::lower_bound(rmm::exec_policy(handle.get_stream_view()),
                                      cumulative_degrees.begin(),
                                      cumulative_degrees.end(),
                                      degree_sum_threshold - degree_sum);
      if (last == cumulative_degrees.end()) {
        last = cumulative_degrees.begin() + (cumulative_degrees.size() - 1);
      }

      new_roots.resize(num_new_roots + thrust::distance(cumulative_degrees.begin(), last) + 1,
                       handle.get_stream_view());
      vertex_t tmp_num_scanned{0};
      edge_t tmp_degree_sum{0};
      raft::update_host(&tmp_num_scanned,
                        scan_counts.data() + thrust::distance(cumulative_degrees.begin(), last),
                        size_t{1},
                        handle.get_stream());
      raft::update_host(
        &tmp_degree_sum,
        cumulative_degrees.data() + thrust::distance(cumulative_degrees.begin(), last),
        size_t{1},
        handle.get_stream());
      handle.get_stream_view().synchronize();
      num_scanned += tmp_num_scanned;
      degree_sum += tmp_degree_sum;
    } else {
      num_scanned += scan_size;
    }

    if (static_cast<vertex_t>(new_roots.size()) >= max_new_roots ||
        degree_sum >= degree_sum_threshold) {
      break;
    }
  }

  new_roots.shrink_to_fit(handle.get_stream_view());

  return std::make_tuple(std::move(new_roots), num_scanned, degree_sum);
}

// FIXME: to silence the spurious warning (missing return statement ...) due to the nvcc bug
// (https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-
// function-in-constexpr-if-fun)
template <typename GraphViewType>
struct v_op_t {
  using vertex_type = typename GraphViewType::vertex_type;

  vertex_partition_device_t<GraphViewType> vertex_partition{};
  vertex_type const *level_components{};
  decltype(thrust::make_zip_iterator(thrust::make_tuple(
    static_cast<vertex_type *>(nullptr), static_cast<vertex_type *>(nullptr)))) edge_buffer_first{};
  // FIXME: we can use cuda::atomic instead but currently on a system with x86 + GPU, this requires
  // placing the atomic barrier on managed memory and this adds additional complication.
  size_t *num_edge_inserts{};
  size_t next_bucket_idx{};

  template <bool multi_gpu = GraphViewType::is_multi_gpu>
  __device__ std::enable_if_t<multi_gpu, thrust::optional<thrust::tuple<size_t, std::byte>>>
  operator()(thrust::tuple<vertex_type, vertex_type> tagged_v, int v_val /* dummy */) const
  {
    auto tag = thrust::get<1>(tagged_v);
    auto v_offset =
      vertex_partition.get_local_vertex_offset_from_vertex_nocheck(thrust::get<0>(tagged_v));
    // FIXME: better switch to atomic_ref after
    // https://github.com/nvidia/libcudacxx/milestone/2
    auto old =
      atomicCAS(level_components + v_offset, invalid_component_id<vertex_type>::value, tag);
    if (old != invalid_component_id<vertex_type>::value && old != tag) {  // conflict
      // FIXME: potential overflow? we need to count dst for square root P GPUs then do
      // reduction to figure out actual maximum... this is too pessimistic and also
      // expensive, actually, we know the graph is symmetric and we alredy have in==out
      // dgree, so just need to sum the degree for the vertices in this GPU... yeah... but
      // that's 1 over square root P of the total edges so not small... (but still not
      // prohibitive especially for large P so not a bad thing....)
      static_assert(sizeof(unsigned long long int) == sizeof(size_t));
      auto edge_idx = atomicAdd(reinterpret_cast<unsigned long long int *>(num_edge_inserts),
                                static_cast<unsigned long long int>(1));
      *(edge_buffer_first + edge_idx) = thrust::make_tuple(tag, old);
    }
    return (old == invalid_component_id<vertex_type>::value)
             ? thrust::optional<thrust::tuple<size_t, std::byte>>{thrust::make_tuple(
                 next_bucket_idx, std::byte{0} /* dummy */)}
             : thrust::nullopt;
  }

  template <bool multi_gpu = GraphViewType::is_multi_gpu>
  __device__ std::enable_if_t<!multi_gpu, thrust::optional<thrust::tuple<size_t, std::byte>>>
  operator()(thrust::tuple<vertex_type, vertex_type> tagged_v, int v_val /* dummy */) const
  {
    return thrust::optional<thrust::tuple<size_t, std::byte>>{
      thrust::make_tuple(next_bucket_idx, std::byte{0} /* dummy */)};
  }
};

template <typename GraphViewType>
void weakly_connected_components_impl(raft::handle_t const &handle,
                                      GraphViewType const &push_graph_view,
                                      typename GraphViewType::vertex_type *components,
                                      bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

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

  size_t num_levels{0};
  graph_t<vertex_t,
          edge_t,
          typename GraphViewType::weight_type,
          GraphViewType::is_adj_matrix_transposed,
          GraphViewType::is_multi_gpu>
    level_graph(handle);
  rmm::device_uvector<vertex_t> level_renumber_map(0, handle.get_stream_view());
  std::vector<rmm::device_uvector<vertex_t>> level_component_vectors{};
  // vertex ID in this level to the component ID in the finer level
  std::vector<rmm::device_uvector<vertex_t>> level_renumber_map_vectors{};
  std::vector<vertex_t> level_local_vertex_first_vectors{};
  while (true) {
    auto level_graph_view = num_levels == 0 ? push_graph_view : level_graph.view();
    vertex_partition_device_t<GraphViewType> vertex_partition(level_graph_view);
    level_component_vectors.push_back(rmm::device_uvector<vertex_t>(
      num_levels == 0 ? vertex_t{0} : level_graph_view.get_number_of_local_vertices(),
      handle.get_stream_view()));
    level_renumber_map_vectors.push_back(std::move(level_renumber_map));
    level_local_vertex_first_vectors.push_back(level_graph_view.get_local_vertex_first());
    auto level_components =
      num_levels == 0 ? components : level_component_vectors[num_levels].data();
    ++num_levels;
    auto degrees = level_graph_view.compute_out_degrees(handle);
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

    // 2-2. initialize new root candidates

    // Vertices are first partitioned to high-degree vertices and low-degree vertices, we can reach
    // degree_sum_threshold with fewer high-degree vertices leading to a higher compression ratio.
    // The degree threshold is set to ceil(sqrt(degree_sum_threshold * 2)); this guarantees the
    // compression ratio of at least 50% (ignoring rounding errors) even if all the selected roots
    // fall into a single connected component as there will be at least as many non-root vertices in
    // the connected component (assuming there are no multi-edges, if there are multi-edges, we may
    // not get 50% compression in # vertices but still get compression in # edges). the remaining
    // low-degree vertices will be randomly shuffled so comparable ratios of vertices will be
    // selected as roots in the remaining connected components.

    rmm::device_uvector<vertex_t> new_root_candidates(
      level_graph_view.get_number_of_local_vertices(), handle.get_stream_view());
    new_root_candidates.resize(
      thrust::distance(
        new_root_candidates.begin(),
        thrust::copy_if(
          rmm::exec_policy(handle.get_stream_view()),
          thrust::make_counting_iterator(level_graph_view.get_local_vertex_first()),
          thrust::make_counting_iterator(level_graph_view.get_local_vertex_last()),
          new_root_candidates.begin(),
          [vertex_partition, level_components] __device__(auto v) {
            return level_components[vertex_partition.get_local_vertex_offset_from_vertex_nocheck(
                     v)] == invalid_component_id<vertex_t>::value;
          })),
      handle.get_stream_view());
    auto high_degree_partition_last = thrust::stable_partition(
      rmm::exec_policy(handle.get_stream_view()),
      new_root_candidates.begin(),
      new_root_candidates.end(),
      [vertex_partition,
       degrees   = degrees.data(),
       threshold = static_cast<edge_t>(
         ceil(sqrt(static_cast<double>(degree_sum_threshold) * 2.0)))] __device__(auto v) {
        return degrees[vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v)] >=
               threshold;
      });
    thrust::shuffle(rmm::exec_policy(handle.get_stream_view()),
                    high_degree_partition_last,
                    new_root_candidates.end(),
                    thrust::default_random_engine());

    double constexpr max_new_roots_ratio =
      0.1;  // to avoid selecting all the vertices as roots leading to zero compression
    auto max_new_roots = std::max(
      static_cast<vertex_t>(new_root_candidates.size() * max_new_roots_ratio), vertex_t{1});

    // 2-3. initialize vertex frontier

    VertexFrontier<vertex_t,
                   vertex_t,
                   GraphViewType::is_multi_gpu,
                   static_cast<size_t>(Bucket::num_buckets)>
      vertex_frontier(handle);
    vertex_t next_candidate_offset{0};
    edge_t edge_count{0};

    auto edge_buffer =
      allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(0, handle.get_stream());
    // FIXME: we can use cuda::atomic instead but currently on a system with x86 + GPU, this
    // requires placing the atomic variable on managed memory and this make it less attractive.
    rmm::device_scalar<size_t> num_edge_inserts(size_t{0}, handle.get_stream_view());

    rmm::device_uvector<vertex_t> col_components(
      GraphViewType::is_multi_gpu ? level_graph_view.get_number_of_local_adj_matrix_partition_cols()
                                  : vertex_t{0},
      handle.get_stream_view());
    if (GraphViewType::is_multi_gpu) {
      thrust::fill(rmm::exec_policy(handle.get_stream_view()),
                   col_components.begin(),
                   col_components.end(),
                   invalid_component_id<vertex_t>::value);
    }

    // 2.3 iterate till every vertex gets visited

    while (true) {
      if (edge_count < degree_sum_threshold) {
        auto [new_roots, num_scanned, degree_sum] =
          accumulate_new_roots(handle,
                               vertex_partition,
                               level_components,
                               degrees.data(),
                               new_root_candidates.data() + next_candidate_offset,
                               new_root_candidates.data() + new_root_candidates.size(),
                               max_new_roots,
                               degree_sum_threshold - edge_count);
        next_candidate_offset += num_scanned;
        edge_count += degree_sum;

        thrust::sort(
          rmm::exec_policy(handle.get_stream_view()), new_roots.begin(), new_roots.end());

        thrust::for_each(
          rmm::exec_policy(handle.get_stream_view()),
          new_roots.begin(),
          new_roots.end(),
          [vertex_partition, components = level_components] __device__(auto c) {
            components[vertex_partition.get_local_vertex_offset_from_vertex_nocheck(c)] = c;
          });

        auto pair_first =
          thrust::make_zip_iterator(thrust::make_tuple(new_roots.begin(), new_roots.begin()));
        vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur))
          .insert(pair_first, pair_first + new_roots.size());
      }

      if (vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).aggregate_size() == 0) {
        break;
      }

      if (GraphViewType::is_multi_gpu) {
        copy_to_adj_matrix_col(
          handle,
          level_graph_view,
          thrust::get<0>(vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur))
                           .begin()
                           .get_iterator_tuple()),
          thrust::get<0>(vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur))
                           .end()
                           .get_iterator_tuple()),
          level_components,
          col_components.begin());
      }

      // FIXME: if we use cuco::static_map (no duplicates, ideally we need static_set),
      // edge_buffer size cannot exceed (# local roots * # aggregate roots)
      resize_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        edge_buffer,
        num_edge_inserts.value(handle.get_stream_view()) + edge_count +
          (GraphViewType::is_multi_gpu ? local_vertex_in_degree_sum : edge_t{0}),
        handle.get_stream());

      update_frontier_v_push_if_out_nbr(
        handle,
        level_graph_view,
        vertex_frontier,
        static_cast<size_t>(Bucket::cur),
        std::vector<size_t>{static_cast<size_t>(Bucket::next)},
        thrust::make_counting_iterator(0) /* dummy */,
        thrust::make_counting_iterator(0) /* dummy */,
        [col_components = GraphViewType::is_multi_gpu ? col_components.data() : level_components,
         col_first      = level_graph_view.get_local_adj_matrix_partition_col_first(),
         edge_buffer_first =
           get_dataframe_buffer_begin<thrust::tuple<vertex_t, vertex_t>>(edge_buffer),
         num_edge_inserts = num_edge_inserts.data()] __device__(auto tagged_src,
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
            static_assert(sizeof(unsigned long long int) == sizeof(size_t));
            auto edge_idx = atomicAdd(reinterpret_cast<unsigned long long int *>(num_edge_inserts),
                                      static_cast<unsigned long long int>(1));
            *(edge_buffer_first + edge_idx) = thrust::make_tuple(tag, old);
          }
          return (old == invalid_component_id<vertex_t>::value) ? thrust::optional<vertex_t>{tag}
                                                                : thrust::nullopt;
        },
        reduce_op::null(),
        thrust::make_constant_iterator(0) /* dummy */,
        thrust::make_discard_iterator() /* dummy */,
        v_op_t<GraphViewType>{
          vertex_partition,
          level_components,
          get_dataframe_buffer_begin<thrust::tuple<vertex_t, vertex_t>>(edge_buffer),
          num_edge_inserts.data(),
          static_cast<size_t>(Bucket::next)});

      // FIXME: if we maintain sorted & unique edge_buffer elements, we can run sort & unique to
      // the newly added edges and run merge & unique (this is unnecessary if we use
      // cuco::static_map (no duplicates, ideally we need static_set)

      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).clear();
      vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).shrink_to_fit();
      vertex_frontier.swap_buckets(static_cast<size_t>(Bucket::cur),
                                   static_cast<size_t>(Bucket::next));
      edge_count = thrust::transform_reduce(
        rmm::exec_policy(handle.get_stream_view()),
        thrust::get<0>(vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur))
                         .begin()
                         .get_iterator_tuple()),
        thrust::get<0>(
          vertex_frontier.get_bucket(static_cast<size_t>(Bucket::cur)).end().get_iterator_tuple()),
        [vertex_partition, degrees = degrees.data()] __device__(auto v) {
          return degrees[vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v)];
        },
        edge_t{0},
        thrust::plus<edge_t>());
    }

    if (auto num_inserts = num_edge_inserts.value(handle.get_stream_view()); num_inserts > 0) {
      resize_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        edge_buffer, static_cast<size_t>(num_inserts * 2), handle.get_stream());
      auto input_first = get_dataframe_buffer_begin<thrust::tuple<vertex_t, vertex_t>>(edge_buffer);
      auto output_first = thrust::make_zip_iterator(
                            thrust::make_tuple(thrust::get<1>(input_first.get_iterator_tuple()),
                                               thrust::get<0>(input_first.get_iterator_tuple()))) +
                          num_inserts;
      thrust::copy(rmm::exec_policy(handle.get_stream_view()),
                   input_first,
                   input_first + num_inserts,
                   output_first);
      auto edge_first = get_dataframe_buffer_begin<thrust::tuple<vertex_t, vertex_t>>(edge_buffer);
      thrust::sort(
        rmm::exec_policy(handle.get_stream_view()), edge_first, edge_first + num_inserts * 2);
      auto last = thrust::unique(
        rmm::exec_policy(handle.get_stream_view()), edge_first, edge_first + num_inserts * 2);
      resize_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
        edge_buffer, static_cast<size_t>(thrust::distance(edge_first, last)), handle.get_stream());
      shrink_to_fit_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(edge_buffer,
                                                                        handle.get_stream());
      std::tie(level_graph, level_renumber_map) =
        create_graph_from_edgelist<vertex_t,
                                   edge_t,
                                   weight_t,
                                   GraphViewType::is_adj_matrix_transposed,
                                   GraphViewType::is_multi_gpu>(
          handle,
          std::nullopt,
          std::move(std::get<0>(edge_buffer)),
          std::move(std::get<1>(edge_buffer)),
          rmm::device_uvector<weight_t>(size_t{0}, handle.get_stream_view()),
          graph_properties_t{true, false, false},
          true);
    } else {
      break;
    }
  }

  for (size_t i = 0; i < num_levels - 1; ++i) {
    size_t coarser_level = num_levels - 1 - i;
    size_t finer_level   = coarser_level - 1;

    rmm::device_uvector<vertex_t> coarser_local_vertices(
      level_renumber_map_vectors[coarser_level].size(), handle.get_stream_view());
    thrust::sequence(rmm::exec_policy(handle.get_stream_view()),
                     coarser_local_vertices.begin(),
                     coarser_local_vertices.end(),
                     level_local_vertex_first_vectors[coarser_level]);
    relabel<vertex_t, GraphViewType::is_multi_gpu>(
      handle,
      std::make_tuple(coarser_local_vertices.data(),
                      level_renumber_map_vectors[coarser_level].data()),
      coarser_local_vertices.size(),
      level_component_vectors[coarser_level].data(),
      level_component_vectors[coarser_level].size(),
      false);
    relabel<vertex_t, GraphViewType::is_multi_gpu>(
      handle,
      std::make_tuple(level_renumber_map_vectors[coarser_level].data(),
                      level_component_vectors[coarser_level].data()),
      level_renumber_map_vectors[coarser_level].size(),
      finer_level == 0 ? components : level_component_vectors[finer_level].data(),
      finer_level == 0 ? push_graph_view.get_number_of_local_vertices()
                       : level_component_vectors[finer_level].size(),
      true);
  }
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
