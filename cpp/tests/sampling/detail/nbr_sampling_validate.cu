/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

// utilities for testing / verification of Nbr Sampling functionality:
//
namespace cugraph {
namespace test {

// FIXME: Consider moving this to thrust_tuple_utils and making it
//        generic for any tuple that supports < operator
struct ArithmeticZipLess {
  template <typename left_t, typename right_t>
  __device__ bool operator()(left_t const& left, right_t const& right)
  {
    if constexpr (cugraph::is_thrust_tuple_of_arithmetic<left_t>::value) {
      // Need a more generic solution, for now I can just check cuda::std::tuple_size
      if (cuda::std::get<0>(left) < cuda::std::get<0>(right)) return true;
      if (cuda::std::get<0>(right) < cuda::std::get<0>(left)) return false;

      if constexpr (cuda::std::tuple_size<left_t>::value > 2) {
        if (cuda::std::get<1>(left) < cuda::std::get<1>(right)) return true;
        if (cuda::std::get<1>(right) < cuda::std::get<1>(left)) return false;
        return cuda::std::get<2>(left) < cuda::std::get<2>(right);
      } else {
        return cuda::std::get<1>(left) < cuda::std::get<1>(right);
      }
    } else {
      return false;
    }
  }
};

// FIXME: Consider moving this to thrust_tuple_utils and making it
//        generic for any tuple that supports < operator
struct ArithmeticZipEqual {
  template <typename vertex_t, typename weight_t>
  __device__ bool operator()(cuda::std::tuple<vertex_t, vertex_t, weight_t> const& left,
                             cuda::std::tuple<vertex_t, vertex_t, weight_t> const& right)
  {
    return (cuda::std::get<0>(left) == cuda::std::get<0>(right)) &&
           (cuda::std::get<1>(left) == cuda::std::get<1>(right)) &&
           (cuda::std::get<2>(left) == cuda::std::get<2>(right));
  }

  template <typename vertex_t>
  __device__ bool operator()(cuda::std::tuple<vertex_t, vertex_t> const& left,
                             cuda::std::tuple<vertex_t, vertex_t> const& right)
  {
    return (cuda::std::get<0>(left) == cuda::std::get<0>(right)) &&
           (cuda::std::get<1>(left) == cuda::std::get<1>(right));
  }
};

template <typename vertex_t, typename weight_t>
bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> src,
  raft::device_span<vertex_t const> dst,
  std::optional<raft::device_span<weight_t const>> wgt,
  raft::device_span<vertex_t const> subgraph_src,
  raft::device_span<vertex_t const> subgraph_dst,
  std::optional<raft::device_span<weight_t const>> subgraph_wgt)
{
  if (wgt.has_value() != subgraph_wgt.has_value()) { return false; }

  rmm::device_uvector<vertex_t> src_v(src.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(dst.size(), handle.get_stream());
  raft::copy(src_v.data(), src.data(), src.size(), handle.get_stream());
  raft::copy(dst_v.data(), dst.data(), dst.size(), handle.get_stream());

  size_t num_invalids{0};
  if (wgt) {
    rmm::device_uvector<weight_t> wgt_v(wgt->size(), handle.get_stream());
    raft::copy(wgt_v.data(), wgt->data(), wgt->size(), handle.get_stream());

    auto graph_iter = thrust::make_zip_iterator(src_v.begin(), dst_v.begin(), wgt_v.begin());
    thrust::sort(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipLess{});
    auto graph_iter_end = thrust::unique(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipEqual{});
    auto new_size = cuda::std::distance(graph_iter, graph_iter_end);

    src_v.resize(new_size, handle.get_stream());
    dst_v.resize(new_size, handle.get_stream());
    wgt_v.resize(new_size, handle.get_stream());

    auto subgraph_iter =
      thrust::make_zip_iterator(subgraph_src.begin(), subgraph_dst.begin(), subgraph_wgt->begin());
    num_invalids =
      thrust::count_if(handle.get_thrust_policy(),
                       subgraph_iter,
                       subgraph_iter + subgraph_src.size(),
                       [graph_iter, new_size] __device__(auto tup) {
                         return (thrust::binary_search(
                                   thrust::seq, graph_iter, graph_iter + new_size, tup) == false);
                       });
  } else {
    auto graph_iter = thrust::make_zip_iterator(src_v.begin(), dst_v.begin());
    thrust::sort(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipLess{});
    auto graph_iter_end = thrust::unique(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipEqual{});
    auto new_size = cuda::std::distance(graph_iter, graph_iter_end);

    src_v.resize(new_size, handle.get_stream());
    dst_v.resize(new_size, handle.get_stream());

    auto subgraph_iter = thrust::make_zip_iterator(subgraph_src.begin(), subgraph_dst.begin());
    num_invalids =
      thrust::count_if(handle.get_thrust_policy(),
                       subgraph_iter,
                       subgraph_iter + subgraph_src.size(),
                       [graph_iter, new_size] __device__(auto tup) {
                         return (thrust::binary_search(
                                   thrust::seq, graph_iter, graph_iter + new_size, tup) == false);
                       });
  }

  return (num_invalids == 0);
}

template bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> src,
  raft::device_span<int32_t const> dst,
  std::optional<raft::device_span<float const>> wgt,
  raft::device_span<int32_t const> subgraph_src,
  raft::device_span<int32_t const> subgraph_dst,
  std::optional<raft::device_span<float const>> subgraph_wgt);

template bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> src,
  raft::device_span<int32_t const> dst,
  std::optional<raft::device_span<double const>> wgt,
  raft::device_span<int32_t const> subgraph_src,
  raft::device_span<int32_t const> subgraph_dst,
  std::optional<raft::device_span<double const>> subgraph_wgt);

template bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> src,
  raft::device_span<int64_t const> dst,
  std::optional<raft::device_span<float const>> wgt,
  raft::device_span<int64_t const> subgraph_src,
  raft::device_span<int64_t const> subgraph_dst,
  std::optional<raft::device_span<float const>> subgraph_wgt);

template bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> src,
  raft::device_span<int64_t const> dst,
  std::optional<raft::device_span<double const>> wgt,
  raft::device_span<int64_t const> subgraph_src,
  raft::device_span<int64_t const> subgraph_dst,
  std::optional<raft::device_span<double const>> subgraph_wgt);

template <typename vertex_t, typename weight_t>
bool validate_sampling_depth(raft::handle_t const& handle,
                             rmm::device_uvector<vertex_t>&& d_src,
                             rmm::device_uvector<vertex_t>&& d_dst,
                             std::optional<rmm::device_uvector<weight_t>>&& d_wgt,
                             rmm::device_uvector<vertex_t>&& d_source_vertices,
                             int max_depth)
{
  graph_t<vertex_t, vertex_t, false, false> graph(handle);
  std::optional<rmm::device_uvector<vertex_t>> number_map{std::nullopt};
  std::tie(graph, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, number_map) =
    create_graph_from_edgelist<vertex_t, vertex_t, weight_t, int32_t, int32_t, false, false>(
      handle,
      std::nullopt,
      std::move(d_src),
      std::move(d_dst),
      std::move(d_wgt),
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      graph_properties_t{false, true},
      true);

  auto graph_view = graph.view();

  //  Renumber sources
  renumber_ext_vertices<vertex_t, false>(handle,
                                         d_source_vertices.data(),
                                         d_source_vertices.size(),
                                         number_map->data(),
                                         graph_view.local_vertex_partition_range_first(),
                                         graph_view.local_vertex_partition_range_last());

  rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), d_distances.begin(), d_distances.end(), vertex_t{max_depth + 1});

  rmm::device_uvector<vertex_t> d_local_distances(graph_view.number_of_vertices(),
                                                  handle.get_stream());

  std::vector<vertex_t> h_source_vertices(d_source_vertices.size());
  raft::update_host(h_source_vertices.data(),
                    d_source_vertices.data(),
                    d_source_vertices.size(),
                    handle.get_stream());

  for (size_t i = 0; i < d_source_vertices.size(); ++i) {
    if (h_source_vertices[i] != cugraph::invalid_vertex_id<vertex_t>::value) {
      // Do BFS
      cugraph::bfs<vertex_t, vertex_t, false>(handle,
                                              graph_view,
                                              d_local_distances.data(),
                                              nullptr,
                                              d_source_vertices.data() + i,
                                              size_t{1},
                                              bool{false},
                                              vertex_t{max_depth});

      auto tuple_iter = thrust::make_zip_iterator(d_distances.begin(), d_local_distances.begin());

      thrust::transform(handle.get_thrust_policy(),
                        tuple_iter,
                        tuple_iter + d_distances.size(),
                        d_distances.begin(),
                        [] __device__(auto tuple) {
                          return cuda::std::min(cuda::std::get<0>(tuple), cuda::std::get<1>(tuple));
                        });
    }
  }

  return (thrust::count_if(handle.get_thrust_policy(),
                           d_distances.begin(),
                           d_distances.end(),
                           [max_depth] __device__(auto d) { return d > max_depth; }) == 0);
}

template bool validate_sampling_depth(raft::handle_t const& handle,
                                      rmm::device_uvector<int32_t>&& d_src,
                                      rmm::device_uvector<int32_t>&& d_dst,
                                      std::optional<rmm::device_uvector<float>>&& d_wgt,
                                      rmm::device_uvector<int32_t>&& d_source_vertices,
                                      int max_depth);

template bool validate_sampling_depth(raft::handle_t const& handle,
                                      rmm::device_uvector<int32_t>&& d_src,
                                      rmm::device_uvector<int32_t>&& d_dst,
                                      std::optional<rmm::device_uvector<double>>&& d_wgt,
                                      rmm::device_uvector<int32_t>&& d_source_vertices,
                                      int max_depth);

template bool validate_sampling_depth(raft::handle_t const& handle,
                                      rmm::device_uvector<int64_t>&& d_src,
                                      rmm::device_uvector<int64_t>&& d_dst,
                                      std::optional<rmm::device_uvector<float>>&& d_wgt,
                                      rmm::device_uvector<int64_t>&& d_source_vertices,
                                      int max_depth);

template bool validate_sampling_depth(raft::handle_t const& handle,
                                      rmm::device_uvector<int64_t>&& d_src,
                                      rmm::device_uvector<int64_t>&& d_dst,
                                      std::optional<rmm::device_uvector<double>>&& d_wgt,
                                      rmm::device_uvector<int64_t>&& d_source_vertices,
                                      int max_depth);

template <typename vertex_t, typename edge_time_t>
bool validate_temporal_integrity(raft::handle_t const& handle,
                                 raft::device_span<vertex_t const> srcs,
                                 raft::device_span<vertex_t const> dsts,
                                 raft::device_span<edge_time_t const> edge_times,
                                 raft::device_span<vertex_t const> source_vertices)
{
  // Sampling doesn't return paths.  All I can do is determine if an edge
  // with time t could have been legally selected, not whether it was correct to
  // actually be selected.

  // for each entry in srcs that is not in source_vertices, there needs to exist
  // an entry in dsts that has a corresponding edge_time less than the time for this src.
  //
  //  I think I can do the following:
  //    1) compute  the minimum edge_time for each dst
  //    2) Foreach src, search for dst in minimum time tuple.  If it exists check the time, if it
  //    does not exist verify that it exists in source_vertices

  rmm::device_uvector<vertex_t> sorted_dsts(dsts.size(), handle.get_stream());
  rmm::device_uvector<edge_time_t> sorted_dst_times(edge_times.size(), handle.get_stream());

  raft::copy(sorted_dsts.begin(), dsts.begin(), dsts.size(), handle.get_stream());
  raft::copy(sorted_dst_times.begin(), edge_times.begin(), edge_times.size(), handle.get_stream());

  thrust::sort(handle.get_thrust_policy(),
               thrust::make_zip_iterator(sorted_dsts.begin(), sorted_dst_times.begin()),
               thrust::make_zip_iterator(sorted_dsts.end(), sorted_dst_times.end()));

  sorted_dsts.resize(thrust::distance(sorted_dsts.begin(),
                                      thrust::reduce_by_key(handle.get_thrust_policy(),
                                                            sorted_dsts.begin(),
                                                            sorted_dsts.end(),
                                                            sorted_dst_times.begin(),
                                                            sorted_dsts.begin(),
                                                            sorted_dst_times.begin(),
                                                            thrust::equal_to<vertex_t>(),
                                                            thrust::minimum<edge_time_t>())
                                        .first),
                     handle.get_stream());
  sorted_dst_times.resize(sorted_dsts.size(), handle.get_stream());

  auto error_count = thrust::count_if(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(srcs.begin(), dsts.begin(), edge_times.begin()),
    thrust::make_zip_iterator(srcs.end(), dsts.end(), edge_times.end()),
    [min_dsts = raft::device_span<vertex_t const>{sorted_dsts.data(), sorted_dsts.size()},
     min_dst_times =
       raft::device_span<edge_time_t const>{sorted_dst_times.data(), sorted_dst_times.size()},
     source_vertices] __device__(auto t) {
      vertex_t src     = cuda::std::get<0>(t);
      vertex_t dst     = cuda::std::get<1>(t);
      edge_time_t time = cuda::std::get<2>(t);

      bool vertex_is_source =
        thrust::find(thrust::seq, source_vertices.begin(), source_vertices.end(), src) !=
        source_vertices.end();

      if (vertex_is_source) {
        return false;
      } else {
        auto pos = thrust::lower_bound(thrust::seq, min_dsts.begin(), min_dsts.end(), dst);
        if ((pos == min_dsts.end()) || (*pos != dst)) {
          printf(". dst %d not found in min_dsts\n", (int)dst);
          return true;
        } else {
          return time < min_dst_times[thrust::distance(min_dsts.begin(), pos)];
        }
      }
    });

  return error_count == 0;
}

template bool validate_temporal_integrity(raft::handle_t const& handle,
                                          raft::device_span<int32_t const> src,
                                          raft::device_span<int32_t const> dst,
                                          raft::device_span<int32_t const> edge_time,
                                          raft::device_span<int32_t const> source_vertices);

template bool validate_temporal_integrity(raft::handle_t const& handle,
                                          raft::device_span<int32_t const> src,
                                          raft::device_span<int32_t const> dst,
                                          raft::device_span<int64_t const> edge_time,
                                          raft::device_span<int32_t const> source_vertices);

template bool validate_temporal_integrity(raft::handle_t const& handle,
                                          raft::device_span<int64_t const> src,
                                          raft::device_span<int64_t const> dst,
                                          raft::device_span<int32_t const> edge_time,
                                          raft::device_span<int64_t const> source_vertices);

template bool validate_temporal_integrity(raft::handle_t const& handle,
                                          raft::device_span<int64_t const> src,
                                          raft::device_span<int64_t const> dst,
                                          raft::device_span<int64_t const> edge_time,
                                          raft::device_span<int64_t const> source_vertices);

}  // namespace test
}  // namespace cugraph
