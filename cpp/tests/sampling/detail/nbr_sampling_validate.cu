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

#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
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
//        generic for any typle that supports < operator
struct ArithmeticZipLess {
  template <typename left_t, typename right_t>
  __device__ bool operator()(left_t const& left, right_t const& right)
  {
    if constexpr (cugraph::is_thrust_tuple_of_arithmetic<left_t>::value) {
      // Need a more generic solution, for now I can just check thrust::tuple_size
      if (thrust::get<0>(left) < thrust::get<0>(right)) return true;
      if (thrust::get<0>(right) < thrust::get<0>(left)) return false;

      if constexpr (thrust::tuple_size<left_t>::value > 2) {
        if (thrust::get<1>(left) < thrust::get<1>(right)) return true;
        if (thrust::get<1>(right) < thrust::get<1>(left)) return false;
        return thrust::get<2>(left) < thrust::get<2>(right);
      } else {
        return thrust::get<1>(left) < thrust::get<1>(right);
      }
    } else {
      return false;
    }
  }
};

// FIXME: Consider moving this to thrust_tuple_utils and making it
//        generic for any typle that supports < operator
struct ArithmeticZipEqual {
  template <typename vertex_t, typename weight_t>
  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t, weight_t> const& left,
                             thrust::tuple<vertex_t, vertex_t, weight_t> const& right)
  {
    return (thrust::get<0>(left) == thrust::get<0>(right)) &&
           (thrust::get<1>(left) == thrust::get<1>(right)) &&
           (thrust::get<2>(left) == thrust::get<2>(right));
  }

  template <typename vertex_t>
  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> const& left,
                             thrust::tuple<vertex_t, vertex_t> const& right)
  {
    return (thrust::get<0>(left) == thrust::get<0>(right)) &&
           (thrust::get<1>(left) == thrust::get<1>(right));
  }
};

template <typename vertex_t, typename weight_t>
bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t> const& src,
  rmm::device_uvector<vertex_t> const& dst,
  std::optional<rmm::device_uvector<weight_t>> const& wgt,
  rmm::device_uvector<vertex_t> const& subgraph_src,
  rmm::device_uvector<vertex_t> const& subgraph_dst,
  std::optional<rmm::device_uvector<weight_t>> const& subgraph_wgt)
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

    auto graph_iter =
      thrust::make_zip_iterator(thrust::make_tuple(src_v.begin(), dst_v.begin(), wgt_v.begin()));
    thrust::sort(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipLess{});
    auto graph_iter_end = thrust::unique(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipEqual{});
    auto new_size = cuda::std::distance(graph_iter, graph_iter_end);

    src_v.resize(new_size, handle.get_stream());
    dst_v.resize(new_size, handle.get_stream());
    wgt_v.resize(new_size, handle.get_stream());

    auto subgraph_iter = thrust::make_zip_iterator(
      thrust::make_tuple(subgraph_src.begin(), subgraph_dst.begin(), subgraph_wgt->begin()));
    num_invalids =
      thrust::count_if(handle.get_thrust_policy(),
                       subgraph_iter,
                       subgraph_iter + subgraph_src.size(),
                       [graph_iter, new_size] __device__(auto tup) {
                         return (thrust::binary_search(
                                   thrust::seq, graph_iter, graph_iter + new_size, tup) == false);
                       });
  } else {
    auto graph_iter = thrust::make_zip_iterator(thrust::make_tuple(src_v.begin(), dst_v.begin()));
    thrust::sort(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipLess{});
    auto graph_iter_end = thrust::unique(
      handle.get_thrust_policy(), graph_iter, graph_iter + src_v.size(), ArithmeticZipEqual{});
    auto new_size = cuda::std::distance(graph_iter, graph_iter_end);

    src_v.resize(new_size, handle.get_stream());
    dst_v.resize(new_size, handle.get_stream());

    auto subgraph_iter =
      thrust::make_zip_iterator(thrust::make_tuple(subgraph_src.begin(), subgraph_dst.begin()));
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
  rmm::device_uvector<int32_t> const& src,
  rmm::device_uvector<int32_t> const& dst,
  std::optional<rmm::device_uvector<float>> const& wgt,
  rmm::device_uvector<int32_t> const& subgraph_src,
  rmm::device_uvector<int32_t> const& subgraph_dst,
  std::optional<rmm::device_uvector<float>> const& subgraph_wgt);

template bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& src,
  rmm::device_uvector<int32_t> const& dst,
  std::optional<rmm::device_uvector<double>> const& wgt,
  rmm::device_uvector<int32_t> const& subgraph_src,
  rmm::device_uvector<int32_t> const& subgraph_dst,
  std::optional<rmm::device_uvector<double>> const& subgraph_wgt);

template bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& src,
  rmm::device_uvector<int64_t> const& dst,
  std::optional<rmm::device_uvector<float>> const& wgt,
  rmm::device_uvector<int64_t> const& subgraph_src,
  rmm::device_uvector<int64_t> const& subgraph_dst,
  std::optional<rmm::device_uvector<float>> const& subgraph_wgt);

template bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& src,
  rmm::device_uvector<int64_t> const& dst,
  std::optional<rmm::device_uvector<double>> const& wgt,
  rmm::device_uvector<int64_t> const& subgraph_src,
  rmm::device_uvector<int64_t> const& subgraph_dst,
  std::optional<rmm::device_uvector<double>> const& subgraph_wgt);

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

      auto tuple_iter = thrust::make_zip_iterator(
        thrust::make_tuple(d_distances.begin(), d_local_distances.begin()));

      thrust::transform(handle.get_thrust_policy(),
                        tuple_iter,
                        tuple_iter + d_distances.size(),
                        d_distances.begin(),
                        [] __device__(auto tuple) {
                          return cuda::std::min(thrust::get<0>(tuple), thrust::get<1>(tuple));
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

}  // namespace test
}  // namespace cugraph
