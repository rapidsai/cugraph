/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "generators/scramble.cuh"

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <numeric>

namespace cugraph {

namespace detail {

template <typename T>
rmm::device_uvector<T> append_all(raft::handle_t const& handle,
                                  std::vector<rmm::device_uvector<T>>&& input)
{
  auto size = std::transform_reduce(
    input.begin(), input.end(), size_t{0}, std::plus<size_t>{}, [](auto const& element) {
      return element.size();
    });

  rmm::device_uvector<T> output(size, handle.get_stream());
  auto output_iter = output.begin();

  for (auto& element : input) {
    raft::copy(output_iter, element.begin(), element.size(), handle.get_stream());
    output_iter += element.size();
  }

  return output;
}

}  // namespace detail

template <typename vertex_t>
rmm::device_uvector<vertex_t> scramble_vertex_ids(raft::handle_t const& handle,
                                                  rmm::device_uvector<vertex_t>&& vertices,
                                                  size_t lgN)
{
  thrust::transform(handle.get_thrust_policy(),
                    vertices.begin(),
                    vertices.end(),
                    vertices.begin(),
                    [lgN] __device__(auto v) { return detail::scramble(v, lgN); });

  return std::move(vertices);
}

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
combine_edgelists(raft::handle_t const& handle,
                  std::vector<rmm::device_uvector<vertex_t>>&& sources,
                  std::vector<rmm::device_uvector<vertex_t>>&& dests,
                  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& optional_d_weights,
                  bool remove_multi_edges)
{
  CUGRAPH_EXPECTS(sources.size() == dests.size(),
                  "sources and dests vertex lists must be the same size");

  if (optional_d_weights) {
    CUGRAPH_EXPECTS(sources.size() == optional_d_weights.value().size(),
                    "has_weights is specified, sources and weights must be the same size");

    thrust::for_each_n(
      thrust::host,
      thrust::make_zip_iterator(sources.begin(), dests.begin(), optional_d_weights.value().begin()),
      sources.size(),
      [](auto tuple) {
        CUGRAPH_EXPECTS(cuda::std::get<0>(tuple).size() != cuda::std::get<1>(tuple).size(),
                        "source vertex and dest vertex uvectors must be same size");
        CUGRAPH_EXPECTS(cuda::std::get<0>(tuple).size() != cuda::std::get<2>(tuple).size(),
                        "source vertex and weights uvectors must be same size");
      });
  } else {
    thrust::for_each_n(thrust::host,
                       thrust::make_zip_iterator(sources.begin(), dests.begin()),
                       sources.size(),
                       [](auto tuple) {
                         CUGRAPH_EXPECTS(
                           cuda::std::get<0>(tuple).size() == cuda::std::get<1>(tuple).size(),
                           "source vertex and dest vertex uvectors must be same size");
                       });
  }

  std::vector<rmm::device_uvector<weight_t>> d_weights;

  rmm::device_uvector<vertex_t> srcs_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts_v(0, handle.get_stream());
  rmm::device_uvector<weight_t> weights_v(0, handle.get_stream());

  srcs_v = detail::append_all<vertex_t>(handle, std::move(sources));
  dsts_v = detail::append_all<vertex_t>(handle, std::move(dests));

  if (optional_d_weights) {
    weights_v = detail::append_all(handle, std::move(optional_d_weights.value()));
  }

  if (remove_multi_edges) {
    size_t number_of_edges{srcs_v.size()};

    if (optional_d_weights) {
      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(srcs_v.begin(), dsts_v.begin(), weights_v.begin()),
                   thrust::make_zip_iterator(srcs_v.end(), dsts_v.end(), weights_v.end()));

      auto pair_first = thrust::make_zip_iterator(srcs_v.begin(), dsts_v.begin());
      auto end_iter   = thrust::unique_by_key(
        handle.get_thrust_policy(), pair_first, pair_first + srcs_v.size(), weights_v.begin());

      number_of_edges = cuda::std::distance(pair_first, cuda::std::get<0>(end_iter));
    } else {
      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(srcs_v.begin(), dsts_v.begin()),
                   thrust::make_zip_iterator(srcs_v.end(), dsts_v.end()));

      auto pair_first = thrust::make_zip_iterator(srcs_v.begin(), dsts_v.begin());

      auto end_iter = thrust::unique(handle.get_thrust_policy(),
                                     thrust::make_zip_iterator(srcs_v.begin(), dsts_v.begin()),
                                     thrust::make_zip_iterator(srcs_v.end(), dsts_v.end()));

      number_of_edges = cuda::std::distance(pair_first, end_iter);
    }

    srcs_v.resize(number_of_edges, handle.get_stream());
    srcs_v.shrink_to_fit(handle.get_stream());
    dsts_v.resize(number_of_edges, handle.get_stream());
    dsts_v.shrink_to_fit(handle.get_stream());

    if (optional_d_weights) {
      weights_v.resize(number_of_edges, handle.get_stream());
      weights_v.shrink_to_fit(handle.get_stream());
    }
  }

  return std::make_tuple(
    std::move(srcs_v),
    std::move(dsts_v),
    optional_d_weights
      ? std::move(std::optional<rmm::device_uvector<weight_t>>(std::move(weights_v)))
      : std::nullopt);
}

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
symmetrize_edgelist_from_triangular(raft::handle_t const& handle,
                                    rmm::device_uvector<vertex_t>&& d_src_v,
                                    rmm::device_uvector<vertex_t>&& d_dst_v,
                                    std::optional<rmm::device_uvector<weight_t>>&& d_weight_v,
                                    bool check_diagonal,
                                    std::optional<large_buffer_type_t> large_buffer_type)
{
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  std::optional<size_t> num_diagonals{std::nullopt};
  if (check_diagonal) {
    auto edge_first = thrust::make_zip_iterator(d_src_v.begin(), d_dst_v.begin());
    num_diagonals   = thrust::count_if(handle.get_thrust_policy(),
                                     edge_first,
                                     edge_first + d_src_v.size(),
                                     cuda::proclaim_return_type<bool>([] __device__(auto e) {
                                       return cuda::std::get<0>(e) == cuda::std::get<1>(e);
                                     }));
  }

  auto old_size = d_src_v.size();
  auto new_size = old_size * size_t{2} - (num_diagonals ? *num_diagonals : size_t{0});
  auto new_srcs = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<vertex_t>(
                                        new_size, handle.get_stream())
                                    : rmm::device_uvector<vertex_t>(new_size, handle.get_stream());
  thrust::copy(handle.get_thrust_policy(), d_src_v.begin(), d_src_v.end(), new_srcs.begin());
  d_src_v.resize(0, handle.get_stream());
  d_src_v.shrink_to_fit(handle.get_stream());
  auto new_dsts = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<vertex_t>(
                                        new_size, handle.get_stream())
                                    : rmm::device_uvector<vertex_t>(new_size, handle.get_stream());
  thrust::copy(handle.get_thrust_policy(), d_dst_v.begin(), d_dst_v.end(), new_dsts.begin());
  d_dst_v.resize(0, handle.get_stream());
  d_dst_v.shrink_to_fit(handle.get_stream());
  auto new_weights =
    d_weight_v
      ? std::make_optional(
          large_buffer_type
            ? large_buffer_manager::allocate_memory_buffer<weight_t>(new_size, handle.get_stream())
            : rmm::device_uvector<weight_t>(new_size, handle.get_stream()))
      : std::nullopt;
  if (new_weights) {
    thrust::copy(
      handle.get_thrust_policy(), d_weight_v->begin(), d_weight_v->end(), new_weights->begin());
    d_weight_v = std::nullopt;
  }

  if (new_weights) {
    auto edge_first =
      thrust::make_zip_iterator(new_srcs.begin(), new_dsts.begin(), new_weights->begin());
    if (check_diagonal) {
      thrust::copy_if(
        handle.get_thrust_policy(),
        edge_first,
        edge_first + old_size,
        thrust::make_zip_iterator(new_dsts.begin(), new_srcs.begin(), new_weights->begin()) +
          old_size,
        cuda::proclaim_return_type<bool>(
          [] __device__(auto e) { return cuda::std::get<0>(e) != cuda::std::get<1>(e); }));
    } else {
      thrust::copy(
        handle.get_thrust_policy(),
        edge_first,
        edge_first + old_size,
        thrust::make_zip_iterator(new_dsts.begin(), new_srcs.begin(), new_weights->begin()) +
          old_size);
    }
  } else {
    auto edge_first = thrust::make_zip_iterator(new_srcs.begin(), new_dsts.begin());
    if (check_diagonal) {
      thrust::copy_if(handle.get_thrust_policy(),
                      edge_first,
                      edge_first + old_size,
                      thrust::make_zip_iterator(new_dsts.begin(), new_srcs.begin()) + old_size,
                      cuda::proclaim_return_type<bool>([] __device__(auto e) {
                        return cuda::std::get<0>(e) != cuda::std::get<1>(e);
                      }));
    } else {
      thrust::copy(handle.get_thrust_policy(),
                   edge_first,
                   edge_first + old_size,
                   thrust::make_zip_iterator(new_dsts.begin(), new_srcs.begin()) + old_size);
    }
  }

  return std::make_tuple(std::move(new_srcs), std::move(new_dsts), std::move(new_weights));
}

}  // namespace cugraph
