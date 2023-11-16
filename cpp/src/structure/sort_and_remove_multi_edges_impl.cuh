/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <optional>

namespace cugraph {

namespace detail {

template <typename vertex_t>
void sort_and_remove_multi_edges(raft::handle_t const& handle,
                                 rmm::device_uvector<vertex_t>& edgelist_srcs /* [INOUT] */,
                                 rmm::device_uvector<vertex_t>& edgelist_dsts /* [INOUT] */)
{
  auto edge_first =
    thrust::make_zip_iterator(thrust::make_tuple(edgelist_srcs.begin(), edgelist_dsts.begin()));
  thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());
  edgelist_srcs.resize(
    thrust::distance(edge_first,
                     thrust::unique(handle.get_thrust_policy(),
                                    edge_first,
                                    edge_first + edgelist_srcs.size(),
                                    [] __device__(auto lhs, auto rhs) {
                                      return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) &&
                                             (thrust::get<1>(lhs) == thrust::get<1>(rhs));
                                    })),
    handle.get_stream());
  edgelist_dsts.resize(edgelist_srcs.size(), handle.get_stream());
}

template <typename vertex_t, typename A>
void sort_and_remove_multi_edges(raft::handle_t const& handle,
                                 rmm::device_uvector<vertex_t>& edgelist_srcs /* [INOUT] */,
                                 rmm::device_uvector<vertex_t>& edgelist_dsts /* [INOUT] */,
                                 rmm::device_uvector<A>& edgelist_a /* [INOUT] */)
{
  auto edge_first = thrust::make_zip_iterator(
    thrust::make_tuple(edgelist_srcs.begin(), edgelist_dsts.begin(), edgelist_a.begin()));
  thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());
  edgelist_srcs.resize(
    thrust::distance(edge_first,
                     thrust::unique(handle.get_thrust_policy(),
                                    edge_first,
                                    edge_first + edgelist_srcs.size(),
                                    [] __device__(auto lhs, auto rhs) {
                                      return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) &&
                                             (thrust::get<1>(lhs) == thrust::get<1>(rhs));
                                    })),
    handle.get_stream());
  edgelist_dsts.resize(edgelist_srcs.size(), handle.get_stream());
  edgelist_a.resize(edgelist_srcs.size(), handle.get_stream());
}

template <typename vertex_t, typename A, typename B>
void sort_and_remove_multi_edges(raft::handle_t const& handle,
                                 rmm::device_uvector<vertex_t>& edgelist_srcs /* [INOUT] */,
                                 rmm::device_uvector<vertex_t>& edgelist_dsts /* [INOUT] */,
                                 rmm::device_uvector<A>& edgelist_a /* [INOUT] */,
                                 rmm::device_uvector<B>& edgelist_b /* [INOUT] */)
{
  auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
    edgelist_srcs.begin(), edgelist_dsts.begin(), edgelist_a.begin(), edgelist_b.begin()));
  thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());
  edgelist_srcs.resize(
    thrust::distance(edge_first,
                     thrust::unique(handle.get_thrust_policy(),
                                    edge_first,
                                    edge_first + edgelist_srcs.size(),
                                    [] __device__(auto lhs, auto rhs) {
                                      return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) &&
                                             (thrust::get<1>(lhs) == thrust::get<1>(rhs));
                                    })),
    handle.get_stream());
  edgelist_dsts.resize(edgelist_srcs.size(), handle.get_stream());
  edgelist_a.resize(edgelist_srcs.size(), handle.get_stream());
  edgelist_b.resize(edgelist_srcs.size(), handle.get_stream());
}

template <typename vertex_t, typename A, typename B, typename C>
void sort_and_remove_multi_edges(raft::handle_t const& handle,
                                 rmm::device_uvector<vertex_t>& edgelist_srcs /* [INOUT] */,
                                 rmm::device_uvector<vertex_t>& edgelist_dsts /* [INOUT] */,
                                 rmm::device_uvector<A>& edgelist_a /* [INOUT] */,
                                 rmm::device_uvector<B>& edgelist_b /* [INOUT] */,
                                 rmm::device_uvector<C>& edgelist_c /* [INOUT] */)
{
  auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(edgelist_srcs.begin(),
                                                                 edgelist_dsts.begin(),
                                                                 edgelist_a.begin(),
                                                                 edgelist_b.begin(),
                                                                 edgelist_c.begin()));
  thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());
  edgelist_srcs.resize(
    thrust::distance(edge_first,
                     thrust::unique(handle.get_thrust_policy(),
                                    edge_first,
                                    edge_first + edgelist_srcs.size(),
                                    [] __device__(auto lhs, auto rhs) {
                                      return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) &&
                                             (thrust::get<1>(lhs) == thrust::get<1>(rhs));
                                    })),
    handle.get_stream());
  edgelist_dsts.resize(edgelist_srcs.size(), handle.get_stream());
  edgelist_a.resize(edgelist_srcs.size(), handle.get_stream());
  edgelist_b.resize(edgelist_srcs.size(), handle.get_stream());
  edgelist_c.resize(edgelist_srcs.size(), handle.get_stream());
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>>
sort_and_remove_multi_edges(raft::handle_t const& handle,
                            rmm::device_uvector<vertex_t>&& edgelist_srcs,
                            rmm::device_uvector<vertex_t>&& edgelist_dsts,
                            std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                            std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
                            std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types)
{
  if (edgelist_weights) {
    if (edgelist_edge_ids) {
      if (edgelist_edge_types) {
        detail::sort_and_remove_multi_edges(handle,
                                            edgelist_srcs,
                                            edgelist_dsts,
                                            *edgelist_weights,
                                            *edgelist_edge_ids,
                                            *edgelist_edge_types);
      } else {
        detail::sort_and_remove_multi_edges(
          handle, edgelist_srcs, edgelist_dsts, *edgelist_weights, *edgelist_edge_ids);
      }
    } else {
      if (edgelist_edge_types) {
        detail::sort_and_remove_multi_edges(
          handle, edgelist_srcs, edgelist_dsts, *edgelist_weights, *edgelist_edge_types);
      } else {
        detail::sort_and_remove_multi_edges(
          handle, edgelist_srcs, edgelist_dsts, *edgelist_weights);
      }
    }
  } else {
    if (edgelist_edge_ids) {
      if (edgelist_edge_types) {
        detail::sort_and_remove_multi_edges(
          handle, edgelist_srcs, edgelist_dsts, *edgelist_edge_ids, *edgelist_edge_types);
      } else {
        detail::sort_and_remove_multi_edges(
          handle, edgelist_srcs, edgelist_dsts, *edgelist_edge_ids);
      }
    } else {
      if (edgelist_edge_types) {
        detail::sort_and_remove_multi_edges(
          handle, edgelist_srcs, edgelist_dsts, *edgelist_edge_types);
      } else {
        detail::sort_and_remove_multi_edges(handle, edgelist_srcs, edgelist_dsts);
      }
    }
  }

  return std::make_tuple(std::move(edgelist_srcs),
                         std::move(edgelist_dsts),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types));
}

}  // namespace cugraph
