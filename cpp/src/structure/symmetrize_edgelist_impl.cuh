/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cugraph/detail/shuffle_wrappers.hpp>

#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <optional>

namespace cugraph {

namespace {

// compare after flipping major & minor
template <typename vertex_t, typename weight_t>
struct compare_upper_triangular_edges_as_lower_triangular_t {
  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t, weight_t> const& lhs,
                             thrust::tuple<vertex_t, vertex_t, weight_t> const& rhs) const
  {
    return thrust::make_tuple(thrust::get<1>(lhs), thrust::get<0>(lhs), thrust::get<2>(lhs)) <
           thrust::make_tuple(thrust::get<1>(rhs), thrust::get<0>(rhs), thrust::get<2>(rhs));
  }
};

// if in the upper triangular region, flip major & minor before comparison.
// if major & minor coincide (after flip if upper triangular), lower triangular edges are less than
// upper triangular edges
template <typename vertex_t, typename weight_t>
struct compare_lower_and_upper_triangular_edges_t {
  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t, weight_t> const& lhs,
                             thrust::tuple<vertex_t, vertex_t, weight_t> const& rhs) const
  {
    auto lhs_in_lower = thrust::get<0>(lhs) > thrust::get<1>(lhs);
    auto rhs_in_lower = thrust::get<0>(rhs) > thrust::get<1>(rhs);
    return thrust::make_tuple(
             lhs_in_lower ? thrust::get<0>(lhs) : thrust::get<1>(lhs),
             lhs_in_lower ? thrust::get<1>(lhs) : thrust::get<0>(lhs),
             !lhs_in_lower,  // lower triangular edges comes before upper triangular edges
             thrust::get<2>(lhs)) <
           thrust::make_tuple(
             rhs_in_lower ? thrust::get<0>(rhs) : thrust::get<1>(rhs),
             rhs_in_lower ? thrust::get<1>(rhs) : thrust::get<0>(rhs),
             !rhs_in_lower,  // lower triangular edges comes before upper triangular edges
             thrust::get<2>(rhs));
  }
};

template <typename EdgeIterator>
struct symmetrize_op_t {
  bool reciprocal{false};

  __device__ void operator()(
    EdgeIterator edge_first,
    size_t lower_run_length,
    size_t upper_run_length,
    uint8_t* include_first /* size = lower_run_length + upper_run_Length */) const
  {
    using weight_t = typename thrust::
      tuple_element<2, typename thrust::iterator_traits<EdgeIterator>::value_type>::type;

    auto min_run_length = lower_run_length < upper_run_length ? lower_run_length : upper_run_length;
    auto max_run_length = lower_run_length < upper_run_length ? upper_run_length : lower_run_length;
    for (size_t i = 0; i < max_run_length; ++i) {
      if (i < min_run_length) {
        thrust::get<2>(*(edge_first + i)) = (thrust::get<2>(*(edge_first + i)) +
                                             thrust::get<2>(*(edge_first + lower_run_length + i))) /
                                            weight_t{2.0};  // average
        *(include_first + i)                    = true;
        *(include_first + lower_run_length + i) = false;
      } else {
        if (lower_run_length > upper_run_length) {
          *(include_first + i) = !reciprocal;
        } else {
          *(include_first + lower_run_length + i) = !reciprocal;
        }
      }
    }
  }
};

template <typename EdgeIterator>
struct update_edge_weights_and_flags_t {
  EdgeIterator edge_first{};
  uint8_t* include_first{nullptr};  // 0: remove 1: include
  size_t num_edges{0};
  symmetrize_op_t<EdgeIterator> op{};

  __device__ void operator()(size_t i) const
  {
    bool first_in_run{};
    if (i == 0) {
      first_in_run = true;
    } else {
      auto cur       = *(edge_first + i);
      auto prev      = *(edge_first + (i - 1));
      auto cur_pair  = thrust::get<0>(cur) > thrust::get<1>(cur)
                         ? thrust::make_tuple(thrust::get<0>(cur), thrust::get<1>(cur))
                         : thrust::make_tuple(thrust::get<1>(cur), thrust::get<0>(cur));
      auto prev_pair = thrust::get<0>(prev) > thrust::get<1>(prev)
                         ? thrust::make_tuple(thrust::get<0>(prev), thrust::get<1>(prev))
                         : thrust::make_tuple(thrust::get<1>(prev), thrust::get<0>(prev));
      first_in_run   = cur_pair != prev_pair;
    }

    if (first_in_run) {
      auto first = *(edge_first + i);
      size_t lower_run_length{0};
      size_t upper_run_length{0};
      auto pair_first = thrust::get<0>(first) > thrust::get<1>(first)
                          ? thrust::make_tuple(thrust::get<0>(first), thrust::get<1>(first))
                          : thrust::make_tuple(thrust::get<1>(first), thrust::get<0>(first));
      while (i + lower_run_length < num_edges) {
        auto cur = *(edge_first + i + lower_run_length);
        if ((thrust::get<0>(cur) > thrust::get<1>(cur)) &&
            (thrust::make_tuple(thrust::get<0>(cur), thrust::get<1>(cur)) == pair_first)) {
          ++lower_run_length;
        } else {
          break;
        }
      }
      while (i + lower_run_length + upper_run_length < num_edges) {
        auto cur = *(edge_first + i + lower_run_length + upper_run_length);
        if ((thrust::get<0>(cur) < thrust::get<1>(cur)) &&
            (thrust::make_tuple(thrust::get<1>(cur), thrust::get<0>(cur)) == pair_first)) {
          ++upper_run_length;
        } else {
          break;
        }
      }

      op(edge_first + i, lower_run_length, upper_run_length, include_first + i);
    }
  }
};

template <typename vertex_t>
struct to_lower_triangular_t {
  __device__ thrust::tuple<vertex_t, vertex_t> operator()(thrust::tuple<vertex_t, vertex_t> e) const
  {
    return thrust::get<0>(e) > thrust::get<1>(e)
             ? e
             : thrust::make_tuple(thrust::get<1>(e), thrust::get<0>(e));
  }
};

}  // namespace

namespace detail {

// FIXME: This function should be modified to support edge id/type
template <typename vertex_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
symmetrize_edgelist(raft::handle_t const& handle,
                    rmm::device_uvector<vertex_t>&& edgelist_majors,
                    rmm::device_uvector<vertex_t>&& edgelist_minors,
                    std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                    bool reciprocal)
{
  // 1. separate lower triangular, diagonal (self-loop), and upper triangular edges

  size_t num_lower_triangular_edges{0};
  size_t num_diagonal_edges{0};
  if (edgelist_weights) {
    auto edge_first            = thrust::make_zip_iterator(thrust::make_tuple(
      edgelist_majors.begin(), edgelist_minors.begin(), (*edgelist_weights).begin()));
    auto lower_triangular_last = thrust::partition(handle.get_thrust_policy(),
                                                   edge_first,
                                                   edge_first + edgelist_majors.size(),
                                                   [] __device__(auto e) {
                                                     auto major = thrust::get<0>(e);
                                                     auto minor = thrust::get<1>(e);
                                                     return major > minor;
                                                   });
    num_lower_triangular_edges =
      static_cast<size_t>(thrust::distance(edge_first, lower_triangular_last));
    auto diagonal_last = thrust::partition(handle.get_thrust_policy(),
                                           edge_first + num_lower_triangular_edges,
                                           edge_first + edgelist_majors.size(),
                                           [] __device__(auto e) {
                                             auto major = thrust::get<0>(e);
                                             auto minor = thrust::get<1>(e);
                                             return major == minor;
                                           });
    num_diagonal_edges =
      static_cast<size_t>(thrust::distance(lower_triangular_last, diagonal_last));
  } else {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
    auto lower_triangular_last = thrust::partition(handle.get_thrust_policy(),
                                                   edge_first,
                                                   edge_first + edgelist_majors.size(),
                                                   [] __device__(auto e) {
                                                     auto major = thrust::get<0>(e);
                                                     auto minor = thrust::get<1>(e);
                                                     return major > minor;
                                                   });
    num_lower_triangular_edges =
      static_cast<size_t>(thrust::distance(edge_first, lower_triangular_last));
    auto diagonal_last = thrust::partition(handle.get_thrust_policy(),
                                           edge_first + num_lower_triangular_edges,
                                           edge_first + edgelist_majors.size(),
                                           [] __device__(auto e) {
                                             auto major = thrust::get<0>(e);
                                             auto minor = thrust::get<1>(e);
                                             return major == minor;
                                           });
    num_diagonal_edges =
      static_cast<size_t>(thrust::distance(lower_triangular_last, diagonal_last));
  }

  rmm::device_uvector<vertex_t> diagonal_majors(num_diagonal_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> upper_triangular_majors(
    edgelist_majors.size() - num_lower_triangular_edges - num_diagonal_edges, handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               edgelist_majors.begin() + num_lower_triangular_edges,
               edgelist_majors.begin() + num_lower_triangular_edges + num_diagonal_edges,
               diagonal_majors.begin());
  thrust::copy(handle.get_thrust_policy(),
               edgelist_majors.begin() + num_lower_triangular_edges + num_diagonal_edges,
               edgelist_majors.end(),
               upper_triangular_majors.begin());
  edgelist_majors.resize(num_lower_triangular_edges, handle.get_stream());
  edgelist_majors.shrink_to_fit(handle.get_stream());
  auto lower_triangular_majors = std::move(edgelist_majors);

  rmm::device_uvector<vertex_t> upper_triangular_minors(upper_triangular_majors.size(),
                                                        handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               edgelist_minors.begin() + num_lower_triangular_edges + num_diagonal_edges,
               edgelist_minors.end(),
               upper_triangular_minors.begin());
  edgelist_minors.resize(lower_triangular_majors.size(), handle.get_stream());
  edgelist_minors.shrink_to_fit(handle.get_stream());
  auto lower_triangular_minors = std::move(edgelist_minors);

  auto diagonal_weights = edgelist_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                               diagonal_majors.size(), handle.get_stream())
                                           : std::nullopt;
  auto upper_triangular_weights = edgelist_weights
                                    ? std::make_optional<rmm::device_uvector<weight_t>>(
                                        upper_triangular_majors.size(), handle.get_stream())
                                    : std::nullopt;
  if (edgelist_weights) {
    thrust::copy(handle.get_thrust_policy(),
                 (*edgelist_weights).begin() + num_lower_triangular_edges,
                 (*edgelist_weights).begin() + num_lower_triangular_edges + num_diagonal_edges,
                 (*diagonal_weights).begin());
    thrust::copy(handle.get_thrust_policy(),
                 (*edgelist_weights).begin() + num_lower_triangular_edges + num_diagonal_edges,
                 (*edgelist_weights).end(),
                 (*upper_triangular_weights).begin());
    (*edgelist_weights).resize(lower_triangular_majors.size(), handle.get_stream());
    (*edgelist_weights).shrink_to_fit(handle.get_stream());
  }
  auto lower_triangular_weights = std::move(edgelist_weights);

  // 2. shuffle the (to-be-flipped) upper triangular edges

  if constexpr (multi_gpu) {
    std::tie(
      upper_triangular_minors, upper_triangular_majors, upper_triangular_weights, std::ignore) =
      detail::shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                         vertex_t,
                                                                         weight_t,
                                                                         int32_t>(
        handle,
        std::move(upper_triangular_minors),
        std::move(upper_triangular_majors),
        std::move(upper_triangular_weights),
        std::nullopt);
  }

  // 3. merge the lower triangular and the (flipped) upper triangular edges

  rmm::device_uvector<vertex_t> merged_lower_triangular_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> merged_lower_triangular_minors(0, handle.get_stream());
  auto merged_lower_triangular_weights =
    edgelist_weights ? std::make_optional<rmm::device_uvector<weight_t>>(0, handle.get_stream())
                     : std::nullopt;
  if (edgelist_weights) {
    auto lower_triangular_edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(lower_triangular_majors.begin(),
                                                   lower_triangular_minors.begin(),
                                                   (*lower_triangular_weights).begin()));
    thrust::sort(handle.get_thrust_policy(),
                 lower_triangular_edge_first,
                 lower_triangular_edge_first + lower_triangular_majors.size());
    auto upper_triangular_edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      upper_triangular_majors.begin(),
      upper_triangular_minors.begin(),
      (*upper_triangular_weights)
        .begin()));  // do not flip here to use "lower_triangular = major > minor"
    thrust::sort(handle.get_thrust_policy(),
                 upper_triangular_edge_first,
                 upper_triangular_edge_first + upper_triangular_majors.size(),
                 compare_upper_triangular_edges_as_lower_triangular_t<vertex_t, weight_t>{});

    merged_lower_triangular_majors.resize(
      lower_triangular_majors.size() + upper_triangular_majors.size(), handle.get_stream());
    merged_lower_triangular_minors.resize(merged_lower_triangular_majors.size(),
                                          handle.get_stream());
    (*merged_lower_triangular_weights)
      .resize(merged_lower_triangular_majors.size(), handle.get_stream());
    auto merged_first =
      thrust::make_zip_iterator(thrust::make_tuple(merged_lower_triangular_majors.begin(),
                                                   merged_lower_triangular_minors.begin(),
                                                   (*merged_lower_triangular_weights).begin()));
    thrust::merge(handle.get_thrust_policy(),
                  lower_triangular_edge_first,
                  lower_triangular_edge_first + lower_triangular_majors.size(),
                  upper_triangular_edge_first,
                  upper_triangular_edge_first + upper_triangular_majors.size(),
                  merged_first,
                  compare_lower_and_upper_triangular_edges_t<vertex_t, weight_t>{});

    lower_triangular_majors.resize(0, handle.get_stream());
    lower_triangular_majors.shrink_to_fit(handle.get_stream());
    lower_triangular_minors.resize(0, handle.get_stream());
    lower_triangular_minors.shrink_to_fit(handle.get_stream());
    (*lower_triangular_weights).resize(0, handle.get_stream());
    (*lower_triangular_weights).shrink_to_fit(handle.get_stream());

    upper_triangular_majors.resize(0, handle.get_stream());
    upper_triangular_majors.shrink_to_fit(handle.get_stream());
    upper_triangular_minors.resize(0, handle.get_stream());
    upper_triangular_minors.shrink_to_fit(handle.get_stream());
    (*upper_triangular_weights).resize(0, handle.get_stream());
    (*upper_triangular_weights).shrink_to_fit(handle.get_stream());

    rmm::device_uvector<uint8_t> includes(merged_lower_triangular_majors.size(),
                                          handle.get_stream());
    symmetrize_op_t<decltype(merged_first)> symm_op{reciprocal};
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(merged_lower_triangular_majors.size()),
      update_edge_weights_and_flags_t<decltype(merged_first)>{
        merged_first, includes.data(), merged_lower_triangular_majors.size(), symm_op});

    auto merged_edge_and_flag_first =
      thrust::make_zip_iterator(thrust::make_tuple(merged_lower_triangular_majors.begin(),
                                                   merged_lower_triangular_minors.begin(),
                                                   (*merged_lower_triangular_weights).begin(),
                                                   includes.begin()));
    merged_lower_triangular_majors.resize(
      thrust::distance(
        merged_edge_and_flag_first,
        thrust::remove_if(handle.get_thrust_policy(),
                          merged_edge_and_flag_first,
                          merged_edge_and_flag_first + merged_lower_triangular_majors.size(),
                          [] __device__(auto t) { return !thrust::get<3>(t); })),
      handle.get_stream());
    merged_lower_triangular_majors.shrink_to_fit(handle.get_stream());
    merged_lower_triangular_minors.resize(merged_lower_triangular_majors.size(),
                                          handle.get_stream());
    merged_lower_triangular_minors.shrink_to_fit(handle.get_stream());
    (*merged_lower_triangular_weights)
      .resize(merged_lower_triangular_majors.size(), handle.get_stream());
    (*merged_lower_triangular_weights).shrink_to_fit(handle.get_stream());

    auto merged_major_minor_range_first = thrust::make_zip_iterator(thrust::make_tuple(
      merged_lower_triangular_majors.begin(), merged_lower_triangular_minors.begin()));
    thrust::transform(handle.get_thrust_policy(),
                      merged_major_minor_range_first,
                      merged_major_minor_range_first + merged_lower_triangular_majors.size(),
                      merged_major_minor_range_first,
                      to_lower_triangular_t<vertex_t>{});
  } else {
    auto lower_triangular_edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(lower_triangular_majors.begin(), lower_triangular_minors.begin()));
    thrust::sort(handle.get_thrust_policy(),
                 lower_triangular_edge_first,
                 lower_triangular_edge_first + lower_triangular_majors.size());
    auto upper_triangular_edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      upper_triangular_minors.begin(), upper_triangular_majors.begin()));  // flip
    thrust::sort(handle.get_thrust_policy(),
                 upper_triangular_edge_first,
                 upper_triangular_edge_first + upper_triangular_majors.size());

    merged_lower_triangular_majors.resize(
      reciprocal ? std::min(num_lower_triangular_edges, upper_triangular_majors.size())
                 : num_lower_triangular_edges + upper_triangular_majors.size(),
      handle.get_stream());
    merged_lower_triangular_minors.resize(merged_lower_triangular_majors.size(),
                                          handle.get_stream());
    auto merged_first = thrust::make_zip_iterator(thrust::make_tuple(
      merged_lower_triangular_majors.begin(), merged_lower_triangular_minors.begin()));
    auto merged_last =
      reciprocal
        ? thrust::set_intersection(handle.get_thrust_policy(),
                                   lower_triangular_edge_first,
                                   lower_triangular_edge_first + lower_triangular_majors.size(),
                                   upper_triangular_edge_first,
                                   upper_triangular_edge_first + upper_triangular_majors.size(),
                                   merged_first)
        : thrust::set_union(handle.get_thrust_policy(),
                            lower_triangular_edge_first,
                            lower_triangular_edge_first + lower_triangular_majors.size(),
                            upper_triangular_edge_first,
                            upper_triangular_edge_first + upper_triangular_majors.size(),
                            merged_first);

    lower_triangular_majors.resize(0, handle.get_stream());
    lower_triangular_majors.shrink_to_fit(handle.get_stream());
    lower_triangular_minors.resize(0, handle.get_stream());
    lower_triangular_minors.shrink_to_fit(handle.get_stream());

    upper_triangular_majors.resize(0, handle.get_stream());
    upper_triangular_majors.shrink_to_fit(handle.get_stream());
    upper_triangular_minors.resize(0, handle.get_stream());
    upper_triangular_minors.shrink_to_fit(handle.get_stream());

    merged_lower_triangular_majors.resize(thrust::distance(merged_first, merged_last),
                                          handle.get_stream());
    merged_lower_triangular_majors.shrink_to_fit(handle.get_stream());
    merged_lower_triangular_minors.resize(merged_lower_triangular_majors.size(),
                                          handle.get_stream());
    merged_lower_triangular_minors.shrink_to_fit(handle.get_stream());
  }

  // 4. symmetrize from the merged lower triangular edges & diagonal edges

  upper_triangular_majors.resize(merged_lower_triangular_majors.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               merged_lower_triangular_minors.begin(),
               merged_lower_triangular_minors.end(),
               upper_triangular_majors.begin());
  upper_triangular_minors.resize(upper_triangular_majors.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               merged_lower_triangular_majors.begin(),
               merged_lower_triangular_majors.end(),
               upper_triangular_minors.begin());
  if (edgelist_weights) {
    (*upper_triangular_weights).resize(upper_triangular_majors.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 (*merged_lower_triangular_weights).begin(),
                 (*merged_lower_triangular_weights).end(),
                 (*upper_triangular_weights).begin());
  }

  if constexpr (multi_gpu) {
    std::tie(
      upper_triangular_majors, upper_triangular_minors, upper_triangular_weights, std::ignore) =
      detail::shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                         vertex_t,
                                                                         weight_t,
                                                                         int32_t>(
        handle,
        std::move(upper_triangular_majors),
        std::move(upper_triangular_minors),
        std::move(upper_triangular_weights),
        std::nullopt);
  }

  edgelist_majors  = std::move(merged_lower_triangular_majors);
  edgelist_minors  = std::move(merged_lower_triangular_minors);
  edgelist_weights = std::move(merged_lower_triangular_weights);

  edgelist_majors.resize(
    edgelist_majors.size() + diagonal_majors.size() + upper_triangular_majors.size(),
    handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               diagonal_majors.begin(),
               diagonal_majors.end(),
               edgelist_majors.end() - diagonal_majors.size() - upper_triangular_majors.size());
  thrust::copy(handle.get_thrust_policy(),
               upper_triangular_majors.begin(),
               upper_triangular_majors.end(),
               edgelist_majors.end() - upper_triangular_majors.size());
  upper_triangular_majors.resize(0, handle.get_stream());
  upper_triangular_majors.shrink_to_fit(handle.get_stream());

  edgelist_minors.resize(edgelist_majors.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               diagonal_majors.begin(),
               diagonal_majors.end(),
               edgelist_minors.end() - diagonal_majors.size() -
                 upper_triangular_minors.size());  // minors == majors if diagonal
  thrust::copy(handle.get_thrust_policy(),
               upper_triangular_minors.begin(),
               upper_triangular_minors.end(),
               edgelist_minors.end() - upper_triangular_minors.size());
  diagonal_majors.resize(0, handle.get_stream());
  diagonal_majors.shrink_to_fit(handle.get_stream());
  upper_triangular_minors.resize(0, handle.get_stream());
  upper_triangular_minors.shrink_to_fit(handle.get_stream());

  if (edgelist_weights) {
    (*edgelist_weights).resize(edgelist_majors.size(), handle.get_stream());
    thrust::copy(
      handle.get_thrust_policy(),
      (*diagonal_weights).begin(),
      (*diagonal_weights).end(),
      (*edgelist_weights).end() - (*diagonal_weights).size() - (*upper_triangular_weights).size());
    thrust::copy(handle.get_thrust_policy(),
                 (*upper_triangular_weights).begin(),
                 (*upper_triangular_weights).end(),
                 (*edgelist_weights).end() - (*upper_triangular_weights).size());
    (*diagonal_weights).resize(0, handle.get_stream());
    (*diagonal_weights).shrink_to_fit(handle.get_stream());
    (*upper_triangular_weights).resize(0, handle.get_stream());
    (*upper_triangular_weights).shrink_to_fit(handle.get_stream());
  }

  return std::make_tuple(
    std::move(edgelist_majors), std::move(edgelist_minors), std::move(edgelist_weights));
}

}  // namespace detail

template <typename vertex_t, typename weight_t, bool store_transposed, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
symmetrize_edgelist(raft::handle_t const& handle,
                    rmm::device_uvector<vertex_t>&& edgelist_srcs,
                    rmm::device_uvector<vertex_t>&& edgelist_dsts,
                    std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                    bool reciprocal)
{
  rmm::device_uvector<vertex_t> edgelist_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minors(0, handle.get_stream());
  std::tie(edgelist_majors, edgelist_minors, edgelist_weights) =
    detail::symmetrize_edgelist<vertex_t, weight_t, multi_gpu>(
      handle,
      store_transposed ? std::move(edgelist_dsts) : std::move(edgelist_srcs),
      store_transposed ? std::move(edgelist_srcs) : std::move(edgelist_dsts),
      std::move(edgelist_weights),
      reciprocal);

  return std::make_tuple(store_transposed ? std::move(edgelist_minors) : std::move(edgelist_majors),
                         store_transposed ? std::move(edgelist_majors) : std::move(edgelist_minors),
                         std::move(edgelist_weights));
}

}  // namespace cugraph
