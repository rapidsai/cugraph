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

#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>

#include <cstddef>

namespace cugraph {

namespace detail {

template <typename GraphViewType, typename T, typename EdgePropertyOutputWrapper>
void fill_edge_property(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        T input,
                        EdgePropertyOutputWrapper edge_property_output)
{
  static_assert(std::is_same_v<T, typename EdgePropertyOutputWrapper::value_type>);

  using edge_t = typename GraphViewType::edge_type;

  auto edge_mask_view = graph_view.edge_mask_view();

  auto value_firsts = edge_property_output.value_firsts();
  auto edge_counts  = edge_property_output.edge_counts();
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition_e_mask =
      edge_mask_view
        ? thrust::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : thrust::nullopt;

    if constexpr (cugraph::has_packed_bool_element<
                    std::remove_reference_t<decltype(value_firsts[i])>,
                    T>()) {
      static_assert(std::is_arithmetic_v<T>, "unimplemented for thrust::tuple types.");
      auto packed_input = input ? packed_bool_full_mask() : packed_bool_empty_mask();
      auto rem          = edge_counts[i] % packed_bools_per_word();
      if (edge_partition_e_mask) {
        auto input_first =
          thrust::make_zip_iterator(value_firsts[i], (*edge_partition_e_mask).value_first());
        thrust::transform(handle.get_thrust_policy(),
                          input_first,
                          input_first + packed_bool_size(static_cast<size_t>(edge_counts[i] - rem)),
                          value_firsts[i],
                          [packed_input] __device__(thrust::tuple<T, uint32_t> pair) {
                            auto old_value = thrust::get<0>(pair);
                            auto mask      = thrust::get<1>(pair);
                            return (old_value & ~mask) | (packed_input & mask);
                          });
        if (rem > 0) {
          thrust::transform(
            handle.get_thrust_policy(),
            input_first + packed_bool_size(static_cast<size_t>(edge_counts[i] - rem)),
            input_first + packed_bool_size(static_cast<size_t>(edge_counts[i])),
            value_firsts[i] + packed_bool_size(static_cast<size_t>(edge_counts[i] - rem)),
            [packed_input, rem] __device__(thrust::tuple<T, uint32_t> pair) {
              auto old_value = thrust::get<0>(pair);
              auto mask      = thrust::get<1>(pair);
              return ((old_value & ~mask) | (packed_input & mask)) & packed_bool_partial_mask(rem);
            });
        }
      } else {
        thrust::fill_n(handle.get_thrust_policy(),
                       value_firsts[i],
                       packed_bool_size(static_cast<size_t>(edge_counts[i] - rem)),
                       packed_input);
        if (rem > 0) {
          thrust::fill_n(
            handle.get_thrust_policy(),
            value_firsts[i] + packed_bool_size(static_cast<size_t>(edge_counts[i] - rem)),
            1,
            packed_input & packed_bool_partial_mask(rem));
        }
      }
    } else {
      if (edge_partition_e_mask) {
        thrust::transform_if(handle.get_thrust_policy(),
                             thrust::make_constant_iterator(input),
                             thrust::make_constant_iterator(input) + edge_counts[i],
                             thrust::make_counting_iterator(edge_t{0}),
                             value_firsts[i],
                             thrust::identity<T>{},
                             [edge_partition_e_mask = *edge_partition_e_mask] __device__(edge_t i) {
                               return edge_partition_e_mask.get(i);
                             });
      } else {
        thrust::fill_n(
          handle.get_thrust_policy(), value_firsts[i], static_cast<size_t>(edge_counts[i]), input);
      }
    }
  }
}

}  // namespace detail

/**
 * @brief Fill graph edge property values to the input value.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam T Type of the edge property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input Edge property values will be set to @p input.
 * @param edge_property_output edge_property_t class object to store edge property values (for the
 * edges assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename T>
void fill_edge_property(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        T input,
                        edge_property_t<GraphViewType, T>& edge_property_output,
                        bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  detail::fill_edge_property(handle, graph_view, input, edge_property_output.mutable_view());
}

}  // namespace cugraph
