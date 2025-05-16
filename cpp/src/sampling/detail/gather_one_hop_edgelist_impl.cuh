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

#pragma once

#include "common_utilities.cuh"
#include "prims/extract_transform_if_v_frontier_outgoing_e.cuh"
#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/kv_store.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/vertex_frontier.cuh"
#include "structure/detail/structure_utils.cuh"
#include "utilities/collect_comm.cuh"
#include "utilities/tuple_with_optionals_dispatching.hpp"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <cuda.h>

#include <cstddef>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_properties_t, typename label_t>
struct format_gather_edges_return_t {
  using edge_properties_tup_type =
    std::conditional_t<std::is_same_v<edge_properties_t, cuda::std::nullopt_t>,
                       thrust::tuple<>,
                       std::conditional_t<std::is_arithmetic_v<edge_properties_t>,
                                          thrust::tuple<edge_properties_t>,
                                          edge_properties_t>>;

  using return_type =
    std::conditional_t<std::is_same_v<label_t, cuda::std::nullopt_t>,
                       decltype(cugraph::thrust_tuple_cat(thrust::tuple<vertex_t, vertex_t>{},
                                                          edge_properties_tup_type{})),
                       decltype(cugraph::thrust_tuple_cat(thrust::tuple<vertex_t, vertex_t>{},
                                                          edge_properties_tup_type{},
                                                          thrust::tuple<label_t>{}))>;

  return_type __device__ format_result(vertex_t src,
                                       vertex_t dst,
                                       edge_properties_t edge_properties,
                                       label_t label) const
  {
    edge_properties_tup_type edge_properties_tup{};

    if constexpr (!std::is_same_v<edge_properties_t, cuda::std::nullopt_t>) {
      if constexpr (std::is_arithmetic_v<edge_properties_t>) {
        thrust::get<0>(edge_properties_tup) = edge_properties;
      } else {
        edge_properties_tup = edge_properties;
      }
    }

    std::conditional_t<std::is_same_v<label_t, cuda::std::nullopt_t>,
                       thrust::tuple<>,
                       thrust::tuple<label_t>>
      label_tup{};
    if constexpr (!std::is_same_v<label_t, cuda::std::nullopt_t>) {
      thrust::get<0>(label_tup) = label;
    }
    return thrust_tuple_cat(thrust::make_tuple(src, dst), edge_properties_tup, label_tup);
  }
};

template <typename key_t, typename vertex_t, typename edge_properties_t, typename label_t>
struct return_edges_with_properties_e_op
  : public format_gather_edges_return_t<vertex_t, edge_properties_t, label_t> {
  typename format_gather_edges_return_t<vertex_t, edge_properties_t, label_t>::return_type
    __device__
    operator()(key_t optionally_tagged_src,
               vertex_t dst,
               cuda::std::nullopt_t,
               cuda::std::nullopt_t,
               edge_properties_t edge_properties) const
  {
    static_assert(std::is_same_v<key_t, vertex_t> ||
                  std::is_same_v<key_t, thrust::tuple<vertex_t, int32_t>>);

    if constexpr (std::is_same_v<key_t, vertex_t>) {
      return format_result(optionally_tagged_src, dst, edge_properties, label_t{});
    } else {
      return format_result(thrust::get<0>(optionally_tagged_src),
                           dst,
                           edge_properties,
                           thrust::get<1>(optionally_tagged_src));
    }
  }
};

template <typename vertex_t, typename edge_time_t, typename edge_properties_t, typename label_t>
struct return_temporal_edges_with_properties_e_op
  : public format_gather_edges_return_t<vertex_t, edge_properties_t, label_t> {
  typename format_gather_edges_return_t<vertex_t, edge_properties_t, label_t>::return_type
    __device__
    operator()(thrust::tuple<vertex_t, edge_time_t> tagged_src,
               vertex_t dst,
               cuda::std::nullopt_t,
               cuda::std::nullopt_t,
               edge_properties_t edge_properties) const
  {
    return format_result(thrust::get<0>(tagged_src), dst, edge_properties, label_t{});
  }
};

template <typename vertex_t, typename edge_time_t, typename edge_properties_t, typename label_t>
struct return_indirect_edges_with_properties_e_op
  : public format_gather_edges_return_t<vertex_t, edge_properties_t, label_t> {
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>>
    kv_store_view_;

  return_indirect_edges_with_properties_e_op(
    kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
      size_t const*,
      thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>> kv_store_view)
    : format_gather_edges_return_t<vertex_t, edge_properties_t, label_t>(),
      kv_store_view_(kv_store_view)
  {
  }

  typename format_gather_edges_return_t<vertex_t, edge_properties_t, label_t>::return_type
    __device__
    operator()(thrust::tuple<vertex_t, size_t> tagged_src,
               vertex_t dst,
               cuda::std::nullopt_t,
               cuda::std::nullopt_t,
               edge_properties_t edge_properties) const
  {
    auto tuple = kv_store_view_.find(thrust::get<1>(tagged_src));

    label_t src_label{thrust::get<1>(tuple)};

    return format_result(thrust::get<0>(tagged_src), dst, edge_properties, src_label);
  }
};

struct type_filtered_edges_with_properties_pred_op {
  raft::device_span<uint8_t const> gather_flags_{nullptr, size_t{0}};

  template <typename key_t, typename vertex_t, typename edge_properties_t>
  bool __device__ operator()(key_t tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             edge_properties_t edge_properties) const
  {
    using edge_type_t = int32_t;

    if constexpr (std::is_arithmetic_v<edge_properties_t>) {
      return (gather_flags_[edge_properties] == static_cast<uint8_t>(true));
    } else {
      static_assert(thrust::tuple_size<edge_properties_t>::value > 1);
      static_assert(std::is_same_v<edge_type_t, std::tuple_element_t<0, edge_properties_t>>);

      return (gather_flags_[thrust::get<0>(edge_properties)] == static_cast<uint8_t>(true));
    }
  }
};

struct simple_time_filtered_edges_with_properties_pred_op {
  cuda::std::optional<raft::device_span<uint8_t const>> optional_gather_flags_{cuda::std::nullopt};

  template <typename vertex_t, typename edge_time_t, typename edge_properties_t>
  bool __device__ operator()(thrust::tuple<vertex_t, edge_time_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             edge_properties_t edge_properties) const
  {
    static_assert(std::is_arithmetic<edge_properties_t>::value ||
                  cugraph::is_thrust_tuple_of_arithmetic<edge_properties_t>::value);

    vertex_t src{thrust::get<0>(tagged_src)};
    edge_time_t src_time{thrust::get<1>(tagged_src)};
    edge_time_t edge_time{};

    if constexpr (std::is_arithmetic_v<edge_properties_t>) {
      edge_time = edge_properties;
    } else {
      edge_time = thrust::get<0>(edge_properties);
    }

    if (src_time < edge_time) {
      if (optional_gather_flags_) {
        if constexpr (cugraph::is_thrust_tuple_of_arithmetic<edge_properties_t>::value) {
          if constexpr (thrust::tuple_size<edge_properties_t>::value > 1) {
            if constexpr (std::is_integral_v<thrust::tuple_element<1, edge_properties_t>>) {
              return ((*optional_gather_flags_)[thrust::get<1>(edge_properties)] ==
                      static_cast<uint8_t>(true));
            }
          }
        }
      } else {
        return true;
      }
    }

    return false;
  }
};

template <typename vertex_t,
          typename edge_time_t,
          typename edge_type_t,
          typename edge_properties_t,
          typename label_t>
struct label_time_filtered_edges_with_properties_pred_op {
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>>
    kv_store_view_;

  cuda::std::optional<raft::device_span<uint8_t const>> optional_gather_flags_{std::nullopt};

  bool __device__ operator()(thrust::tuple<vertex_t, size_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             edge_properties_t edge_properties) const
  {
    if constexpr (cugraph::is_thrust_tuple_of_arithmetic<edge_properties_t>::value) {
      auto edge_time = thrust::get<0>(edge_properties);

      if constexpr (std::is_integral_v<decltype(edge_time)>) {
        vertex_t src{thrust::get<0>(tagged_src)};
        size_t label_time_id{thrust::get<1>(tagged_src)};

        auto tuple = kv_store_view_.find(label_time_id);

        edge_time_t src_time{thrust::get<0>(tuple)};
        label_t src_label{thrust::get<1>(tuple)};

        if (src_time < edge_time) {
          if (optional_gather_flags_) {
            auto edge_type = thrust::get<1>(edge_properties);
            if constexpr (std::is_integral_v<decltype(edge_type)>) {
              return ((*optional_gather_flags_)[thrust::get<1>(edge_properties)] ==
                      static_cast<uint8_t>(true));
            }
          } else {
            return true;
          }
        }
      }
    }
    return false;
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename tag_t,
          typename label_t,
          bool multi_gpu>
struct gather_one_hop_edgelist_functor_t {
  raft::handle_t const& handle;
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view;
  key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list;
  std::optional<raft::host_span<uint8_t const>> gather_flags;
  bool do_expensive_check{false};

  template <bool... Flags, typename TupleType>
  auto operator()(TupleType edge_properties)
  {
    using key_t =
      std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;

    auto return_result =
      std::make_tuple(rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      std::optional<rmm::device_uvector<edge_type_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<weight_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<label_t>>{std::nullopt});

    auto edge_value_view           = concatenate_views(edge_properties);
    using edge_property_elements_t = typename decltype(edge_value_view)::value_type;

    if (gather_flags) {
      if constexpr (std::tuple_size_v<TupleType> > 0) {
        if constexpr (std::is_same_v<std::tuple_element_t<0, TupleType>,
                                     edge_property_view_t<edge_t, edge_type_t const*>>) {
          rmm::device_uvector<uint8_t> d_gather_flags(gather_flags->size(), handle.get_stream());
          raft::update_device(
            d_gather_flags.data(), gather_flags->data(), gather_flags->size(), handle.get_stream());

          auto output_buffer = cugraph::extract_transform_if_v_frontier_outgoing_e(
            handle,
            graph_view,
            key_list,
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            edge_value_view,
            return_edges_with_properties_e_op<key_t, vertex_t, edge_property_elements_t, label_t>{},
            type_filtered_edges_with_properties_pred_op{
              raft::device_span<uint8_t const>{d_gather_flags.data(), d_gather_flags.size()}},
            do_expensive_check);

          move_results<0, 0, true, true, Flags..., !std::is_same_v<tag_t, void>>(output_buffer,
                                                                                 return_result);
        } else {
          CUGRAPH_FAIL("If gather_flags specified edge_type must be first property");
        }
      } else {
        CUGRAPH_FAIL("If gather_flags specified edge_type must be first property");
      }
    } else {
      auto output_buffer = cugraph::extract_transform_v_frontier_outgoing_e(
        handle,
        graph_view,
        key_list,
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_value_view,
        return_edges_with_properties_e_op<key_t, vertex_t, edge_property_elements_t, label_t>{},
        do_expensive_check);

      move_results<0, 0, true, true, Flags..., !std::is_same_v<tag_t, void>>(output_buffer,
                                                                             return_result);
    }

    return return_result;
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename label_t,
          bool multi_gpu>
struct temporal_simple_gather_one_hop_edgelist_functor_t {
  raft::handle_t const& handle;
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view;
  key_bucket_t<vertex_t, edge_time_t, multi_gpu, false> const& key_list;
  std::optional<raft::host_span<uint8_t const>> gather_flags;
  bool do_expensive_check{false};

  template <bool... Flags, typename TupleType>
  auto operator()(TupleType edge_properties)
  {
    auto return_result =
      std::make_tuple(rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_type_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<weight_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<label_t>>{std::nullopt});

    if constexpr (std::tuple_size_v<TupleType> == 0) {
      CUGRAPH_FAIL("Edge time must be specified");
    } else if constexpr (!std::is_same_v<std::tuple_element_t<0, TupleType>,
                                         edge_property_view_t<edge_t, edge_time_t const*>>) {
      CUGRAPH_FAIL("Edge time must be first");
    } else {
      auto edge_value_view           = concatenate_views(edge_properties);
      using edge_property_elements_t = typename decltype(edge_value_view)::value_type;

      if (gather_flags) {
        if constexpr (1 < std::tuple_size_v<TupleType>) {
          if constexpr (std::is_same_v<std::tuple_element_t<1, TupleType>,
                                       edge_property_view_t<edge_t, edge_type_t const*>>) {
            rmm::device_uvector<uint8_t> d_gather_flags(gather_flags->size(), handle.get_stream());
            raft::update_device(d_gather_flags.data(),
                                gather_flags->data(),
                                gather_flags->size(),
                                handle.get_stream());

            auto output_buffer = cugraph::extract_transform_if_v_frontier_outgoing_e(
              handle,
              graph_view,
              key_list,
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_value_view,
              return_temporal_edges_with_properties_e_op<vertex_t,
                                                         edge_time_t,
                                                         edge_property_elements_t,
                                                         label_t>{},
              simple_time_filtered_edges_with_properties_pred_op{
                raft::device_span<uint8_t const>{d_gather_flags.data(), d_gather_flags.size()}},
              do_expensive_check);

            move_results<0, 0, true, true, Flags..., true>(output_buffer, return_result);
          } else {
            CUGRAPH_FAIL("If gather_flags specified edge_type must be first property");
          }
        } else {
          CUGRAPH_FAIL("If gather_flags specified edge_type must be first property");
        }
      } else {
        auto output_buffer = cugraph::extract_transform_if_v_frontier_outgoing_e(
          handle,
          graph_view,
          key_list,
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_value_view,
          return_temporal_edges_with_properties_e_op<vertex_t,
                                                     edge_time_t,
                                                     edge_property_elements_t,
                                                     label_t>{},
          simple_time_filtered_edges_with_properties_pred_op{},
          do_expensive_check);

        move_results<0, 0, true, true, Flags..., false>(output_buffer, return_result);
      }
    }

    return return_result;
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename label_t,
          bool multi_gpu>
struct temporal_label_gather_one_hop_edgelist_functor_t {
  raft::handle_t const& handle;
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view;
  key_bucket_t<vertex_t, size_t, multi_gpu, false> const& key_list;
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>>
    kv_store_view;

  std::optional<raft::host_span<uint8_t const>> gather_flags;
  bool do_expensive_check{false};

  template <bool... Flags, typename TupleType>
  auto operator()(TupleType edge_properties)
  {
    auto return_result =
      std::make_tuple(rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_type_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<weight_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<label_t>>{std::nullopt});

    auto edge_value_view           = concatenate_views(edge_properties);
    using edge_property_elements_t = typename decltype(edge_value_view)::value_type;

    if (gather_flags) {
      if constexpr (1 < std::tuple_size_v<TupleType>)
        if constexpr (std::is_same_v<std::tuple_element_t<1, TupleType>,
                                     edge_property_view_t<edge_t, edge_type_t const*>>) {
          rmm::device_uvector<uint8_t> d_gather_flags(gather_flags->size(), handle.get_stream());
          raft::update_device(
            d_gather_flags.data(), gather_flags->data(), gather_flags->size(), handle.get_stream());

          auto output_buffer = cugraph::extract_transform_if_v_frontier_outgoing_e(
            handle,
            graph_view,
            key_list,
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            edge_value_view,
            return_indirect_edges_with_properties_e_op<vertex_t,
                                                       edge_time_t,
                                                       edge_property_elements_t,
                                                       label_t>(kv_store_view),
            label_time_filtered_edges_with_properties_pred_op<vertex_t,
                                                              edge_time_t,
                                                              edge_type_t,
                                                              edge_property_elements_t,
                                                              label_t>{
              kv_store_view,
              raft::device_span<uint8_t const>{d_gather_flags.data(), d_gather_flags.size()}},
            do_expensive_check);

          move_results<0, 0, true, true, Flags..., true>(output_buffer, return_result);
        } else {
          CUGRAPH_FAIL("If gather_flags specified edge_type must be second property");
        }
      else {
        CUGRAPH_FAIL("If gather_flags specified edge_type must be second property");
      }
    } else {
      auto output_buffer = cugraph::extract_transform_if_v_frontier_outgoing_e(
        handle,
        graph_view,
        key_list,
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_value_view,
        return_indirect_edges_with_properties_e_op<vertex_t,
                                                   edge_time_t,
                                                   edge_property_elements_t,
                                                   label_t>(kv_store_view),
        label_time_filtered_edges_with_properties_pred_op<vertex_t,
                                                          edge_time_t,
                                                          edge_type_t,
                                                          edge_property_elements_t,
                                                          label_t>{kv_store_view,
                                                                   cuda::std::nullopt},
        do_expensive_check);

      move_results<0, 0, true, true, Flags..., true>(output_buffer, return_result);
    }

    return return_result;
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<label_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_end_time_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<edge_time_t const>> active_major_times,
  std::optional<raft::device_span<label_t const>> active_major_labels,
  std::optional<raft::host_span<uint8_t const>> gather_flags,
  bool do_expensive_check)
{
  assert(!gather_flags || edge_type_view);

  rmm::device_uvector<vertex_t> result_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> result_weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> result_ids{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> result_types{std::nullopt};
  std::optional<rmm::device_uvector<edge_time_t>> result_start_times{std::nullopt};
  std::optional<rmm::device_uvector<edge_time_t>> result_end_times{std::nullopt};
  std::optional<rmm::device_uvector<label_t>> result_labels{std::nullopt};

  if (active_major_labels && active_major_times) {
    using tag_t = size_t;

    // FIXME: Currently can't use multiple attributes in the tag.  Here's a hack
    rmm::device_uvector<size_t> vertex_label_time_ids(active_majors.size(), handle.get_stream());

    size_t starting_id{0};
    if (multi_gpu) {
      auto sizes = cugraph::host_scalar_allgather(
        handle.get_comms(), active_majors.size(), handle.get_stream());
      std::exclusive_scan(sizes.begin(), sizes.end(), sizes.begin(), size_t{0});
      starting_id = sizes[handle.get_comms().get_rank()];
    }

    thrust::sequence(handle.get_thrust_policy(),
                     vertex_label_time_ids.begin(),
                     vertex_label_time_ids.end(),
                     starting_id);

    kv_store_t<size_t, thrust::tuple<edge_time_t, label_t>, true> kv_store{handle.get_stream()};
    if (multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

      auto all_minor_keys =
        device_allgatherv(handle,
                          minor_comm,
                          raft::device_span<size_t const>{vertex_label_time_ids.data(),
                                                          vertex_label_time_ids.size()});
      auto all_minor_times =
        device_allgatherv(handle,
                          minor_comm,
                          raft::device_span<edge_time_t const>{active_major_times->data(),
                                                               active_major_times->size()});
      auto all_minor_labels = device_allgatherv(
        handle,
        minor_comm,
        raft::device_span<label_t const>{active_major_labels->data(), active_major_labels->size()});

      CUGRAPH_EXPECTS(
        thrust::is_sorted(handle.get_thrust_policy(), all_minor_keys.begin(), all_minor_keys.end()),
        "need to SORT!");

      kv_store = kv_store_t<size_t, thrust::tuple<edge_time_t, label_t>, true>(
        all_minor_keys.begin(),
        all_minor_keys.end(),
        thrust::make_zip_iterator(all_minor_times.begin(), all_minor_labels.begin()),
        thrust::make_tuple(edge_time_t{-1}, label_t{-1}),
        true,
        handle.get_stream());

    } else {
      kv_store = kv_store_t<size_t, thrust::tuple<edge_time_t, label_t>, true>(
        vertex_label_time_ids.begin(),
        vertex_label_time_ids.end(),
        thrust::make_zip_iterator(active_major_times->begin(),
                                  active_major_labels->begin()),  // multi_gpu is different
        thrust::make_tuple(edge_time_t{-1}, label_t{-1}),
        true,
        handle.get_stream());
    }

    cugraph::vertex_frontier_t<vertex_t, size_t, multi_gpu, false> vertex_label_frontier(handle, 1);
    vertex_label_frontier.bucket(0).insert(
      thrust::make_zip_iterator(active_majors.begin(), vertex_label_time_ids.begin()),
      thrust::make_zip_iterator(active_majors.end(), vertex_label_time_ids.end()));

    temporal_label_gather_one_hop_edgelist_functor_t<vertex_t,
                                                     edge_t,
                                                     weight_t,
                                                     edge_type_t,
                                                     edge_time_t,
                                                     label_t,
                                                     multi_gpu>
      gather_functor{
        handle,
        graph_view,
        vertex_label_frontier.bucket(0),
        kv_binary_search_store_device_view_t<decltype(kv_store.view())>{kv_store.view()},
        gather_flags,
        do_expensive_check};

    std::tie(result_srcs,
             result_dsts,
             result_start_times,
             result_types,
             result_weights,
             result_ids,
             result_end_times,
             result_labels) = tuple_with_optionals_dispatch(gather_functor,
                                                            edge_start_time_view,
                                                            edge_type_view,
                                                            edge_weight_view,
                                                            edge_id_view,
                                                            edge_end_time_view);
  } else if (active_major_labels) {
    using tag_t = label_t;

    cugraph::vertex_frontier_t<vertex_t, label_t, multi_gpu, false> vertex_label_frontier(handle,
                                                                                          1);
    vertex_label_frontier.bucket(0).insert(
      thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
      thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    gather_one_hop_edgelist_functor_t<vertex_t,
                                      edge_t,
                                      weight_t,
                                      edge_type_t,
                                      edge_time_t,
                                      tag_t,
                                      label_t,
                                      multi_gpu>
      gather_functor{
        handle, graph_view, vertex_label_frontier.bucket(0), gather_flags, do_expensive_check};

    std::tie(result_srcs,
             result_dsts,
             result_types,
             result_weights,
             result_ids,
             result_start_times,
             result_end_times,
             result_labels) = tuple_with_optionals_dispatch(gather_functor,
                                                            edge_type_view,
                                                            edge_weight_view,
                                                            edge_id_view,
                                                            edge_start_time_view,
                                                            edge_end_time_view);

  } else if (active_major_times) {
    using tag_t = label_t;

    cugraph::vertex_frontier_t<vertex_t, edge_time_t, multi_gpu, false> vertex_time_frontier(handle,
                                                                                             1);
    vertex_time_frontier.bucket(0).insert(
      thrust::make_zip_iterator(active_majors.begin(), active_major_times->begin()),
      thrust::make_zip_iterator(active_majors.end(), active_major_times->end()));

    temporal_simple_gather_one_hop_edgelist_functor_t<vertex_t,
                                                      edge_t,
                                                      weight_t,
                                                      edge_type_t,
                                                      edge_time_t,
                                                      label_t,
                                                      multi_gpu>
      gather_functor{
        handle, graph_view, vertex_time_frontier.bucket(0), gather_flags, do_expensive_check};

    std::tie(result_srcs,
             result_dsts,
             result_start_times,
             result_types,
             result_weights,
             result_ids,
             result_end_times,
             result_labels) = tuple_with_optionals_dispatch(gather_functor,
                                                            edge_start_time_view,
                                                            edge_type_view,
                                                            edge_weight_view,
                                                            edge_id_view,
                                                            edge_end_time_view);

  } else {
    using tag_t = void;

    cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, false> vertex_frontier(handle, 1);
    vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

    gather_one_hop_edgelist_functor_t<vertex_t,
                                      edge_t,
                                      weight_t,
                                      edge_type_t,
                                      edge_time_t,
                                      tag_t,
                                      label_t,
                                      multi_gpu>
      gather_functor{
        handle, graph_view, vertex_frontier.bucket(0), gather_flags, do_expensive_check};

    std::tie(result_srcs,
             result_dsts,
             result_types,
             result_weights,
             result_ids,
             result_start_times,
             result_end_times,
             result_labels) = tuple_with_optionals_dispatch(gather_functor,
                                                            edge_type_view,
                                                            edge_weight_view,
                                                            edge_id_view,
                                                            edge_start_time_view,
                                                            edge_end_time_view);
  }

  return std::make_tuple(std::move(result_srcs),
                         std::move(result_dsts),
                         std::move(result_weights),
                         std::move(result_ids),
                         std::move(result_types),
                         std::move(result_start_times),
                         std::move(result_end_times),
                         std::move(result_labels));
}

}  // namespace detail
}  // namespace cugraph
