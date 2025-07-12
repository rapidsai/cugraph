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
#include "cugraph/arithmetic_variant_types.hpp"
#include "gather_one_hop_functions.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_if_v_frontier_outgoing_e.cuh"
#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/kv_store.cuh"
#include "prims/transform_gather_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/vertex_frontier.cuh"
#include "structure/detail/structure_utils.cuh"
#include "utilities/collect_comm.cuh"
#include "utilities/tuple_with_optionals_dispatching.hpp"

#include <cugraph/edge_property.hpp>
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

struct return_edges_with_single_property_op {
  template <typename vertex_t, typename edge_property_t>
  cuda::std::tuple<vertex_t, vertex_t, edge_property_t> __device__
  operator()(vertex_t src,
             vertex_t dst,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t,
             edge_property_t edge_property) const
  {
    return cuda::std::make_tuple(src, dst, edge_property);
  }

  template <typename vertex_t, typename label_t, typename edge_property_t>
  cuda::std::tuple<vertex_t, vertex_t, edge_property_t, label_t> __device__
  operator()(cuda::std::tuple<vertex_t, label_t> tagged_src,
             vertex_t dst,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t,
             edge_property_t edge_property) const
  {
    return cuda::std::make_tuple(
      cuda::std::get<0>(tagged_src), dst, edge_property, cuda::std::get<1>(tagged_src));
  }
};

template <typename vertex_t, typename edge_time_t>
struct label_time_filtered_edges_with_properties_pred_op {
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<cuda::std::tuple<edge_time_t const*, int32_t const*>>>>
    kv_store_view_;

  cuda::std::optional<raft::device_span<uint8_t const>> optional_gather_flags_{std::nullopt};

  bool __device__ operator()(cuda::std::tuple<vertex_t, size_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             edge_time_t edge_time) const
  {
    size_t label_time_id{cuda::std::get<1>(tagged_src)};

    auto tuple = kv_store_view_.find(label_time_id);

    edge_time_t src_time{cuda::std::get<0>(tuple)};

    return (src_time < edge_time);
  }

  bool __device__ operator()(cuda::std::tuple<vertex_t, size_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             cuda::std::tuple<int32_t, edge_time_t> edge_type_and_time) const
  {
    vertex_t src{cuda::std::get<0>(tagged_src)};
    size_t label_time_id{cuda::std::get<1>(tagged_src)};

    edge_time_t edge_time = cuda::std::get<1>(edge_type_and_time);

    auto tuple = kv_store_view_.find(label_time_id);

    edge_time_t src_time{cuda::std::get<0>(tuple)};
    int32_t src_label{cuda::std::get<1>(tuple)};

    if (src_time < edge_time) {
      auto edge_type = cuda::std::get<0>(edge_type_and_time);
      return ((*optional_gather_flags_)[edge_type] == static_cast<uint8_t>(true));
    }
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

          auto output_buffer = simple_gather_one_hop_edgelist(
            handle,
            graph_view,
            key_list,
            cuda::std::make_optional(
              raft::device_span<uint8_t const>{d_gather_flags.data(), d_gather_flags.size()}),
            edge_value_view,
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
      auto output_buffer = simple_gather_one_hop_edgelist(
        handle, graph_view, key_list, cuda::std::nullopt, edge_value_view, do_expensive_check);

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

            auto output_buffer = temporal_simple_gather_one_hop_edgelist(
              handle,
              graph_view,
              key_list,
              cuda::std::make_optional(
                raft::device_span<uint8_t const>{d_gather_flags.data(), d_gather_flags.size()}),
              edge_value_view,
              do_expensive_check);

            move_results<0, 0, true, true, Flags..., false>(output_buffer, return_result);
          } else {
            CUGRAPH_FAIL("If gather_flags specified edge_type must be first property");
          }
        } else {
          CUGRAPH_FAIL("If gather_flags specified edge_type must be first property");
        }
      } else {
        auto output_buffer = temporal_simple_gather_one_hop_edgelist(
          handle, graph_view, key_list, cuda::std::nullopt, edge_value_view, do_expensive_check);

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

    if constexpr (0 == std::tuple_size_v<TupleType>) {
      CUGRAPH_FAIL("Edge time must be specified as a property");
    } else if constexpr (!std::is_same_v<typename std::tuple_element_t<0, TupleType>::value_type,
                                         edge_time_t>) {
      CUGRAPH_FAIL("Edge time must be specified as first property");
    } else {
      auto edge_value_view           = concatenate_views(edge_properties);
      using edge_property_elements_t = typename decltype(edge_value_view)::value_type;

      if (gather_flags) {
        if constexpr (1 < std::tuple_size_v<TupleType>)
          if constexpr (std::is_same_v<typename std::tuple_element_t<1, TupleType>::value_type,
                                       edge_type_t>) {
            rmm::device_uvector<uint8_t> d_gather_flags(gather_flags->size(), handle.get_stream());
            raft::update_device(d_gather_flags.data(),
                                gather_flags->data(),
                                gather_flags->size(),
                                handle.get_stream());

            auto output_buffer = temporal_label_gather_one_hop_edgelist(
              handle,
              graph_view,
              key_list,
              cuda::std::make_optional(
                raft::device_span<uint8_t const>{d_gather_flags.data(), d_gather_flags.size()}),
              kv_store_view,
              edge_value_view,
              do_expensive_check);

            move_results<0, 0, true, true, Flags..., true>(output_buffer, return_result);
          } else {
            CUGRAPH_FAIL("If gather_flags specified edge_type must be second property");
          }
        else {
          CUGRAPH_FAIL("If gather_flags specified edge_type must be second property");
        }
      } else {
        auto output_buffer = temporal_label_gather_one_hop_edgelist(handle,
                                                                    graph_view,
                                                                    key_list,
                                                                    cuda::std::nullopt,
                                                                    kv_store_view,
                                                                    edge_value_view,
                                                                    do_expensive_check);

        move_results<0, 0, true, true, Flags..., true>(output_buffer, return_result);
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

// NEW IMPLEMENTATION
#if 0
template <typename vertex_t, typename edge_t, typename tag_t, bool multi_gpu>
std::conditional_t<std::is_same_v<tag_t, void>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              std::vector<arithmetic_device_uvector_t>>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              std::vector<arithmetic_device_uvector_t>,
                              rmm::device_uvector<tag_t>>>
simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  raft::host_span<edge_arithmetic_property_view_t<edge_t, vertex_t>> edge_property_views,
  bool do_expensive_check)
{
  using key_t =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  using label_t = std::conditional_t<std::is_same_v<tag_t, void>, cuda::std::nullopt_t, tag_t>;

  if constexpr (std::is_same_v<T1, tag_t>) {
    return gather_flags ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2, T3, T4, T5>,
                                                              label_t>{},
                            type_filtered_edges_with_properties_pred_op{*gather_flags},
                            do_expensive_check)
                        : cugraph::extract_transform_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2, T3, T4, T5>,
                                                              label_t>{},
                            do_expensive_check);
  } else {
    CUGRAPH_EXPECTS(!gather_flags, "gather_flags can only be specified with edge type");

    return cugraph::extract_transform_v_frontier_outgoing_e(
      handle,
      graph_view,
      key_list,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_value_view,
      return_edges_with_properties_e_op<key_t,
                                        vertex_t,
                                        thrust::tuple<T1, T2, T3, T4, T5>,
                                        label_t>{},
      do_expensive_check);
  }
}
#endif

#if 0
// Struggling with complexity, trying what might be a simpler approach
template <typename vertex_t, typename edge_t, typename edge_time_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<edge_t, vertex_t>> edge_property_views,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_time_view,
  std::optional<edge_arithmetic_property_view_t<edge_t, vertex_t>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<edge_time_t const>> active_major_times,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::host_span<uint8_t const>> gather_flags,
  bool do_expensive_check)
{
  constexpr bool store_transposed = false;

  CUGRAPH_EXPECTS(!gather_flags || edge_type_view, "If gather_flags is specified a type view must be specified");

  if (edge_time_view)
    CUGRAPH_EXPECTS(edge_property_views.size() > 0,
                    "Time view specified as parameter but not included in properties");

  if (edge_type_view) {
    if (edge_time_view)
      CUGRAPH_EXPECTS(edge_property_views.size() > 1,
                      "Time and type views specified as parameters but not included in properties");
    else
      CUGRAPH_EXPECTS(edge_property_views.size() > 0,
                      "Type view specified as parameter but not included in properties");
  }

  rmm::device_uvector<vertex_t> result_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(0, handle.get_stream());
  std::vector<arithmetic_device_uvector_t> result_properties{};
  std::optional<rmm::device_uvector<int32_t>> result_labels{std::nullopt};

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

    kv_store_t<size_t, thrust::tuple<edge_time_t, int32_t>, true> kv_store{handle.get_stream()};
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
        raft::device_span<int32_t const>{active_major_labels->data(), active_major_labels->size()});

      CUGRAPH_EXPECTS(
        thrust::is_sorted(handle.get_thrust_policy(), all_minor_keys.begin(), all_minor_keys.end()),
        "need to SORT!");

      kv_store = kv_store_t<size_t, thrust::tuple<edge_time_t, int32_t>, true>(
        all_minor_keys.begin(),
        all_minor_keys.end(),
        thrust::make_zip_iterator(all_minor_times.begin(), all_minor_labels.begin()),
        thrust::make_tuple(edge_time_t{-1}, int32_t{-1}),
        true,
        handle.get_stream());

    } else {
      kv_store = kv_store_t<size_t, thrust::tuple<edge_time_t, int32_t>, true>(
        vertex_label_time_ids.begin(),
        vertex_label_time_ids.end(),
        thrust::make_zip_iterator(active_major_times->begin(),
                                  active_major_labels->begin()),  // multi_gpu is different
        thrust::make_tuple(edge_time_t{-1}, int32_t{-1}),
        true,
        handle.get_stream());
    }

    cugraph::vertex_frontier_t<vertex_t, size_t, multi_gpu, false> vertex_label_frontier(handle, 1);
    auto& key_list = vertex_label_frontier.bucket(0);
    key_list.insert(thrust::make_zip_iterator(active_majors.begin(), vertex_label_time_ids.begin()),
                    thrust::make_zip_iterator(active_majors.end(), vertex_label_time_ids.end()));

    if (edge_property_views.size() == 1) {
      cugraph::variant_type_dispatch(
        edge_property_views[0],
        [&handle,
         &graph_view,
         &key_list,
         &result_srcs,
         &result_dsts,
         &result_properties,
         &result_labels,
         do_expensive_check](auto& property_view) {
          using T = typename decltype(property_view)::value_type;
          if constexpr (std::is_same_v<T, edge_time_t>) {
            rmm::device_uvector<T> tmp(0, handle.get_stream());

            std::tie(result_srcs, result_dsts, tmp, result_labels) =
              gather_flags
                ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                    handle,
                    graph_view,
                    key_list,
                    edge_src_dummy_property_t{}.view(),
                    edge_dst_dummy_property_t{}.view(),
                    view_concat(edge_type_view, property_view),
                    return_edges_with_single_property_op{},
                    label_time_filtered_edges_with_properties_pred_op<vertex_t, edge_time_t>{
                      kv_store_view, gather_flags},
                    do_expensive_check)
                : cugraph::extract_transform_if_v_frontier_outgoing_e(
                    handle,
                    graph_view,
                    key_list,
                    edge_src_dummy_property_t{}.view(),
                    edge_dst_dummy_property_t{}.view(),
                    property_view,
                    return_edges_with_single_property_op{},
                    label_time_filtered_edges_with_properties_pred_op<vertex_t, edge_time_t>{
                      kv_store_view, gather_flags},
                    do_expensive_check);

            result_properties.push_back(std::move(tmp));
          } else {
            CUGRAPH_FAIL("Property type must match edge_time_t");
          }
        });
    } else {
      std::optional<cugraph::edge_multi_index_property_t<edge_t, vertex_t>> multi_edge_indices{
        std::nullopt};

      cugraph::edge_bucket_t<vertex_t, edge_t, !store_transposed, true, false> edge_list(
        handle, graph_view.is_multigraph());

      if (graph_view.is_multigraph()) {
        cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                    graph_view);
        cugraph::edge_arithmetic_property_view_t<edge_t, vertex_t> multi_index_property_view =
          multi_index_property.view();

        rmm::device_uvector<edge_t> tmp(0, handle.get_stream());

        // Don't I need the transform_if variant here?
        std::tie(result_srcs, result_dsts, tmp, result_labels) =
          gather_flags ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                           handle,
                           graph_view,
                           key_list,
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           view_concat(edge_type_view, multi_index_property_view),
                           return_edges_with_single_property_op{},
                           label_time_filtered_edges_with_properties_pred_op<vertex_t, edge_time_t>{
                             kv_store_view, gather_flags},
                           do_expensive_check)
                       : cugraph::extract_transform_if_v_frontier_outgoing_e(
                           handle,
                           graph_view,
                           key_list,
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           multi_index_property_view,
                           return_edges_with_single_property_op{},
                           label_time_filtered_edges_with_properties_pred_op<vertex_t, edge_time_t>{
                             kv_store_view, gather_flags},
                           do_expensive_check);

        *multi_edge_indices = std::move(multi_index_property);
        edge_list.insert(result_srcs.begin(),
                         result_srcs.end(),
                         result_dsts.begin(),
                         std::make_optional(tmp.begin()));
      } else {
        cugraph::edge_arithmetic_property_view_t<edge_t, vertex_t> dummy_property_view =
          cugraph::edge_dummy_property_view_t{};

        std::tie(result_srcs, result_dsts, std::ignore, result_labels) =
          gather_flags ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                           handle,
                           graph_view,
                           key_list,
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_type_view,
                           multi_index_property_view,
                           return_edges_with_single_property_op{},
                           label_time_filtered_edges_with_properties_pred_op<vertex_t, edge_time_t>{
                             kv_store_view, gather_flags},
                           do_expensive_check)
                       : cugraph::extract_transform_v_frontier_outgoing_e(
                           handle,
                           graph_view,
                           key_list,
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_dummy_property_view_t{},
                           return_edges_with_single_property_op{},
                           do_expensive_check);

        edge_list.insert(result_srcs.begin(),
                         result_srcs.end(),
                         result_dsts.begin(),
                         std::optional<edge_t*>{std::nullopt});
      }

      std::for_each(
        edge_property_views.begin(),
        edge_property_views.end(),
        [&handle, &graph_view, &edge_list, &result_properties](auto edge_property_view) {
          cugraph::variant_type_dispatch(
            edge_property_view,
            [&handle, &graph_view, &edge_list, &result_properties](auto property_view) {
              using T = typename decltype(property_view)::value_type;

              if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
                CUGRAPH_FAIL("Should not have a property of type cuda::std::nullopt");
              } else {
                rmm::device_uvector<T> tmp(edge_list.size(), handle.get_stream());

                cugraph::transform_gather_e(handle,
                                            graph_view,
                                            edge_list,
                                            edge_src_dummy_property_t{}.view(),
                                            edge_dst_dummy_property_t{}.view(),
                                            property_view,
                                            return_edges_with_single_property_op{},
                                            tmp.begin());

                result_properties.push_back(arithmetic_device_uvector_t{std::move(tmp)});
              }
            });
        });
    }
  } else if (active_major_labels) {
    using tag_t = int32_t;

    cugraph::vertex_frontier_t<vertex_t, int32_t, multi_gpu, false> vertex_label_frontier(handle,
                                                                                          1);
    auto& key_list = vertex_label_frontier.bucket(0);
    key_list.insert(thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
                    thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    if (edge_property_views.size() == 0) {
      std::tie(result_srcs, result_dsts, std::ignore, result_labels) =
        cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                         graph_view,
                                                         key_list,
                                                         edge_src_dummy_property_t{}.view(),
                                                         edge_dst_dummy_property_t{}.view(),
                                                         edge_dummy_property_view_t{},
                                                         return_edges_with_single_property_op{},
                                                         do_expensive_check);

    } else if (edge_property_views.size() == 1) {
      cugraph::variant_type_dispatch(edge_property_views[0],
                                     [&handle,
                                      &graph_view,
                                      &key_list,
                                      &result_srcs,
                                      &result_dsts,
                                      &result_properties,
                                      &result_labels,
                                      do_expensive_check](auto& property_view) {
                                       using T = typename decltype(property_view)::value_type;
                                       if constexpr (std::is_same_v<T, edge_time_t>) {
                                         rmm::device_uvector<T> tmp(0, handle.get_stream());

                                         std::tie(result_srcs, result_dsts, tmp, result_labels) =
                                           cugraph::extract_transform_v_frontier_outgoing_e(
                                             handle,
                                             graph_view,
                                             key_list,
                                             edge_src_dummy_property_t{}.view(),
                                             edge_dst_dummy_property_t{}.view(),
                                             property_view,
                                             return_edges_with_single_property_op{},
                                             do_expensive_check);

                                         result_properties.push_back(std::move(tmp));
                                       } else {
                                         CUGRAPH_FAIL("Property type must match edge_time_t");
                                       }
                                     });
    } else {
      std::optional<cugraph::edge_multi_index_property_t<edge_t, vertex_t>> multi_edge_indices{
        std::nullopt};

      cugraph::edge_bucket_t<vertex_t, edge_t, !store_transposed, true, false> edge_list(
        handle, graph_view.is_multigraph());

      if (graph_view.is_multigraph()) {
        cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                    graph_view);
        cugraph::edge_arithmetic_property_view_t<edge_t, vertex_t> multi_index_property_view =
          multi_index_property.view();

        rmm::device_uvector<edge_t> tmp(0, handle.get_stream());

        // Don't I need the transform_if variant here?
        std::tie(result_srcs, result_dsts, tmp, result_labels) =
          cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                           graph_view,
                                                           key_list,
                                                           edge_src_dummy_property_t{}.view(),
                                                           edge_dst_dummy_property_t{}.view(),
                                                           multi_index_property_view,
                                                           return_edges_with_single_property_op{},
                                                           do_expensive_check);

        *multi_edge_indices = std::move(multi_index_property);
        edge_list.insert(result_srcs.begin(),
                         result_srcs.end(),
                         result_dsts.begin(),
                         std::make_optional(tmp.begin()));
      } else {
        cugraph::edge_arithmetic_property_view_t<edge_t, vertex_t> dummy_property_view =
          cugraph::edge_dummy_property_view_t{};

        std::tie(result_srcs, result_dsts, std::ignore, result_labels) =
          cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                           graph_view,
                                                           key_list,
                                                           edge_src_dummy_property_t{}.view(),
                                                           edge_dst_dummy_property_t{}.view(),
                                                           edge_dummy_property_view_t{},
                                                           return_edges_with_single_property_op{},
                                                           do_expensive_check);

        edge_list.insert(result_srcs.begin(),
                         result_srcs.end(),
                         result_dsts.begin(),
                         std::optional<edge_t*>{std::nullopt});
      }

      std::for_each(
        edge_property_views.begin(),
        edge_property_views.end(),
        [&handle, &graph_view, &edge_list, &result_properties](auto edge_property_view) {
          cugraph::variant_type_dispatch(
            edge_property_view,
            [&handle, &graph_view, &edge_list, &result_properties](auto property_view) {
              using T = typename decltype(property_view)::value_type;

              if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
                CUGRAPH_FAIL("Should not have a property of type cuda::std::nullopt");
              } else {
                rmm::device_uvector<T> tmp(edge_list.size(), handle.get_stream());

                cugraph::transform_gather_e(handle,
                                            graph_view,
                                            edge_list,
                                            edge_src_dummy_property_t{}.view(),
                                            edge_dst_dummy_property_t{}.view(),
                                            property_view,
                                            return_edges_with_single_property_op{},
                                            tmp.begin());

                result_properties.push_back(arithmetic_device_uvector_t{std::move(tmp)});
              }
            });
        });
    }
  } else if (active_major_times) {
#if 0
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
#endif
  } else {
#if 0
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
#endif
  }

  return std::make_tuple(std::move(result_srcs),
                         std::move(result_dsts),
                         std::move(result_properties),
                         std::move(result_labels));
}
#endif

#if 0
template <typename vertex_t, typename edge_t, typename edge_time_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<edge_t, vertex_t>> edge_property_views,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_time_view,
  std::optional<edge_arithmetic_property_view_t<edge_t, vertex_t>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<edge_time_t const>> active_major_times,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::host_span<uint8_t const>> gather_flags,
  bool do_expensive_check)
{
  constexpr bool store_transposed = false;

  rmm::device_uvector<vertex_t> result_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(0, handle.get_stream());
  std::vector<arithmetic_device_uvector_t> result_properties{};
  std::optional<rmm::device_uvector<int32_t>> result_labels{std::nullopt};

  if (edge_property_views.size() == 0) {
    CUGRAPH_EXPECTS(!edge_time_view,
                    "Can't specify temporal filtering without edge time as a property");
    CUGRAPH_EXPECTS(!edge_type_view,
                    "Can't specify type filtering without edge type as a property");

    std::tie(result_srcs, result_dsts, std::ignore, result_labels) =
      cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                       graph_view,
                                                       key_list,
                                                       edge_src_dummy_property_t{}.view(),
                                                       edge_dst_dummy_property_t{}.view(),
                                                       edge_dummy_property_view_t{},
                                                       return_edges_with_single_property_op{},
                                                       do_expensive_check);
  } else if (edge_property_views.size() == 1) {
    if (edge_time_view) {
      CUGRAPH_EXPECTS(
        !edge_type_view,
        "Can't specify time and type filtering without both edge time and type as a property");

      cugraph::variant_type_dispatch(
        edge_property_views[0],
        [&handle,
         &graph_view,
         &key_list,
         &result_srcs,
         &result_dsts,
         &result_properties,
         &result_labels,
         do_expensive_check](auto& property_view) {
          using T = typename decltype(property_view)::value_type;
          if constexpr (std::is_same_v<T, edge_time_t>) {
            rmm::device_uvector<T> tmp(0, handle.get_stream());

            std::tie(result_srcs, result_dsts, tmp, result_labels) =
              gather_flags
                ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                    handle,
                    graph_view,
                    key_list,
                    edge_src_dummy_property_t{}.view(),
                    edge_dst_dummy_property_t{}.view(),
                    view_concat(edge_type_view, property_view),
                    return_edges_with_single_property_op{},
                    label_time_filtered_edges_with_properties_pred_op<vertex_t, edge_time_t>{
                      kv_store_view, gather_flags},
                    do_expensive_check)
                : cugraph::extract_transform_if_v_frontier_outgoing_e(
                    handle,
                    graph_view,
                    key_list,
                    edge_src_dummy_property_t{}.view(),
                    edge_dst_dummy_property_t{}.view(),
                    property_view,
                    return_edges_with_single_property_op{},
                    label_time_filtered_edges_with_properties_pred_op<vertex_t, edge_time_t>{
                      kv_store_view, gather_flags},
                    do_expensive_check);

            result_properties.push_back(std::move(tmp));
          } else {
            CUGRAPH_FAIL("Property type must match edge_time_t");
          }
        });

    } else if (edge_type_view) {
    } else {
      cugraph::variant_type_dispatch(edge_property_views[0],
                                     [&handle,
                                      &graph_view,
                                      &key_list,
                                      &result_srcs,
                                      &result_dsts,
                                      &result_properties,
                                      &result_labels,
                                      do_expensive_check](auto& property_view) {
                                       using T = typename decltype(property_view)::value_type;
                                       rmm::device_uvector<T> tmp(0, handle.get_stream());

                                       std::tie(result_srcs, result_dsts, tmp, result_labels) =
                                         cugraph::extract_transform_v_frontier_outgoing_e(
                                           handle,
                                           graph_view,
                                           key_list,
                                           edge_src_dummy_property_t{}.view(),
                                           edge_dst_dummy_property_t{}.view(),
                                           property_view,
                                           return_edges_with_single_property_op{},
                                           do_expensive_check);

                                       result_properties.push_back(std::move(tmp));
                                     });
    }
  } else {
  }

  return std::make_tuple(std::move(result_srcs),
                         std::move(result_dsts),
                         std::move(result_properties),
                         std::move(result_labels));
}
#endif

}  // namespace detail
}  // namespace cugraph
