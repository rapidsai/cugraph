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

#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/vertex_frontier.cuh"
#include "structure/detail/structure_utils.cuh"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/tuple.h>

namespace cugraph {
namespace detail {

struct return_edges_with_properties_e_op {
  template <typename key_t, typename vertex_t, typename EdgeProperties>
  auto __host__ __device__ operator()(key_t optionally_tagged_src,
                                      vertex_t dst,
                                      cuda::std::nullopt_t,
                                      cuda::std::nullopt_t,
                                      EdgeProperties edge_properties) const
  {
    static_assert(std::is_same_v<key_t, vertex_t> ||
                  std::is_same_v<key_t, thrust::tuple<vertex_t, int32_t>>);

    // FIXME: A solution using thrust_tuple_cat would be more flexible here
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      vertex_t src{optionally_tagged_src};

      if constexpr (std::is_same_v<EdgeProperties, cuda::std::nullopt_t>) {
        return thrust::make_tuple(src, dst);
      } else if constexpr (std::is_arithmetic<EdgeProperties>::value) {
        return thrust::make_tuple(src, dst, edge_properties);
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 2)) {
        return thrust::make_tuple(
          src, dst, thrust::get<0>(edge_properties), thrust::get<1>(edge_properties));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 3)) {
        return thrust::make_tuple(src,
                                  dst,
                                  thrust::get<0>(edge_properties),
                                  thrust::get<1>(edge_properties),
                                  thrust::get<2>(edge_properties));
      }
    } else {
      vertex_t src{thrust::get<0>(optionally_tagged_src)};
      int32_t label{thrust::get<1>(optionally_tagged_src)};

      src = thrust::get<0>(optionally_tagged_src);
      if constexpr (std::is_same_v<EdgeProperties, cuda::std::nullopt_t>) {
        return thrust::make_tuple(src, dst, label);
      } else if constexpr (std::is_arithmetic<EdgeProperties>::value) {
        return thrust::make_tuple(src, dst, edge_properties, label);
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 2)) {
        return thrust::make_tuple(
          src, dst, thrust::get<0>(edge_properties), thrust::get<1>(edge_properties), label);
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 3)) {
        return thrust::make_tuple(src,
                                  dst,
                                  thrust::get<0>(edge_properties),
                                  thrust::get<1>(edge_properties),
                                  thrust::get<2>(edge_properties),
                                  label);
      }
    }
  }
};

template <bool has_weight,
          bool has_edge_id,
          bool has_edge_type,
          typename label_t,
          typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename tag_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<label_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> const& vertex_frontier,
  std::optional<raft::host_span<uint8_t const>> gather_flags,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
  std::optional<rmm::device_uvector<label_t>> labels{std::nullopt};

  using edge_value_t = std::conditional_t<
    has_weight,
    std::conditional_t<
      has_edge_id,
      std::conditional_t<has_edge_type,
                         thrust::tuple<weight_t, edge_t, edge_type_t>,
                         thrust::tuple<weight_t, edge_t>>,
      std::conditional_t<has_edge_type, thrust::tuple<weight_t, edge_type_t>, weight_t>>,
    std::conditional_t<
      has_edge_id,
      std::conditional_t<has_edge_type, thrust::tuple<edge_t, edge_type_t>, edge_t>,
      std::conditional_t<has_edge_type, edge_type_t, cuda::std::nullopt_t>>>;

  using edge_value_view_t =
    edge_property_view_type_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_value_t>;

  edge_value_view_t edge_value_view{};
  if constexpr (has_weight) {
    if constexpr (has_edge_id) {
      if constexpr (has_edge_type) {
        edge_value_view = view_concat(*edge_weight_view, *edge_id_view, *edge_type_view);
      } else {
        edge_value_view = view_concat(*edge_weight_view, *edge_id_view);
      }
    } else {
      if constexpr (has_edge_type) {
        edge_value_view = view_concat(*edge_weight_view, *edge_type_view);
      } else {
        edge_value_view = *edge_weight_view;
      }
    }
  } else {
    if constexpr (has_edge_id) {
      if constexpr (has_edge_type) {
        edge_value_view = view_concat(*edge_id_view, *edge_type_view);
      } else {
        edge_value_view = *edge_id_view;
      }
    } else {
      if constexpr (has_edge_type) { edge_value_view = *edge_type_view; }
    }
  }

  auto output_buffer =
    cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                     graph_view,
                                                     vertex_frontier.bucket(0),
                                                     edge_src_dummy_property_t{}.view(),
                                                     edge_dst_dummy_property_t{}.view(),
                                                     edge_value_view,
                                                     return_edges_with_properties_e_op{},
                                                     do_expensive_check);

  majors = std::move(std::get<0>(output_buffer));
  minors = std::move(std::get<1>(output_buffer));
  if constexpr (has_weight) {
    if constexpr (has_edge_id) {
      if constexpr (has_edge_type) {
        edge_weights = std::move(std::get<2>(output_buffer));
        edge_ids     = std::move(std::get<3>(output_buffer));
        edge_types   = std::move(std::get<4>(output_buffer));
        if constexpr (std::is_same_v<tag_t, label_t>) {
          labels = std::move(std::get<5>(output_buffer));
        }
      } else {
        edge_weights = std::move(std::get<2>(output_buffer));
        edge_ids     = std::move(std::get<3>(output_buffer));
        if constexpr (std::is_same_v<tag_t, label_t>) {
          labels = std::move(std::get<4>(output_buffer));
        }
      }
    } else {
      if constexpr (has_edge_type) {
        edge_weights = std::move(std::get<2>(output_buffer));
        edge_types   = std::move(std::get<3>(output_buffer));
        if constexpr (std::is_same_v<tag_t, label_t>) {
          labels = std::move(std::get<4>(output_buffer));
        }
      } else {
        edge_weights = std::move(std::get<2>(output_buffer));
        if constexpr (std::is_same_v<tag_t, label_t>) {
          labels = std::move(std::get<3>(output_buffer));
        }
      }
    }
  } else {
    if constexpr (has_edge_id) {
      if constexpr (has_edge_type) {
        edge_ids   = std::move(std::get<2>(output_buffer));
        edge_types = std::move(std::get<3>(output_buffer));
        if constexpr (std::is_same_v<tag_t, label_t>) {
          labels = std::move(std::get<4>(output_buffer));
        }
      } else {
        edge_ids = std::move(std::get<2>(output_buffer));
        if constexpr (std::is_same_v<tag_t, label_t>) {
          labels = std::move(std::get<3>(output_buffer));
        }
      }
    } else {
      if constexpr (has_edge_type) {
        edge_types = std::move(std::get<2>(output_buffer));
        if constexpr (std::is_same_v<tag_t, label_t>) {
          labels = std::move(std::get<3>(output_buffer));
        }
      } else {
        if constexpr (std::is_same_v<tag_t, label_t>) {
          labels = std::move(std::get<2>(output_buffer));
        }
      }
    }
  }

  if (gather_flags) {
    assert(edge_types);

    rmm::device_uvector<uint8_t> d_gather_flags(gather_flags->size(), handle.get_stream());
    raft::update_device(
      d_gather_flags.data(), gather_flags->data(), gather_flags->size(), handle.get_stream());

    size_t new_size{};
    if constexpr (has_weight) {
      if constexpr (has_edge_id) {
        if constexpr (std::is_same_v<tag_t, label_t>) {
          auto tuple_first = thrust::make_zip_iterator(majors.begin(),
                                                       minors.begin(),
                                                       edge_weights->begin(),
                                                       edge_ids->begin(),
                                                       edge_types->begin(),
                                                       labels->begin());
          new_size         = static_cast<size_t>(cuda::std::distance(
            tuple_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              tuple_first,
              tuple_first + majors.size(),
              [gather_flags = raft::device_span<uint8_t const>(
                 d_gather_flags.data(), d_gather_flags.size())] __device__(auto tup) {
                auto type = thrust::get<4>(tup);
                return gather_flags[type] == static_cast<uint8_t>(false);
              })));
        } else {
          auto tuple_first = thrust::make_zip_iterator(majors.begin(),
                                                       minors.begin(),
                                                       edge_weights->begin(),
                                                       edge_ids->begin(),
                                                       edge_types->begin());
          new_size         = static_cast<size_t>(cuda::std::distance(
            tuple_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              tuple_first,
              tuple_first + majors.size(),
              [gather_flags = raft::device_span<uint8_t const>(
                 d_gather_flags.data(), d_gather_flags.size())] __device__(auto tup) {
                auto type = thrust::get<4>(tup);
                return gather_flags[type] == static_cast<uint8_t>(false);
              })));
        }
      } else {
        if constexpr (std::is_same_v<tag_t, label_t>) {
          auto tuple_first = thrust::make_zip_iterator(majors.begin(),
                                                       minors.begin(),
                                                       edge_weights->begin(),
                                                       edge_types->begin(),
                                                       labels->begin());
          new_size         = static_cast<size_t>(cuda::std::distance(
            tuple_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              tuple_first,
              tuple_first + majors.size(),
              [gather_flags = raft::device_span<uint8_t const>(
                 d_gather_flags.data(), d_gather_flags.size())] __device__(auto tup) {
                auto type = thrust::get<3>(tup);
                return gather_flags[type] == static_cast<uint8_t>(false);
              })));
        } else {
          auto tuple_first = thrust::make_zip_iterator(
            majors.begin(), minors.begin(), edge_weights->begin(), edge_types->begin());
          new_size = static_cast<size_t>(cuda::std::distance(
            tuple_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              tuple_first,
              tuple_first + majors.size(),
              [gather_flags = raft::device_span<uint8_t const>(
                 d_gather_flags.data(), d_gather_flags.size())] __device__(auto tup) {
                auto type = thrust::get<3>(tup);
                return gather_flags[type] == static_cast<uint8_t>(false);
              })));
        }
      }
    } else {
      if constexpr (has_edge_id) {
        if constexpr (std::is_same_v<tag_t, label_t>) {
          auto tuple_first = thrust::make_zip_iterator(majors.begin(),
                                                       minors.begin(),
                                                       edge_ids->begin(),
                                                       edge_types->begin(),
                                                       labels->begin());
          new_size         = static_cast<size_t>(cuda::std::distance(
            tuple_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              tuple_first,
              tuple_first + majors.size(),
              [gather_flags = raft::device_span<uint8_t const>(
                 d_gather_flags.data(), d_gather_flags.size())] __device__(auto tup) {
                auto type = thrust::get<3>(tup);
                return gather_flags[type] == static_cast<uint8_t>(false);
              })));
        } else {
          auto tuple_first = thrust::make_zip_iterator(
            majors.begin(), minors.begin(), edge_ids->begin(), edge_types->begin());
          new_size = static_cast<size_t>(cuda::std::distance(
            tuple_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              tuple_first,
              tuple_first + majors.size(),
              [gather_flags = raft::device_span<uint8_t const>(
                 d_gather_flags.data(), d_gather_flags.size())] __device__(auto tup) {
                auto type = thrust::get<3>(tup);
                return gather_flags[type] == static_cast<uint8_t>(false);
              })));
        }
      } else {
        if constexpr (std::is_same_v<tag_t, label_t>) {
          auto tuple_first = thrust::make_zip_iterator(
            majors.begin(), minors.begin(), edge_types->begin(), labels->begin());
          new_size = static_cast<size_t>(cuda::std::distance(
            tuple_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              tuple_first,
              tuple_first + majors.size(),
              [gather_flags = raft::device_span<uint8_t const>(
                 d_gather_flags.data(), d_gather_flags.size())] __device__(auto tup) {
                auto type = thrust::get<2>(tup);
                return gather_flags[type] == static_cast<uint8_t>(false);
              })));
        } else {
          auto tuple_first =
            thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_types->begin());
          new_size = static_cast<size_t>(cuda::std::distance(
            tuple_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              tuple_first,
              tuple_first + majors.size(),
              [gather_flags = raft::device_span<uint8_t const>(
                 d_gather_flags.data(), d_gather_flags.size())] __device__(auto tup) {
                auto type = thrust::get<2>(tup);
                return gather_flags[type] == static_cast<uint8_t>(false);
              })));
        }
      }
    }

    majors.resize(new_size, handle.get_stream());
    majors.shrink_to_fit(handle.get_stream());
    minors.resize(new_size, handle.get_stream());
    minors.shrink_to_fit(handle.get_stream());
    if constexpr (has_weight) {
      edge_weights->resize(new_size, handle.get_stream());
      edge_weights->shrink_to_fit(handle.get_stream());
    }
    if constexpr (has_edge_id) {
      edge_ids->resize(new_size, handle.get_stream());
      edge_ids->shrink_to_fit(handle.get_stream());
    }
    if constexpr (has_edge_type) {
      edge_types->resize(new_size, handle.get_stream());
      edge_types->shrink_to_fit(handle.get_stream());
    }
    if constexpr (std::is_same_v<tag_t, label_t>) {
      labels->resize(new_size, handle.get_stream());
      labels->shrink_to_fit(handle.get_stream());
    }
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(edge_weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(labels));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<label_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<label_t const>> active_major_labels,
  std::optional<raft::host_span<uint8_t const>> gather_flags,
  bool do_expensive_check)
{
  assert(!gather_flags || edge_type_view);

  if (active_major_labels) {
    using tag_t = label_t;

    cugraph::vertex_frontier_t<vertex_t, label_t, multi_gpu, false> vertex_label_frontier(handle,
                                                                                          1);
    vertex_label_frontier.bucket(0).insert(
      thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
      thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    if (edge_weight_view) {
      bool constexpr has_weight = true;
      if (edge_id_view) {
        bool constexpr has_edge_id = true;
        if (edge_type_view) {
          bool constexpr has_edge_type = true;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_label_frontier,
            gather_flags,
            do_expensive_check);
        } else {
          bool constexpr has_edge_type = false;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_label_frontier,
            gather_flags,
            do_expensive_check);
        }
      } else {
        bool constexpr has_edge_id = false;
        if (edge_type_view) {
          bool constexpr has_edge_type = true;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_label_frontier,
            gather_flags,
            do_expensive_check);
        } else {
          bool constexpr has_edge_type = false;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_label_frontier,
            gather_flags,
            do_expensive_check);
        }
      }
    } else {
      bool constexpr has_weight = false;
      if (edge_id_view) {
        bool constexpr has_edge_id = true;
        if (edge_type_view) {
          bool constexpr has_edge_type = true;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_label_frontier,
            gather_flags,
            do_expensive_check);
        } else {
          bool constexpr has_edge_type = false;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_label_frontier,
            gather_flags,
            do_expensive_check);
        }
      } else {
        bool constexpr has_edge_id = false;
        if (edge_type_view) {
          bool constexpr has_edge_type = true;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_label_frontier,
            gather_flags,
            do_expensive_check);
        } else {
          bool constexpr has_edge_type = false;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_label_frontier,
            gather_flags,
            do_expensive_check);
        }
      }
    }
  } else {
    using tag_t = void;

    cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, false> vertex_frontier(handle, 1);
    vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

    if (edge_weight_view) {
      bool constexpr has_weight = true;
      if (edge_id_view) {
        bool constexpr has_edge_id = true;
        if (edge_type_view) {
          bool constexpr has_edge_type = true;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_frontier,
            gather_flags,
            do_expensive_check);
        } else {
          bool constexpr has_edge_type = false;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_frontier,
            gather_flags,
            do_expensive_check);
        }
      } else {
        bool constexpr has_edge_id = false;
        if (edge_type_view) {
          bool constexpr has_edge_type = true;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_frontier,
            gather_flags,
            do_expensive_check);
        } else {
          bool constexpr has_edge_type = false;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_frontier,
            gather_flags,
            do_expensive_check);
        }
      }
    } else {
      bool constexpr has_weight = false;
      if (edge_id_view) {
        bool constexpr has_edge_id = true;
        if (edge_type_view) {
          bool constexpr has_edge_type = true;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_frontier,
            gather_flags,
            do_expensive_check);
        } else {
          bool constexpr has_edge_type = false;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_frontier,
            gather_flags,
            do_expensive_check);
        }
      } else {
        bool constexpr has_edge_id = false;
        if (edge_type_view) {
          bool constexpr has_edge_type = true;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_frontier,
            gather_flags,
            do_expensive_check);
        } else {
          bool constexpr has_edge_type = false;
          return gather_one_hop_edgelist<has_weight, has_edge_id, has_edge_type, label_t>(
            handle,
            graph_view,
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            vertex_frontier,
            gather_flags,
            do_expensive_check);
        }
      }
    }
  }
}

}  // namespace detail
}  // namespace cugraph
