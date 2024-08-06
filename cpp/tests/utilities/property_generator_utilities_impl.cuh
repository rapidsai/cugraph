/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/property_generator_kernels.cuh"
#include "utilities/property_generator_utilities.hpp"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cuco/hash_functions.cuh>

#include <tuple>
#include <vector>

namespace cugraph {
namespace test {

template <typename GraphViewType, typename property_t>
property_t generate<GraphViewType, property_t>::initial_value(int32_t init)
{
  return detail::make_property_value<property_t>(init);
}

template <typename GraphViewType, typename property_t>
typename generate<GraphViewType, property_t>::property_buffer_type
generate<GraphViewType, property_t>::vertex_property(raft::handle_t const& handle,
                                                     rmm::device_uvector<vertex_type> const& labels,
                                                     int32_t hash_bin_count)
{
  auto data = cugraph::allocate_dataframe_buffer<property_t>(labels.size(), handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    labels.begin(),
                    labels.end(),
                    cugraph::get_dataframe_buffer_begin(data),
                    detail::vertex_property_transform<vertex_type, property_t>{hash_bin_count});
  return data;
}

template <typename GraphViewType, typename property_t>
typename generate<GraphViewType, property_t>::property_buffer_type
generate<GraphViewType, property_t>::vertex_property(raft::handle_t const& handle,
                                                     thrust::counting_iterator<vertex_type> begin,
                                                     thrust::counting_iterator<vertex_type> end,
                                                     int32_t hash_bin_count)
{
  auto length = thrust::distance(begin, end);
  auto data   = cugraph::allocate_dataframe_buffer<property_t>(length, handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    begin,
                    end,
                    cugraph::get_dataframe_buffer_begin(data),
                    detail::vertex_property_transform<vertex_type, property_t>{hash_bin_count});
  return data;
}

template <typename GraphViewType, typename property_t>
cugraph::edge_src_property_t<GraphViewType, property_t>
generate<GraphViewType, property_t>::src_property(raft::handle_t const& handle,
                                                  GraphViewType const& graph_view,
                                                  property_buffer_type const& property)
{
  auto output_property =
    cugraph::edge_src_property_t<GraphViewType, property_t>(handle, graph_view);
  update_edge_src_property(handle,
                           graph_view,
                           cugraph::get_dataframe_buffer_begin(property),
                           output_property.mutable_view());
  return output_property;
}

template <typename GraphViewType, typename property_t>
cugraph::edge_dst_property_t<GraphViewType, property_t>
generate<GraphViewType, property_t>::dst_property(raft::handle_t const& handle,
                                                  GraphViewType const& graph_view,
                                                  property_buffer_type const& property)
{
  auto output_property =
    cugraph::edge_dst_property_t<GraphViewType, property_t>(handle, graph_view);
  update_edge_dst_property(handle,
                           graph_view,
                           cugraph::get_dataframe_buffer_begin(property),
                           output_property.mutable_view());
  return output_property;
}

template <typename GraphViewType, typename property_t>
cugraph::edge_property_t<GraphViewType, property_t>
generate<GraphViewType, property_t>::edge_property(raft::handle_t const& handle,
                                                   GraphViewType const& graph_view,
                                                   int32_t hash_bin_count)
{
  auto output_property = cugraph::edge_property_t<GraphViewType, property_t>(handle, graph_view);
  cugraph::transform_e(handle,
                       graph_view,
                       cugraph::edge_src_dummy_property_t{}.view(),
                       cugraph::edge_dst_dummy_property_t{}.view(),
                       cugraph::edge_dummy_property_t{}.view(),
                       detail::edge_property_transform<vertex_type, property_t>{hash_bin_count},
                       output_property.mutable_view());
  return output_property;
}

}  // namespace test
}  // namespace cugraph
