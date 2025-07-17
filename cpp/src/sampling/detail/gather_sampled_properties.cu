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

#include "prims/edge_bucket.cuh"
#include "prims/transform_gather_e.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

namespace cugraph {
namespace detail {

namespace {

struct return_edge_property_t {
  template <typename key_t, typename vertex_t, typename T>
  T __device__
  operator()(key_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, T edge_property) const
  {
    return edge_property;
  }
};

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::vector<arithmetic_device_uvector_t> gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_bucket_t<vertex_t, edge_t, true, multi_gpu, false>& edge_list,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views)
{
  std::vector<arithmetic_device_uvector_t> result_properties{};

  std::for_each(edge_property_views.begin(),
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
                                                    return_edge_property_t{},
                                                    tmp.begin());

                        result_properties.push_back(arithmetic_device_uvector_t{std::move(tmp)});
                      }
                    });
                });

  return result_properties;
}

template std::vector<arithmetic_device_uvector_t> gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  cugraph::edge_bucket_t<int32_t, int32_t, true, false, false>& edge_list,
  raft::host_span<edge_arithmetic_property_view_t<int32_t>> edge_property_views);

template std::vector<arithmetic_device_uvector_t> gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  cugraph::edge_bucket_t<int64_t, int64_t, true, false, false>& edge_list,
  raft::host_span<edge_arithmetic_property_view_t<int64_t>> edge_property_views);

template std::vector<arithmetic_device_uvector_t> gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  cugraph::edge_bucket_t<int32_t, int32_t, true, true, false>& edge_list,
  raft::host_span<edge_arithmetic_property_view_t<int32_t>> edge_property_views);

template std::vector<arithmetic_device_uvector_t> gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  cugraph::edge_bucket_t<int64_t, int64_t, true, true, false>& edge_list,
  raft::host_span<edge_arithmetic_property_view_t<int64_t>> edge_property_views);

}  // namespace detail
}  // namespace cugraph
