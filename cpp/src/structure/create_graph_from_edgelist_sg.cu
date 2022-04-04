/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <structure/create_graph_from_edgelist_impl.cuh>

namespace cugraph {

// explicit instantiations

template std::tuple<cugraph::graph_t<int32_t, int32_t, float, false, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int32_t, int32_t, float, true, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int32_t, int32_t, double, false, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int32_t, int32_t, double, true, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int32_t, int64_t, float, false, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int32_t, int64_t, float, true, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int32_t, int64_t, double, false, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int32_t, int64_t, double, true, false>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int64_t, int64_t, float, false, false>,
                    std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int64_t, int64_t, float, true, false>,
                    std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int64_t, int64_t, double, false, false>,
                    std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int64_t, int64_t, double, true, false>,
                    std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

}  // namespace cugraph
