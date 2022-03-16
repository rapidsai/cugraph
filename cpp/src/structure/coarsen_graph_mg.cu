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
#include <structure/coarsen_graph_impl.cuh>

namespace cugraph {

// MG instantiation

template std::tuple<graph_t<int32_t, int32_t, float, true, true>, rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, float, true, true> const& graph_view,
              int32_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, float, false, true>, rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
              int32_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int64_t, float, true, true>, rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int64_t, float, true, true> const& graph_view,
              int32_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int64_t, float, false, true>, rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
              int32_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int64_t, int64_t, float, true, true>, rmm::device_uvector<int64_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, float, true, true> const& graph_view,
              int64_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int64_t, int64_t, float, false, true>, rmm::device_uvector<int64_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
              int64_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, double, true, true>, rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, double, true, true> const& graph_view,
              int32_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, double, false, true>, rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
              int32_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int64_t, double, true, true>, rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int64_t, double, true, true> const& graph_view,
              int32_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int64_t, double, false, true>, rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
              int32_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int64_t, int64_t, double, true, true>, rmm::device_uvector<int64_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, double, true, true> const& graph_view,
              int64_t const* labels,
              bool do_expensive_check);

template std::tuple<graph_t<int64_t, int64_t, double, false, true>, rmm::device_uvector<int64_t>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
              int64_t const* labels,
              bool do_expensive_check);

}  // namespace cugraph
