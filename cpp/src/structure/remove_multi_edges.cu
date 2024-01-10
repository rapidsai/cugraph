/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <structure/remove_multi_edges_impl.cuh>

namespace cugraph {

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int32_t>&& edgelist_srcs,
                   rmm::device_uvector<int32_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<float>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   bool keep_min_value_edge);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int32_t>&& edgelist_srcs,
                   rmm::device_uvector<int32_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<float>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   bool keep_min_value_edge);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int64_t>&& edgelist_srcs,
                   rmm::device_uvector<int64_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<float>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   bool keep_min_value_edge);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int32_t>&& edgelist_srcs,
                   rmm::device_uvector<int32_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<double>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   bool keep_min_value_edge);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int32_t>&& edgelist_srcs,
                   rmm::device_uvector<int32_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<double>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   bool keep_min_value_edge);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
remove_multi_edges(raft::handle_t const& handle,
                   rmm::device_uvector<int64_t>&& edgelist_srcs,
                   rmm::device_uvector<int64_t>&& edgelist_dsts,
                   std::optional<rmm::device_uvector<double>>&& edgelist_weights,
                   std::optional<rmm::device_uvector<int64_t>>&& edgelist_edge_ids,
                   std::optional<rmm::device_uvector<int32_t>>&& edgelist_edge_types,
                   bool keep_min_value_edge);

}  // namespace cugraph
