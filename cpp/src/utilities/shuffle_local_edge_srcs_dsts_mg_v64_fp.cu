/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "shuffle_local_edge_srcs_dsts.cuh"

namespace cugraph {

template
std::tuple<rmm::device_uvector<int64_t>, dataframe_buffer_type_t<float>>
shuffle_local_edge_src_value_pairs<int64_t, float>(raft::handle_t const& handle,
                                   rmm::device_uvector<int64_t>&& edge_srcs,
                                   dataframe_buffer_type_t<float>&& edge_values,
                                   raft::host_span<int64_t const> vertex_partition_range_lasts,
                                   bool store_transposed);

template
std::tuple<rmm::device_uvector<int64_t>, dataframe_buffer_type_t<double>>
shuffle_local_edge_src_value_pairs<int64_t, double>(raft::handle_t const& handle,
                                   rmm::device_uvector<int64_t>&& edge_srcs,
                                   dataframe_buffer_type_t<double>&& edge_values,
                                   raft::host_span<int64_t const> vertex_partition_range_lasts,
                                   bool store_transposed);

template
std::tuple<rmm::device_uvector<int64_t>, dataframe_buffer_type_t<float>>
shuffle_local_edge_dst_value_pairs<int64_t, float>(raft::handle_t const& handle,
                                   rmm::device_uvector<int64_t>&& edge_dsts,
                                   dataframe_buffer_type_t<float>&& edge_values,
                                   raft::host_span<int64_t const> vertex_partition_range_lasts,
                                   bool store_transposed);

template
std::tuple<rmm::device_uvector<int64_t>, dataframe_buffer_type_t<double>>
shuffle_local_edge_dst_value_pairs<int64_t, double>(raft::handle_t const& handle,
                                   rmm::device_uvector<int64_t>&& edge_dsts,
                                   dataframe_buffer_type_t<double>&& edge_values,
                                   raft::host_span<int64_t const> vertex_partition_range_lasts,
                                   bool store_transposed);

}  // namespace cugraph
