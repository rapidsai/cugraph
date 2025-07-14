/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "shuffle_vertex_pairs.cuh"

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<cugraph::arithmetic_device_uvector_t>,
                    std::vector<size_t>>
shuffle_ext_edges(raft::handle_t const& handle,
                  rmm::device_uvector<int64_t>&& edge_srcs,
                  rmm::device_uvector<int64_t>&& edge_dsts,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  std::optional<large_buffer_type_t> large_buffer_type);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<cugraph::arithmetic_device_uvector_t>,
                    std::vector<size_t>>
shuffle_int_edges(raft::handle_t const& handle,
                  rmm::device_uvector<int64_t>&& majors,
                  rmm::device_uvector<int64_t>&& minors,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  raft::host_span<int64_t const> vertex_partition_range_lasts,
                  std::optional<large_buffer_type_t> large_buffer_type);

}  // namespace cugraph
