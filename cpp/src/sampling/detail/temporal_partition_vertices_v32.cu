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

#include "temporal_partition_vertices_impl.cuh"

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
temporal_partition_vertices(raft::handle_t const& handle,
                            raft::device_span<int32_t const> vertices,
                            raft::device_span<int32_t const> vertex_times,
                            std::optional<raft::device_span<int32_t const>> vertex_labels);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
temporal_partition_vertices(raft::handle_t const& handle,
                            raft::device_span<int32_t const> vertices,
                            raft::device_span<int64_t const> vertex_times,
                            std::optional<raft::device_span<int32_t const>> vertex_labels);

}  // namespace detail
}  // namespace cugraph
