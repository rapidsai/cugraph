/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace c_api {

template <typename vertex_t, typename edge_t>
rmm::device_uvector<vertex_t> expand_sparse_offsets(raft::device_span<edge_t const> offsets,
                                                    vertex_t base_vertex_id,
                                                    rmm::cuda_stream_view const& stream);

}  // namespace c_api
}  // namespace cugraph
