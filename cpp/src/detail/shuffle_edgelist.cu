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
#include <detail/shuffle_wrappers_impl.cuh>

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<int32_t>&& d_edgelist_majors,
                           rmm::device_uvector<int32_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<float>>&& d_edgelist_weights);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<int32_t>&& d_edgelist_majors,
                           rmm::device_uvector<int32_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<double>>&& d_edgelist_weights);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<int64_t>&& d_edgelist_majors,
                           rmm::device_uvector<int64_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<float>>&& d_edgelist_weights);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<int64_t>&& d_edgelist_majors,
                           rmm::device_uvector<int64_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<double>>&& d_edgelist_weights);

}  // namespace detail
}  // namespace cugraph
