/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <structure/transpose_edgelist_impl.cuh>

namespace cugraph {

namespace detail {

// MG instantiation

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
transpose_edgelist<int32_t, float, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& edgelist_majors,
  rmm::device_uvector<int32_t>&& edgelist_minors,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
transpose_edgelist<int32_t, double, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& edgelist_majors,
  rmm::device_uvector<int32_t>&& edgelist_minors,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
transpose_edgelist<int64_t, float, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& edgelist_majors,
  rmm::device_uvector<int64_t>&& edgelist_minors,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
transpose_edgelist<int64_t, double, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& edgelist_majors,
  rmm::device_uvector<int64_t>&& edgelist_minors,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights);

}  // namespace detail

}  // namespace cugraph
