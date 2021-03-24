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

// Andrei Schaffer, aschaffer@nvidia.com
//
#include <algorithms.hpp>
#include <experimental/random_walks.cuh>

namespace cugraph {
namespace experimental {
// template explicit instantiation directives (EIDir's):
//
// SG FP32{
template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>, rmm::device_uvector<int32_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int32_t, int32_t, float, false, false> const& gview,
               rmm::device_uvector<int32_t> const& d_start,
               int32_t max_depth);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>, rmm::device_uvector<int64_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int32_t, int64_t, float, false, false> const& gview,
               rmm::device_uvector<int32_t> const& d_start,
               int64_t max_depth);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>, rmm::device_uvector<int64_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int64_t, int64_t, float, false, false> const& gview,
               rmm::device_uvector<int64_t> const& d_start,
               int64_t max_depth);
//}
//
// SG FP64{
template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>, rmm::device_uvector<int32_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int32_t, int32_t, double, false, false> const& gview,
               rmm::device_uvector<int32_t> const& d_start,
               int32_t max_depth);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>, rmm::device_uvector<int64_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int32_t, int64_t, double, false, false> const& gview,
               rmm::device_uvector<int32_t> const& d_start,
               int64_t max_depth);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>, rmm::device_uvector<int64_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int64_t, int64_t, double, false, false> const& gview,
               rmm::device_uvector<int64_t> const& d_start,
               int64_t max_depth);
//}
}  // namespace experimental
}  // namespace cugraph
