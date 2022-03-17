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

#include <cugraph/algorithms.hpp>

#include "nbr_sampling_impl.cuh"

namespace cugraph {
// template explicit instantiation directives (EIDir's):
//
// SG FP32{
template std::tuple<std::tuple<rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
uniform_nbr_sample(raft::handle_t const& handle,
                   graph_view_t<int32_t, int32_t, float, false, true> const& gview,
                   int32_t const* ptr_d_start,
                   int32_t const* ptr_d_ranks,
                   size_t num_starting_vs,
                   std::vector<int> const& h_fan_out,
                   bool flag_replacement);

template std::tuple<std::tuple<rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int64_t>>,
                    std::vector<size_t>>
uniform_nbr_sample(raft::handle_t const& handle,
                   graph_view_t<int32_t, int64_t, float, false, true> const& gview,
                   int32_t const* ptr_d_start,
                   int32_t const* ptr_d_ranks,
                   size_t num_starting_vs,
                   std::vector<int> const& h_fan_out,
                   bool flag_replacement);

template std::tuple<std::tuple<rmm::device_uvector<int64_t>,
                               rmm::device_uvector<int64_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int64_t>>,
                    std::vector<size_t>>
uniform_nbr_sample(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, float, false, true> const& gview,
                   int64_t const* ptr_d_start,
                   int32_t const* ptr_d_ranks,
                   size_t num_starting_vs,
                   std::vector<int> const& h_fan_out,
                   bool flag_replacement);
//}
//
// SG FP64{
template std::tuple<std::tuple<rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
uniform_nbr_sample(raft::handle_t const& handle,
                   graph_view_t<int32_t, int32_t, double, false, true> const& gview,
                   int32_t const* ptr_d_start,
                   int32_t const* ptr_d_ranks,
                   size_t num_starting_vs,
                   std::vector<int> const& h_fan_out,
                   bool flag_replacement);

template std::tuple<std::tuple<rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int64_t>>,
                    std::vector<size_t>>
uniform_nbr_sample(raft::handle_t const& handle,
                   graph_view_t<int32_t, int64_t, double, false, true> const& gview,
                   int32_t const* ptr_d_start,
                   int32_t const* ptr_d_ranks,
                   size_t num_starting_vs,
                   std::vector<int> const& h_fan_out,
                   bool flag_replacement);

template std::tuple<std::tuple<rmm::device_uvector<int64_t>,
                               rmm::device_uvector<int64_t>,
                               rmm::device_uvector<int32_t>,
                               rmm::device_uvector<int64_t>>,
                    std::vector<size_t>>
uniform_nbr_sample(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, double, false, true> const& gview,
                   int64_t const* ptr_d_start,
                   int32_t const* ptr_d_ranks,
                   size_t num_starting_vs,
                   std::vector<int> const& h_fan_out,
                   bool flag_replacement);
//}

}  // namespace cugraph
