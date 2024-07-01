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

// Andrei Schaffer, aschaffer@nvidia.com
//
#include "random_walks.cuh"

namespace cugraph {
// template explicit instantiation directives (EIDir's):
//
// SG FP32{
template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>, rmm::device_uvector<int32_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int32_t, int32_t, false, false> const& gview,
               std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
               int32_t const* ptr_d_start,
               int32_t num_paths,
               int32_t max_depth,
               bool use_padding,
               std::unique_ptr<sampling_params_t> sampling_strategy);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>, rmm::device_uvector<int64_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int32_t, int64_t, false, false> const& gview,
               std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
               int32_t const* ptr_d_start,
               int64_t num_paths,
               int64_t max_depth,
               bool use_padding,
               std::unique_ptr<sampling_params_t> sampling_strategy);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>, rmm::device_uvector<int64_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int64_t, int64_t, false, false> const& gview,
               std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
               int64_t const* ptr_d_start,
               int64_t num_paths,
               int64_t max_depth,
               bool use_padding,
               std::unique_ptr<sampling_params_t> sampling_strategy);
//}
//
// SG FP64{
template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>, rmm::device_uvector<int32_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int32_t, int32_t, false, false> const& gview,
               std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
               int32_t const* ptr_d_start,
               int32_t num_paths,
               int32_t max_depth,
               bool use_padding,
               std::unique_ptr<sampling_params_t> sampling_strategy);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>, rmm::device_uvector<int64_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int32_t, int64_t, false, false> const& gview,
               std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
               int32_t const* ptr_d_start,
               int64_t num_paths,
               int64_t max_depth,
               bool use_padding,
               std::unique_ptr<sampling_params_t> sampling_strategy);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>, rmm::device_uvector<int64_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<int64_t, int64_t, false, false> const& gview,
               std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
               int64_t const* ptr_d_start,
               int64_t num_paths,
               int64_t max_depth,
               bool use_padding,
               std::unique_ptr<sampling_params_t> sampling_strategy);
//}

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
  convert_paths_to_coo(raft::handle_t const& handle,
                       int32_t coalesced_sz_v,
                       int32_t num_paths,
                       rmm::device_buffer&& d_coalesced_v,
                       rmm::device_buffer&& d_sizes);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<int64_t>>
  convert_paths_to_coo(raft::handle_t const& handle,
                       int64_t coalesced_sz_v,
                       int64_t num_paths,
                       rmm::device_buffer&& d_coalesced_v,
                       rmm::device_buffer&& d_sizes);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
  convert_paths_to_coo(raft::handle_t const& handle,
                       int64_t coalesced_sz_v,
                       int64_t num_paths,
                       rmm::device_buffer&& d_coalesced_v,
                       rmm::device_buffer&& d_sizes);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>>
query_rw_sizes_offsets(raft::handle_t const& handle, int32_t num_paths, int32_t const* ptr_d_sizes);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>>
query_rw_sizes_offsets(raft::handle_t const& handle, int64_t num_paths, int64_t const* ptr_d_sizes);

}  // namespace cugraph
