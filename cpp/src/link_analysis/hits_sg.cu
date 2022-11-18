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

#include <link_analysis/hits_impl.cuh>

namespace cugraph {

// SG instantiation
template std::tuple<float, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  float* const hubs,
  float* const authorities,
  float epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

template std::tuple<double, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  double* const hubs,
  double* const authorities,
  double epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

template std::tuple<float, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, false> const& graph_view,
  float* const hubs,
  float* const authorities,
  float epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

template std::tuple<double, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, false> const& graph_view,
  double* const hubs,
  double* const authorities,
  double epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

template std::tuple<float, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  float* const hubs,
  float* const authorities,
  float epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

template std::tuple<double, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  double* const hubs,
  double* const authorities,
  double epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

}  // namespace cugraph
