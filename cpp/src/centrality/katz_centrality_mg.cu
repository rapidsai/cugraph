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
#include <centrality/katz_centrality_impl.cuh>

namespace cugraph {

// MG instantiation

template void katz_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  float const* betas,
  float* katz_centralities,
  float alpha,
  float beta,
  float epsilon,
  size_t max_iterations,
  bool has_initial_guess,
  bool normalize,
  bool do_expensive_check);

template void katz_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  double const* betas,
  double* katz_centralities,
  double alpha,
  double beta,
  double epsilon,
  size_t max_iterations,
  bool has_initial_guess,
  bool normalize,
  bool do_expensive_check);

template void katz_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  float const* betas,
  float* katz_centralities,
  float alpha,
  float beta,
  float epsilon,
  size_t max_iterations,
  bool has_initial_guess,
  bool normalize,
  bool do_expensive_check);

template void katz_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  double const* betas,
  double* katz_centralities,
  double alpha,
  double beta,
  double epsilon,
  size_t max_iterations,
  bool has_initial_guess,
  bool normalize,
  bool do_expensive_check);

template void katz_centrality(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  float const* betas,
  float* katz_centralities,
  float alpha,
  float beta,
  float epsilon,
  size_t max_iterations,
  bool has_initial_guess,
  bool normalize,
  bool do_expensive_check);

template void katz_centrality(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  double const* betas,
  double* katz_centralities,
  double alpha,
  double beta,
  double epsilon,
  size_t max_iterations,
  bool has_initial_guess,
  bool normalize,
  bool do_expensive_check);

}  // namespace cugraph
