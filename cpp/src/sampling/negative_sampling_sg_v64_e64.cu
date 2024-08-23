/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "negative_sampling_impl.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/sampling_functions.hpp>

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> negative_sampling(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<raft::device_span<float const>> src_bias,
  std::optional<raft::device_span<float const>> dst_bias,
  size_t num_samples,
  bool remove_duplicates,
  bool remove_existing_edges,
  bool exact_number_of_samples,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> negative_sampling(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<raft::device_span<double const>> src_bias,
  std::optional<raft::device_span<double const>> dst_bias,
  size_t num_samples,
  bool remove_duplicates,
  bool remove_existing_edges,
  bool exact_number_of_samples,
  bool do_expensive_check);

}  // namespace cugraph
