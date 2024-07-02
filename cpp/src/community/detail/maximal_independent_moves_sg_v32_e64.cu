/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include "maximal_independent_moves.cuh"

namespace cugraph {
namespace detail {

template rmm::device_uvector<int32_t> maximal_independent_moves(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, false> const& decision_graph_view,
  raft::random::RngState& rng_state);

}  // namespace detail
}  // namespace cugraph
