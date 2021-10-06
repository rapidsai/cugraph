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

#include <structure/relabel_impl.cuh>

namespace cugraph {

// SG instantiation

template void relabel<int32_t, false>(
  raft::handle_t const& handle,
  std::tuple<int32_t const*, int32_t const*> old_new_label_pairs,
  int32_t num_label_pairs,
  int32_t* labels,
  int32_t num_labels,
  bool skip_missing_labels,
  bool do_expensive_check);

template void relabel<int64_t, false>(
  raft::handle_t const& handle,
  std::tuple<int64_t const*, int64_t const*> old_new_label_pairs,
  int64_t num_label_pairs,
  int64_t* labels,
  int64_t num_labels,
  bool skip_missing_labels,
  bool do_expensive_check);

}  // namespace cugraph
