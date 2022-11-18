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

#include <cores/core_number_impl.cuh>

namespace cugraph {

// MG instantiation

template void core_number(raft::handle_t const& handle,
                          graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                          int32_t* core_numbers,
                          k_core_degree_type_t degree_type,
                          size_t k_first,
                          size_t k_last,
                          bool do_expensive_check);

template void core_number(raft::handle_t const& handle,
                          graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                          int64_t* core_numbers,
                          k_core_degree_type_t degree_type,
                          size_t k_first,
                          size_t k_last,
                          bool do_expensive_check);

template void core_number(raft::handle_t const& handle,
                          graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                          int64_t* core_numbers,
                          k_core_degree_type_t degree_type,
                          size_t k_first,
                          size_t k_last,
                          bool do_expensive_check);

}  // namespace cugraph
