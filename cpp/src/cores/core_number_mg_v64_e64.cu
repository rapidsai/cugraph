/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cores/core_number_impl.cuh"

namespace cugraph {

// MG instantiation

template void core_number(raft::handle_t const& handle,
                          graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                          int64_t* core_numbers,
                          k_core_degree_type_t degree_type,
                          size_t k_first,
                          size_t k_last,
                          bool do_expensive_check);

}  // namespace cugraph
