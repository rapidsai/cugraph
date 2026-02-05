/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/relabel_impl.cuh"

namespace cugraph {

// MG instantiation

template void relabel<int32_t, true>(raft::handle_t const& handle,
                                     std::tuple<int32_t const*, int32_t const*> old_new_label_pairs,
                                     int32_t num_label_pairs,
                                     int32_t* labels,
                                     int32_t num_labels,
                                     bool skip_missing_labels,
                                     bool do_expensive_check);

}  // namespace cugraph
