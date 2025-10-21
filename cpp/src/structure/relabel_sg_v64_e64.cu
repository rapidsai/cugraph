/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/relabel_impl.cuh"

namespace cugraph {

// SG instantiation

template void relabel<int64_t, false>(
  raft::handle_t const& handle,
  std::tuple<int64_t const*, int64_t const*> old_new_label_pairs,
  int64_t num_label_pairs,
  int64_t* labels,
  int64_t num_labels,
  bool skip_missing_labels,
  bool do_expensive_check);

}  // namespace cugraph
