/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/relabel_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT void relabel<int64_t, true>(
  raft::handle_t const& handle,
  std::tuple<int64_t const*, int64_t const*> old_new_label_pairs,
  int64_t num_label_pairs,
  int64_t* labels,
  int64_t num_labels,
  bool skip_missing_labels,
  bool do_expensive_check);

}  // namespace cugraph
