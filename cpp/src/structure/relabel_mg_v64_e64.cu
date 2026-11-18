/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template CUGRAPH_EXPORT rmm::device_uvector<int64_t> relabel<int64_t, true>(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> old_labels,
  raft::device_span<int64_t const> new_labels,
  rmm::device_uvector<int64_t>&& labels,
  std::optional<raft::host_span<int64_t const>> vertex_partition_range_lasts,
  bool skip_missing_labels,
  bool do_expensive_check);

}  // namespace cugraph
