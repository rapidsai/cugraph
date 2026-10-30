/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/flatten_dendrogram_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

template CUGRAPH_EXPORT vertex_property_t<int64_t, int64_t> flatten_dendrogram(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, true> const&,
  dendrogram_t<int64_t> const&,
  bool,
  bool);
template CUGRAPH_EXPORT void flatten_dendrogram(raft::handle_t const&,
                                                graph_view_t<int64_t, int64_t, false, true> const&,
                                                dendrogram_t<int64_t> const&,
                                                int64_t*);

}  // namespace cugraph
