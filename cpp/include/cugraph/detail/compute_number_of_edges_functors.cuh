/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

namespace cugraph {
namespace detail {

// Converts a uint32_t bitmap offset to a vertex_t by adding range_first.
// Used in the bitmap code path of extract_transform_if_v_frontier_e.
template <typename vertex_t>
struct bitmap_offset_to_vertex_op_t {
  vertex_t range_first;
  __device__ vertex_t operator()(uint32_t v_offset) const
  {
    return range_first + static_cast<vertex_t>(v_offset);
  }
};

// Maps a linear index to either a sparse-range vertex or a hypersparse DCS vertex.
// Used in per_v_transform_reduce_dst_key_aggregated_outgoing_e.
template <typename vertex_t>
struct sparse_hypersparse_major_op_t {
  vertex_t major_sparse_range_size;
  vertex_t major_range_first;
  vertex_t const* dcs_nzd_vertices;
  __device__ vertex_t operator()(vertex_t i) const
  {
    if (i < major_sparse_range_size) { return major_range_first + i; }
    return *(dcs_nzd_vertices + (i - major_sparse_range_size));
  }
};

}  // namespace detail
}  // namespace cugraph
