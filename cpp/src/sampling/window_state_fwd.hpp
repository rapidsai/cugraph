/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file window_state_fwd.hpp
 * @brief Forward declaration of window_state_t for use in non-CUDA compilation units
 *
 * This header provides a forward declaration of window_state_t that can be included
 * in .cpp files without pulling in CUDA dependencies.
 */

#include <cugraph/utilities/packed_bool_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>

namespace cugraph {
namespace detail {

/**
 * @brief State for incremental window updates (Optimization C)
 *
 * Maintains sorted edge indices and current window bounds for efficient
 * incremental mask updates when sliding the window.
 */
template <typename edge_t, typename time_stamp_t>
struct window_state_t {
  rmm::device_uvector<edge_t> sorted_edge_indices;
  rmm::device_uvector<time_stamp_t> sorted_edge_times;
  // Packed edge mask (uint32 words) persisted across calls to enable O(ΔE) updates (Optimization C)
  rmm::device_uvector<uint32_t> edge_mask_words;
  size_t current_start_idx{0};
  size_t current_end_idx{0};
  bool initialized{false};

  window_state_t(rmm::cuda_stream_view stream)
    : sorted_edge_indices(0, stream), sorted_edge_times(0, stream), edge_mask_words(0, stream)
  {
  }

  void ensure_edge_mask_size(edge_t num_edges, rmm::cuda_stream_view stream)
  {
    auto required_words =
      static_cast<size_t>(cugraph::packed_bool_size(static_cast<size_t>(num_edges)));
    if (edge_mask_words.size() != required_words) {
      edge_mask_words.resize(required_words, stream);
    }
  }
};

}  // namespace detail
}  // namespace cugraph
