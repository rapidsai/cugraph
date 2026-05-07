/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <raft/core/device_span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

template <typename key_t>
void launch_sort(raft::device_span<key_t> keys, rmm::cuda_stream_view stream);

template <typename key_t>
void launch_stable_sort(raft::device_span<key_t> keys, rmm::cuda_stream_view stream);

template <typename key_t, typename value_t>
void launch_sort_by_key(raft::device_span<key_t> keys,
                        raft::device_span<value_t> values,
                        rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph
