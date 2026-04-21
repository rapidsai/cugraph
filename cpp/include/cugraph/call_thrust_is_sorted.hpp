/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <cstddef>

namespace cugraph {

void call_thrust_is_sorted(raft::handle_t const& handle,
                           raft::device_span<size_t const> subgraph_offsets);

}  // namespace cugraph
