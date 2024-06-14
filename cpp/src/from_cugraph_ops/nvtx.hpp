/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

namespace cugraph::ops::utils {

/** at runtime disable the creation of nvtx ranges */
void disable_nvtx_ranges();

/** at runtime enable the creation of nvtx ranges (default is, disabled) */
void enable_nvtx_ranges();

/**
 * @brief Push a named nvtx range
 * @param name range name
 */
void push_range(const char* name);

/** Pop the latest range */
void pop_range();

}  // namespace cugraph::ops::utils
