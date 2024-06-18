/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "macros.hpp"

namespace cugraph::ops::utils {

/** number of threads per warp */
static constexpr int WARP_SIZE = 32;

/** minimum CUDA version required for warp shfl sync functions */
static constexpr int CUDA_VER_WARP_SHFL = 9000;

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 *
 * @tparam IntT supposed to be only integers for now!
 *
 * @param[in] a dividend
 * @param[in] b divisor
 */
template <typename IntT>
constexpr CUGRAPH_OPS_HD IntT ceil_div(IntT a, IntT b)
{
  return (a + b - 1) / b;
}

/**
 * @brief Provide an alignment function ie. ceil(a / b) * b
 *
 * @tparam IntT supposed to be only integers for now!
 *
 * @param[in] a dividend
 * @param[in] b divisor
 */
template <typename IntT>
constexpr CUGRAPH_OPS_HD IntT align_to(IntT a, IntT b)
{
  return ceil_div(a, b) * b;
}

}  // namespace cugraph::ops::utils
