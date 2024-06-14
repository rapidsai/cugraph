/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "macros.hpp"

#include <cstdint>

namespace cugraph::ops::utils {

/** number of threads per warp */
static constexpr int WARP_SIZE = 32;

/** max number of threads in a block */
static constexpr int MAX_TPB = 1024;

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

/**
 * @brief Provide an alignment function ie. (a / b) * b
 *
 * @tparam IntT supposed to be only integers for now!
 *
 * @param[in] a dividend
 * @param[in] b divisor
 */
template <typename IntT>
constexpr CUGRAPH_OPS_HD IntT align_down(IntT a, IntT b)
{
  return (a / b) * b;
}

/**
 * @brief Check if the input is a power of 2
 *
 * @tparam IntT data type (checked only for strictly positive integers)
 *
 * @param[in] num input
 */
template <typename IntT>
constexpr CUGRAPH_OPS_HD bool is_po2(IntT num)
{
  return (num && !(num & (num - IntT{1})));
}

// recursion is fine in `constexpr` function
// NOLINTBEGIN(misc-no-recursion)
/**
 * @brief Give logarithm of the number to base-2
 *
 * @tparam IntT data type (checked only for strictly positive integers)
 *
 * @param[in] num input number
 * @param[in] ret value returned during each recursion
 */
template <typename IntT>
constexpr CUGRAPH_OPS_HD IntT log2(IntT num, IntT ret = IntT{0})
{
  return num <= IntT{1} ? ret : log2(num >> IntT{1}, ++ret);
}
// NOLINTEND(misc-no-recursion)

/**
 * @brief Runtime-only version of `log2`, see those docs
 *
 * @tparam IntT data type (checked only for strictly positive integers, <= 8 bytes)
 *
 * @param[in] num input number
 */
template <typename IntT>
CUGRAPH_OPS_HD inline IntT log2_rt(IntT num)
{
#if defined(__CUDA_ARCH__)
  if (sizeof(IntT) == 8) return 63 - __clzll(static_cast<int64_t>(num));
  // anything else is up-casted (or unchanged) to `int`
  return 31 - __clz(static_cast<int>(num));
#elif defined(__GNUC__) || defined(__GNUG__) || defined(__ICL) || defined(__clang__)
  if (sizeof(IntT) == 8) return 63 - __builtin_clzll(static_cast<uint64_t>(num));
  // anything else is up-casted (or unchanged) to `uint32_t`
  return 31 - __builtin_clz(static_cast<uint32_t>(num));
#else
  IntT out = 0, x = num;
  while (x >>= 1)
    ++out;
  return out;
#endif
}

/**
 * @brief Computes the number of bits needed to express the input number
 *
 * @tparam IntT data type (supports only strictly positive integers)
 *
 * @param[in] num input number
 */
template <typename IntT>
constexpr CUGRAPH_OPS_HD IntT num_bits(IntT num)
{
  return log2<IntT>(num) + !is_po2<IntT>(num);
}

/**
 * @brief Runtime-only version of `num_bits`, see those docs
 *
 * @tparam IntT data type (supports only strictly positive integers, <= 8 bytes)
 *
 * @param[in] num input number
 */
template <typename IntT>
CUGRAPH_OPS_HD inline IntT num_bits_rt(IntT num)
{
  return log2_rt<IntT>(num) + !is_po2<IntT>(num);
}

/**
 * @brief Computes the next closest power-of-2 of the input number
 *
 * @tparam IntT data type (supports only strictly positive integers)
 *
 * @param[in] num input number
 */
template <typename IntT>
constexpr CUGRAPH_OPS_HD IntT next_po2(IntT num)
{
  auto bits = num_bits<IntT>(num);
  return IntT{1} << bits;
}

}  // namespace cugraph::ops::utils
