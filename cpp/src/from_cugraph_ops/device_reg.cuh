/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "casts.cuh"
#include "macros.hpp"

namespace cugraph::ops::utils {

/**
 * @brief helper method to fill a set of registers with a constant
 *
 * @tparam DataT  data type
 * @tparam VECLEN number of registers to be filled
 *
 * @param[out] vec the vector to be filled
 * @param[in]  val the value to be filled
 */
template <typename DataT, int VECLEN>
__device__ inline void fill_reg(DataT (&vec)[VECLEN], DataT val)
{
  CUGRAPH_OPS_UNROLL_N(VECLEN)
  for (int i = 0; i < VECLEN; ++i) {
    vec[i] = val;
  }
}
template <typename DataT, int VECLEN>
__device__ inline void fill_reg(DataT (&vec)[VECLEN], DataT* val)
{
  CUGRAPH_OPS_UNROLL_N(VECLEN)
  for (int i = 0; i < VECLEN; ++i) {
    vec[i] = val[i];
  }
}

/**
 * @brief helper to copy register array  for the purpose of casting
 *
 * @tparam DataT  data type
 * @tparam VECLEN number of registers in both arrays
 *
 * @param[out] out_vec output array to be filled
 * @param[in]  in_vec  input array to copy values from
 */
template <int VECLEN, typename OutDataT, typename InDataT>
__device__ __forceinline__ void cast_reg(OutDataT* out_vec, const InDataT* in_vec)
{
  CUGRAPH_OPS_UNROLL_N(VECLEN)
  for (int j = 0; j < VECLEN; ++j) {
    cast(out_vec[j], in_vec[j]);
  }
}
template <typename OutDataT, typename InDataT>
__device__ __forceinline__ void cast_reg(OutDataT& out_vec, const InDataT& in_vec)
{
  cast(out_vec, in_vec);
}

/**
 * @brief helper to copy register array from a larger array to a smaller one
 *
 * @tparam DataT  data type
 * @tparam OUTLEN number of registers in the output array
 * @tparam INLEN  number of registers in the input array
 *
 * @param[out] out_vec output array to be filled
 * @param[in]  in_vec  input array to copy values from
 * @param[in]  i       offset within input array where to begin copying
 */
template <typename OutDataT, typename InDataT, int OUTLEN, int INLEN>
__device__ __forceinline__ void copy_reg(OutDataT (&out_vec)[OUTLEN],
                                         const InDataT (&in_vec)[INLEN],
                                         int i)
{
  CUGRAPH_OPS_UNROLL_N(OUTLEN)
  for (int j = 0; j < OUTLEN; ++j) {
    cast(out_vec[j], in_vec[i + j]);
  }
}

template <typename DataT, int OUTLEN, int INLEN>
__device__ __forceinline__ void copy_reg(DataT (&out_vec)[OUTLEN],
                                         const DataT (&in_vec)[INLEN],
                                         int i)
{
  CUGRAPH_OPS_UNROLL_N(OUTLEN)
  for (int j = 0; j < OUTLEN; ++j) {
    out_vec[j] = in_vec[i + j];
  }
}

}  // namespace cugraph::ops::utils
