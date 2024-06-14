/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace cugraph::ops::utils {

void __device__ __forceinline__ cast(float& out, const float& in) { out = in; }
void __device__ __forceinline__ cast(double& out, const double& in) { out = in; }
void __device__ __forceinline__ cast(int16_t& out, const int16_t& in) { out = in; }
void __device__ __forceinline__ cast(int32_t& out, const int32_t& in) { out = in; }
void __device__ __forceinline__ cast(int64_t& out, const int64_t& in) { out = in; }
void __device__ __forceinline__ cast(uint16_t& out, const uint16_t& in) { out = in; }
void __device__ __forceinline__ cast(uint32_t& out, const uint32_t& in) { out = in; }
void __device__ __forceinline__ cast(uint64_t& out, const uint64_t& in) { out = in; }

#if __CUDA_ARCH__ >= 700
void __device__ __forceinline__ cast(__half& out, const __half& in) { out = in; }
void __device__ __forceinline__ cast(float& out, const __half& in) { out = __half2float(in); }
void __device__ __forceinline__ cast(__half& out, const float& in) { out = __float2half(in); }
#endif

#if __CUDA_ARCH__ >= 800
void __device__ __forceinline__ cast(__nv_bfloat16& out, const __nv_bfloat16& in) { out = in; }
void __device__ __forceinline__ cast(float& out, const __nv_bfloat16& in)
{
  out = __bfloat162float(in);
}
void __device__ __forceinline__ cast(__nv_bfloat16& out, const float& in)
{
  out = __float2bfloat16(in);
}
#endif

}  // namespace cugraph::ops::utils
