/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <string>

namespace cugraph::ops::utils {

__device__ constexpr int sm_arch()
{
#ifdef __CUDA_ARCH__
  return __CUDA_ARCH__;
#else
  return -1;
#endif
}

template <typename DataT>
__device__ constexpr bool valid_data_type_for_arch()
{
  if (std::is_same<DataT, __half>::value && (sm_arch() < 700)) { return false; }
  if (std::is_same<DataT, __nv_bfloat16>::value && (sm_arch() < 800)) { return false; }
  return true;
}

template <typename DataT>
__host__ __device__ constexpr int32_t get_native_veclen()
{
  // simplified alignment checks: use VECLEN for packed types of reduced precision
  // otherwise 1 (e.g. FP16 would have up to 2 FP8 could have up to 4)
  // we usually already use rather many registers and don't have too many dimensions
  // to iterative over to begin with, so this should simplify things for now
  if (static_cast<int32_t>(sizeof(DataT)) >= 4) { return int32_t{1}; }
  return static_cast<int32_t>(4 / static_cast<int32_t>(sizeof(DataT)));
}

template <typename DataT>
void copy(DataT* out, const DataT* in, size_t len, cudaMemcpyKind kind)
{
  RAFT_CUDA_TRY(cudaMemcpy(out, in, sizeof(DataT) * len, kind));
}

template <typename DataT>
void copy_async(DataT* out, const DataT* in, size_t len, cudaMemcpyKind kind, cudaStream_t stream)
{
  RAFT_CUDA_TRY(cudaMemcpyAsync(out, in, sizeof(DataT) * len, kind, stream));
}

template <typename DataT>
void copy_async_no_throw(
  DataT* out, const DataT* in, size_t len, cudaMemcpyKind kind, cudaStream_t stream) noexcept
{
  RAFT_CUDA_TRY_NO_THROW(cudaMemcpyAsync(out, in, sizeof(DataT) * len, kind, stream));
}

inline void sync(cudaStream_t stream = nullptr) { RAFT_CUDA_TRY(cudaStreamSynchronize(stream)); }

inline void sync_no_throw(cudaStream_t stream = nullptr) noexcept
{
  RAFT_CUDA_TRY_NO_THROW(cudaStreamSynchronize(stream));
}

template <typename DataT>
inline void memset(DataT* out, size_t len, uint8_t byte_value = 0)
{
  RAFT_CUDA_TRY(cudaMemset(out, byte_value, len * sizeof(DataT)));
}

template <typename DataT>
inline void memset_async(DataT* out, size_t len, cudaStream_t stream, uint8_t byte_value = 0)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(out, byte_value, len * sizeof(DataT), stream));
}

template <typename DataT>
inline void memset_async_no_throw(DataT* out,
                                  size_t len,
                                  cudaStream_t stream,
                                  uint8_t byte_value = 0) noexcept
{
  RAFT_CUDA_TRY_NO_THROW(cudaMemsetAsync(out, byte_value, len * sizeof(DataT), stream));
}

inline int get_sm_count()
{
  int dev_id;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  int mp_count;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, dev_id));
  return mp_count;
}

template <class FuncT>
inline int get_max_blocks_per_sm(FuncT func, int block_size, size_t dynamic_smem_size)
{
  int nblks;
  RAFT_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblks, func, block_size, dynamic_smem_size));
  return nblks;
}

template <class FuncT>
inline int get_max_grid_blocks(FuncT func, int block_size, size_t dynamic_smem_size)
{
  return get_max_blocks_per_sm(func, block_size, dynamic_smem_size) * get_sm_count();
}

template <class FuncT>
inline size_t get_static_smem_size(FuncT func)
{
  cudaFuncAttributes attrs;
  RAFT_CUDA_TRY(cudaFuncGetAttributes(&attrs, func));
  return attrs.sharedSizeBytes;
}

inline void get_max_smem_sizes(size_t& smem_block, size_t& smem_optin)
{
  int dev_id, smem_blk, smem_max;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&smem_blk, cudaDevAttrMaxSharedMemoryPerBlock, dev_id));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
  smem_block = static_cast<size_t>(smem_blk);
  smem_optin = static_cast<size_t>(smem_max);
}

inline int get_max_smem_per_sm()
{
  int dev_id;
  cudaGetDevice(&dev_id);
  int available_smem;
  cudaDeviceGetAttribute(&available_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id);
  return available_smem;
}

inline int get_max_smem_per_block()
{
  int dev_id;
  cudaGetDevice(&dev_id);
  int available_smem;
  cudaDeviceGetAttribute(&available_smem, cudaDevAttrMaxSharedMemoryPerBlock, dev_id);
  return available_smem;
}

template <typename FuncT>
inline void set_smem_optin(int32_t required_size, FuncT func)
{
  // opt-in with actual required size
  RAFT_CUDA_TRY(
    cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, required_size));
}

template <typename DataT, int ALIGN>
inline bool is_aligned(std::initializer_list<const void*> ptrs,
                       std::initializer_list<size_t> sizes = {})
{
  bool ret = ALIGN % sizeof(DataT) == 0;
  for (const auto* p : ptrs)
    ret = ret && reinterpret_cast<uintptr_t>(p) % ALIGN == 0;
  for (auto s : sizes)
    ret = ret && (s * sizeof(DataT)) % ALIGN == 0;
  return ret;
}

template <typename DataT>
void print_array(
  const DataT* a, size_t len, const std::string& name, bool is_host = false, int precision = 0)
{
  auto ss      = std::cout.precision();
  auto s_flags = std::cout.flags();
  if (precision > 0) { std::cout << std::fixed << std::setprecision(precision); }
  std::cout << "Array '" << name << "' [size " << len << "]: ";
  if (len <= 0) std::cout << "No elements!" << std::endl;
  DataT* h_d2a = nullptr;
  const DataT* h_a;
  if (!is_host) {
    host_alloc(h_d2a, len, nullptr);
    copy(h_d2a, a, len, cudaMemcpyDeviceToHost);
    h_a = h_d2a;
  } else {
    h_a = a;
  }
  std::cout << h_a[0];
  for (size_t i = 1; i < len; ++i)
    std::cout << ", " << h_a[i];
  std::cout << std::endl;
  if (!is_host) host_free(h_d2a, nullptr);
  if (precision > 0) {
    std::cout << std::setprecision(ss);
    std::cout.flags(s_flags);
  }
}

}  // namespace cugraph::ops::utils
