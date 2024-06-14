/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "device_reg.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace cugraph::ops::utils {

enum class BufferType { kGlobal, kShared };

// st_shared is a wrapper around the st.shared PTX instruction.
// There is a template specialization for values of N=1, 2, 4, 8, 16.
template <int N>
__device__ inline void st_shared(void* addr, const uint8_t (&x)[N]);

template <>
__device__ inline void st_shared(void* addr, const uint8_t (&x)[1])
{
  using addr_t     = uint8_t;
  uint32_t x_store = x[0];
  auto s1          = __cvta_generic_to_shared(reinterpret_cast<addr_t*>(addr));
  asm volatile("st.shared.u8 [%0], {%1};" : : "l"(s1), "r"(x_store));
}

template <>
__device__ inline void st_shared(void* addr, const uint8_t (&x)[2])
{
  using addr_t   = uint16_t;
  addr_t x_store = *reinterpret_cast<const addr_t*>(x);
  auto s1        = __cvta_generic_to_shared(reinterpret_cast<addr_t*>(addr));
  asm volatile("st.shared.u16 [%0], {%1};" : : "l"(s1), "h"(x_store));
}

template <>
__device__ inline void st_shared(void* addr, const uint8_t (&x)[4])
{
  using addr_t    = uint32_t;
  using store_t   = uint32_t;
  store_t x_store = *reinterpret_cast<const store_t*>(x);
  auto s1         = __cvta_generic_to_shared(reinterpret_cast<addr_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x_store));
}

template <>
__device__ inline void st_shared(void* addr, const uint8_t (&x)[8])
{
  using store_t              = uint32_t;
  const store_t* x_store_arr = reinterpret_cast<const store_t*>(x);
  auto s1                    = __cvta_generic_to_shared(reinterpret_cast<void*>(addr));
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};"
               :
               : "l"(s1), "r"(x_store_arr[0]), "r"(x_store_arr[1]));
}

template <>
__device__ inline void st_shared(void* addr, const uint8_t (&x)[16])
{
  using store_t              = uint32_t;
  const store_t* x_store_arr = reinterpret_cast<const store_t*>(x);
  auto s1                    = __cvta_generic_to_shared(reinterpret_cast<void*>(addr));
  asm volatile(
    "st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
    :
    : "l"(s1), "r"(x_store_arr[0]), "r"(x_store_arr[1]), "r"(x_store_arr[2]), "r"(x_store_arr[3]));
}

// ld_shared is a wrapper around the ld.shared PTX instruction.
// There is a template specialization for values of N=1, 2, 4, 8, 16.
template <int N>
__device__ inline void ld_shared(uint8_t (&x)[N], const void* addr);

template <>
__device__ inline void ld_shared(uint8_t (&x)[1], const void* addr)
{
  using addr_t = uint8_t;
  uint32_t x_store;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const addr_t*>(addr));
  asm volatile("ld.shared.u8 {%0}, [%1];" : "=r"(x_store) : "l"(s1));
  x[0] = x_store;
}

template <>
__device__ inline void ld_shared(uint8_t (&x)[2], const void* addr)
{
  using addr_t    = uint16_t;
  addr_t* x_store = reinterpret_cast<addr_t*>(x);
  auto s1         = __cvta_generic_to_shared(reinterpret_cast<const addr_t*>(addr));
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(*x_store) : "l"(s1));
}

template <>
__device__ inline void ld_shared(uint8_t (&x)[4], const void* addr)
{
  using addr_t     = uint32_t;
  using store_t    = uint32_t;
  store_t* x_store = reinterpret_cast<store_t*>(x);
  auto s1          = __cvta_generic_to_shared(reinterpret_cast<const addr_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(*x_store) : "l"(s1));
}

template <>
__device__ inline void ld_shared(uint8_t (&x)[8], const void* addr)
{
  using store_t        = uint32_t;
  store_t* x_store_arr = reinterpret_cast<store_t*>(x);
  auto s1              = __cvta_generic_to_shared(addr);
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
               : "=r"(x_store_arr[0]), "=r"(x_store_arr[1])
               : "l"(s1));
}

template <>
__device__ inline void ld_shared(uint8_t (&x)[16], const void* addr)
{
  using store_t        = uint32_t;
  store_t* x_store_arr = reinterpret_cast<store_t*>(x);
  auto s1              = __cvta_generic_to_shared(addr);
  asm volatile(
    "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
    : "=r"(x_store_arr[0]), "=r"(x_store_arr[1]), "=r"(x_store_arr[2]), "=r"(x_store_arr[3])
    : "l"(s1));
}

// st_global is a wrapper around the st.global PTX instruction.
// There is a template specialization for values of N=1, 2, 4, 8, 16.
template <int N>
__device__ inline void st_global(void* addr, const uint8_t (&x)[N]);

template <>
__device__ inline void st_global(void* addr, const uint8_t (&x)[1])
{
  using addr_t     = uint8_t;
  uint32_t x_store = x[0];
  auto* s1         = reinterpret_cast<addr_t*>(addr);
  asm volatile("st.global.u8 [%0], {%1};" : : "l"(s1), "r"(x_store));
}

template <>
__device__ inline void st_global(void* addr, const uint8_t (&x)[2])
{
  using addr_t   = uint16_t;
  addr_t x_store = *reinterpret_cast<const addr_t*>(x);
  auto* s1       = reinterpret_cast<addr_t*>(addr);
  asm volatile("st.global.u16 [%0], {%1};" : : "l"(s1), "h"(x_store));
}

template <>
__device__ inline void st_global(void* addr, const uint8_t (&x)[4])
{
  using addr_t    = uint32_t;
  using store_t   = uint32_t;
  store_t x_store = *reinterpret_cast<const store_t*>(x);
  auto* s1        = reinterpret_cast<addr_t*>(addr);
  asm volatile("st.global.u32 [%0], {%1};" : : "l"(s1), "r"(x_store));
}

template <>
__device__ inline void st_global(void* addr, const uint8_t (&x)[8])
{
  using store_t              = uint32_t;
  const store_t* x_store_arr = reinterpret_cast<const store_t*>(x);
  auto* s1                   = reinterpret_cast<void*>(addr);
  asm volatile("st.global.v2.u32 [%0], {%1, %2};"
               :
               : "l"(s1), "r"(x_store_arr[0]), "r"(x_store_arr[1]));
}

template <>
__device__ inline void st_global(void* addr, const uint8_t (&x)[16])
{
  using store_t              = uint32_t;
  const store_t* x_store_arr = reinterpret_cast<const store_t*>(x);
  auto* s1                   = reinterpret_cast<void*>(addr);
  asm volatile(
    "st.global.v4.u32 [%0], {%1, %2, %3, %4};"
    :
    : "l"(s1), "r"(x_store_arr[0]), "r"(x_store_arr[1]), "r"(x_store_arr[2]), "r"(x_store_arr[3]));
}

// ld_shared is a wrapper around the ld.shared PTX instruction.
// There is a template specialization for values of N=1, 2, 4, 8, 16.
template <int N>
__device__ inline void ld_global(uint8_t (&x)[N], const void* addr);
template <int N>
__device__ inline void ld_global_evict_last(uint8_t (&x)[N], const void* addr);
template <int N>
__device__ inline void ld_global_evict_first(uint8_t (&x)[N], const void* addr);

template <>
__device__ inline void ld_global(uint8_t (&x)[1], const void* addr)
{
  using addr_t = uint8_t;
  uint32_t x_store;
  const auto* s1 = reinterpret_cast<const addr_t*>(addr);
  asm volatile("ld.global.u8 {%0}, [%1];" : "=r"(x_store) : "l"(s1));
  x[0] = x_store;
}
template <>
__device__ inline void ld_global_evict_last(uint8_t (&x)[1], const void* addr)
{
  using addr_t = uint8_t;
  uint32_t x_store;
  const auto* s1 = reinterpret_cast<const addr_t*>(addr);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_last.b64 %tmp;\n\t"
    "ld.global.L2::cache_hint.L1::no_allocate.u8 {%0}, [%1], %tmp;\n\t}"
    : "=r"(x_store)
    : "l"(s1));
  x[0] = x_store;
}
template <>
__device__ inline void ld_global_evict_first(uint8_t (&x)[1], const void* addr)
{
  using addr_t = uint8_t;
  uint32_t x_store;
  const auto* s1 = reinterpret_cast<const addr_t*>(addr);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_last.b64 %tmp;\n\t"
    "ld.global.L2::cache_hint.L1::evict_last.u8 {%0}, [%1], %tmpy;\n\t}"
    : "=r"(x_store)
    : "l"(s1));
  x[0] = x_store;
}

template <>
__device__ inline void ld_global(uint8_t (&x)[2], const void* addr)
{
  using addr_t    = uint16_t;
  addr_t* x_store = reinterpret_cast<addr_t*>(x);
  const auto* s1  = reinterpret_cast<const addr_t*>(addr);
  asm volatile("ld.global.u16 %0, [%1];" : "=h"(*x_store) : "l"(s1));
}
template <>
__device__ inline void ld_global_evict_last(uint8_t (&x)[2], const void* addr)
{
  using addr_t    = uint16_t;
  addr_t* x_store = reinterpret_cast<addr_t*>(x);
  const auto* s1  = reinterpret_cast<const addr_t*>(addr);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_last.b64 %tmp;\n\t"
    "ld.global.L2::cache_hint.L1::evict_last.u16 %0, [%1], %tmp;\n\t}"
    : "=h"(*x_store)
    : "l"(s1));
}
template <>
__device__ inline void ld_global_evict_first(uint8_t (&x)[2], const void* addr)
{
  using addr_t    = uint16_t;
  addr_t* x_store = reinterpret_cast<addr_t*>(x);
  const auto* s1  = reinterpret_cast<const addr_t*>(addr);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_first.b64 %tmp;\n\t"
    "ld.global.L2::cache_hint.L1::no_allocate.u16 %0, [%1], %tmp;\n\t}"
    : "=h"(*x_store)
    : "l"(s1));
}

template <>
__device__ inline void ld_global(uint8_t (&x)[4], const void* addr)
{
  using addr_t     = uint32_t;
  using store_t    = uint32_t;
  store_t* x_store = reinterpret_cast<store_t*>(x);
  const auto* s1   = reinterpret_cast<const addr_t*>(addr);
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(*x_store) : "l"(s1));
}
template <>
__device__ inline void ld_global_evict_last(uint8_t (&x)[4], const void* addr)
{
  using addr_t     = uint32_t;
  using store_t    = uint32_t;
  store_t* x_store = reinterpret_cast<store_t*>(x);
  const auto* s1   = reinterpret_cast<const addr_t*>(addr);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_last.b64 %tmp;\n\t"
    "ld.global.L2::cache_hint.L1::evict_last.u32 %0, [%1], %tmp;\n\t}"
    : "=r"(*x_store)
    : "l"(s1));
}
template <>
__device__ inline void ld_global_evict_first(uint8_t (&x)[4], const void* addr)
{
  using addr_t     = uint32_t;
  using store_t    = uint32_t;
  store_t* x_store = reinterpret_cast<store_t*>(x);
  const auto* s1   = reinterpret_cast<const addr_t*>(addr);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_first.b64 %tmp;\n\t"
    "ld.global.L2::cache_hint.L1::no_allocate.u32 %0, [%1], %tmp;\n\t}"
    : "=r"(*x_store)
    : "l"(s1));
}

template <>
__device__ inline void ld_global(uint8_t (&x)[8], const void* addr)
{
  using store_t        = uint32_t;
  store_t* x_store_arr = reinterpret_cast<store_t*>(x);
  asm volatile("ld.global.v2.u32 {%0, %1}, [%2];"
               : "=r"(x_store_arr[0]), "=r"(x_store_arr[1])
               : "l"(addr));
}
template <>
__device__ inline void ld_global_evict_last(uint8_t (&x)[8], const void* addr)
{
  using store_t        = uint32_t;
  store_t* x_store_arr = reinterpret_cast<store_t*>(x);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_last.b64 %tmp;\n\t"
    "ld.global.L2::cache_hint.L1::evict_last.v2.u32 {%0, %1}, [%2], %tmp;\n\t}"
    : "=r"(x_store_arr[0]), "=r"(x_store_arr[1])
    : "l"(addr));
}
template <>
__device__ inline void ld_global_evict_first(uint8_t (&x)[8], const void* addr)
{
  using store_t        = uint32_t;
  store_t* x_store_arr = reinterpret_cast<store_t*>(x);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_first.b64 %tmp;\n\t"
    "ld.global.L2::cache_hint.L1::no_allocate.v2.u32 {%0, %1}, [%2], %tmp;\n\t}"
    : "=r"(x_store_arr[0]), "=r"(x_store_arr[1])
    : "l"(addr));
}

template <>
__device__ inline void ld_global(uint8_t (&x)[16], const void* addr)
{
  using store_t        = uint32_t;
  store_t* x_store_arr = reinterpret_cast<store_t*>(x);
  asm volatile(
    "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
    : "=r"(x_store_arr[0]), "=r"(x_store_arr[1]), "=r"(x_store_arr[2]), "=r"(x_store_arr[3])
    : "l"(addr));
}
template <>
__device__ inline void ld_global_evict_last(uint8_t (&x)[16], const void* addr)
{
  using store_t        = uint32_t;
  store_t* x_store_arr = reinterpret_cast<store_t*>(x);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_last.b64 %tmp;\n\t"
    "ld.global.L2::cache_hint.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4], %tmp;\n\t}"
    : "=r"(x_store_arr[0]), "=r"(x_store_arr[1]), "=r"(x_store_arr[2]), "=r"(x_store_arr[3])
    : "l"(addr));
}
template <>
__device__ inline void ld_global_evict_first(uint8_t (&x)[16], const void* addr)
{
  using store_t        = uint32_t;
  store_t* x_store_arr = reinterpret_cast<store_t*>(x);
  asm volatile(
    "{\n\t"
    ".reg .u64 %tmp;\n\t"
    "createpolicy.fractional.L2::evict_first.b64 %tmp;\n"
    "ld.global.L2::cache_hint.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4], %tmp;\n\t}"
    : "=r"(x_store_arr[0]), "=r"(x_store_arr[1]), "=r"(x_store_arr[2]), "=r"(x_store_arr[3])
    : "l"(addr));
}

/**
 * @defgroup SmemStores Shared memory store operations
 * @{
 *
 * @brief Stores to shared memory (both vectorized and non-vectorized forms)
 *
 * @param[out] addr shared memory address
 * @param[in]  x    data to be stored at this address
 */
// sts can be used to store to shared memory any type DataT
// if sizeof(DataT) == 1, 2, 4, 8, 16.
template <typename DataT, typename MathT, int N>
__device__ inline void sts(DataT* addr, const MathT (&x)[N])
{
  using arr_t = const uint8_t(&)[sizeof(DataT) * N];
  DataT x_data[N];
  utils::cast_reg<N>(x_data, x);
  st_shared(static_cast<void*>(addr), reinterpret_cast<arr_t>(x_data));
}
template <typename DataT, typename MathT>
__device__ inline void sts(DataT* addr, const MathT& x)
{
  using arr_t = const uint8_t(&)[sizeof(DataT)];  // N = 1
  DataT x_data;
  utils::cast_reg(x_data, x);
  st_shared(static_cast<void*>(addr), reinterpret_cast<arr_t>(x_data));
}

/** @} */

/**
 * @defgroup SmemLoads Shared memory load operations
 * @{
 *
 * @brief Loads from shared memory (both vectorized and non-vectorized forms)
 *
 * @param[out] x    the data to be loaded
 * @param[in]  addr shared memory address from where to load
 */

template <typename DataT, typename MathT, int N>
__device__ inline void lds(MathT (&x)[N], const DataT* addr)
{
  using arr_t = uint8_t(&)[sizeof(DataT) * N];
  DataT x_data[N];
  ld_shared(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
  utils::cast_reg<N>(x, x_data);
}
template <typename DataT, typename MathT>
__device__ inline void lds(MathT& x, const DataT* addr)
{
  using arr_t = uint8_t(&)[sizeof(DataT)];  // N = 1
  DataT x_data;
  ld_shared(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
  utils::cast_reg(x, x_data);
}

/** @} */

/**
 * @defgroup GlobalLoads Global cached load operations
 * @{
 *
 * @brief Load from global memory with caching at L1 level
 *
 * @param[out] x    data to be loaded from global memory
 * @param[in]  addr address in global memory from where to load
 */
template <typename DataT, typename MathT, int N>
__device__ inline void ldg(MathT (&x)[N], const DataT* addr)
{
  using arr_t = uint8_t(&)[sizeof(DataT) * N];
  DataT x_data[N];
  ld_global(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
  utils::cast_reg<N>(x, x_data);
}
template <typename DataT, typename MathT>
__device__ inline void ldg(MathT& x, const DataT* addr)
{
  using arr_t = uint8_t(&)[sizeof(DataT)];  // N = 1
  DataT x_data;
  ld_global(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
  utils::cast_reg(x, x_data);
}
template <typename DataT, typename MathT, int N>
__device__ inline void ldg_evict_first(MathT (&x)[N], const DataT* addr)
{
  using arr_t = uint8_t(&)[sizeof(DataT) * N];
  DataT x_data[N];
#if (__CUDA_ARCH__ >= 800)
  ld_global_evict_first(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
#else
  ld_global(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
#endif
  utils::cast_reg<N>(x, x_data);
}
template <typename DataT, typename MathT>
__device__ inline void ldg_evict_first(DataT& x, const DataT* addr)
{
  using arr_t = uint8_t(&)[sizeof(DataT)];  // N = 1
  DataT x_data;
#if (__CUDA_ARCH__ >= 800)
  ld_global_evict_first(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
#else
  ld_global(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
#endif
  utils::cast_reg(x, x_data);
}
template <typename DataT, typename MathT, int N>
__device__ inline void ldg_evict_last(MathT (&x)[N], const DataT* addr)
{
  using arr_t = uint8_t(&)[sizeof(DataT) * N];
  DataT x_data[N];
#if (__CUDA_ARCH__ >= 800)
  ld_global_evict_last(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
#else
  ld_global(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
#endif
  utils::cast_reg<N>(x, x_data);
}
template <typename DataT, typename MathT>
__device__ inline void ldg_evict_last(MathT& x, const DataT* addr)
{
  using arr_t = uint8_t(&)[sizeof(DataT)];  // N = 1
  DataT x_data;
#if (__CUDA_ARCH__ >= 800)
  ld_global_evict_last(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
#else
  ld_global(reinterpret_cast<arr_t>(x_data), static_cast<const void*>(addr));
#endif
  utils::cast_reg(x, x_data);
}
/** @} */

/**
 * @defgroup GmemStores Global memory store operations
 * @{
 *
 * @brief Stores to global memory (both vectorized and non-vectorized forms)
 *
 * @param[out] addr global memory address
 * @param[in]  x    data to be stored at this address
 */
// sts can be used to store to shared memory any type DataT
// if sizeof(DataT) == 1, 2, 4, 8, 16.
template <typename DataT, typename MathT, int N>
__device__ inline void stg(DataT* addr, const MathT (&x)[N])
{
  using arr_t = const uint8_t(&)[sizeof(DataT) * N];
  DataT x_data[N];
  utils::cast_reg<N>(x_data, x);
  st_global(static_cast<void*>(addr), reinterpret_cast<arr_t>(x_data));
}
template <typename DataT, typename MathT>
__device__ inline void stg(DataT* addr, const MathT& x)
{
  using arr_t = const uint8_t(&)[sizeof(DataT)];  // N = 1
  DataT x_data;
  utils::cast_reg(x_data, x);
  st_global(static_cast<void*>(addr), reinterpret_cast<arr_t>(x_data));
}

/** @} */

/**
 * @defgroup SmemLoads Shared/Global memory load operations
 * @{
 *
 * @brief Loads from global or shared memory (both vectorized and non-vectorized forms)
 *
 * @param[out] x    the data to be loaded
 * @param[in]  addr memory address from where to load
 */
template <BufferType TYPE, typename DataT, typename MathT, int N>
__device__ inline void ld_generic(MathT (&x)[N], const DataT* addr)
{
  if constexpr (TYPE == BufferType::kGlobal)
    ldg(x, addr);
  else
    lds(x, addr);
}
template <BufferType TYPE, typename DataT, typename MathT>
__device__ inline void ld_generic(MathT& x, const DataT* addr)
{
  if constexpr (TYPE == BufferType::kGlobal)
    ldg(x, addr);
  else
    lds(x, addr);
}

/**
 * @defgroup GmemStores Global/Shared memory store operations
 * @{
 *
 * @brief Stores to global or shared memory (both vectorized and non-vectorized forms)
 *
 * @param[out] addr global or shared memory address
 * @param[in]  x    data to be stored at this address
 */
template <BufferType TYPE, typename DataT, typename MathT, int N>
__device__ inline void st_generic(DataT* addr, const MathT (&x)[N])
{
  if constexpr (TYPE == BufferType::kGlobal)
    stg(addr, x);
  else
    sts(addr, x);
}
template <BufferType TYPE, typename DataT, typename MathT>
__device__ inline void st_generic(DataT* addr, const MathT& x)
{
  if constexpr (TYPE == BufferType::kGlobal)
    stg(addr, x);
  else
    sts(addr, x);
}
/** @} */
}  // namespace cugraph::ops::utils
