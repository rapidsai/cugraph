/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#define CUGRAPH_OPS_STRINGIFY_DETAIL(x) #x
#define CUGRAPH_OPS_STRINGIFY(x)        CUGRAPH_OPS_STRINGIFY_DETAIL(x)

#define CUGRAPH_OPS_UNROLL _Pragma("unroll")
#if defined(__clang__) && defined(__CUDA__)
// clang wants pragma unroll without parentheses
#define CUGRAPH_OPS_UNROLL_N(n) _Pragma(CUGRAPH_OPS_STRINGIFY(unroll n))
#else
// nvcc / nvrtc want pragma unroll with parentheses
#define CUGRAPH_OPS_UNROLL_N(n) _Pragma(CUGRAPH_OPS_STRINGIFY(unroll(n)))
#endif

#if defined(__clang__)
#define CUGRAPH_OPS_CONSTEXPR_D constexpr
#else
#define CUGRAPH_OPS_CONSTEXPR_D constexpr __device__
#endif

#if defined(__CUDACC__) || defined(__CUDA__)
#define CUGRAPH_OPS_HD __host__ __device__
#else
#define CUGRAPH_OPS_HD
#endif

// The CUGRAPH_OPS_KERNEL specificies that a kernel has hidden visibility
//
// cugraph-ops needs to ensure that the visibility of its CUGRAPH_OPS_KERNEL function
// templates have hidden visibility ( default is weak visibility).
//
// When kernels have weak visibility it means that if two dynamic libraries
// both contain identical instantiations of a kernel/template, then the linker
// will discard one of the two instantiations and use only one of them.
//
// Do to unique requirements of how the CUDA works this de-deduplication
// can lead to the wrong kernels being called ( SM version being wrong ),
// silently no kernel being called at all, or cuda runtime errors being
// thrown.
//
// https://github.com/rapidsai/raft/issues/1722
#ifndef CUGRAPH_OPS_KERNEL
#define CUGRAPH_OPS_KERNEL __global__ static
#endif
