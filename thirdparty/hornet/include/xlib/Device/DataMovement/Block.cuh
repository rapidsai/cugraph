#pragma once

namespace xlib {
namespace block {

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const double* __restrict__ d_in,
                      int                        num_items,
                      double*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const float* __restrict__ d_in,
                      int                       num_items,
                      float*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const int64_t* __restrict__ d_in,
                      int                         num_items,
                      int64_t*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const uint64_t* __restrict__ d_in,
                      int                          num_items,
                      uint64_t*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const int* __restrict__ d_in,
                      int                     num_items,
                      int*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const unsigned* __restrict__ d_in,
                      int                          num_items,
                      unsigned*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const int16_t* __restrict__ d_in,
                      int                         num_items,
                      int16_t*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const uint16_t* __restrict__ d_in,
                      int                          num_items,
                      uint16_t*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const char* __restrict__ d_in,
                      int                      num_items,
                      char*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const int8_t* __restrict__ d_in,
                      int                        num_items,
                      int8_t*       __restrict__ smem_out);

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const unsigned char* __restrict__ d_in,
                      int                               num_items,
                      unsigned char*       __restrict__ smem_out);

} // namespace block
} // namespace xlib

#include "impl/Block.i.cuh"
