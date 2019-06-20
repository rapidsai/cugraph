#pragma once

#include "Host/Numeric.hpp"   //xlib::is_power2

namespace xlib {
namespace block {

template<unsigned START, unsigned END, unsigned BLOCK_SIZE, typename = void>
struct StrideOp;

template<unsigned START, unsigned END, unsigned BLOCK_SIZE>
struct StrideOp<START, END, BLOCK_SIZE,
                typename std::enable_if<(END - START > BLOCK_SIZE)>::type> {

    template<typename T>
    __device__ __forceinline__
    static void copy(const T* __restrict__ d_in,
                     int                   num_items,
                     T*       __restrict__ smem_out) {
        const auto MID = (START + END) / 2u;
        if (num_items < MID)
            StrideOp<START, MID, BLOCK_SIZE>::copy(d_in, num_items, smem_out);
        else
            StrideOp<MID, END, BLOCK_SIZE>::copy(d_in, num_items, smem_out);
    }
};

template<unsigned START, unsigned END, unsigned BLOCK_SIZE>
struct StrideOp<START, END, BLOCK_SIZE,
                typename std::enable_if<(END - START <= BLOCK_SIZE)>::type> {

    template<typename T>
    __device__ __forceinline__
    static void copy(const T* d_in,
                     int      num_items,
                     T*       smem_out) {
        const int LOOPS = END / BLOCK_SIZE;

        d_in     += threadIdx.x;
        smem_out += threadIdx.x;
        #pragma unroll
        for (int i = 0; i < LOOPS; i++)
            smem_out[i * BLOCK_SIZE] = d_in[i * BLOCK_SIZE];
        if (LOOPS * BLOCK_SIZE + threadIdx.x < num_items)
            smem_out[LOOPS * BLOCK_SIZE] = d_in[LOOPS * BLOCK_SIZE];
    }
};

namespace detail {

template<unsigned BLOCK_SIZE, typename T, typename = void>
__device__ __forceinline__
void stride_load_smem_aux(const T* __restrict__ d_in,
                          int                   num_items,
                          T*       __restrict__ smem_out) {

    for (int i = threadIdx.x; i < num_items; i += BLOCK_SIZE) {
        smem_out = *d_in;
        smem_out += BLOCK_SIZE;
        d_in     += BLOCK_SIZE;
    }
}

template<unsigned BLOCK_SIZE, typename T,
         typename V1, typename V2, typename... VArgs>
__device__ __forceinline__
void stride_load_smem_aux(const T* __restrict__ d_in,
                          int                   num_items,
                          T*       __restrict__ smem_out) {

    const int RATIO = sizeof(V1) / sizeof(T);
    const int LOOP  = num_items / (RATIO * BLOCK_SIZE);
    static_assert(RATIO > 0, "Ratio must be greater than zero");
    static_assert(xlib::is_power2(sizeof(T)) &&
                  sizeof(T) >= 1 && sizeof(T) <= 16,
                  "size(T) constraits");

    auto in_ptr  = reinterpret_cast<const V1*>(d_in) + threadIdx.x;
    auto out_ptr = reinterpret_cast<V1*>(smem_out) + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < LOOP; i++)
        out_ptr[i * BLOCK_SIZE] = in_ptr[i * BLOCK_SIZE];

    const int STEP = LOOP * RATIO * BLOCK_SIZE;
    const int REM  = num_items - STEP;
    stride_load_smem_aux<BLOCK_SIZE, T, V2, VArgs...>
        (d_in + STEP, REM, smem_out + STEP);
}

} // namespace detail

//==============================================================================

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const double* __restrict__ d_in,
                      int                        num_items,
                      double*       __restrict__ smem_out) {

    detail::stride_load_smem_aux<BLOCK_SIZE, double2>
        (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const float* __restrict__ d_in,
                      int                       num_items,
                      float*       __restrict__ smem_out) {

    detail::stride_load_smem_aux<BLOCK_SIZE, float4, float2>
        (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const int64_t* __restrict__ d_in,
                      int                         num_items,
                      int64_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux<BLOCK_SIZE, longlong2>
        (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const uint64_t* __restrict__ d_in,
                      int                          num_items,
                      uint64_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux<BLOCK_SIZE, ulonglong2>
        (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const int* __restrict__ d_in,
                      int                     num_items,
                      int*       __restrict__ smem_out) {

    detail::stride_load_smem_aux<BLOCK_SIZE, int4, int2>
    (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const unsigned* __restrict__ d_in,
                      int                          num_items,
                      unsigned*       __restrict__ smem_out) {

    detail::stride_load_smem_aux<BLOCK_SIZE, uint4, uint2>
        (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const int16_t* __restrict__ d_in,
                      int                         num_items,
                      int16_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux<BLOCK_SIZE, int4, short4, short2>
        (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const uint16_t* __restrict__ d_in,
                      int                          num_items,
                      uint16_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux<BLOCK_SIZE, uint4, ushort4, ushort2>
        (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const char* __restrict__ d_in,
                      int                      num_items,
                      char*       __restrict__ smem_out) {

    detail::stride_load_smem_aux
        <BLOCK_SIZE, char, int4, int2, char4, char2>
        (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const int8_t* __restrict__ d_in,
                      int                        num_items,
                      int8_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux
        <BLOCK_SIZE, int8_t, int4, int2, char4, char2>
        (d_in, num_items, smem_out);
}

template<unsigned BLOCK_SIZE>
__device__ __forceinline__
void stride_load_smem(const unsigned char* __restrict__ d_in,
                      int                               num_items,
                      unsigned char*       __restrict__ smem_out) {

    detail::stride_load_smem_aux
        <BLOCK_SIZE, unsigned char, uint4, uint2, uchar4, uchar2>
        (d_in, num_items, smem_out);
}

} // namespace block
} // namespace xlib
