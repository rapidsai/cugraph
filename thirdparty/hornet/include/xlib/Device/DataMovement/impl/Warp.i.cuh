#pragma once

#include "Host/Numeric.hpp"   //xlib::is_power2

namespace xlib {
namespace warp {
namespace detail {

template<typename T, typename = void>
__device__ __forceinline__
void stride_load_smem_aux(const T* __restrict__ d_in,
                          int                   num_items,
                          T*       __restrict__ smem_out) {
    #pragma unroll
    for (int i = 0; i < num_items; i += xlib::WARP_SIZE) {
        auto index = xlib::lane_id() + i;
        if (index < num_items)
            smem_out[index] = d_in[index];
    }
}

template<typename T,
         typename V1, typename V2, typename... VArgs>
__device__ __forceinline__
void stride_load_smem_aux(const T* __restrict__ d_in,
                          int                   num_items,
                          T*       __restrict__ smem_out) {

    const int RATIO = sizeof(V1) / sizeof(T);
    const int LOOP  = num_items / (RATIO * xlib::WARP_SIZE);
    static_assert(RATIO > 0, "Ratio must be greater than zero");
    static_assert(xlib::is_power2(sizeof(T)) &&
                  sizeof(T) >= 1 && sizeof(T) <= 16,
                  "size(T) constraits");

    auto in_ptr  = reinterpret_cast<const V1*>(d_in) + xlib::lane_id();
    auto out_ptr = reinterpret_cast<V1*>(smem_out) + xlib::lane_id();
    #pragma unroll
    for (int i = 0; i < LOOP; i++)
        out_ptr[i * xlib::WARP_SIZE] = in_ptr[i * xlib::WARP_SIZE];

    const int STEP = LOOP * RATIO * xlib::WARP_SIZE;
    const int REM  = num_items - STEP;
    stride_load_smem_aux<T, V2, VArgs...>(d_in + STEP, REM, smem_out + STEP);
}

} // namespace detail

//==============================================================================
//==============================================================================

template<>
__device__ __forceinline__
void stride_load_smem<double>(const double* __restrict__ d_in,
                              int                        num_items,
                              double*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <double2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<float>(const float* __restrict__ d_in,
                             int                       num_items,
                             float*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <float4, float2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<int64_t>(const int64_t* __restrict__ d_in,
                               int                         num_items,
                               int64_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <longlong2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<uint64_t>(const uint64_t* __restrict__ d_in,
                                int                          num_items,
                                uint64_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <ulonglong2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<int>(const int* __restrict__ d_in,
                           int                     num_items,
                           int*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <int4, int2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<unsigned>(const unsigned* __restrict__ d_in,
                                int                          num_items,
                                unsigned*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <uint4, uint2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<int16_t>(const int16_t* __restrict__ d_in,
                               int                         num_items,
                               int16_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <int4, short4, short2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<uint16_t>(const uint16_t* __restrict__ d_in,
                                int                          num_items,
                                uint16_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <uint4, ushort4, ushort2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<char>(const char* __restrict__ d_in,
                            int                      num_items,
                            char*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <char, int4, int2, char4, char2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<int8_t>(const int8_t* __restrict__ d_in,
                              int                        num_items,
                              int8_t*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <int8_t, int4, int2, char4, char2>
        (d_in, num_items, smem_out);
}

template<>
__device__ __forceinline__
void stride_load_smem<unsigned char>
                                (const unsigned char* __restrict__ d_in,
                                 int                               num_items,
                                 unsigned char*       __restrict__ smem_out) {

    detail::stride_load_smem_aux <unsigned char, uint4, uint2, uchar4, uchar2>
        (d_in, num_items, smem_out);
}

template<typename T>
__device__ __forceinline__
void stride_load_smem(const T* __restrict__ d_in,
                      int                   num_items,
                      T*       __restrict__ smem_out) {
    static_assert(sizeof(T) != sizeof(T), "NOT IMPLEMENTED");
}

} // namespace warp
} // namespace xlib
