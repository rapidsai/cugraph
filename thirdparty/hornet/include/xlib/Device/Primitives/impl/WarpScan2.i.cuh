/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date October, 2017
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "Device/Util/Atomic.cuh"   //xlib::atomicAdd
#include "Device/Util/Basic.cuh"    //xlib::member_mask, xlib::shfl_xor
#include "Device/Util/PTX.cuh"      //xlib::lane_id
#include "Host/Metaprogramming.hpp" //xlib::Log2

namespace xlib {
namespace detail {

#define WARP_INCLUSIVE_SCAN_AUX(ASM_OP, ASM_T, ASM_CL)                         \
    _Pragma("unroll")                                                          \
    for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {                 \
        asm(                                                                   \
            "{"                                                                \
            ".reg ."#ASM_T" r1;"                                               \
            ".reg .pred p;"                                                    \
            "shfl.sync.up.b32 r1|p, %1, %2, %3, %4;"                           \
            "@p "#ASM_OP"."#ASM_T" r1, r1, %1;"                                \
            "mov."#ASM_T" %0, r1;"                                             \
            "}"                                                                \
            : "="#ASM_CL(value) : #ASM_CL(value), "r"(STEP), "r"(max_lane),    \
              "r"(member_mask));                                               \
    }

#define WARP_INCLUSIVE_SCAN_AUX2(ASM_OP, ASM_T, ASM_CL)                        \
    _Pragma("unroll")                                                          \
    for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {                 \
        asm(                                                                   \
            "{"                                                                \
            ".reg .u32 lo;"                                                    \
            ".reg .u32 hi;"                                                    \
            ".reg .pred p;"                                                    \
            "mov.b64 {lo, hi}, %1;"                                            \
            "shfl.sync.up.b32 lo|p, lo, %2, %3, %4;"                           \
            "shfl.sync.up.b32 hi|p, hi, %2, %3, %4;"                           \
            "mov.b64 %0, {lo, hi};"                                            \
            "@p "#ASM_OP"."#ASM_T" %0, %0, %1;"                                \
            "}"                                                                \
            : "="#ASM_CL(value) : #ASM_CL(value), "r"(STEP), "r"(max_lane),    \
              "r"(member_mask));                                               \
    }

#define WARP_INCLUSIVE_SCAN(ASM_OP, ASM_T, ASM_CL)                             \
    const unsigned    max_lane = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;     \
    const unsigned member_mask = xlib::member_mask<WARP_SZ>();                 \
    WARP_INCLUSIVE_SCAN_AUX(ASM_OP, ASM_T, ASM_CL)

#define WARP_INCLUSIVE_SCAN2(ASM_OP, ASM_T, ASM_CL)                            \
    const unsigned    max_lane = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;     \
    const unsigned member_mask = xlib::member_mask<WARP_SZ>();                 \
    WARP_INCLUSIVE_SCAN_AUX2(ASM_OP, ASM_T, ASM_CL)

//==============================================================================

template<unsigned WARP_SZ, typename T>
struct WarpInclusiveScanHelper {

    __device__ __forceinline__
    static void add(T& value) {
        const unsigned member_mask = xlib::member_mask<WARP_SZ>();
        int vlane = lane_id<WARP_SZ>();

        #pragma unroll
        for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {
            auto tmp = xlib::shfl_up(member_mask, value, STEP, WARP_SZ);
            if (vlane >= STEP)
                value += tmp;
        }
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, int> {

    __device__ __forceinline__
    static void add(int& value) {
        WARP_INCLUSIVE_SCAN(add, s32, r)
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, unsigned> {

    __device__ __forceinline__
    static void add(unsigned& value) {
        WARP_INCLUSIVE_SCAN(add, u32, r)
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, float> {

    __device__ __forceinline__
    static void add(float& value) {
        WARP_INCLUSIVE_SCAN(add, f32, f)
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, double> {
    __device__ __forceinline__
    static void add(double& value) {
        WARP_INCLUSIVE_SCAN2(add, f64, d)
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, long int> {
    __device__ __forceinline__
    static void add(double& value) {
        WARP_INCLUSIVE_SCAN2(add, s64, l)
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, long unsigned> {
    __device__ __forceinline__
    static void add(double& value) {
        WARP_INCLUSIVE_SCAN2(add, u64, l)
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, long long int> {
    __device__ __forceinline__
    static void add(double& value) {
        WARP_INCLUSIVE_SCAN2(add, s64, l)
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpInclusiveScanHelper<WARP_SZ, long long unsigned> {
    __device__ __forceinline__
    static void add(double& value) {
        WARP_INCLUSIVE_SCAN2(add, u64, l)
    }
};

#undef WARP_INCLUSIVE_SCAN_AUX
#undef WARP_INCLUSIVE_SCAN_AUX2
#undef WARP_INCLUSIVE_SCAN
#undef WARP_INCLUSIVE_SCAN2

} // namespace detail

//==============================================================================
///////////////////////
// WarpInclusiveScan //
///////////////////////

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpInclusiveScan<WARP_SZ>::add(T& value) {
    detail::WarpInclusiveScanHelper<WARP_SZ, T>::add(value);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpInclusiveScan<WARP_SZ>::add(T& value, T& total) {
    detail::WarpInclusiveScanHelper<WARP_SZ, T>::add(value);
    const unsigned member_mask = xlib::member_mask<WARP_SZ>();
    total = xlib::shfl(member_mask, value, WARP_SZ - 1, WARP_SZ);
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpInclusiveScan<WARP_SZ>::add(T& value, R* pointer) {
    detail::WarpInclusiveScanHelper<WARP_SZ, T>::add(value);
    if (lane_id<WARP_SZ>() == WARP_SZ - 1)
        *pointer = value;
}

//==============================================================================
///////////////////////
// WarpExclusiveScan //
///////////////////////

namespace detail {

template<unsigned WARP_SZ, typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) <= 4>::type
last_step(T& value, unsigned member_mask) {
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;
    asm(
        "{"
        ".reg .pred p;"
        "shfl.sync.up.b32 %0|p, %1, %2, %3, %4;"
        "@!p mov.b32 %0, 0;"
        "}"
        : "=r"(value) : "r"(value), "r"(1), "r"(MASK), "r"(member_mask));
}

template<unsigned WARP_SZ, typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 8>::type
last_step(T& value, unsigned member_mask) {
    const int MASK = ((-1 << Log2<WARP_SZ>::value) & 31) << 8;
    auto& tmp = reinterpret_cast<uint64_t&>(value);
    asm(
        "{"
        ".reg .u32 lo;"
        ".reg .u32 hi;"
        ".reg .pred p;"
        "mov.b64 {lo, hi}, %1;"
        "shfl.sync.up.b32 lo|p, lo, %2, %3, %4;"
        "shfl.sync.up.b32 hi|p, hi, %2, %3, %4;"
        "mov.b64 %0, {lo, hi};"
        "@!p mov.b64 %0, 0;"
        "}"
        : "=l"(tmp) : "l"(tmp), "r"(1), "r"(MASK), "r"(member_mask));
}

template<unsigned WARP_SZ, typename T>
__device__ __forceinline__
typename std::enable_if<(sizeof(T) > 8)>::type
last_step(T& value, unsigned member_mask) {
    xlib::shfl_up(member_mask, value, 1, WARP_SZ);
    if (lane_id<WARP_SZ>() == 0)
        value = 0;
}

} //namespace detail

//==============================================================================

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::add(T& value) {
    detail::WarpInclusiveScanHelper<WARP_SZ, T>::add(value);

    const unsigned member_mask =  xlib::member_mask<WARP_SZ>();
    detail::last_step<WARP_SZ>(value, member_mask);
}


template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::add(T& value, T& total) {
    detail::WarpInclusiveScanHelper<WARP_SZ, T>::add(value);

    const unsigned member_mask =  xlib::member_mask<WARP_SZ>();
    total = xlib::shfl(member_mask, value, WARP_SZ - 1, WARP_SZ);
    detail::last_step<WARP_SZ>(value, member_mask);
}

//------------------------------------------------------------------------------

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpExclusiveScan<WARP_SZ>::add(T& value, R* total_ptr) {
    T total;
    WarpExclusiveScan<WARP_SZ>::add(value, total);
    if (lane_id<WARP_SZ>() == WARP_SZ - 1)
        *total_ptr = total;
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
T WarpExclusiveScan<WARP_SZ>::atomicAdd(T& value, R* total_ptr) {
    T total, old;
    WarpExclusiveScan<WARP_SZ>::add(value, total);

    const unsigned member_mask = xlib::member_mask<WARP_SZ>();
    if (lane_id<WARP_SZ>() == WARP_SZ - 1)
        old = atomic::add(total_ptr, total);
    return xlib::shfl(member_mask, old, WARP_SZ - 1);
}

} // namespace xlib
