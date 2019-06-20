/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date December, 2017
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
#include "WarpReduceMacro.i.cuh"    //WARP_REDUCE_32BIT

namespace xlib {
namespace detail {

__device__ __forceinline__
constexpr int min_max_lane(int STEP) {
    const int MASK_WARP = (1 << (STEP + 1)) - 1;
    return ((31 - MASK_WARP) << 8) | MASK_WARP;
}

//==============================================================================

template<unsigned WARP_SZ, typename T>
struct WarpReduceHelper {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(T& value) {
        #pragma unroll
        for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2)
            value += xlib::shfl_xor(member_mask, value, STEP);
    }

    __device__ __forceinline__
    static void min(T& value) {
        #pragma unroll
        for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {
            auto tmp = xlib::shfl_xor(member_mask, value, STEP);
            value    = tmp < value ? tmp : value;
        }
    }

    __device__ __forceinline__
    static void max(T& value) {
        #pragma unroll
        for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {
            auto tmp = xlib::shfl_xor(member_mask, value, STEP);
            value    = tmp > value ? tmp : value;
        }
    }

    template<typename Lambda>
     __device__ __forceinline__
    static void apply(T& value, const Lambda& lambda) {
        #pragma unroll
        for (int STEP = 1; STEP <= WARP_SZ / 2; STEP = STEP * 2) {
            auto tmp = xlib::shfl_xor(member_mask, value, STEP);
            value    = lambda(value, tmp);
        }
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpReduceHelper<WARP_SZ, int> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(int& value) {
        WARP_REDUCE_32BIT(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void min(int& value) {
        WARP_REDUCE_32BIT(min, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void max(int& value) {
        WARP_REDUCE_32BIT(max, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[2]) {
        WARP_REDUCE_GEN2(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[3]) {
        WARP_REDUCE_GEN3(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[4]) {
        WARP_REDUCE_GEN4(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[5]) {
        WARP_REDUCE_GEN5(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[6]) {
        WARP_REDUCE_GEN6(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[7]) {
        WARP_REDUCE_GEN7(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[8]) {
        WARP_REDUCE_GEN8(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[9]) {
        WARP_REDUCE_GEN9(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[10]) {
        WARP_REDUCE_GEN10(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[11]) {
        WARP_REDUCE_GEN11(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[12]) {
        WARP_REDUCE_GEN12(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[13]) {
        WARP_REDUCE_GEN13(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[14]) {
        WARP_REDUCE_GEN14(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[15]) {
        WARP_REDUCE_GEN15(add, s32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(int (&value)[16]) {
        WARP_REDUCE_GEN16(add, s32, r, min_max_lane(STEP))
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpReduceHelper<WARP_SZ, unsigned> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(unsigned& value) {
        WARP_REDUCE_32BIT(add, u32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void min(unsigned& value) {
        WARP_REDUCE_32BIT(min, u32, r, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void max(unsigned& value) {
        WARP_REDUCE_32BIT(max, u32, r, min_max_lane(STEP))
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpReduceHelper<WARP_SZ, float> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(float& value) {
        WARP_REDUCE_32BIT(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void min(float& value) {
        WARP_REDUCE_32BIT(min, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void max(float& value) {
        WARP_REDUCE_32BIT(max, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[2]) {
        WARP_REDUCE_GEN2(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[3]) {
        WARP_REDUCE_GEN3(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[4]) {
        WARP_REDUCE_GEN4(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[5]) {
        WARP_REDUCE_GEN5(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[6]) {
        WARP_REDUCE_GEN6(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[7]) {
        WARP_REDUCE_GEN7(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[8]) {
        WARP_REDUCE_GEN8(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[9]) {
        WARP_REDUCE_GEN9(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[10]) {
        WARP_REDUCE_GEN10(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[11]) {
        WARP_REDUCE_GEN11(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[12]) {
        WARP_REDUCE_GEN12(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[13]) {
        WARP_REDUCE_GEN13(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[14]) {
        WARP_REDUCE_GEN14(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[15]) {
        WARP_REDUCE_GEN15(add, f32, f, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void add(float (&value)[16]) {
        WARP_REDUCE_GEN16(add, f32, f, min_max_lane(STEP))
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpReduceHelper<WARP_SZ, double> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static  void add(double& value) {
        WARP_REDUCE_64BIT(add, f64, d, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void min(double& value) {
        WARP_REDUCE_64BIT(min, f64, d, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void max(double& value) {
        WARP_REDUCE_64BIT(max, f64, d, min_max_lane(STEP))
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpReduceHelper<WARP_SZ, int64_t> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static  void add(int64_t& value) {
        WARP_REDUCE_64BIT(add, s64, l, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void min(int64_t& value) {
        WARP_REDUCE_64BIT(min, s64, l, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void max(int64_t& value) {
        WARP_REDUCE_64BIT(max, s64, l, min_max_lane(STEP))
    }
};

//------------------------------------------------------------------------------

template<unsigned WARP_SZ>
struct WarpReduceHelper<WARP_SZ, uint64_t> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static  void add(uint64_t& value) {
        WARP_REDUCE_64BIT(add, u64, l, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void min(uint64_t& value) {
        WARP_REDUCE_64BIT(min, u64, l, min_max_lane(STEP))
    }

    __device__ __forceinline__
    static void max(uint64_t& value) {
        WARP_REDUCE_64BIT(max, u64, l, min_max_lane(STEP))
    }
};

#undef WARP_REDUCE_32BIT
#undef WARP_REDUCE_64BIT
#undef WARP_REDUCE_GEN1
#undef WARP_REDUCE_GEN2
#undef WARP_REDUCE_GEN3
#undef WARP_REDUCE_GEN4
#undef WARP_REDUCE_GEN5
#undef WARP_REDUCE_GEN6
#undef WARP_REDUCE_GEN7
#undef WARP_REDUCE_GEN8
#undef WARP_REDUCE_GEN9
#undef WARP_REDUCE_GEN10
#undef WARP_REDUCE_GEN11
#undef WARP_REDUCE_GEN12
#undef WARP_REDUCE_GEN13
#undef WARP_REDUCE_GEN14
#undef WARP_REDUCE_GEN15
#undef WARP_REDUCE_GEN16

} // namespace detail

//==============================================================================
//==============================================================================

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::add(T& value) {
    detail::WarpReduceHelper<VW_SIZE, T>::add(value);
}

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::min(T& value) {
    detail::WarpReduceHelper<VW_SIZE, T>::min(value);
}

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::max(T& value) {
    detail::WarpReduceHelper<VW_SIZE, T>::max(value);
}

template<int VW_SIZE>
template<typename T, int SIZE>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::add(T (&value)[SIZE]) {
    detail::WarpReduceHelper<VW_SIZE, T>::add(value);
}

//==============================================================================

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::addAll(T& value) {
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();
    detail::WarpReduceHelper<VW_SIZE, T>::add(value);
    value = xlib::shfl(member_mask, value, 0, VW_SIZE);
}

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::minAll(T& value) {
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();
    detail::WarpReduceHelper<VW_SIZE, T>::min(value);
    value = xlib::shfl(member_mask, value, 0, VW_SIZE);
}

template<int VW_SIZE>
template<typename T>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::maxAll(T& value) {
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();
    detail::WarpReduceHelper<VW_SIZE, T>::max(value);
    value = xlib::shfl(member_mask, value, 0, VW_SIZE);
}

//==============================================================================

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::add(T& value, R* pointer) {
    detail::WarpReduceHelper<VW_SIZE, T>::add(value);
    if (lane_id<VW_SIZE>() == 0)
        *pointer = value;
}

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::min(T& value, R* pointer) {
    detail::WarpReduceHelper<VW_SIZE, T>::min(value);
    if (lane_id<VW_SIZE>() == 0)
        *pointer = value;
}

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::max(T& value, R* pointer) {
    detail::WarpReduceHelper<VW_SIZE, T>::max(value);
    if (lane_id<VW_SIZE>() == 0)
        *pointer = value;
}

//==============================================================================

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
T WarpReduce<VW_SIZE>::atomicAdd(const T& value, R* pointer) {
    const unsigned member_mask = xlib::member_mask<VW_SIZE>();
    T old, value_tmp = value;
    detail::WarpReduceHelper<VW_SIZE, T>::add(value_tmp);
    if (lane_id<VW_SIZE>() == 0)
        old = atomic::add(value_tmp, pointer);
    return xlib::shfl(member_mask, old, 0, VW_SIZE);
}

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::atomicMin(const T& value, R* pointer) {
    T value_tmp = value;
    detail::WarpReduceHelper<VW_SIZE, T>::min(value_tmp);
    if (lane_id<VW_SIZE>() == 0)
        atomic::min(value_tmp, pointer);
}

template<int VW_SIZE>
template<typename T, typename R>
__device__ __forceinline__
void WarpReduce<VW_SIZE>::atomicMax(const T& value, R* pointer) {
    T value_tmp = value;
    detail::WarpReduceHelper<VW_SIZE, T>::max(value_tmp);
    if (lane_id<VW_SIZE>() == 0)
        atomic::max(value_tmp, pointer);
}

} // namespace xlib
