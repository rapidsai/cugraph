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

template<int WARP_SZ, typename T>
struct WarpSegReduceHelper;

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, int> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(int& value, int max_lane) {
        WARP_REDUCE_32BIT(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void min(int& value, int max_lane) {
        WARP_REDUCE_32BIT(min, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void max(int& value, int max_lane) {
        WARP_REDUCE_32BIT(max, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[2], int max_lane) {
        WARP_REDUCE_GEN2(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[3], int max_lane) {
        WARP_REDUCE_GEN3(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[4], int max_lane) {
        WARP_REDUCE_GEN4(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[5], int max_lane) {
        WARP_REDUCE_GEN5(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[6], int max_lane) {
        WARP_REDUCE_GEN6(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[7], int max_lane) {
        WARP_REDUCE_GEN7(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[8], int max_lane) {
        WARP_REDUCE_GEN8(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[9], int max_lane) {
        WARP_REDUCE_GEN9(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[10], int max_lane) {
        WARP_REDUCE_GEN10(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[11], int max_lane) {
        WARP_REDUCE_GEN11(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[12], int max_lane) {
         WARP_REDUCE_GEN12(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[13], int max_lane) {
        WARP_REDUCE_GEN13(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[14], int max_lane) {
        WARP_REDUCE_GEN14(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[15], int max_lane) {
        WARP_REDUCE_GEN15(add, s32, r, max_lane)
    }

    __device__ __forceinline__
    static void add(int (&value)[16], int max_lane) {
        WARP_REDUCE_GEN16(add, s32, r, max_lane)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, unsigned> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(unsigned& value, int max_lane) {
        WARP_REDUCE_32BIT(add, u32, r, max_lane)
    }

    __device__ __forceinline__
    static void min(unsigned& value, int max_lane) {
        WARP_REDUCE_32BIT(min, u32, r, max_lane)
    }

    __device__ __forceinline__
    static void max(unsigned& value, int max_lane) {
        WARP_REDUCE_32BIT(max, u32, r, max_lane)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, float> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(float& value, int max_lane) {
        WARP_REDUCE_32BIT(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void min(float& value, int max_lane) {
        WARP_REDUCE_32BIT(min, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void max(float& value, int max_lane) {
        WARP_REDUCE_32BIT(max, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[2], int max_lane) {
        WARP_REDUCE_GEN2(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[3], int max_lane) {
        WARP_REDUCE_GEN3(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[4], int max_lane) {
        WARP_REDUCE_GEN4(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[5], int max_lane) {
        WARP_REDUCE_GEN5(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[6], int max_lane) {
        WARP_REDUCE_GEN6(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[7], int max_lane) {
        WARP_REDUCE_GEN7(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[8], int max_lane) {
        WARP_REDUCE_GEN8(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[9], int max_lane) {
        WARP_REDUCE_GEN9(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[10], int max_lane) {
        WARP_REDUCE_GEN10(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[11], int max_lane) {
        WARP_REDUCE_GEN11(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[12], int max_lane) {
        WARP_REDUCE_GEN12(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[13], int max_lane) {
        WARP_REDUCE_GEN13(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[14], int max_lane) {
        WARP_REDUCE_GEN14(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[15], int max_lane) {
        WARP_REDUCE_GEN15(add, f32, f, max_lane)
    }

    __device__ __forceinline__
    static void add(float (&value)[16], int max_lane) {
        WARP_REDUCE_GEN16(add, f32, f, max_lane)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, double> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(double& value, int max_lane) {
        WARP_REDUCE_64BIT(add, f64, d, max_lane)
    }

    __device__ __forceinline__
    static void min(double& value, int max_lane) {
        WARP_REDUCE_64BIT(min, f64, d, max_lane)
    }

    __device__ __forceinline__
    static void max(double& value, int max_lane) {
        WARP_REDUCE_64BIT(max, f64, d, max_lane)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, int64_t> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(int64_t& value, int max_lane) {
        WARP_REDUCE_64BIT(add, s64, l, max_lane)
    }

    __device__ __forceinline__
    static void min(int64_t& value, int max_lane) {
        WARP_REDUCE_64BIT(min, s64, l, max_lane)
    }

    __device__ __forceinline__
    static void max(int64_t& value, int max_lane) {
        WARP_REDUCE_64BIT(max, s64, l, max_lane)
    }
};

//------------------------------------------------------------------------------

template<int WARP_SZ>
struct WarpSegReduceHelper<WARP_SZ, uint64_t> {
    static const unsigned member_mask = xlib::member_mask<WARP_SZ>();

    __device__ __forceinline__
    static void add(uint64_t& value, int max_lane) {
        WARP_REDUCE_64BIT(add, u64, l, max_lane)
    }

    __device__ __forceinline__
    static void min(uint64_t& value, int max_lane) {
        WARP_REDUCE_64BIT(min, u64, l, max_lane)
    }

    __device__ __forceinline__
    static void max(uint64_t& value, int max_lane) {
        WARP_REDUCE_64BIT(max, u64, l, max_lane)
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

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::add(T& value, int max_lane) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::add(value, max_lane);
}

template<int WARP_SZ>
template<typename T, int SIZE>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::add(T (&value)[SIZE], int max_lane) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::add(value, max_lane);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::min(T& value, int max_lane) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::min(value, max_lane);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::max(T& value, int max_lane) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::max(value, max_lane);
}

//==============================================================================

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::add(const T& value, R* pointer,
                                       bool pivot, int max_lane) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::add(value, max_lane);
    if (pivot)
        *pointer = value;
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::min(const T& value, R* pointer,
                                       bool pivot, int max_lane) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::min(value, max_lane);
    if (pivot)
        *pointer = value;
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::max(const T& value, R* pointer,
                                       bool pivot, int max_lane) {
    detail::WarpSegReduceHelper<WARP_SZ, T>::max(value, max_lane);
    if (pivot)
        *pointer = value;
}

//==============================================================================

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicAdd(const T& value, R* pointer, bool pivot, int max_lane) {
    auto value_tmp = value;
    WarpSegmentedReduce<WARP_SZ>::add(value_tmp, max_lane);
    if (pivot) {
        if (lane_id() != 0 && max_lane == xlib::WARP_SIZE)
            *pointer = value_tmp;
        else
            xlib::atomic::add(value_tmp, pointer);
    }
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicMin(const T& value, R* pointer, bool pivot, int max_lane) {
    auto value_tmp = value;
    WarpSegmentedReduce::min(value_tmp, max_lane);
    if (pivot) {
        if (lane_id() != 0 && max_lane == xlib::WARP_SIZE)
            *pointer = value_tmp;
        else
            xlib::atomic::min(value_tmp, pointer);
    }
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicMax(const T& value, R* pointer, bool pivot, int max_lane) {
    auto value_tmp = value;
    WarpSegmentedReduce::max(value_tmp, max_lane);
    if (pivot) {
        if (lane_id() != 0 && max_lane == xlib::WARP_SIZE - 1)
            *pointer = value_tmp;
        else
            xlib::atomic::max(value_tmp, pointer);
    }
}

//==============================================================================
//==============================================================================

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::mask_add(T& value, unsigned mask) {
    int  max_lane = xlib::max_lane<WARP_SZ>(mask);
    detail::WarpSegReduceHelper<WARP_SZ, T>::add(value, max_lane);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::mask_min(T& value, unsigned mask) {
    int  max_lane = xlib::max_lane<WARP_SZ>(mask);
    detail::WarpSegReduceHelper<WARP_SZ, T>::min(value, max_lane);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>::mask_max(T& value, unsigned mask) {
    int  max_lane = xlib::max_lane<WARP_SZ>(mask);
    detail::WarpSegReduceHelper<WARP_SZ, T>::max(value, max_lane);
}

//==============================================================================

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicAdd(const T& value, R* pointer, unsigned mask) {
    auto value_tmp = value;
    int  max_lane  = xlib::max_lane<WARP_SZ>(mask);
    WarpSegmentedReduce<WARP_SZ>::add(value_tmp, max_lane);

    if (lanemask_eq() & mask) {
        if (lane_id() != 0 && lanemask_gt() & mask)
            *pointer = value_tmp;
        else
            xlib::atomic::add(value_tmp, pointer);
    }
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicMin(const T& value, R* pointer, unsigned mask) {
    auto value_tmp = value;
    int  max_lane  = xlib::max_lane<WARP_SZ>(mask);
    WarpSegmentedReduce::min(value_tmp, max_lane);

    if (lanemask_eq() & mask) {
        if (lane_id() != 0 && lanemask_gt() & mask)
            *pointer = value_tmp;
        else
            xlib::atomic::min(value_tmp, pointer);
    }
}

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicMax(const T& value, R* pointer, unsigned mask) {
    auto value_tmp = value;
    int  max_lane  = xlib::max_lane<WARP_SZ>(mask);
    WarpSegmentedReduce::max(value_tmp, max_lane);

    if (lanemask_eq() & mask) {
        if (lane_id() != 0 && lanemask_gt() & mask)
            *pointer = value_tmp;
        else
            xlib::atomic::max(value_tmp, pointer);
    }
}

//==============================================================================
//==============================================================================

template<int WARP_SZ>
template<typename T, typename R>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::atomicAdd(const T& value, R* pointer, bool normal_store, bool atomic_store,
            int max_lane) {
    auto value_tmp = value;
    WarpSegmentedReduce<WARP_SZ>::add(value_tmp, max_lane);
    if (normal_store)
        *pointer = value_tmp;
    else if (atomic_store)
        xlib::atomic::add(value_tmp, pointer);
}

template<int WARP_SZ>
template<typename T>
__device__ __forceinline__
void WarpSegmentedReduce<WARP_SZ>
::conditional_add(T& left, T& right, int predicate, int max_lane) {
    const unsigned member_mask = xlib::member_mask<WARP_SZ>();
    _Pragma("unroll")
    for (int STEP = 0; STEP < xlib::Log2<WARP_SZ>::value; STEP++) {
        asm(
            "{\n\t\t"
            ".reg .s32 r1;\n\t\t"
            ".reg .pred p, q, s;\n\t\t"
            "shfl.sync.down.b32 r1|p, %0, %2, %3, %4;\n\t\t"
            "setp.ne.and.b32 s|q, %5, 0, p;\n\t\t"
            "@s add.s32 %1, r1, %1;\n\t\t"
            "@q add.s32 %0, r1, %0;\n\t"
            "}"
            : "+r"(left), "+r"(right) : "r"(1 << STEP),
              "r"(max_lane), "r"(member_mask), "r"(predicate));

        /*int tmp = __shfl_down_sync(member_mask, left, 1 << STEP);

        if (xlib::lane_id() + (1 << STEP) <= max_lane) {
            if (predicate)
                right += tmp;
            else
                left += tmp;
        }*/
    }
}

} // namespace xlib
