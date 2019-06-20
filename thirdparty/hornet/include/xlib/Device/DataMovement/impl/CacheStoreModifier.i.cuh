/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
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
 *
 * @file
 */
#pragma once

#include <type_traits>  //std::remove_cv

namespace xlib {

#define Store_MACRO(CACHE_MOD, ptx_modifier)                                   \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, ulonglong2>                                       \
    (ulonglong2* pointer, ulonglong2 value) {                                  \
                                                                               \
    asm volatile("st."#ptx_modifier".v2.u64 [%0], {%1, %2};"                   \
                    : : "l"(pointer), "l"(value.x), "l"(value.y));             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, uint4>(uint4* pointer, uint4 value) {             \
    asm volatile("st."#ptx_modifier".v4.u32 [%0], {%1, %2, %3, %4};"           \
                    : : "l"(pointer), "r"(value.x), "r"(value.y),              \
                        "r"(value.z), "r"(value.w));                           \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, uint2>(uint2* pointer, uint2 value) {             \
    asm volatile("st."#ptx_modifier".v2.u32 [%0], {%1, %2};"                   \
                    : : "l"(pointer), "r"(value.x), "r"(value.y));             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, ushort4>(ushort4* pointer, ushort4 value) {       \
    asm volatile("st."#ptx_modifier".v4.u16 [%0], {%1, %2, %3, %4};"           \
                    : : "l"(pointer), "h"(value.x), "h"(value.y),              \
                          "h"(value.z), "h"(value.w));                         \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, ushort2>(ushort2* pointer, ushort2 value) {       \
    asm volatile("st."#ptx_modifier".v2.u16 [%0], {%1, %2};"                   \
                    : : "l"(pointer), "h"(value.x), "h"(value.y));             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, longlong2>                                        \
    (longlong2* pointer, longlong2 value) {                                    \
                                                                               \
    asm volatile("st."#ptx_modifier".v2.s64 [%0], {%1, %2};"                   \
                    : : "l"(pointer), "l"(value.x), "l"(value.y));             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, int4>(int4* pointer, int4 value) {                \
    asm volatile("st."#ptx_modifier".v4.s32 [%0], {%1, %2, %3, %4};"           \
                    : : "l"(pointer), "r"(value.x), "r"(value.y),              \
                          "r"(value.z), "r"(value.w));                         \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, int2>(int2* pointer, int2 value) {                \
    asm volatile("st."#ptx_modifier".v2.s32 [%0], {%1, %2};"                   \
                    : : "l"(pointer), "r"(value.x), "r"(value.y));             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, short4>(short4* pointer, short4 value) {          \
    asm volatile("st."#ptx_modifier".v4.s16 [%0], {%1, %2, %3, %4};"           \
                    : : "l"(pointer), "h"(value.x), "h"(value.y),              \
                          "h"(value.z), "h"(value.w));                         \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, short2>(short2* pointer, short2 value) {          \
    asm volatile("st."#ptx_modifier".v2.s16 [%0], {%1, %2};"                   \
                    : : "l"(pointer), "h"(value.x), "h"(value.y));             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, unsigned long long>                               \
    (unsigned long long* pointer, unsigned long long value) {                  \
                                                                               \
    asm volatile("st."#ptx_modifier".u64 [%0], %1;"                            \
                    : : "l"(pointer), "l"(value));                             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, unsigned>(unsigned* pointer, unsigned value) {    \
    asm volatile("st."#ptx_modifier".u32 [%0], %1;"                            \
                    : : "l"(pointer), "r"(value));                             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, unsigned short>                                   \
    (unsigned short* pointer, unsigned short value) {                          \
                                                                               \
    asm volatile("st."#ptx_modifier".u16 [%0], %1;"                            \
                    : : "l"(pointer),                                          \
                    "h"(static_cast<unsigned short>(value)));                  \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, long long int>                                    \
    (long long int* pointer, long long int value) {                            \
                                                                               \
    asm volatile("st."#ptx_modifier".s64 [%0], %1;"                            \
                    : : "l"(pointer), "l"(value));                             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, int>(int* pointer, int value) {                   \
    asm volatile("st."#ptx_modifier".s32 [%0], %1;"                            \
                    : : "l"(pointer), "r"(value));                             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, short>(short* pointer, short value) {             \
    asm volatile("st."#ptx_modifier".s16 [%0], %1;"                            \
                    : : "l"(pointer), "h"(value));                             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, char>(char* pointer, char value) {                \
    asm volatile("st."#ptx_modifier".s8 [%0], %1;"                             \
                    : : "l"(pointer), "h"(static_cast<short>(value)));         \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, char2>(char2* pointer, char2 value) {             \
    StoreSupport<CACHE_MOD>(reinterpret_cast<short*>(pointer),                 \
                            reinterpret_cast<short&>(value));                  \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, char4>(char4* pointer, char4 value) {             \
    StoreSupport<CACHE_MOD>(reinterpret_cast<int*>(pointer),                   \
                            reinterpret_cast<int&>(value));                    \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, unsigned char>                                    \
    (unsigned char* pointer, unsigned char value) {                            \
                                                                               \
    asm volatile("st."#ptx_modifier".u8 [%0], %1;"                             \
                    : : "l"(pointer),                                          \
                    "h"(static_cast<unsigned short>(value)));                  \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, uchar2>(uchar2* pointer, uchar2 value) {          \
    StoreSupport<CACHE_MOD>(reinterpret_cast<unsigned short*>(pointer),        \
                            reinterpret_cast<unsigned short&>(value));         \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, uchar4>(uchar4* pointer, uchar4 value) {          \
   StoreSupport<CACHE_MOD>(reinterpret_cast<unsigned*>(pointer),               \
                           reinterpret_cast<unsigned&>(value));                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, double2>(double2* pointer, double2 value) {       \
    asm volatile("st."#ptx_modifier".v2.f64 [%0], {%1, %2};"                   \
                    : : "l"(pointer), "d"(value.x), "d"(value.y));             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, double>(double* pointer, double value) {          \
    asm volatile("st."#ptx_modifier".f64 [%0], %1;"                            \
                    : : "l"(pointer), "d"(value));                             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, float4>(float4* pointer, float4 value) {          \
    asm volatile("st."#ptx_modifier".v4.f32 [%0], {%1, %2, %3, %4};"           \
                    : : "l"(pointer), "f"(value.x), "f"(value.y),              \
                        "f"(value.z), "f"(value.w));                           \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, float2>(float2* pointer, float2 value) {          \
    asm volatile("st."#ptx_modifier".v2.f32 [%0], {%1, %2};"                   \
                    : : "l"(pointer), "f"(value.x), "f"(value.y));             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
void StoreSupport<CACHE_MOD, float>(float* pointer, float value) {             \
    asm volatile("st."#ptx_modifier".f32 [%0], %1;"                            \
                    : : "l"(pointer), "f"(value));                             \
}

//==============================================================================
//==============================================================================

template<CacheModifier MODIFIER = DF>
struct ThreadStore;

template<CacheModifier MODIFIER>
struct ThreadStore {
    template<typename T, typename R>
    __device__ __forceinline__
    static void op(T* pointer, R value) {
        static_assert(sizeof(T) != sizeof(T), "NOT IMPLEMENTED");
    }
};

template<>
struct ThreadStore<DF> {
    template<typename T, typename R>
    __device__ __forceinline__
    static void op(T* pointer, R value) {
        *pointer = value;
    }
};

//==============================================================================
//==============================================================================

template<CacheModifier M, typename T>
__device__ __forceinline__  void StoreSupport(T* pointer, T value);

#define StoreStruct_MACRO(CACHE_MOD)                                           \
                                                                               \
template<>                                                                     \
struct ThreadStore<CACHE_MOD> {                                                \
    template<typename T, typename R>                                           \
    __device__ __forceinline__                                                 \
    static void op(T* pointer, R value) {                                      \
        return StoreSupport<CACHE_MOD>(                                        \
            const_cast<typename std::remove_cv<T>::type*>(pointer),            \
            value);                                                            \
    }                                                                          \
};

StoreStruct_MACRO(WB)
StoreStruct_MACRO(CG)
StoreStruct_MACRO(CS)
StoreStruct_MACRO(CV)

Store_MACRO(WB, global.wb)
Store_MACRO(CG, global.cg)
Store_MACRO(CS, global.cs)
Store_MACRO(CV, global.volatile)

#undef StoreStruct_MACRO
#undef Store_MACRO

//==============================================================================
//==============================================================================

template<CacheModifier MODIFIER, typename T, typename R>
__device__ __forceinline__
void Store(T* pointer, R value) {
    static_assert(std::is_same<typename std::remove_cv<T>::type,
                               typename std::remove_cv<R>::type>::value,
                  "Different Type: T != R");
    ThreadStore<MODIFIER>::op(pointer, value);
}

} // namespace xlib
