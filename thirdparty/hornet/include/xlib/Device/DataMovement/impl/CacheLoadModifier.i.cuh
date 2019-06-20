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

#define Load_MACRO(CACHE_MOD, ptx_modifier)                                    \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
ulonglong2 LoadSupport<CACHE_MOD, ulonglong2>(ulonglong2* pointer) {           \
    ulonglong2 ret;                                                            \
    asm volatile("ld."#ptx_modifier".v2.u64 {%0, %1}, [%2];"                   \
                    : "=l"(ret.x), "=l"(ret.y) : "l"(pointer));                \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
uint4 LoadSupport<CACHE_MOD, uint4>(uint4* pointer) {                          \
    uint4 ret;                                                                 \
    asm volatile("ld."#ptx_modifier".v4.u32 {%0, %1, %2, %3}, [%4];"           \
                    : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)       \
                    : "l"(pointer));                                           \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
uint2 LoadSupport<CACHE_MOD, uint2>(uint2* pointer) {                          \
    uint2 ret;                                                                 \
    asm volatile("ld."#ptx_modifier".v2.u32 {%0, %1}, [%2];"                   \
                    : "=r"(ret.x), "=r"(ret.y) : "l"(pointer));                \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
ushort4 LoadSupport<CACHE_MOD, ushort4>(ushort4* pointer) {                    \
    ushort4 ret;                                                               \
    asm volatile("ld."#ptx_modifier".v4.u16 {%0, %1, %2, %3}, [%4];"           \
                    : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w)       \
                    : "l"(pointer));                                           \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
ushort2 LoadSupport<CACHE_MOD, ushort2>(ushort2* pointer) {                    \
    ushort2 ret;                                                               \
    asm volatile("ld."#ptx_modifier".v2.u16 {%0, %1}, [%2];"                   \
                    : "=h"(ret.x), "=h"(ret.y) : "l"(pointer));                \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
longlong2 LoadSupport<CACHE_MOD, longlong2>(longlong2* pointer) {              \
    longlong2 ret;                                                             \
    asm volatile("ld."#ptx_modifier".v2.s64 {%0, %1}, [%2];"                   \
                    : "=l"(ret.x), "=l"(ret.y) : "l"(pointer));                \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
int4 LoadSupport<CACHE_MOD, int4>(int4* pointer) {                             \
    int4 ret;                                                                  \
    asm volatile("ld."#ptx_modifier".v4.s32 {%0, %1, %2, %3}, [%4];"           \
                    : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)       \
                    : "l"(pointer));                                           \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
int2 LoadSupport<CACHE_MOD, int2>(int2* pointer) {                             \
    int2 ret;                                                                  \
    asm volatile("ld."#ptx_modifier".v2.s32 {%0, %1}, [%2];"                   \
                    : "=r"(ret.x), "=r"(ret.y) : "l"(pointer));                \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
short4 LoadSupport<CACHE_MOD, short4>(short4* pointer) {                       \
    short4 ret;                                                                \
    asm volatile("ld."#ptx_modifier".v4.s16 {%0, %1, %2, %3}, [%4];"           \
                    : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w)       \
                    : "l"(pointer));                                           \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
short2 LoadSupport<CACHE_MOD, short2>(short2* pointer) {                       \
    short2 ret;                                                                \
    asm volatile("ld."#ptx_modifier".v2.s16 {%0, %1}, [%2];"                   \
                    : "=h"(ret.x), "=h"(ret.y) : "l"(pointer));                \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
long long unsigned LoadSupport<CACHE_MOD, long long unsigned>                  \
    (long long unsigned* pointer) {                                            \
                                                                               \
    long long unsigned ret;                                                    \
    asm volatile("ld."#ptx_modifier".u64 %0, [%1];"                            \
                    : "=l"(ret) : "l"(pointer));                               \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
unsigned LoadSupport<CACHE_MOD, unsigned>(unsigned* pointer) {                 \
    unsigned ret;                                                              \
    asm volatile("ld."#ptx_modifier".u32 %0, [%1];"                            \
                    : "=r"(ret) : "l"(pointer));                               \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
unsigned short LoadSupport<CACHE_MOD, unsigned short>                          \
    (unsigned short* pointer) {                                                \
                                                                               \
    unsigned short ret;                                                        \
    asm volatile("ld."#ptx_modifier".u16 %0, [%1];"                            \
                    : "=h"(ret) : "l"(pointer));                               \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
unsigned char LoadSupport<CACHE_MOD, unsigned char>(unsigned char* pointer) {  \
    unsigned short ret;                                                        \
    asm volatile("ld."#ptx_modifier".u8 %0, [%1];"                             \
                    :"=h"(ret) : "l"(pointer));                                \
    return static_cast<unsigned char>(ret);                                    \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
    uchar2 LoadSupport<CACHE_MOD, uchar2>(uchar2* pointer) {                   \
    unsigned short ret = LoadSupport<CACHE_MOD>(                               \
                                reinterpret_cast<unsigned short*>(pointer));   \
    return reinterpret_cast<uchar2&>(ret);                                     \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
uchar4 LoadSupport<CACHE_MOD, uchar4>(uchar4* pointer) {                       \
    unsigned ret = LoadSupport<CACHE_MOD>(                                     \
                                reinterpret_cast<unsigned*>(pointer));         \
    return reinterpret_cast<uchar4&>(ret);                                     \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
long long int LoadSupport<CACHE_MOD, long long int>(long long int* pointer) {  \
    long long int ret;                                                         \
    asm volatile("ld."#ptx_modifier".s64 %0, [%1];"                            \
                    : "=l"(ret) : "l"(pointer));                               \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
int LoadSupport<CACHE_MOD, int>(int* pointer) {                                \
    int ret;                                                                   \
    asm volatile("ld."#ptx_modifier".s32 %0, [%1];"                            \
                    : "=r"(ret) : "l"(pointer));                               \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
short LoadSupport<CACHE_MOD, short>(short* pointer) {                          \
    short ret;                                                                 \
    asm volatile("ld."#ptx_modifier".s16 %0, [%1];"                            \
                    : "=h"(ret) : "l"(pointer));                               \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
char LoadSupport<CACHE_MOD, char>(char* pointer) {                             \
    short ret;                                                                 \
    asm volatile("ld."#ptx_modifier".s8 %0, [%1];"                             \
                    : "=h"(ret) : "l"(pointer));                               \
    return static_cast<char>(ret);                                             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
    char2 LoadSupport<CACHE_MOD, char2>(char2* pointer) {                      \
    short ret = LoadSupport<CACHE_MOD>(reinterpret_cast<short*>(pointer));     \
    return reinterpret_cast<char2&>(ret);                                      \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
char4 LoadSupport<CACHE_MOD, char4>(char4* pointer) {                          \
    int ret = LoadSupport<CACHE_MOD>(reinterpret_cast<int*>(pointer));         \
    return reinterpret_cast<char4&>(ret);                                      \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
double2 LoadSupport<CACHE_MOD, double2>(double2* pointer) {                    \
    double2 ret;                                                               \
    asm volatile("ld."#ptx_modifier".v2.f64 {%0, %1}, [%2];"                   \
                    : "=d"(ret.x), "=d"(ret.y) : "l"(pointer));                \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
float4 LoadSupport<CACHE_MOD, float4>(float4* pointer) {                       \
    float4 ret;                                                                \
    asm volatile("ld."#ptx_modifier".v4.f32 {%0, %1, %2, %3}, [%4];"           \
                    : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)       \
                    : "l"(pointer));                                           \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
float2 LoadSupport<CACHE_MOD, float2>(float2* pointer) {                       \
    float2 ret;                                                                \
    asm volatile("ld."#ptx_modifier".v2.f32 {%0, %1}, [%2];"                   \
                    : "=f"(ret.x), "=f"(ret.y) : "l"(pointer));                \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
float LoadSupport<CACHE_MOD, float>(float* pointer) {                          \
    float ret;                                                                 \
    asm volatile("ld."#ptx_modifier".f32 %0, [%1];"                            \
                    : "=f"(ret) : "l"(pointer));                               \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
double LoadSupport<CACHE_MOD, double>(double* pointer) {                       \
    double ret;                                                                \
    asm volatile("ld."#ptx_modifier".f64 %0, [%1];"                            \
                    : "=d"(ret) : "l"(pointer));                               \
    return ret;                                                                \
}

//==============================================================================
//==============================================================================

template<CacheModifier MODIFIER = DF>
struct ThreadLoad;

template<CacheModifier MODIFIER>
struct ThreadLoad {
    template<typename T>
    static __device__ __forceinline__ T op(T* pointer) {
        static_assert(sizeof(T) != sizeof(T), "NOT IMPLEMENTED");
        return *pointer;
    }
};

template<>
struct ThreadLoad<DF> {
    template<typename T>
    __device__ __forceinline__
    static T op(T* pointer) {
        return *pointer;
    }
};

template<>
struct ThreadLoad<NC> {
    template<typename T>
    __device__ __forceinline__
    static T op(T* pointer) {
        return __ldg(pointer);
    }
};

//==============================================================================

template<CacheModifier M, typename T>
__device__ __forceinline__  T LoadSupport(T* pointer);

#define LoadStruct_MACRO(CACHE_MOD)                                            \
                                                                               \
template<>                                                                     \
struct ThreadLoad<CACHE_MOD> {                                                 \
    template<typename T>                                                       \
    __device__ __forceinline__                                                 \
    static T op(T* pointer) {                                                  \
        return LoadSupport<CACHE_MOD>(                                         \
               const_cast<typename std::remove_cv<T>::type*>(pointer));        \
    }                                                                          \
};

LoadStruct_MACRO(CA)
LoadStruct_MACRO(CG)
LoadStruct_MACRO(CS)
LoadStruct_MACRO(CV)
LoadStruct_MACRO(NC_CA)
LoadStruct_MACRO(NC_CG)
LoadStruct_MACRO(NC_CS)

Load_MACRO(CA, global.ca)
Load_MACRO(CG, global.cg)
Load_MACRO(CS, global.cs)
Load_MACRO(CV, global.volatile)
Load_MACRO(NC_CA, global.ca.nc)
Load_MACRO(NC_CG, global.cg.nc)
Load_MACRO(NC_CS, global.cs.nc)

#undef LoadStruct_MACRO
#undef Load_MACRO

//==============================================================================
//==============================================================================

template<CacheModifier MODIFIER, typename T>
__device__ __forceinline__
T Load(T* pointer) {
    return ThreadLoad<MODIFIER>::op(pointer);
}

} // namespace xlib
