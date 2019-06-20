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

#include "Base/Device/Util/include/Definition.cuh"
#if defined(ARCH_DEF)


enum MEM_SPACE { GLOBAL, SHARED };

namespace data_movement {
namespace warp {

template<int SIZE, typename T>
void __device__ __forceinline__ computeGlobalOffset(T* __restrict__ &SMem_ptr,
                                                    T* __restrict__ &Glob_ptr);

/**
* Always ordered
*/
template<int SIZE,
         cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         typename T>
void __device__ __forceinline__ SharedToGlobal(T* __restrict__ SMem,
                                               T* __restrict__ Pointer);

/**
* Always ordered
*/
template<int SIZE,
         cub::CacheLoadModifier M = cub::CacheLoadModifier::LOAD_DEFAULT,
         typename T>
void __device__ __forceinline__ GlobalToShared(T* __restrict__ Pointer,
                                              T* __restrict__ SMem);


/**
* Always ordered
*//*
template<int SIZE,
         cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         typename T>
void __device__ __forceinline__ SharedToGlobal(T* __restrict__ SMem,
                                               T* __restrict__ Pointer);

/**
* Always ordered
*//*
template<int SIZE,
         cub::CacheLoadModifier M = cub::CacheLoadModifier::LOAD_DEFAULT,
         typename T>
void __device__ __forceinline__ GlobalToShared(T* __restrict__ Pointer,
                                              T* __restrict__ SMem);*/

} //@warp_ordered_adv

//------------------------------------------------------------------------------

namespace warp_ordered {

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
        typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ SMem,
                                            T* __restrict__ Pointer);

template<cub::CacheLoadModifier M = cub::CacheLoadModifier::LOAD_DEFAULT,
        typename T, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T* __restrict__ SMem,
                                            T (&Queue)[SIZE]);

/**
* Can involve bank conflict
*/
template<typename T, int SIZE>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem);

/**
* Can involve bank conflict
*/
template<typename T, int SIZE>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]);

} //@ordered

namespace warp_ordered_adv {

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
        typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ SMem,
                                            T* __restrict__ SMemThread,
                                            T* __restrict__ Pointer);

template<cub::CacheLoadModifier M = cub::CacheLoadModifier::LOAD_DEFAULT,
        typename T, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T* __restrict__ SMem,
                                            T* __restrict__ SMemThread,
                                            T (&Queue)[SIZE]);

/**
* Can involve bank conflict
*/
template<typename T, int SIZE>
__device__ __forceinline__ void RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem);

/**
* Can involve bank conflict
*/
template<typename T, int SIZE>
__device__ __forceinline__ void SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]);

} //@warp_ordered_adv

//------------------------------------------------------------------------------

namespace unordered {

template<cub::CacheStoreModifier M = cub::CacheStoreModifier::STORE_DEFAULT,
         typename T, int SIZE>
__device__ __forceinline__ void RegToGlobal(T (&Queue)[SIZE],
                                            T* __restrict__ Pointer);

template<cub::CacheLoadModifier M = cub::CacheLoadModifier::LOAD_DEFAULT,
         typename T, int SIZE>
__device__ __forceinline__ void GlobalToReg(T* __restrict__ Pointer,
                                            T (&Queue)[SIZE]);

template<int SIZE, typename T>
void __device__ __forceinline__ RegToShared(T (&Queue)[SIZE],
                                            T* __restrict__ SMem);

template<int SIZE, typename T>
void __device__ __forceinline__ SharedToReg(T* __restrict__ SMem,
                                            T (&Queue)[SIZE]);

} //@unordered
} //@data_movement

#include "impl/DataMovementWarp.i.cuh"
#include "impl/DataMovementWarpUnordered.i.cuh"
#include "impl/DataMovementWarpOrdered.i.cuh"

#endif
