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

namespace xlib {

/**
 *  DF: default (standard memory access)
 *  CG: cache global
 *  CS: cache streaming (read/store one or few times)
 *  CV: volatile (no optimized)
 *
 *  CA: cache all levels (L1, L2, etc)
 *  NC: read-only memory (standard memory access --> ldg)
 *  NC_CA: read-only memory and cache all levels
 *  NC_CG: read-only memory and cache global
 *  NC_CS: read-only memory and cache streaming
 *
 *  WB: Write Back
 */
enum CacheModifier { DF, CG, CS, CV,                // LOAD/STORE
                     CA, NC, NC_CA, NC_CG, NC_CS,   // Only LOAD
                     WB                             // Only STORE
                   };

template<CacheModifier MODIFIER = DF, typename T>
__device__ __forceinline__
T Load(T* pointer);

template<CacheModifier MODIFIER = DF, typename T, typename R>
__device__ __forceinline__
void Store(T* pointer, R value);

} // namespace xlib

#include "impl/CacheLoadModifier.i.cuh"
#include "impl/CacheStoreModifier.i.cuh"
