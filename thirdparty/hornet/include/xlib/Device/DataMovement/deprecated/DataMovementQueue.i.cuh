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
#include "Base/Device/Primitives/WarpScan.cuh"
#include "Base/Device/DataMovement/DataMovementDynamic.cuh"

namespace data_movement {
namespace Queue {

template<MODE QM,
         cub::CacheStoreModifier M,
         typename T, typename R, int SIZE>
__device__ __forceinline__
void Store(T (&Queue)[SIZE],
           const int size,
           T* __restrict__ queue_ptr,
           R* __restrict__ queue_size_ptr) {

    using namespace primitives;
    using namespace data_movement::dynamic;
    int th_offset = size;
    int warp_offset = WarpExclusiveScan<>::AddAtom(th_offset, queue_size_ptr);
    if (QM == MODE::SIMPLE) {
        thread::RegToGlobal_Simple<M>(Queue, size,
                                   queue_ptr + warp_offset + th_offset);
    } else if (QM == MODE::UNROLL) {
        thread::RegToGlobal_Unroll<M>(Queue, size,
                                   queue_ptr + warp_offset + th_offset);
    }
}

} //@Queue
} //@data_movement
