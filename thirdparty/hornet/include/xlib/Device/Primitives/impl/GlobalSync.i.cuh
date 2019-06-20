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

#include "Device/DataMovement/CacheModifier.cuh"

namespace xlib {
/*
template<unsigned BLOCK_SIZE>
__device__ __forceinline__ void globalSync_v2() {
    volatile auto volatile_ptr = GlobalSyncArray;
    __syncthreads(); //-> all warps conclude their work

    if (blockIdx.x == 0) {
        if (threadIdx.x == 0)
            volatile_ptr[blockIdx.x] = 1;

        for (int id = threadIdx.x; id < gridDim.x; id += BLOCK_SIZE)
            while ( Load<CG>(GlobalSyncArray + id) == 0 );

        __syncthreads();

        for (int id = threadIdx.x; id < gridDim.x; id += BLOCK_SIZE)
            Store<CG>(GlobalSyncArray + id, 0u);
    }
    else {
        if (threadIdx.x == 0) {
            volatile_ptr[blockIdx.x] = 1;
            while ( Load<CG>(GlobalSyncArray + blockIdx.x) == 1 );
        }
        __syncthreads();
    }
}*/

#if !defined(GLOBAL_SYNC)

__device__ __forceinline__
void global_sync() {
    if (blockIdx.x == 0) {
        int limit = gridDim.x - 1;
        for (int id = threadIdx.x; id < limit; id += blockDim.x)
            while ( Load<CV>(global_sync_array + id) == 0 ); // {
            //    __threadfence_block();
            //}

        __syncthreads();

        for (int id = threadIdx.x; id < limit; id += blockDim.x)
            Store<CG>(global_sync_array + id, 0u);
    }
    else {
        if (threadIdx.x == 0) {
            int idx = blockIdx.x - 1;
            volatile auto volatile_ptr = global_sync_array;
            volatile_ptr[idx] = 1;
            while ( Load<CV>(global_sync_array + idx) == 1 ) {
                ;//__threadfence_block();
            }
        }
        __syncthreads();
    }
}

#endif
} // namespace xlib
