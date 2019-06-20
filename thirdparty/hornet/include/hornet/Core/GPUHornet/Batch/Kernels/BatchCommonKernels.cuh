/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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
#include "Core/DataLayout/DataLayoutDev.cuh"
#include "Core/GPUHornet/HornetDevice.cuh"
#include <Device/Util/DeviceQueue.cuh>

namespace hornets_nest {
namespace gpu {

template<typename HornetDevice>
__global__
void markUniqueKernel(HornetDevice              hornet,
                         const vid_t* __restrict__ d_batch_src,
                         const vid_t* __restrict__ d_batch_dst,
                         int                       batch_size,
                         bool*        __restrict__ d_flags) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < batch_size - 1; i += stride) {
        auto batch_src = d_batch_src[i];
        auto batch_dst = d_batch_dst[i];

        d_flags[i] = batch_src != d_batch_src[i + 1] ||
                     batch_dst != d_batch_dst[i + 1];
    }
    if (id == 0)
        d_flags[batch_size - 1] = true;
}

//==============================================================================
template<typename HornetDevice>
__global__
void markDuplicateSorted(HornetDevice              hornet,
                         const vid_t* __restrict__ d_batch_src,
                         const vid_t* __restrict__ d_batch_dst,
                         int                       batch_size,
                         bool*        __restrict__ d_flags) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < batch_size - 1; i += stride) {
        /*auto batch_src = d_batch_src[i];
        auto batch_dst = d_batch_dst[i];

        auto vertex_src = hornet.vertex(batch_src);
        auto    adj_ptr = vertex_src.neighbor_ptr();*/
        //WORK ONLY FOR NOT WEIGHTED GRAPHS --> degree iterator
        //d_flags[i] = xlib::binary_search(adj_ptr, vertex_src.degree(),
        //                           batch_dst) != -1;
    }
}

//------------------------------------------------------------------------------

template<typename HornetDevice>
__global__
void vertexDegreeKernel(HornetDevice              hornet,
                        const vid_t* __restrict__ d_array,
                        int                       size,
                        degree_t*   __restrict__  d_degrees) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (auto i = id; i < size; i += stride)
        d_degrees[i] = hornet.vertex(d_array[i]).degree();
}


template<int BLOCK_SIZE, typename HornetDevice>
__global__
void bulkMarkDuplicate(HornetDevice              hornet,
                       const int*   __restrict__ d_prefixsum,
                       const int*   __restrict__ d_batch_offsets,//offset in batch
                       const vid_t* __restrict__ d_batch_unique_src,
                       const vid_t* __restrict__ d_batch_dst,
                       int                       batch_size,
                       bool*        __restrict__ d_flags) {

    const auto& lambda = [&] (int pos, degree_t offset) {
                    auto     vertex = hornet.vertex(d_batch_unique_src[pos]);
                    assert(offset < vertex.degree());
                    auto        dst = vertex.edge(offset).dst_id();
                    int start = d_batch_offsets[pos];
                    int end   = d_batch_offsets[pos + 1];
                    int found = xlib::lower_bound_left(
                            d_batch_dst + start,
                            end - start,
                            dst);
                    if (found >= 0 && (dst == d_batch_dst[start + found])) {
                        d_flags[start + found] = false;
                    }

                };
    xlib::simpleBinarySearchLB<BLOCK_SIZE>(d_prefixsum, batch_size, nullptr, lambda);
}

//==============================================================================

template<typename HornetDevice>
__global__
void buildQueueKernel(HornetDevice               hornet,
                      const vid_t* __restrict__  d_unique,
                      const int*   __restrict__  d_counts,
                      int                        num_uniques,
                      vid_t*       __restrict__  queue_id,
                      void*        __restrict__ *queue_old_ptr,
                      degree_t*    __restrict__  queue_old_degree,
                      degree_t*    __restrict__  queue_new_degree,
                      int*         __restrict__  d_queue_size,
                      bool                       is_insert,
                      degree_t*    __restrict__  d_old_degrees = nullptr) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    xlib::DeviceQueueOffset queue(d_queue_size);

    for (auto i = id; i < num_uniques; i += stride) {
        vid_t        src = d_unique[i];
        auto     request = d_counts[i];
        auto       vnode = hornet.vertex(src);
        auto  new_degree = is_insert ? vnode.degree() + request :
                                       vnode.degree() - request;
        assert(new_degree < EDGES_PER_BLOCKARRAY);

        if (d_old_degrees != nullptr)
            d_old_degrees[i] = vnode.degree();

        bool flag = is_insert ? new_degree > vnode.limit() :
                                new_degree <= vnode.limit() / 2;
        if (flag) {
            int offset = queue.offset();
            queue_id[offset]         = src;
            queue_old_ptr[offset]    = vnode.neighbor_ptr();
            queue_old_degree[offset] = vnode.degree();
            queue_new_degree[offset] = new_degree;
        }
        else {
            vnode.set_degree(new_degree);
            hornet[src] = vnode;
        }
    }
}

template<typename HornetDevice>
__global__
void updateVertexDataKernel(HornetDevice                   hornet,
                            const vid_t*     __restrict__  d_queue_id,
                            const degree_t*  __restrict__  d_queue_new_degree,
                            void*                         *d_queue_new_ptr,
                            int                            queue_size) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < queue_size; i += stride) {
        auto data =  AoSData<size_t, void*>
                                    (static_cast<size_t>(d_queue_new_degree[i]),
                                     d_queue_new_ptr[i]);
        hornet[d_queue_id[i]] = data;
    }
}

} // namespace gpu
} // namespace hornets_nest
