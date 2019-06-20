/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
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
#include <Host/Algorithm.hpp>                     //xlib::binary_search
#include <Device/Primitives/BinarySearchLB.cuh>   //xlib::binarySearchLB

namespace hornets_nest {
namespace gpu {

template<typename HornetDevice>
__global__
void collectOldDegreeKernel(HornetDevice              hornet,
                            const vid_t* __restrict__ d_unique,
                            int                       num_uniques,
                            degree_t*    __restrict__ d_degree_old,
                            eoff_t*      __restrict__ d_inverse_pos) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride) {
        auto        src = d_unique[i];
        d_degree_old[i] = hornet.vertex(src).degree();
        d_inverse_pos[src] = i;
    }
}

template<typename HornetDevice>
__global__
void deleteEdgesKernel(HornetDevice                 hornet,
                       BatchUpdate                  batch_update,
                       const degree_t* __restrict__ d_degree_old_prefix,
                       const eoff_t*   __restrict__ d_inverse_pos,
                       bool*           __restrict__ d_flags,
                       bool                         sorted) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < batch_update.size(); i += stride) {
        vid_t src = batch_update.src(i);
        vid_t dst = batch_update.dst(i);

        auto src_vertex = hornet.vertex(src);
        auto    adj_ptr = src_vertex.neighbor_ptr();

        int pos;
        if (sorted)
            pos = xlib::binary_search(adj_ptr, src_vertex.degree(), dst);
        else {
            for (degree_t j = 0; j < src_vertex.degree(); j++) {
                if (adj_ptr[j] == dst) {
                    pos = j;
                    break;
                }
            }
        }
        int inverse_pos = d_inverse_pos[src];
        d_flags[d_degree_old_prefix[inverse_pos] + pos] = false;
        //printf("del %d %d \t%d\t%d\t%d\n",
        //    src, dst, d_degree_old_prefix[inverse_pos], pos,
        //    d_degree_old_prefix[inverse_pos] + pos);
    }
}

//collect d_ptrs_array, d_degree_new and update hornet degree
template<typename HornetDevice>
__global__
void collectDataKernel(HornetDevice              hornet,
                       const vid_t* __restrict__ d_unique,
                       degree_t*    __restrict__ d_count,
                       int                       num_uniques,
                       degree_t*    __restrict__ d_degree_new,
                       void**       __restrict__ d_ptrs_array) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride) {
        auto        src = d_unique[i];
        auto     vertex = hornet.vertex(src);
        auto new_degree = vertex.degree() - d_count[i];

        d_ptrs_array[i] = vertex.neighbor_ptr();
        d_degree_new[i] = new_degree;

        hornet[src] = AoSData<size_t, void*>(static_cast<size_t>(new_degree),
                                             vertex.neighbor_ptr());
    }
}

} // namespace gpu
} // namespace hornets_nest
