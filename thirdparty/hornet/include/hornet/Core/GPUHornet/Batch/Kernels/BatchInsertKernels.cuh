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
#include <Host/Algorithm.hpp>
#include <Device/Primitives/BinarySearchLB.cuh>
#include <Device/Util/DeviceProperties.cuh>

namespace hornets_nest {
namespace gpu {

template<typename HornetDevice>
__global__
void mergeAdjListKernel(HornetDevice                 hornet,
                        const degree_t* __restrict__ d_old_degrees,
                        const vid_t*    __restrict__ d_unique_src,
                        const int*      __restrict__ d_counts_ps,
                        int                          num_uniques,
                        const vid_t*    __restrict__ d_batch_dst) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride) {
        auto    vertex = hornet.vertex(d_unique_src[i]);
        auto  left_ptr = vertex.neighbor_ptr();
        auto left_size = d_old_degrees[i];

        int      start = d_counts_ps[i];
        int        end = d_counts_ps[i + 1];
        int right_size = end - start;
        auto right_ptr = d_batch_dst + start;

        xlib::inplace_merge(left_ptr, left_size, right_ptr, right_size);
    }
}

template<unsigned BLOCK_SIZE, typename HornetDevice>
__global__
void bulkCopyAdjLists(HornetDevice                 hornet,
                      const degree_t* __restrict__ d_prefixsum,
                      int                          prefixsum_size,
                      const vid_t*    __restrict__ d_batch_dst,
                      const vid_t*    __restrict__ d_unique,
                      const degree_t* __restrict__ d_old_degrees) {

    const auto& lambda = [&] (int pos, degree_t offset) {
                            auto        vertex = hornet.vertex(d_unique[pos]);
                            auto    vertex_ptr = vertex.neighbor_ptr() +
                                                 d_old_degrees[pos];
                            auto  batch_offset = d_prefixsum[pos] + offset;
                            vertex_ptr[offset] = d_batch_dst[batch_offset];
                        };
    xlib::simpleBinarySearchLB<BLOCK_SIZE>(d_prefixsum, prefixsum_size, nullptr, lambda);
}

} // namespace gpu
} // namespace hornets_nest
