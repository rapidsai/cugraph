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
#include "Core/GPUHornet/HornetDevice.cuh"
#include "Core/DataLayout/DataLayoutDev.cuh"

namespace hornets_nest {
namespace gpu {

template<unsigned BLOCK_SIZE, int NUM_ETYPES,
         typename... EdgeTypes>
__global__
void copySparseToContinuosKernel(const degree_t* __restrict__  d_prefixsum,
                                 int                           num_items,
                                 void*           __restrict__ *d_ptrs_array,
                                 void*           __restrict__  d_tmp) {

    const int ITEMS_PER_BLOCK = xlib::smem_per_block<int, BLOCK_SIZE>();
    using DataLayout = BestLayoutDevPitchAux<PITCH<EdgeTypes...>,
                                             TypeList< vid_t, EdgeTypes...> >;
    __shared__ degree_t smem[ITEMS_PER_BLOCK];
    auto to_write = DataLayout(d_tmp);

    const auto& lambda = [&] (int pos, degree_t offset) {
                    int       tmp_offset = d_prefixsum[pos] + offset;
                    auto      edge_block = DataLayout(d_ptrs_array[pos]);
                    to_write[tmp_offset] = edge_block[offset];
                };
    xlib::binarySearchLB<BLOCK_SIZE>(d_prefixsum, num_items, smem, lambda);
}

template<unsigned BLOCK_SIZE, typename... EdgeTypes>
__global__
void copySparseToContinuosKernel(const degree_t* __restrict__ d_prefixsum,
                                 int                          num_items,
                                 void*           __restrict__ *d_ptrs_array,
                                 const int*      __restrict__ d_offsets,
                                 void*           __restrict__ d_tmp) {

    const int ITEMS_PER_BLOCK = xlib::smem_per_block<int, BLOCK_SIZE>();
    using DataLayout = BestLayoutDevPitchAux<PITCH<EdgeTypes...>,
                                             TypeList<vid_t, EdgeTypes...>>;
    __shared__ degree_t smem[ITEMS_PER_BLOCK];
    auto to_write = DataLayout(d_tmp);

    const auto& lambda = [&] (int pos, degree_t offset) {
            int       tmp_offset = d_prefixsum[pos] + offset + d_offsets[pos];
            auto      edge_block = DataLayout(d_ptrs_array[pos]);
            to_write[tmp_offset] = edge_block[offset];
        };
    xlib::binarySearchLB<BLOCK_SIZE>(d_prefixsum, num_items, smem, lambda);
}

template<unsigned BLOCK_SIZE, typename... EdgeTypes>
__global__
void copyContinuosToSparseKernel(
                              const degree_t* __restrict__  d_degree_new_prefix,
                              int                           num_items,
                              void*           __restrict__  d_tmp,
                              void*           __restrict__ *d_ptrs_array) {

    const int ITEMS_PER_BLOCK = xlib::smem_per_block<int, BLOCK_SIZE>();
    using DataLayout = BestLayoutDevPitchAux<PITCH<EdgeTypes...>,
                                             TypeList<vid_t, EdgeTypes...> >;
    __shared__ degree_t smem[ITEMS_PER_BLOCK];
    auto to_write = DataLayout(d_tmp);

    const auto& lambda = [&] (int pos, degree_t offset) {
            int       tmp_offset = d_degree_new_prefix[pos] + offset;
            auto      edge_block = DataLayout(d_ptrs_array[pos]);
            to_write[tmp_offset] = edge_block[offset];
        };
    xlib::binarySearchLB<BLOCK_SIZE>(d_degree_new_prefix, num_items, smem,
                                     lambda);
}

template<unsigned BLOCK_SIZE, typename... EdgeTypes>
__global__
void copySparseToSparseKernel(const degree_t* __restrict__  d_prefixsum,
                              int                           work_size,
                              void*                        *d_old_ptrs,
                              void*                        *d_new_ptrs) {
    //const int ITEMS_PER_BLOCK = xlib::smem_per_block<int, BLOCK_SIZE>();
    using DataLayout = BestLayoutDevPitchAux<PITCH<EdgeTypes...>,
                                             TypeList< vid_t, EdgeTypes... >>;

    //__shared__ degree_t smem[ITEMS_PER_BLOCK];
    const auto& lambda = [&] (int pos, degree_t offset) {
                                auto block_old = DataLayout(d_old_ptrs[pos]);
                                auto block_new = DataLayout(d_new_ptrs[pos]);
                                block_new[offset] = block_old[offset];
                            };
    xlib::simpleBinarySearchLB<BLOCK_SIZE>(d_prefixsum, work_size, nullptr, lambda);
}

template<bool = true>
__global__
void scatterDegreeKernel(const vid_t* __restrict__ d_unique,
                         const int*   __restrict__ d_counts,
                         int                       num_uniques,
                         int*         __restrict__ d_batch_offsets) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride)
        d_batch_offsets[d_unique[i]] = d_counts[i];
}

} // namespace gpu
} // namespace hornets_nest
