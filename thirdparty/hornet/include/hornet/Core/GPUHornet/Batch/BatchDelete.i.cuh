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
#include "Kernels/BatchDeleteKernels.cuh"

namespace hornets_nest {
namespace gpu {

//#define BATCH_DELETE_DEBUG

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
    void HORNET::deleteEdgeBatch(BatchUpdate& batch_update, const BatchProperty batch_prop) noexcept {
        reserveBatchOpResource(batch_update.original_size(), batch_prop);
        const unsigned BLOCK_SIZE = 128;
        int num_uniques = batch_preprocessing(batch_update, batch_prop, false);
        //==========================================================================
        size_t  batch_size = batch_update.size();
        vid_t* d_batch_src = batch_update.src_ptr();
        vid_t* d_batch_dst = batch_update.dst_ptr();
        //--------------------------------------------------------------------------

        gpu::copyToDevice(_d_counts, num_uniques + 1, _d_batch_offset);
        cub_prefixsum.run(_d_batch_offset, num_uniques + 1);
        gpu::memsetZero(_d_counter, num_uniques + 1);
        gpu::memsetZero(_d_flags,        batch_size);

        ///////////////////
        // DELETE KERNEL //
        ///////////////////
        vertexDegreeKernel
            <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
            (device_side(), _d_unique, num_uniques, _d_degree_tmp);
        cub_prefixsum.run(_d_degree_tmp, num_uniques + 1);

        //It is possible that a user tries to delete (s,d) from hornet
        //even if degree(s) = 0
        degree_t degree_tmp_sum;
        gpu::copyToHost(_d_degree_tmp + num_uniques, 1, &degree_tmp_sum);

        if (degree_tmp_sum == 0) {
            return;
        }
        //location of batch edges in graph
        locateEdges<BLOCK_SIZE>
            <<< xlib::ceil_div(degree_tmp_sum, BLOCK_SIZE), BLOCK_SIZE >>>
            (device_side(),
             _d_degree_tmp, _d_batch_offset,
             _d_unique, _d_batch_dst,
             num_uniques + 1, _d_flags, _d_locations);
        cub_select_flag.run(_d_batch_src, batch_size, _d_flags);
        cub_select_flag.run(_d_batch_dst, batch_size, _d_flags);
        batch_size = cub_select_flag.run(_d_locations, batch_size, _d_flags);
        if (batch_size == 0) {
            return;
        }
        num_uniques = cub_runlength.run(d_batch_src, batch_size,
                _d_unique, _d_counts);
        gpu::copyToDevice(_d_counts, num_uniques + 1, _d_batch_offset);
        cub_prefixsum.run(_d_batch_offset, num_uniques + 1);
#ifdef BATCH_DELETE_DEBUG
    xlib::gpu::printArray(_d_batch_src, batch_size, "_d_batch_src:\n");
    xlib::gpu::printArray(_d_batch_dst, batch_size, "_d_batch_dst:\n");
    xlib::gpu::printArray(_d_unique, num_uniques, "_d_unique_src:\n");
    xlib::gpu::printArray(_d_counts, num_uniques, "_d_counts:\n");
    xlib::gpu::printArray(_d_locations, batch_size, "_d_locations:\n");
    std::cout<<"num_uniques "<<num_uniques<<"\n";
    std::cout<<xlib::ceil_div(batch_size, BLOCK_SIZE)<<"\n";
#endif

        cub_sort_pair.run(_d_batch_src, _d_locations, batch_size,
                _d_tmp_sort_src, _d_tmp_sort_dst, _nV, _nV);

        overwriteDeletedEdges<BLOCK_SIZE>
            <<< xlib::ceil_div(batch_size, BLOCK_SIZE), BLOCK_SIZE >>>
            (device_side(),
             _d_batch_offset, _d_locations, _d_counts,
             _d_unique, _d_batch_dst,
             num_uniques + 1, _d_counter);
        CHECK_CUDA_ERROR

        fixInternalRepresentation(num_uniques, false, false);

        if (batch_prop == batch_property::CSR)
            build_batch_csr(batch_update, batch_prop, num_uniques, !_is_sorted);

        _nE -= batch_size;
    }

#undef BATCH_DELETE_DEBUG
} // namespace gpu
} // namespace hornets_nest
