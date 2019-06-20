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
#include "Kernels/BatchOOPDeleteKernels.cuh"
#include <Device/Util/PrintExt.cuh>          //cu::printArray

//#define BATCH_DELETE_DEBUG

namespace hornets_nest {
namespace gpu {

static const unsigned BLOCK_SIZE = 256;

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::deleteOOPEdgeBatch(BatchUpdate& batch_update) noexcept {
    int num_uniques = batch_preprocessing(batch_update, true);
    //==========================================================================
    auto    batch_size = batch_update.size();
    vid_t* d_batch_src = batch_update.src_ptr();
    vid_t* d_batch_dst = batch_update.dst_ptr();

    collectOldDegreeKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
        (device_side(), _d_unique, num_uniques, _d_degree_tmp, _d_inverse_pos);
    CHECK_CUDA_ERROR

#if defined(BATCH_DELETE_DEBUG)
    std::cout << "num_uniques " << num_uniques << std::endl;
    cu::printArray(_d_degree_tmp, num_uniques, "_d_degree_tmp\n");
#endif
    xlib::CubExclusiveSum<degree_t>::srun(_d_degree_tmp, num_uniques + 1);
    //==========================================================================

    degree_t total_degree_old;      //get the total collected degree
    gpu::copyToHost(_d_degree_tmp + num_uniques, 1, &total_degree_old);
    gpu::memsetOne(_d_flags, total_degree_old);

    deleteEdgesKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(batch_size), BLOCK_SIZE >>>
        (device_side(), batch_update, _d_degree_tmp,
         _d_inverse_pos, _d_flags, _is_sorted);
    CHECK_CUDA_ERROR

    collectDataKernel   //modify also the vertices degree
        <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
        (device_side(), _d_unique, _d_counts, num_uniques,
         _d_degree_new, _d_ptrs_array);
    CHECK_CUDA_ERROR

#if defined(BATCH_DELETE_DEBUG)
    cu::printArray(_d_degree_new, num_uniques, "_d_degree_new\n");
#endif

    xlib::CubExclusiveSum<degree_t>::srun(_d_degree_new, num_uniques + 1);

#if defined(BATCH_DELETE_DEBUG)
    cu::printArray(_d_degree_tmp, num_uniques + 1, "_d_degree_old_prefix\n");
    cu::printArray(_d_degree_new, num_uniques + 1, "_d_degree_new_prefix\n");
#endif
    degree_t total_degree_new;                //get the total
    gpu::copyToHost(_d_degree_new + num_uniques, 1, &total_degree_new);

    //==========================================================================
    copySparseToContinuos(_d_degree_tmp, num_uniques + 1, total_degree_old,
                          _d_ptrs_array, _d_tmp);

    xlib::CubSelectFlagged<EdgeAllocT>
                            select(reinterpret_cast<EdgeAllocT*>(_d_tmp),
                                   total_degree_old, _d_flags);
    int tmp_size_new = select.run();
    assert(total_degree_new == tmp_size_new);

    if (total_degree_new > 0) {
        copyContinuosToSparse(_d_degree_new, num_uniques + 1, total_degree_new,
                              _d_tmp, _d_ptrs_array);
    }
    //==========================================================================
    build_batch_csr(batch_update, _batch_prop, num_uniques);
}

} // namespace gpu
} // namespace hornets_nest

#undef BATCH_DELETE_DEBUG
