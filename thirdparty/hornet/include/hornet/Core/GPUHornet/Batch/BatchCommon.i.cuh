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
#include "Kernels/BatchCommonKernels.cuh"
#include <Device/Util/PrintExt.cuh>          //xlib::gpu::printArray

//#define DEBUG_FIXINTERNAL
//#define DEBUG_INSERT

namespace hornets_nest {
namespace gpu {

//#define BATCH_DEBUG

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::copyBatchUpdateData(
        const BatchUpdate& batch_update,
        const BatchProperty batch_prop,
        vid_t * const d_batch_src,
        vid_t * const d_batch_dst,
        size_t * const batch_size) noexcept {
    size_t b_size = batch_update.original_size();
    if (batch_update.type() == BatchType::HOST) {
        host::copyToDevice(batch_update.original_src_ptr(), b_size,
                         d_batch_src);
        host::copyToDevice(batch_update.original_dst_ptr(), b_size,
                         d_batch_dst);
    } else {
        gpu::copyToDevice(batch_update.original_src_ptr(), b_size,
                         d_batch_src);
        gpu::copyToDevice(batch_update.original_dst_ptr(), b_size,
                         d_batch_dst);
    }
    if (batch_prop & batch_property::GEN_INVERSE) {
        gpu::copyToDevice(d_batch_src, b_size,
                               d_batch_dst + b_size);
        gpu::copyToDevice(d_batch_dst, b_size,
                               d_batch_src + b_size);
        b_size *= 2;
    }

#if defined(BATCH_DEBUG)
    xlib::gpu::printArray(d_batch_src, b_size, "Batch Input:\n");
    xlib::gpu::printArray(d_batch_dst, b_size);
#endif

    *batch_size = b_size;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::sort_batch(
        BatchUpdate& batch_update,
        const BatchProperty batch_prop,
        const bool is_insert,
        vid_t ** d_batch_src_ptr,
        vid_t ** d_batch_dst_ptr,
        const size_t batch_size) noexcept {
    //The batch needs to be sorted in the following cases :
    //1. INSERT with REMOVE_CROSS_DUPLICATE
    //2. INSERT with REMOVE_BATCH_DUPLICATE
    //3. DELETE
    if ((batch_prop & batch_property::REMOVE_BATCH_DUPLICATE) ||
            (batch_prop & batch_property::REMOVE_CROSS_DUPLICATE) ||
            (is_insert && _is_sorted) || //legacy
            (!is_insert)) {
        //If duplicates are present within a batch,
        //sort (src, dst) pair to detect duplicates
        cub_sort_pair.run(_d_batch_src, _d_batch_dst, batch_size,
                          _d_tmp_sort_src, _d_tmp_sort_dst, _nV, _nV);
        *d_batch_src_ptr = _d_batch_src;
        *d_batch_dst_ptr = _d_batch_dst;
    } else {
        //User assumes that there are no duplicates within a batch
        cub_sort.run(_d_batch_src, _d_batch_dst, batch_size,
                      _d_tmp_sort_src, _d_tmp_sort_dst, _nV);
        *d_batch_src_ptr = _d_tmp_sort_src;
        *d_batch_dst_ptr = _d_tmp_sort_dst;
    }
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::remove_batch_duplicates(
        vid_t * d_batch_src,
        vid_t * d_batch_dst,
        size_t * const batch_size) noexcept {
    const unsigned BLOCK_SIZE = 128;
    size_t b_size = *batch_size;
    markUniqueKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(b_size), BLOCK_SIZE >>>
        (device_side(), d_batch_src, d_batch_dst, b_size, _d_flags);
    CHECK_CUDA_ERROR
    cub_select_flag.run(d_batch_src, b_size, _d_flags);
    *batch_size = cub_select_flag.run(d_batch_dst, b_size, _d_flags);
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::remove_cross_duplicates(
        BatchUpdate& batch_update,
        vid_t * const d_batch_src,
        vid_t * const d_batch_dst,
        size_t * const batch_size) noexcept {
    const unsigned BLOCK_SIZE = 128;
    size_t b_size = *batch_size;
    gpu::memset(_d_flags, b_size, 0x01);

    int num_uniques = cub_runlength.run(d_batch_src, b_size,
                                        _d_unique, _d_counts);
    gpu::copyToDevice(_d_counts, num_uniques + 1, _d_batch_offset);
    cub_prefixsum.run(_d_batch_offset, num_uniques + 1);

    vertexDegreeKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
        (device_side(), _d_unique, num_uniques, _d_degree_tmp);
    CHECK_CUDA_ERROR

    cub_prefixsum.run(_d_degree_tmp, num_uniques + 1);

#if defined(BATCH_DEBUG)
    xlib::gpu::printArray(_d_degree_tmp, num_uniques + 1, "Degree Prefix:\n");
#endif

    degree_t degree_tmp_sum;
    gpu::copyToHost(_d_degree_tmp + num_uniques, 1, &degree_tmp_sum);

    if (degree_tmp_sum == 0) {
        return;
    }

    bulkMarkDuplicate<BLOCK_SIZE>
        <<< xlib::ceil_div(degree_tmp_sum, BLOCK_SIZE), BLOCK_SIZE >>>
        (device_side(),
         _d_degree_tmp, _d_batch_offset,
         _d_unique, _d_batch_dst,
         num_uniques + 1, _d_flags);
    CHECK_CUDA_ERROR

#if defined(BATCH_DEBUG)
    xlib::gpu::printArray(_d_flags, b_size, "D Flags:\n");
#endif

    cub_select_flag.run(d_batch_src, b_size, _d_flags);
    *batch_size = cub_select_flag.run(d_batch_dst, b_size, _d_flags);
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
int HORNET::batch_preprocessing(
        BatchUpdate& batch_update,
        const BatchProperty batch_prop,
        bool is_insert) noexcept {

    vid_t * d_batch_src = _d_batch_src;
    vid_t * d_batch_dst = _d_batch_dst;
    size_t batch_size;
    copyBatchUpdateData(batch_update, batch_prop, d_batch_src, d_batch_dst, &batch_size);

    ///////////////////
    // SORT + UNIQUE //
    ///////////////////

    sort_batch(batch_update, batch_prop, is_insert, &d_batch_src, &d_batch_dst, batch_size);

    if (batch_prop & batch_property::REMOVE_BATCH_DUPLICATE) {
        remove_batch_duplicates(d_batch_src, d_batch_dst, &batch_size);
    }

    if (batch_prop & batch_property::REMOVE_CROSS_DUPLICATE) {
        if (!is_insert)
            WARNING("REMOVE_CROSS_DUPLICATE enabled with deletion!!")

        remove_cross_duplicates(batch_update, d_batch_src, d_batch_dst, &batch_size);
    }

    batch_update.set_device_ptrs(d_batch_src, d_batch_dst, batch_size);

    int num_uniques = cub_runlength.run(d_batch_src, batch_size,
                                        _d_unique, _d_counts);
#if defined(BATCH_DEBUG)
    xlib::gpu::printArray(d_batch_src, batch_size, "Preprocessing Gives:\n");
    xlib::gpu::printArray(d_batch_dst, batch_size);
    xlib::gpu::printArray(_d_unique, num_uniques, "Unique:\n");
    xlib::gpu::printArray(_d_counts, num_uniques, "Counts:\n");
#endif
    return num_uniques;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
int HORNET::batch_preprocessing(BatchUpdate& batch_update, bool is_insert)
                                noexcept {
    using namespace batch_property;
    const unsigned BLOCK_SIZE = 128;
    size_t batch_size = batch_update.original_size();

    if (batch_update.type() == BatchType::HOST) {
        host::copyToDevice(batch_update.original_src_ptr(), batch_size,
                         _d_batch_src);
        host::copyToDevice(batch_update.original_dst_ptr(), batch_size,
                         _d_batch_dst);
    }
    else {
        gpu::copyToDevice(batch_update.original_src_ptr(), batch_size,
                         _d_batch_src);
        gpu::copyToDevice(batch_update.original_dst_ptr(), batch_size,
                         _d_batch_dst);
    }
    if (_batch_prop & GEN_INVERSE) {
        gpu::copyToDevice(_d_batch_src, batch_size,
                               _d_batch_dst + batch_size);
        gpu::copyToDevice(_d_batch_dst, batch_size,
                               _d_batch_src + batch_size);
        batch_size *= 2;
    }
    ///////////////////
    // SORT + UNIQUE //
    ///////////////////
#if defined(BATCH_DEBUG)
    xlib::gpu::printArray(_d_batch_src, batch_size, "Batch Input:\n");
    xlib::gpu::printArray(_d_batch_dst, batch_size);
#endif
    auto d_batch_src = _d_batch_src;
    auto d_batch_dst = _d_batch_dst;

    if (_batch_prop & REMOVE_BATCH_DUPLICATE || (is_insert && _is_sorted)) {

        cub_sort_pair.run(_d_batch_src, _d_batch_dst, batch_size,
                          _d_tmp_sort_src, _d_tmp_sort_dst, _nV, _nV);
    }
    else {
        cub_sort.run(d_batch_src, d_batch_dst, batch_size,
                      _d_tmp_sort_src, _d_tmp_sort_dst, _nV);

        d_batch_src = _d_tmp_sort_src;
        d_batch_dst = _d_tmp_sort_dst;
    }

    if (_batch_prop & REMOVE_BATCH_DUPLICATE) {
            markUniqueKernel
            <<< xlib::ceil_div<BLOCK_SIZE>(batch_size), BLOCK_SIZE >>>
            (device_side(), _d_batch_src, _d_batch_dst, batch_size, _d_flags);
        CHECK_CUDA_ERROR

        if (_batch_prop & REMOVE_CROSS_DUPLICATE) {
            if (!is_insert)
                WARNING("REMOVE_CROSS_DUPLICATE enabled with deletion!!")

            if (_is_sorted) {
                markDuplicateSorted
                    <<< xlib::ceil_div<BLOCK_SIZE>(batch_size), BLOCK_SIZE >>>
                    (device_side(), _d_batch_src, _d_batch_dst,
                     batch_size, _d_flags);
            }
            //this should improve bulkMarkDuplicate() in case
            //there are many duplicates
            cub_select_flag.run(_d_batch_src, batch_size, _d_flags);
            batch_size = cub_select_flag.run(_d_batch_dst, batch_size,
                                             _d_flags);
            if (_is_sorted) //if sorted already done
                goto L1;
        }
    }

    if (_batch_prop & REMOVE_CROSS_DUPLICATE) {
        if (!is_insert)
            WARNING("REMOVE_CROSS_DUPLICATE enabled with deletion!!")

        if (_is_sorted) {
            markDuplicateSorted
                <<< xlib::ceil_div<BLOCK_SIZE>(batch_size), BLOCK_SIZE >>>
                (device_side(), _d_batch_src, _d_batch_dst,
                 batch_size, _d_flags);
        }
        else {
            vertexDegreeKernel
                <<< xlib::ceil_div<BLOCK_SIZE>(batch_size), BLOCK_SIZE >>>
                (device_side(), d_batch_src, batch_size, _d_degree_tmp);
            CHECK_CUDA_ERROR

            cub_prefixsum.run(_d_degree_tmp, batch_size + 1);

    #if defined(BATCH_DEBUG)
            xlib::gpu::printArray(_d_degree_tmp, batch_size + 1, "Degree Prefix:\n");
    #endif
            int smem = xlib::DeviceProperty::smem_per_block(BLOCK_SIZE);
            int num_blocks = xlib::ceil_div(batch_size, smem);

            bulkMarkDuplicate<BLOCK_SIZE>
                <<< num_blocks, BLOCK_SIZE >>>
                (device_side(), _d_degree_tmp, _d_batch_src, _d_batch_dst,
                 batch_size + 1, _d_flags);
            CHECK_CUDA_ERROR
#if defined(BATCH_DEBUG)
    xlib::gpu::printArray(_d_flags, batch_size, "D Flags:\n");
#endif
        }
        cub_select_flag.run(d_batch_src, batch_size, _d_flags);
        batch_size = cub_select_flag.run(d_batch_dst, batch_size, _d_flags);
    }
L1: batch_update.set_device_ptrs(d_batch_src, d_batch_dst, batch_size);

    //==========================================================================
    int num_uniques = cub_runlength.run(d_batch_src, batch_size,
                                        _d_unique, _d_counts);
#if defined(BATCH_DEBUG)
    xlib::gpu::printArray(d_batch_src, batch_size, "After Preprocessing:\n");
    xlib::gpu::printArray(d_batch_dst, batch_size);
    xlib::gpu::printArray(_d_unique, num_uniques, "Unique:\n");
    xlib::gpu::printArray(_d_counts, num_uniques, "Counts:\n");
#endif
    return num_uniques;
}

//==============================================================================

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::build_batch_csr(
        BatchUpdate& batch_update,
        const BatchProperty batch_prop,
        int num_uniques,
        bool require_prefix_sum) noexcept {
    const unsigned BLOCK_SIZE = 128;
    if (require_prefix_sum)
        xlib::CubExclusiveSum<int>::srun(_d_counts, num_uniques + 1);

    batch_update.set_csr(_d_unique, _d_counts, num_uniques);

    if (batch_prop == batch_property::CSR_WIDE) {
        gpu::memsetZero(_d_wide_csr, _nV + 1);
        scatterDegreeKernel
            <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
            (_d_unique, _d_counts, num_uniques, _d_wide_csr);

        xlib::CubExclusiveSum<int>::srun(_d_wide_csr, _nV + 1);
        batch_update.set_wide_csr(_d_wide_csr);
    }
}

//==============================================================================

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::fixInternalRepresentation(int num_uniques, bool is_insert,
                                       bool get_old_degree) noexcept {
    if (num_uniques == 0) {
        return;
    }
    const unsigned BLOCK_SIZE = 128;
    gpu::memsetZero(_d_queue_size);

    buildQueueKernel
        <<< xlib::ceil_div<BLOCK_SIZE>(num_uniques), BLOCK_SIZE >>>
        (device_side(), _d_unique, _d_counts, num_uniques,
         _d_queue_id, _d_queue_old_ptr, _d_queue_old_degree,
         _d_queue_new_degree, _d_queue_size, is_insert,
         get_old_degree ? _d_degree_tmp : nullptr);
    CHECK_CUDA_ERROR

    int h_num_realloc;
    gpu::copyToHost(_d_queue_size, 1, &h_num_realloc);

#if defined(DEBUG_FIXINTERNAL)
    std::cout << "h_num_realloc: " << h_num_realloc <<std::endl;
#endif
    //==========================================================================
    ////////////////////////////////////////
    // FIX HORNET INTERNAL REPRESENTATION //
    ////////////////////////////////////////
    if (h_num_realloc > 0) {
        gpu::copyToHost(_d_queue_new_degree, h_num_realloc, _h_queue_new_degree);
        gpu::copyToHost(_d_queue_old_ptr, h_num_realloc, _h_queue_old_ptr);
        gpu::copyToHost(_d_queue_old_degree, h_num_realloc, _h_queue_old_degree);

    #if defined(DEBUG_INSERT)
        xlib::gpu::printArray(_d_queue_id, h_num_realloc, "realloc ids:\n");
    #endif
        for (int i = 0; i < h_num_realloc; i++) {
            auto     new_degree = _h_queue_new_degree[i];
            _h_queue_new_ptr[i] = _mem_manager.insert(new_degree).second;
        }
        host::copyToDevice(_h_queue_new_ptr, h_num_realloc, _d_queue_new_ptr);

        //----------------------------------------------------------------------
        // update data structures
        updateVertexDataKernel
            <<< xlib::ceil_div<BLOCK_SIZE>(h_num_realloc), BLOCK_SIZE >>>
            (device_side(), _d_queue_id, _d_queue_new_degree,
             _d_queue_new_ptr, h_num_realloc);
        CHECK_CUDA_ERROR
        //----------------------------------------------------------------------
        //----------------------------------------------------------------------
        // Copy old adj list to new pointers
        degree_t * d_prefixsum;                         //alias

        if (is_insert) {
            d_prefixsum = _d_queue_old_degree;
        } else {
            d_prefixsum = _d_queue_new_degree;
        }
        int      prefixsum_size = h_num_realloc;         //alias
        cub_prefixsum.run(d_prefixsum, h_num_realloc + 1);

        degree_t prefixsum_total;                //get the total
        gpu::copyToHost(d_prefixsum + prefixsum_size, 1, &prefixsum_total);

        if (prefixsum_total > 0) {
            copySparseToSparse(d_prefixsum, prefixsum_size + 1, prefixsum_total,
                               _d_queue_old_ptr, _d_queue_new_ptr);
        }
        for (int i = 0; i < h_num_realloc; i++)
            _mem_manager.remove(_h_queue_old_ptr[i], _h_queue_old_degree[i]);
    }
}

} // namespace gpu
} // namespace hornets_nest

#undef DEBUG_FIXINTERNAL
