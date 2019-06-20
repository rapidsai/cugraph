
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
#include "Kernels/BatchCopyKernels.cuh"
#include "StandardAPI.hpp"
//#define DEBUG_FIXINTERNAL

namespace hornets_nest {
namespace gpu {

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::reserveBatchOpResource(const size_t max_batch_size,
                                  const BatchProperty batch_prop) noexcept {
    size_t max_b_size = max_batch_size;
    if (batch_prop & batch_property::GEN_INVERSE)
        max_b_size *= 2u;

    allocatePrepocessing(max_b_size);

    if (batch_prop & batch_property::IN_PLACE) {
        allocateInPlaceUpdate(max_b_size);
    } else {
        ERROR("Edge Batch operation OUT-OF-PLACE not implemented")
    }

    if (max_b_size > _max_batch_size) {
        _max_batch_size = max_b_size;
    }
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::allocateEdgeInsertion(const size_t max_batch_size,
                                   BatchProperty batch_prop) noexcept {
    size_t max_b_size = max_batch_size;
    _batch_prop = batch_prop;
    if (_batch_prop & batch_property::GEN_INVERSE)
        max_b_size *= 2u;
    auto csr_size = std::min(max_b_size, static_cast<size_t>(_nV));

    allocatePrepocessing(max_b_size);

    if (_batch_prop & batch_property::IN_PLACE)
        allocateInPlaceUpdate(max_b_size);
    else
        ERROR("Edge Batch insertion OUT-OF-PLACE not implemented")

    if (_batch_prop & batch_property::REMOVE_BATCH_DUPLICATE || _is_sorted)
        cub_sort_pair.initialize(max_b_size, false);
}

//==============================================================================

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::allocateInPlaceUpdate(const size_t max_batch_size) noexcept {
    //simple resizing when requested memory is greater than currently owned
    if (max_batch_size > _max_batch_size) {
        gpu::free(_d_locations, _d_batch_offset, _d_counter, _d_queue_new_degree, _d_queue_new_ptr, _d_queue_old_ptr, _d_queue_old_degree, _d_queue_id, _d_queue_size);
        host::freePageLocked(_h_queue_new_ptr, _h_queue_new_degree, _h_queue_old_ptr, _h_queue_old_degree);

        gpu::allocate(_d_locations, max_batch_size);
        gpu::allocate(_d_batch_offset, max_batch_size + 1);
        gpu::allocate(_d_counter,      max_batch_size + 1);
        gpu::allocate(_d_queue_new_degree, max_batch_size + 1);
        gpu::allocate(_d_queue_new_ptr,    max_batch_size);
        gpu::allocate(_d_queue_old_ptr,    max_batch_size);
        gpu::allocate(_d_queue_old_degree, max_batch_size + 1);
        gpu::allocate(_d_queue_id,         max_batch_size);
        gpu::allocate(_d_queue_size, 1);
        host::allocatePageLocked(_h_queue_new_ptr,    max_batch_size);
        host::allocatePageLocked(_h_queue_new_degree, max_batch_size);
        host::allocatePageLocked(_h_queue_old_ptr,    max_batch_size);
        host::allocatePageLocked(_h_queue_old_degree, max_batch_size + 1);
    }
    //_max_batch_size is set to max_batch_size by calling function
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::allocatePrepocessing(const size_t max_batch_size) noexcept {
    cub_prefixsum.resize(max_batch_size);
    cub_runlength.resize(max_batch_size);
    cub_select_flag.resize(max_batch_size);
    cub_sort.resize(max_batch_size);
    cub_sort_pair.resize(max_batch_size);

    if (max_batch_size > _max_batch_size) {

        gpu::free(_d_batch_src, _d_batch_dst, _d_tmp_sort_src, _d_tmp_sort_dst, _d_counts, _d_unique, _d_degree_tmp, _d_flags);

        hornets_nest::gpu::allocate(_d_batch_src,    max_batch_size);
        hornets_nest::gpu::allocate(_d_batch_dst,    max_batch_size);
        hornets_nest::gpu::allocate(_d_tmp_sort_src, max_batch_size);
        hornets_nest::gpu::allocate(_d_tmp_sort_dst, max_batch_size);
        hornets_nest::gpu::allocate(_d_counts,       max_batch_size + 1);
        hornets_nest::gpu::allocate(_d_unique,       max_batch_size);
        hornets_nest::gpu::allocate(_d_degree_tmp,   max_batch_size + 1);
        hornets_nest::gpu::allocate(_d_flags,        max_batch_size);
    }
}

//==============================================================================
//==============================================================================

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::copySparseToContinuos(const degree_t* prefixsum,
                                   int             prefixsum_size,
                                   int             total_sum,
                                   void**          sparse_ptrs,
                                   void*           continuous_array) noexcept {
    const unsigned BLOCK_SIZE = 256;
    int smem       = xlib::DeviceProperty::smem_per_block<int>(BLOCK_SIZE);
    int num_blocks = xlib::ceil_div(total_sum, smem);

    copySparseToContinuosKernel<BLOCK_SIZE, NUM_ETYPES, EdgeTypes...>
        <<< num_blocks, BLOCK_SIZE >>>
        (prefixsum, prefixsum_size, sparse_ptrs, continuous_array);
    CHECK_CUDA_ERROR
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::copySparseToContinuos(const degree_t* prefixsum,
                                   int             prefixsum_size,
                                   int             total_sum,
                                   void**          sparse_ptrs,
                                   const int*      continuos_offsets,
                                   void*           continuous_array) noexcept {
    const unsigned BLOCK_SIZE = 256;
    int smem       = xlib::DeviceProperty::smem_per_block<int>(BLOCK_SIZE);
    int num_blocks = xlib::ceil_div(total_sum, smem);

    copySparseToContinuosKernel<BLOCK_SIZE, EdgeTypes...>
        <<< num_blocks, BLOCK_SIZE >>>
        (prefixsum, prefixsum_size, sparse_ptrs,
         continuos_offsets, continuous_array);
    CHECK_CUDA_ERROR
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::copyContinuosToSparse(const degree_t* prefixsum,
                                   int             prefixsum_size,
                                   int             total_sum,
                                   void*           continuous_array,
                                   void**          sparse_ptrs) noexcept {
    const unsigned BLOCK_SIZE = 256;
    int smem       = xlib::DeviceProperty::smem_per_block<int>(BLOCK_SIZE);
    int num_blocks = xlib::ceil_div(total_sum, smem);

    copyContinuosToSparseKernel<BLOCK_SIZE, EdgeTypes...>
        <<< num_blocks, BLOCK_SIZE >>>
        (prefixsum, prefixsum_size, continuous_array, sparse_ptrs);
    CHECK_CUDA_ERROR
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::copySparseToSparse(const degree_t* d_prefixsum,
                                int             prefixsum_size,
                                int             prefixsum_total,
                                void**          d_old_ptrs,
                                void**          d_new_ptrs)
                                noexcept {
    const unsigned BLOCK_SIZE = 256;
    //int smem       = xlib::DeviceProperty::smem_per_block<int>(BLOCK_SIZE);
    int num_blocks = xlib::ceil_div(prefixsum_total, BLOCK_SIZE);

    copySparseToSparseKernel<BLOCK_SIZE, EdgeTypes...>
        <<< num_blocks, BLOCK_SIZE >>>
        (d_prefixsum, prefixsum_size, d_old_ptrs, d_new_ptrs);
    CHECK_CUDA_ERROR
}

#undef DEBUG_FIXINTERNAL

} // namespace gpu
} // namespace hornets_nest
