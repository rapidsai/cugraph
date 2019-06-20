/**
 * @brief
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
 *
 * @file
 */
#pragma once

#include <Host/Classes/PropertyClass.hpp>   //xlib::PropertyClass

namespace hornets_nest {
namespace gpu {

namespace detail {
    enum class BatchPropEnum { GEN_INVERSE = 1, IN_PLACE = 2, OUT_OF_PLACE = 4,
                               CSR = 8, CSR_WIDE = 16,
                               REMOVE_BATCH_DUPLICATE = 32,
                               REMOVE_CROSS_DUPLICATE = 64 };
} // namespace detail

class BatchProperty : public xlib::PropertyClass<detail::BatchPropEnum,
                                                 BatchProperty> {
public:
    explicit BatchProperty() noexcept = default;
    explicit BatchProperty(const detail::BatchPropEnum& obj) noexcept;
};

/**
 *  RANK: (fastest to slowest)
 *  - in-place unsorted insertion       done   (bulk copy)
 *  - out-of-place sorted deletion      ~done  (bulk copy + compact-by-flag)
 *  - out-of-place unsorted deletion    todo   (bulk copy + compact-by-flag)
 *  - out-of-place sorted insertion     todo   (bulk copy + segmented-sort)
 *  - in-place unsorted deletion        done   (one thread per batch edge)
 *  - in-place sorted deletion          done   (one thread per vertex)
 *  - in-place sorted insertion         done   (one thread per vertex)
 */
namespace batch_property {

    ///@brief Generate Inverse direction for each edge (0, 3) -> (3, 0)
    const BatchProperty GEN_INVERSE (detail::BatchPropEnum::GEN_INVERSE);

    ///@brief Low memory (less performance) insertion and deletion
    const BatchProperty IN_PLACE    (detail::BatchPropEnum::IN_PLACE);

    ///@brief Low memory (less performance) insertion and deletion
    const BatchProperty OUT_OF_PLACE (detail::BatchPropEnum::OUT_OF_PLACE);

    ///@brief Generate 'Wide' CSR representation of the batch.
    ///The size of CSR offsets is graph.nV
    const BatchProperty CSR_WIDE    (detail::BatchPropEnum::CSR_WIDE);

    const BatchProperty CSR         (detail::BatchPropEnum::CSR);

    ///@brief Remove duplicate edges in the batch
    const BatchProperty REMOVE_BATCH_DUPLICATE
                                (detail::BatchPropEnum::REMOVE_BATCH_DUPLICATE);

    const BatchProperty REMOVE_CROSS_DUPLICATE
                                (detail::BatchPropEnum::REMOVE_CROSS_DUPLICATE);
}

enum class BatchType { HOST, DEVICE };

//==============================================================================

template<typename... EdgeTypes>
class BatchUpdateClass {
    template<typename, typename, bool> friend class gpu::Hornet;

public:
    explicit BatchUpdateClass(vid_t* src_array, vid_t* dst_array,
                              EdgeTypes... additional_fiels, int batch_size,
                              BatchType batch_type = BatchType::HOST) noexcept;

    vid_t* original_src_ptr() const noexcept;
    vid_t* original_dst_ptr() const noexcept;
    int    original_size()    const noexcept;

    BatchType type() const noexcept;

    void print() const noexcept;

    HOST_DEVICE int size() const noexcept;

    HOST_DEVICE vid_t* src_ptr() const noexcept;

    HOST_DEVICE vid_t* dst_ptr() const noexcept;

    HOST_DEVICE const eoff_t* csr_offsets_ptr() const noexcept;

    HOST_DEVICE int csr_offsets_size() const noexcept;

    __device__ __forceinline__
    vid_t src(int index) const;

    __device__ __forceinline__
    vid_t dst(int index) const;

    __device__ __forceinline__
    vid_t csr_id(int index) const;

    __device__ __forceinline__
    int csr_offsets(int index) const;

    __device__ __forceinline__
    int csr_src_pos(vid_t vertex_id) const;

    __device__ __forceinline__
    int csr_wide_offsets(vid_t vertex_id) const;

    __device__ __forceinline__
    const eoff_t* csr_wide_offsets_ptr() const;

private:
    BatchType _batch_type    { BatchType::HOST };
    vid_t*    _src_array     { nullptr };   //original
    vid_t*    _dst_array     { nullptr };   //original
    int       _original_size { 0 };

    //device data
    vid_t* _d_src_array { nullptr };
    vid_t* _d_dst_array { nullptr };
    int    _batch_size  { 0 };

    //WARNING: byte pointers
    const byte_t* _field_ptrs[sizeof...(EdgeTypes) + 1] {};

    //CSR representation
    const vid_t*  _d_ids          { nullptr };
    const eoff_t* _d_offsets      { nullptr };
    const eoff_t* _d_wide_offsets { nullptr };
    int           _offsets_size   { 0 };
    const int*    _d_inverse_pos  { nullptr };
    vid_t         _nV             { 0 };

    //--------------------------------------------------------------------------
    void change_size(int d_batch_size) noexcept;

    void set_device_ptrs(vid_t* d_src_array, vid_t* d_dst_array,
                         int d_batch_size) noexcept;

    void set_csr(const vid_t*  d_ids,
                 const eoff_t* d_offsets, int offsets_size,
                 const eoff_t* d_inverse_pos = nullptr) noexcept;

    void set_wide_csr(const vid_t* d_src_array) noexcept;
};

using BatchUpdate = BatchUpdateClass<>;

template<typename T>
using BatchUpdateWeigth = BatchUpdateClass<T>;

} // namespace gpu
} // namespace hornets_nest

#include "impl/BatchUpdate.i.cuh"
