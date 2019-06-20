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
#include "Core/HornetInit.hpp"
#include <Device/Util/SafeCudaAPI.cuh>
#include <Device/Util/PrintExt.cuh>

namespace hornets_nest {
namespace gpu {

inline BatchProperty::BatchProperty(const detail::BatchPropEnum& obj) noexcept :
    xlib::PropertyClass<detail::BatchPropEnum, BatchProperty>(obj) {}

//==============================================================================

template<typename... EdgeTypes>
BatchUpdateClass<EdgeTypes...>
::BatchUpdateClass(vid_t* src_array, vid_t* dst_array,
                   EdgeTypes... additional_fiels,
                   int batch_size, BatchType batch_type)
                   noexcept : _src_array(src_array),
                              _dst_array(dst_array),
                              _original_size(batch_size),
                              _batch_type(batch_type) {
    hornets_nest::bind(_field_ptrs, 0, additional_fiels...);
}

template<typename... EdgeTypes>
vid_t* BatchUpdateClass<EdgeTypes...>::original_src_ptr() const noexcept {
    return _src_array;
}

template<typename... EdgeTypes>
vid_t* BatchUpdateClass<EdgeTypes...>::original_dst_ptr() const noexcept {
    return _dst_array;
}

template<typename... EdgeTypes>
int BatchUpdateClass<EdgeTypes...>::original_size() const noexcept {
    return _original_size;
}

template<typename... EdgeTypes>
BatchType BatchUpdateClass<EdgeTypes...>::type() const noexcept {
    return _batch_type;
}

template<typename... EdgeTypes>
void BatchUpdateClass<EdgeTypes...>::print() const noexcept {
    if (_batch_type == BatchType::HOST) {
        xlib::printArray(_src_array, _original_size,
                         "Source/Destination Arrays:\n");
        xlib::printArray(_dst_array, _original_size);
    }
    else {
        xlib::gpu::printArray(_src_array, _original_size,
                        "Source/Destination Arrays:\n");
        xlib::gpu::printArray(_dst_array, _original_size);
    }
}

//------------------------------------------------------------------------------

template<typename... EdgeTypes>
HOST_DEVICE int BatchUpdateClass<EdgeTypes...>::size() const noexcept {
    return _batch_size;
}

template<typename... EdgeTypes>
HOST_DEVICE vid_t* BatchUpdateClass<EdgeTypes...>::src_ptr() const noexcept {
    return _d_src_array;
}

template<typename... EdgeTypes>
HOST_DEVICE vid_t* BatchUpdateClass<EdgeTypes...>::dst_ptr() const noexcept {
    return _d_dst_array;
}

template<typename... EdgeTypes>
HOST_DEVICE const eoff_t*
BatchUpdateClass<EdgeTypes...>::csr_offsets_ptr() const noexcept {
    assert(_d_offsets != nullptr);
    return _d_offsets;
}

template<typename... EdgeTypes>
HOST_DEVICE int BatchUpdateClass<EdgeTypes...>
::csr_offsets_size() const noexcept {
    assert(_offsets_size != 0);
    return _offsets_size;
}

template<typename... EdgeTypes>
__device__ __forceinline__
vid_t BatchUpdateClass<EdgeTypes...>::src(int index) const {
    assert(index < _batch_size);
    return _d_src_array[index];
}

template<typename... EdgeTypes>
__device__ __forceinline__
vid_t BatchUpdateClass<EdgeTypes...>::dst(int index) const {
    assert(index < _batch_size);
    return _d_dst_array[index];
}

template<typename... EdgeTypes>
__device__ __forceinline__
vid_t BatchUpdateClass<EdgeTypes...>::csr_id(int index) const {
    assert(_d_ids != nullptr);
    assert(index < _offsets_size);
    return _d_ids[index];
}

template<typename... EdgeTypes>
__device__ __forceinline__
int BatchUpdateClass<EdgeTypes...>::csr_offsets(int index) const {
    assert(_d_offsets != nullptr);
    assert(index < _offsets_size);
    return _d_offsets[index];
}

template<typename... EdgeTypes>
__device__ __forceinline__
int BatchUpdateClass<EdgeTypes...>::csr_src_pos(vid_t vertex_id) const {
    assert(_d_inverse_pos != nullptr);
    assert(vertex_id < _nV);
    return _d_inverse_pos[vertex_id];
}

template<typename... EdgeTypes>
__device__ __forceinline__
int BatchUpdateClass<EdgeTypes...>::csr_wide_offsets(vid_t vertex_id) const {
    assert(_d_inverse_pos != nullptr);
    assert(vertex_id < _nV);
    //return _d_offsets[_d_inverse_pos[vertex_id]];
    return _d_wide_offsets[vertex_id];
}

template<typename... EdgeTypes>
__device__ __forceinline__
const eoff_t* BatchUpdateClass<EdgeTypes...>::csr_wide_offsets_ptr() const {
    assert(_d_inverse_pos != nullptr);
    return _d_wide_offsets;
}

template<typename... EdgeTypes>
void BatchUpdateClass<EdgeTypes...>::change_size(int d_batch_size) noexcept {
    _batch_size  = d_batch_size;
}

template<typename... EdgeTypes>
void BatchUpdateClass<EdgeTypes...>
::set_device_ptrs(vid_t* d_src_array, vid_t* d_dst_array,
                  int d_batch_size) noexcept {
    _d_src_array = d_src_array;
    _d_dst_array = d_dst_array;
    _batch_size  = d_batch_size;
}

template<typename... EdgeTypes>
void BatchUpdateClass<EdgeTypes...>
::set_csr(const vid_t*  d_ids,
          const eoff_t* d_offsets, int offsets_size,
          const eoff_t* d_inverse_pos) noexcept {
    _d_offsets     = d_offsets;
    _offsets_size  = offsets_size;
    _d_inverse_pos = d_inverse_pos;
}

template<typename... EdgeTypes>
void BatchUpdateClass<EdgeTypes...>
::set_wide_csr(const eoff_t* d_wide_offsets) noexcept {
    _d_wide_offsets = d_wide_offsets;
}

} // namespace gpu
} // namespace hornets_nest
