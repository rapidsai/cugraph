/**
 * @brief High-level API to access to Hornet data (VertexCsr, EdgeCsr)
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
 *
 * @file
 */
#define HORNET_DEVICE CsrDevice<TypeList<VertexTypes...>,\
                                TypeList<EdgeTypes...>>
namespace hornets_nest {
namespace gpu {

template<typename... VertexTypes, typename... EdgeTypes>
HORNET_DEVICE::CsrDevice(vid_t nV, eoff_t nE,
                         void* d_vertex_ptr, size_t vertex_pitch,
                         void* d_edge_ptr,   size_t edge_pitch) noexcept :
    _nV(nV), _nE(nE),
    BestLayoutDevAux< TypeList< off2_t, VertexTypes...> >
                    (d_vertex_ptr, vertex_pitch),
    BestLayoutDevAux< TypeList< vid_t, EdgeTypes...> >
                    (d_edge_ptr, edge_pitch) {}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
vid_t HORNET_DEVICE::nV() const noexcept {
    return _nV;
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
eoff_t HORNET_DEVICE::nE() const noexcept {
    return _nE;
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
HORNET_DEVICE::VertexT HORNET_DEVICE::vertex(vid_t index) {
    return VertexT(*this, index);
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
HORNET_DEVICE::VertexT HORNET_DEVICE::fake_vertex() {
    return VertexT(*this);
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
HORNET_DEVICE::EdgeT HORNET_DEVICE::edge(eoff_t offset) {
    return EdgeT(*this, offset);
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
AoSData<off2_t, VertexTypes...> HORNET_DEVICE::raw_vertex(vid_t index) const {
    return BestLayoutDevAux< TypeList<off2_t, VertexTypes...> >
            ::operator[](index);
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
AoSData<vid_t, EdgeTypes...> HORNET_DEVICE::raw_edge(eoff_t offset) const {
    return BestLayoutDevAux< TypeList<vid_t, EdgeTypes...> >
            ::operator[](offset);
}

} // namespace gpu
} // namespace hornets_nest

#undef HORNET_DEVICE
