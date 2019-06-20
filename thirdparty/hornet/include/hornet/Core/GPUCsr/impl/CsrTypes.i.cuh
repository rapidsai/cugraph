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
#include "Core/GPUCsr/CsrDevice.cuh"  //HornetDevice
#include "BasicTypes.hpp"             //vid_t

#define       VERTEX VertexCsr<TypeList<VertexTypes...>,TypeList<EdgeTypes...>>
#define         EDGE EdgeCsr<TypeList<VertexTypes...>,TypeList<EdgeTypes...>>
#define HORNETDEVICE CsrDevice<TypeList<VertexTypes...>,\
                               TypeList<EdgeTypes...>>

namespace hornets_nest {
namespace gpu {

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
VERTEX::VertexCsr(HORNETDEVICE& hornet, vid_t index) :
                    _hornet(hornet),
                    _id(index),
                    AoSData<off2_t, VertexTypes...>(hornet.raw_vertex(index)) {
    assert(index < hornet.nV());
    auto v = this->template get<0>();
    _degree = v.y - v.x;
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
VERTEX::VertexCsr(HORNETDEVICE& hornet) : _hornet(hornet) {}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
vid_t VERTEX::id() const {
    return _id;
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
degree_t VERTEX::degree() const {
    return _degree;
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
eoff_t VERTEX::edge_begin() const {
    return this->template get<0>().x;
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
eoff_t VERTEX::edge_end() const {
    return this->template get<0>().y;
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
vid_t VERTEX::neighbor_id(degree_t index) const {
    assert(index < degree());
    return 3;// reinterpret_cast<vid_t*>(this->template get<1>())[index];
}
/*
template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
vid_t* VERTEX::neighbor_ptr() const {
    return static_cast<vid_t*>(this->template get<1>());
}*/

template<typename... VertexTypes, typename... EdgeTypes>
template<int INDEX>
__device__ __forceinline__
typename xlib::SelectType<INDEX, VertexTypes...>::type
VERTEX::field() const {
    return this->template get<INDEX>();
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
EDGE VERTEX::edge(degree_t index) const {
    auto v = this->template get<0>();
    return EDGE(_hornet, v.x + index, _id);
}

//==============================================================================
//==============================================================================

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
EDGE::EdgeCsr(HORNETDEVICE& hornet, eoff_t offset, vid_t src_id) :
                        _hornet(hornet),
                        AoSData<vid_t, EdgeTypes...>(hornet.raw_edge(offset)),
                        _src_id(src_id) {}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
vid_t EDGE::src_id() const {
    return _src_id;
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
vid_t EDGE::dst_id() const {
    return this->template get<0>();
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
EDGE::VertexT EDGE::src() const {
    return VertexT(_hornet, _src_id);
}

template<typename... VertexTypes, typename... EdgeTypes>
__device__ __forceinline__
EDGE::VertexT EDGE::dst() const {
    return VertexT(_hornet, dst_id());
}

template<typename... VertexTypes, typename... EdgeTypes>
template<typename T>
__device__ __forceinline__
EDGE::WeightT EDGE::weight() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 1,
                  "weight is not part of edge type list");
    return this->template get<1>();
}
/*
template<typename... VertexTypes, typename... EdgeTypes>
template<typename T>
__device__ __forceinline__
void EDGE::set_weight(WeightT weight) {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 1,
                  "weight is not part of edge type list");
}*/

template<typename... VertexTypes, typename... EdgeTypes>
template<typename T>
__device__ __forceinline__
EDGE::TimeStamp1T EDGE::time_stamp1() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 2,
                  "time_stamp1 is not part of edge type list");
    return this->template get<2>();
}

template<typename... VertexTypes, typename... EdgeTypes>
template<typename T>
__device__ __forceinline__
EDGE::TimeStamp2T EDGE::time_stamp2() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 3,
                  "time_stamp2 is not part of edge type list");
    return this->template get<3>();
}

template<typename... VertexTypes, typename... EdgeTypes>
template<int INDEX>
__device__ __forceinline__
typename xlib::SelectType<INDEX, EdgeTypes...>::type
EDGE::field() const {
    return this->template get<INDEX>();
}

} // namespace gpu
} // namespace hornets_nest

#undef VERTEX
#undef EDGE
#undef HORNETDEVICE
