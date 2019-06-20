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
#include "BasicTypes.hpp"                   //vid_t
#include "Core/GPUHornet/HornetDevice.cuh"  //HornetDevice

namespace hornets_nest {
namespace gpu {

#define VERTEX Vertex<TypeList<VertexTypes...>,TypeList<EdgeTypes...>,FORCE_SOA>
#define   EDGE Edge<TypeList<VertexTypes...>,TypeList<EdgeTypes...>,FORCE_SOA>
#define HORNETDEVICE HornetDevice<TypeList<VertexTypes...>,\
                                  TypeList<EdgeTypes...>, FORCE_SOA>


template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
VERTEX::Vertex(HORNETDEVICE& hornet, vid_t index) :
                       _hornet(hornet),
                       _id(index),
                       AoSData<size_t, void*, VertexTypes...>(hornet[index]) {
    assert(index < hornet.nV());
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
vid_t VERTEX::id() const {
    return _id;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
degree_t VERTEX::degree() const {
    return this->template get<0>();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
size_t* VERTEX::degree_ptr() const {
    return _hornet.ptr<0>(_id);
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
typename HORNETDEVICE::edgeit_t VERTEX::edge_begin() const {
    return static_cast<edgeit_t>(this->template get<1>());
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
typename HORNETDEVICE::edgeit_t VERTEX::edge_end() const {
    return static_cast<edgeit_t>(this->template get<1>()) + degree();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
void VERTEX::set_degree(size_t degree) {
     this->template get<0>() = degree;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
degree_t VERTEX::limit() const {
    return ::max(static_cast<degree_t>(MIN_EDGES_PER_BLOCK),
                 PREFER_FASTER_UPDATE ? xlib::roundup_pow2(degree() + 1) :
                                        xlib::roundup_pow2(degree()));
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
vid_t VERTEX::neighbor_id(degree_t index) const {
    assert(index < degree());
    return reinterpret_cast<vid_t*>(this->template get<1>())[index];
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
vid_t* VERTEX::neighbor_ptr() const {
    return static_cast<vid_t*>(this->template get<1>());
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
template<typename T>
__device__ __forceinline__
VERTEX::WeightT* VERTEX::edge_weight_ptr() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 1,
                      "edge weight is not part of edge type list");
    return static_cast<WeightT*>(neighbor_ptr() + PITCH<EdgeTypes...>);
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
template<int INDEX>
__device__ __forceinline__
typename xlib::SelectType<INDEX, VertexTypes...>::type
VERTEX::field() const {
    return this->template get<INDEX>();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
EDGE VERTEX::edge(degree_t index) const {
    auto ptr = reinterpret_cast<edgeit_t>(this->template get<1>());
    return EDGE(_hornet, ptr + index, _id);
}

//------------------------------------------------------------------------------

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
void VERTEX::store(degree_t pos, const VERTEX::EdgeT& edge) {
    EdgesLayout data(this->template get<1>());
    data[pos] = edge;
}

//==============================================================================
//==============================================================================

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
EDGE::Edge(HORNETDEVICE& hornet, void* edge_ptr, vid_t src_id, void* ptr) :
        _hornet(hornet),
        AoSData<vid_t, EdgeTypes...>(
             AoSDataHelper<vid_t, EdgeTypes...>(edge_ptr, PITCH<EdgeTypes...>)),
        _src_id(src_id),
        _ptr(ptr) {}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
vid_t EDGE::src_id() const {
    return _src_id;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
vid_t EDGE::dst_id() const {
    return this->template get<0>();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
EDGE::VertexT EDGE::src() const {
    return VertexT(_hornet, _src_id);
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
__device__ __forceinline__
EDGE::VertexT EDGE::dst() const {
    return VertexT(_hornet, dst_id());
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
template<typename T>
__device__ __forceinline__
EDGE::WeightT EDGE::weight() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 1,
                  "weight is not part of edge type list");
    return this->template get<1>();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
template<typename T>
__device__ __forceinline__
void EDGE::set_weight(WeightT weight) {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 1,
                  "weight is not part of edge type list");
    //EdgesLayout data(_ptr);
    //*data.ptr<1>(0) = weight;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
template<typename T>
__device__ __forceinline__
EDGE::TimeStamp1T EDGE::time_stamp1() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 2,
                  "time_stamp1 is not part of edge type list");
    return this->template get<2>();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
template<typename T>
__device__ __forceinline__
EDGE::TimeStamp2T EDGE::time_stamp2() const {
    static_assert(sizeof(T) == sizeof(T) && NUM_ETYPES > 3,
                  "time_stamp2 is not part of edge type list");
    return this->template get<3>();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
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
