/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
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
#include <algorithm>
#include <cassert>

namespace graph {

////////////////////////////////
///         Vertex           ///
////////////////////////////////
template<typename vid_t, typename eoff_t>
inline GraphStd<vid_t, eoff_t>
::Vertex::Vertex(vid_t id, const GraphStd& graph) noexcept : _graph(graph),
                                                            _id(id) {};

template<typename vid_t, typename eoff_t>
inline vid_t GraphStd<vid_t, eoff_t>::Vertex::id() const noexcept {
    return _id;
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::degree_t
GraphStd<vid_t, eoff_t>::Vertex::out_degree() const noexcept {
    return _graph._out_degrees[_id];
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::degree_t
GraphStd<vid_t, eoff_t>::Vertex::in_degree() const noexcept {
    return _graph._in_degrees[_id];
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::Edge
GraphStd<vid_t, eoff_t>::Vertex::edge(int index) const noexcept {
    assert(index < out_degree());
    return Edge(_id, _graph._out_offsets[_id] + index, _graph);
}

template<typename vid_t, typename eoff_t>
inline vid_t
GraphStd<vid_t, eoff_t>::Vertex::neighbor_id(int index) const noexcept {
    assert(index < out_degree());
    return _graph._out_edges[_graph._out_offsets[_id] + index];
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::EdgeIt
GraphStd<vid_t, eoff_t>::Vertex::begin() const noexcept {
    return EdgeIt(_graph._out_edges + _graph._out_offsets[_id], _graph);
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::EdgeIt
GraphStd<vid_t, eoff_t>::Vertex::end() const noexcept {
    return EdgeIt(_graph._out_edges + _graph._out_offsets[_id + 1], _graph);
}
//==============================================================================
////////////////////////////////
///         VertexIt         ///
////////////////////////////////
template<typename vid_t, typename eoff_t>
inline GraphStd<vid_t, eoff_t>::VertexIt
::VertexIt(eoff_t* current, const GraphStd& graph) noexcept :
                    _graph(graph), _current(current) {}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::VertexIt&
GraphStd<vid_t, eoff_t>::VertexIt::VertexIt::operator++ () noexcept {
    _current++;
    return *this;
}

template<typename vid_t, typename eoff_t>
inline bool
GraphStd<vid_t, eoff_t>::VertexIt::operator!= (const VertexIt& it)
                                        const noexcept {
    return _current != it._current;
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::Vertex
GraphStd<vid_t, eoff_t>::VertexIt::operator* () const noexcept {
    return _graph.vertex(static_cast<vid_t>(_current - _graph._out_offsets));
}
//==============================================================================
////////////////////////////////
///         Edge             ///
////////////////////////////////
template<typename vid_t, typename eoff_t>
inline GraphStd<vid_t, eoff_t>
::Edge::Edge(vid_t src_id, eoff_t id, const GraphStd& graph) noexcept : _graph(graph),
                                                         _edge_id(id), _src_id(src_id) {};

template<typename vid_t, typename eoff_t>
inline eoff_t GraphStd<vid_t, eoff_t>::Edge::id() const noexcept {
    return _edge_id;
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::Vertex
GraphStd<vid_t, eoff_t>::Edge::src() const noexcept {
    return _graph.vertex(_src_id);
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::Vertex
GraphStd<vid_t, eoff_t>::Edge::dst() const noexcept {
    return _graph.vertex(_graph._out_edges[_edge_id]);
}

template<typename vid_t, typename eoff_t>
inline vid_t GraphStd<vid_t, eoff_t>::Edge::src_id() const noexcept {
    return _src_id;
}

template<typename vid_t, typename eoff_t>
inline vid_t GraphStd<vid_t, eoff_t>::Edge::dst_id() const noexcept {
    return _graph._out_edges[_edge_id];
}
//==============================================================================
////////////////////////////////
///         EdgeIt           ///
////////////////////////////////
template<typename vid_t, typename eoff_t>
inline GraphStd<vid_t, eoff_t>::EdgeIt
::EdgeIt(vid_t* current, const GraphStd& graph) noexcept :
                           _graph(graph), _current(current) {
    _current_offset = std::lower_bound(
            _graph._out_offsets,
            _graph._out_offsets + _graph._nV + 1,
            _current - _graph._out_edges);
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::EdgeIt&
GraphStd<vid_t, eoff_t>::EdgeIt::EdgeIt::operator++() noexcept {
    eoff_t edge_index = static_cast<eoff_t>(_current - _graph._out_edges);
    if (*(_current_offset+1) <= (edge_index + 1)) {
        _current_offset++;
    }
    _current++;
    return *this;
}

template<typename vid_t, typename eoff_t>
inline bool
GraphStd<vid_t, eoff_t>::EdgeIt::operator!=(const EdgeIt& it) const noexcept {
    return _current != it._current;
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::Edge
GraphStd<vid_t, eoff_t>::EdgeIt::operator*() const noexcept {
    return Edge(
            static_cast<vid_t>(_current_offset - _graph._out_offsets),
            static_cast<eoff_t>(_current - _graph._out_edges), _graph);
}

//==============================================================================
////////////////////////////////
///  VerticesContainer       ///
////////////////////////////////
template<typename vid_t, typename eoff_t>
inline GraphStd<vid_t, eoff_t>::VerticesContainer
::VerticesContainer(const GraphStd& graph) noexcept : _graph(graph) {}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::VertexIt
GraphStd<vid_t, eoff_t>::VerticesContainer::begin() const noexcept {
    return VertexIt(_graph._out_offsets, _graph);
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::VertexIt
GraphStd<vid_t, eoff_t>::VerticesContainer::end() const noexcept {
    return VertexIt(_graph._out_offsets + _graph._nV, _graph);
}

//==============================================================================
////////////////////////////////
///     EdgesContainer       ///
////////////////////////////////
template<typename vid_t, typename eoff_t>
inline GraphStd<vid_t, eoff_t>::EdgesContainer
::EdgesContainer(const GraphStd& graph) noexcept : _graph(graph) {}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::EdgeIt
GraphStd<vid_t, eoff_t>::EdgesContainer::begin() const noexcept {
    return EdgeIt(_graph._out_edges, _graph);
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::EdgeIt
GraphStd<vid_t, eoff_t>::EdgesContainer::end() const noexcept {
    return EdgeIt(_graph._out_edges + _graph._nE, _graph);
}

//==============================================================================
////////////////////////////////
///         GRAPHSTD         ///
////////////////////////////////

template<typename vid_t, typename eoff_t>
inline const eoff_t* GraphStd<vid_t, eoff_t>::csr_out_offsets() const noexcept {
    return _out_offsets;
}

template<typename vid_t, typename eoff_t>
inline const eoff_t* GraphStd<vid_t, eoff_t>::csr_in_offsets() const noexcept {
    return _in_offsets;
}

template<typename vid_t, typename eoff_t>
inline const vid_t* GraphStd<vid_t, eoff_t>::csr_out_edges() const noexcept {
    return _out_edges;
}

template<typename vid_t, typename eoff_t>
inline const vid_t* GraphStd<vid_t, eoff_t>::csr_in_edges() const noexcept {
    return _in_edges;
}

template<typename vid_t, typename eoff_t>
inline const typename GraphStd<vid_t, eoff_t>::degree_t*
GraphStd<vid_t, eoff_t>::out_degrees_ptr() const noexcept {
    return _out_degrees;
}

template<typename vid_t, typename eoff_t>
inline const typename GraphStd<vid_t, eoff_t>::degree_t*
GraphStd<vid_t, eoff_t>::in_degrees_ptr() const noexcept {
    return _in_degrees;
}

template<typename vid_t, typename eoff_t>
inline const typename GraphStd<vid_t, eoff_t>::coo_t*
GraphStd<vid_t, eoff_t>::coo_ptr() const noexcept {
    return _coo_edges;
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::degree_t
GraphStd<vid_t, eoff_t>::out_degree(vid_t index) const noexcept {
    assert(index >= 0 && index < _nV);
    return _out_degrees[index];
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::degree_t
GraphStd<vid_t, eoff_t>::max_out_degree() const noexcept {
    return *std::max_element(_out_degrees, _out_degrees + _nV);
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::degree_t
GraphStd<vid_t, eoff_t>::max_in_degree() const noexcept {
    return *std::max_element(_in_degrees, _in_degrees + _nV);
}

template<typename vid_t, typename eoff_t>
inline vid_t GraphStd<vid_t, eoff_t>::max_out_degree_id() const noexcept {
    return std::distance(_out_degrees,
                         std::max_element(_out_degrees, _out_degrees + _nV));
}

template<typename vid_t, typename eoff_t>
inline vid_t GraphStd<vid_t, eoff_t>::max_in_degree_id() const noexcept {
    return std::distance(_in_degrees,
                         std::max_element(_in_degrees, _in_degrees + _nV));
}


template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::degree_t
GraphStd<vid_t, eoff_t>::in_degree(vid_t index) const noexcept {
    assert(index >= 0 && index < _nV);
    return _in_degrees[index];
}

template<typename vid_t, typename eoff_t>
inline typename GraphStd<vid_t, eoff_t>::Vertex
GraphStd<vid_t, eoff_t>::vertex(vid_t index) const noexcept {
    return Vertex(index, *this);
}

template<typename vid_t, typename eoff_t>
inline bool GraphStd<vid_t, eoff_t>::is_directed() const noexcept {
    return _structure.is_directed();
}

template<typename vid_t, typename eoff_t>
inline bool GraphStd<vid_t, eoff_t>::is_undirected() const noexcept {
    return _structure.is_undirected();
}

} //namespace graph
