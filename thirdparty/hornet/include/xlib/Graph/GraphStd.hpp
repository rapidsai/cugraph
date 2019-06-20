/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
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
 *
 * @file
 */
#pragma once

#include "Graph/GraphBase.hpp"
#include "Host/Classes/Bitmask.hpp"  //xlib::Bitmask
#include <utility>                   //std::pair

namespace graph {

template<typename, typename> class BFS;
template<typename, typename> class WCC;
template<typename, typename> class SCC;

template<typename vid_t = int, typename eoff_t = int>
class GraphStd : public GraphBase<vid_t, eoff_t> {
    using coo_t    = typename std::pair<vid_t, vid_t>;
    using degree_t = int;
    friend class BFS<vid_t, eoff_t>;
    friend class WCC<vid_t, eoff_t>;
    friend class SCC<vid_t, eoff_t>;

public:
    class Edge;
    class VertexIt;
    class EdgeIt;

    //--------------------------------------------------------------------------
    class Vertex {
        template<typename, typename> friend class GraphStd;
    public:
        vid_t    id()                   const noexcept;
        vid_t    neighbor_id(int index) const noexcept;
        degree_t out_degree()           const noexcept;
        degree_t in_degree()            const noexcept;
        Edge     edge(int index)        const noexcept;

        friend inline std::ostream& operator<<(std::ostream& os,
                                               const Vertex& vertex) {
            os << vertex._id;
            return os;
        }

        EdgeIt begin()  const noexcept;
        EdgeIt end()    const noexcept;
    private:
        const GraphStd& _graph;
        const vid_t     _id;
        explicit Vertex(vid_t id, const GraphStd& graph) noexcept;
    };

    class VertexIt : public std::iterator<std::forward_iterator_tag, vid_t> {
        template<typename, typename> friend class GraphStd;
    public:
        VertexIt& operator++()                   noexcept;
        Vertex    operator*()                    const noexcept;
        bool      operator!=(const VertexIt& it) const noexcept;
    private:
        const GraphStd& _graph;
        eoff_t*         _current;
        explicit VertexIt(eoff_t* current, const GraphStd& graph) noexcept;
    };

    class VerticesContainer {
        template<typename, typename> friend class GraphStd;
    public:
        VertexIt begin() const noexcept;
        VertexIt end()   const noexcept;
    private:
        const GraphStd& _graph;

        explicit VerticesContainer(const GraphStd& graph) noexcept;
    };
    //--------------------------------------------------------------------------

    class Edge {
        template<typename, typename> friend class GraphStd;
    public:
        eoff_t id()     const noexcept;
        Vertex src()    const noexcept;
        Vertex dst()    const noexcept;
        vid_t  src_id() const noexcept;
        vid_t  dst_id() const noexcept;

        template<typename>
        friend inline std::ostream& operator<<(std::ostream& os,
                                               const Edge& edge) {
            os << edge._id;
            return os;
        }
    private:
        const GraphStd& _graph;
        const eoff_t    _edge_id;
        vid_t     _src_id;

        explicit Edge(vid_t src_id, eoff_t id, const GraphStd& graph) noexcept;
    };

    class EdgeIt : public std::iterator<std::forward_iterator_tag, vid_t> {
        template<typename, typename> friend class GraphStd;
    public:
        EdgeIt& operator++()                 noexcept;
        Edge    operator*()                  const noexcept;
        bool    operator!=(const EdgeIt& it) const noexcept;
    private:
        const GraphStd& _graph;
        vid_t*          _current;
        eoff_t*         _current_offset;

        explicit EdgeIt(vid_t* current, const GraphStd& graph) noexcept;
    };

    class EdgesContainer {
        template<typename T, typename R> friend class GraphStd;
    public:
        EdgeIt begin() const noexcept;
        EdgeIt end()   const noexcept;
    private:
        const GraphStd& _graph;

        explicit EdgesContainer(const GraphStd& graph) noexcept;
    };
    //--------------------------------------------------------------------------

    /*class InVertexIt :
                        public std::iterator<std::forward_iterator_tag, vid_t> {
        friend class GraphStd<vid_t, eoff_t>::IncomingVerticesContainer;
    public:
        InVertexIt& operator++()                   noexcept;
        IncomingVertex    operator*()                    const noexcept;
        bool      operator!=(const InVertexIt& it) const noexcept;

        void operator=(const InVertexIt&) = delete;
    private:
        const GraphStd& _graph;
        id-t*           _current;
        explicit InVertexIt(const GraphStd& graph) noexcept;
    };

    class IncomingVertex {
    public:
        InVertexIt begin() const noexcept;
        InVertexIt end()   const noexcept;

        Incoming(const Incoming&) = delete;
        Incoming& operator=(const Incoming&& obj) = delete;
    private:
        const GraphStd& _graph;
        explicit Incoming(const GraphStd& graph) noexcept;
    };*/
    //==========================================================================

    VerticesContainer V { *this };
    EdgesContainer    E { *this };

    //explicit GraphStd() = default;

    explicit GraphStd(StructureProp structure = structure_prop::NONE) noexcept;

    explicit GraphStd(const char* filename,
                      const ParsingProp& property
                            = parsing_prop::PRINT_INFO) noexcept;

    explicit GraphStd(StructureProp structure, const char* filename,
                      const ParsingProp& property
                            = parsing_prop::PRINT_INFO) noexcept;

    explicit GraphStd(const eoff_t* csr_offsets, vid_t nV,
                      const vid_t* csr_edges, eoff_t nE) noexcept;

    virtual ~GraphStd() noexcept;                                       //NOLINT
    //--------------------------------------------------------------------------

    Vertex   vertex(vid_t index)     const noexcept;
    Edge     edge  (eoff_t index)    const noexcept;
    degree_t out_degree(vid_t index) const noexcept;
    degree_t in_degree (vid_t index) const noexcept;

    const coo_t*    coo_ptr()         const noexcept;
    const eoff_t*   csr_out_offsets() const noexcept;
    const eoff_t*   csr_in_offsets()  const noexcept;
    const vid_t*    csr_out_edges()   const noexcept;
    const vid_t*    csr_in_edges()    const noexcept;
    const degree_t* out_degrees_ptr() const noexcept;
    const degree_t* in_degrees_ptr()  const noexcept;

    degree_t  max_out_degree()    const noexcept;
    degree_t  max_in_degree()     const noexcept;
    vid_t     max_out_degree_id() const noexcept;
    vid_t     max_in_degree_id()  const noexcept;

    bool      is_directed()       const noexcept;
    bool      is_undirected()     const noexcept;

    void print()                const noexcept override;
    void print_raw()            const noexcept override;
    void print_degree_distrib() const noexcept;
    void print_analysis()       const noexcept;
    void write_analysis(const char* filename) const noexcept;

    void writeBinary(const std::string& filename, bool print = true) const;
    void writeMarket(const std::string& filename, bool print = true) const;
    void writeDimacs10th(const std::string& filename, bool print = true)
                         const;

    using GraphBase<vid_t, eoff_t>::set_structure;

protected:
    xlib::Bitmask _bitmask;
    eoff_t*   _out_offsets { nullptr };
    eoff_t*   _in_offsets  { nullptr };
    vid_t*    _out_edges   { nullptr };
    vid_t*    _in_edges    { nullptr };
    degree_t* _out_degrees { nullptr };
    degree_t* _in_degrees  { nullptr };
    size_t    _coo_size    { 0 };
    static const uint64_t _seed { 0xA599AC3F0FD21B92 };

    using GraphBase<vid_t, eoff_t>::_structure;
    using GraphBase<vid_t, eoff_t>::_prop;
    using GraphBase<vid_t, eoff_t>::_graph_name;
    using GraphBase<vid_t, eoff_t>::_nE;
    using GraphBase<vid_t, eoff_t>::_nV;
    using GraphBase<vid_t, eoff_t>::_directed_to_undirected;
    using GraphBase<vid_t, eoff_t>::_undirected_to_directed;
    using GraphBase<vid_t, eoff_t>::_stored_undirected;

    virtual void allocateAux(const GInfo& ginfo) noexcept;

    void readMarket     (std::ifstream& fin, bool print)   override;
    void readMarketLabel(std::ifstream& fin, bool print)   override;
    void readDimacs9    (std::ifstream& fin, bool print)   override;
    void readDimacs10   (std::ifstream& fin, bool print)   override;
    void readSnap       (std::ifstream& fin, bool print)   override;
    void readKonect     (std::ifstream& fin, bool print)   override;
    void readNetRepo    (std::ifstream& fin)               override;
    void readMPG        (std::ifstream&, bool)             override;
    void readBinary     (const char* filename, bool print) override;

    void COOtoCSR() noexcept override;

private:
    coo_t* _coo_edges { nullptr };

    void allocate(const GInfo& ginfo) noexcept;

    struct GraphAnalysisProp {
        degree_t num_rings      { 0 };
        degree_t max_out_degree { 0 };
        degree_t max_in_degree  { 0 };

        degree_t out_degree_0 { 0 };
        degree_t in_degree_0  { 0 };
        degree_t out_degree_1 { 0 };
        degree_t in_degree_1  { 0 };
        degree_t singleton    { 0 };
        degree_t out_leaf     { 0 };
        degree_t in_leaf      { 0 };
        degree_t max_consec_0 { 0 };

        float std_dev { 0.0f };
        float gini    { 0.0f };
    };

    GraphAnalysisProp _collect_analysis() const noexcept;
};

} // namespace graph

#include "GraphStd.i.hpp"
