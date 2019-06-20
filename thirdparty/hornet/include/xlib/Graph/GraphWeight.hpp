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
 *
 * @file
 */
#pragma once

#include "Graph/GraphStd.hpp"
#include <tuple>

namespace graph {

template<typename, typename> class BFS;
template<typename, typename> class WCC;
template<typename, typename, typename> class BellmanFord;
template<typename, typename, typename> class Dijkstra;
template<typename, typename, typename> class Brim;

template<typename vid_t = int, typename eoff_t = int, typename weight_t = int>
class GraphWeight : public GraphStd<vid_t, eoff_t> {
    using coo_t    = typename std::tuple<vid_t, vid_t, weight_t>;
    using degree_t = int;
    friend class BFS<vid_t, eoff_t>;
    friend class WCC<vid_t, eoff_t>;
    friend class BellmanFord<vid_t, eoff_t, weight_t>;
    friend class Dijkstra<vid_t, eoff_t, weight_t>;
    friend class Brim<vid_t, eoff_t, weight_t>;

public:
    explicit GraphWeight(StructureProp structure = structure_prop::NONE)
                         noexcept;

    explicit GraphWeight(const char* filename,
                         const ParsingProp& property = parsing_prop::PRINT_INFO)
                         noexcept;

    explicit GraphWeight(StructureProp structure, const char* filename,
                         const ParsingProp& property) noexcept;

    explicit GraphWeight(const eoff_t* csr_offsets, vid_t nV,
                         const vid_t* csr_edges, eoff_t nE,
                         const weight_t* csr_weights) noexcept;

    virtual ~GraphWeight() noexcept final;                              //NOLINT
    //--------------------------------------------------------------------------

    using GraphStd<vid_t, eoff_t>::out_degree;
    using GraphStd<vid_t, eoff_t>::in_degree;
    using GraphStd<vid_t, eoff_t>::csr_out_offsets;
    using GraphStd<vid_t, eoff_t>::csr_in_offsets;
    using GraphStd<vid_t, eoff_t>::csr_out_edges;
    using GraphStd<vid_t, eoff_t>::csr_in_edges;
    using GraphStd<vid_t, eoff_t>::out_degrees_ptr;
    using GraphStd<vid_t, eoff_t>::in_degrees_ptr;

    const coo_t*    coo_ptr()         const noexcept;
    const weight_t* out_weights_array() const noexcept;
    const weight_t* in_weights_array()  const noexcept;

    using GraphStd<vid_t, eoff_t>::max_out_degree;
    using GraphStd<vid_t, eoff_t>::max_in_degree;
    using GraphStd<vid_t, eoff_t>::max_out_degree_id;
    using GraphStd<vid_t, eoff_t>::max_in_degree_id;
    using GraphStd<vid_t, eoff_t>::is_directed;

    void print()     const noexcept override;
    void print_raw() const noexcept override;
    void toBinary(const std::string& filename, bool print = true) const;
    void toMarket(const std::string& filename) const;

    using GraphBase<vid_t, eoff_t>::set_structure;
protected:
    using GraphStd<vid_t, eoff_t>::_bitmask;
    using GraphStd<vid_t, eoff_t>::_out_offsets;
    using GraphStd<vid_t, eoff_t>::_in_offsets;
    using GraphStd<vid_t, eoff_t>::_out_edges;
    using GraphStd<vid_t, eoff_t>::_in_edges;
    using GraphStd<vid_t, eoff_t>::_out_degrees;
    using GraphStd<vid_t, eoff_t>::_in_degrees;
    using GraphStd<vid_t, eoff_t>::_coo_size;
    using GraphStd<vid_t, eoff_t>::_seed;

    weight_t*  _out_weights  { nullptr };
    weight_t*  _in_weights   { nullptr };

    using GraphBase<vid_t, eoff_t>::_structure;
    using GraphBase<vid_t, eoff_t>::_prop;
    using GraphBase<vid_t, eoff_t>::_graph_name;
    using GraphBase<vid_t, eoff_t>::_nE;
    using GraphBase<vid_t, eoff_t>::_nV;
    using GraphBase<vid_t, eoff_t>::_directed_to_undirected;
    using GraphBase<vid_t, eoff_t>::_undirected_to_directed;
    using GraphBase<vid_t, eoff_t>::_stored_undirected;

    void allocate(const GInfo& ginfo) noexcept;

    void readMarket  (std::ifstream& fin, bool print)   override;
    void readDimacs9 (std::ifstream& fin, bool print)   override;
    void readDimacs10(std::ifstream& fin, bool print)   override;
    void readSnap    (std::ifstream& fin, bool print)   override;
    void readKonect  (std::ifstream& fin, bool print)   override;
    void readNetRepo (std::ifstream& fin)               override;
    void readMPG     (std::ifstream& fin, bool print)   override;
    void readBinary  (const char* filename, bool print) override;

    void COOtoCSR() noexcept override;

private:
    coo_t* _coo_edges { nullptr };
};

} // namespace graph

#include "GraphWeight.i.hpp"
