/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date June, 2017
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

#include "Host/Classes/PropertyClass.hpp"   //xlib::PropertyClass
#include <string>                           //std::string

namespace graph {

namespace detail {
    enum class ParsingEnum { NONE = 0, RANDOMIZE = 1, SORT = 2,
                             PRINT_INFO = 4, RM_SINGLETON = 8,
                             DIRECTED_BY_DEGREE = 16 };
} // namespace detail

class ParsingProp : public xlib::PropertyClass<detail::ParsingEnum,
                                               ParsingProp> {
    template<typename, typename>           friend class GraphBase;
    template<typename, typename>           friend class GraphStd;
    template<typename, typename, typename> friend class GraphWeight;
public:
    explicit ParsingProp(const detail::ParsingEnum& value) noexcept;
private:
    bool is_sort()         const noexcept;
    bool is_directed_by_degree()    const noexcept;
    bool is_randomize()    const noexcept;
    bool is_print()        const noexcept;
    bool is_rm_singleton() const noexcept;
};

namespace parsing_prop {

///@brief No action (used for empty constructor)
const ParsingProp NONE             ( detail::ParsingEnum::NONE );

///@brief Randomize the label ids of graph vertices (random but reproducible)
const ParsingProp RANDOMIZE        ( detail::ParsingEnum::RANDOMIZE );

///@brief Sort adjacency list by label id
const ParsingProp SORT             ( detail::ParsingEnum::SORT );

///@brief Turn into degree-directed graph (high to low direction) 
const ParsingProp DIRECTED_BY_DEGREE             ( detail::ParsingEnum::DIRECTED_BY_DEGREE );


///@brief Print basic information during graph parsing
const ParsingProp PRINT_INFO       ( detail::ParsingEnum::PRINT_INFO );

///@brief Remove vertices with zero out-degree and zero in-degree
///       (vertex ids are relabeled)
const ParsingProp RM_SINGLETON     ( detail::ParsingEnum::RM_SINGLETON );

} // namespace parsing_prop

//==============================================================================
namespace detail {
    enum class StructureEnum { NONE = 0, DIRECTED = 1, UNDIRECTED = 2,
                               ENABLE_INGOING = 4, COO = 8 };
} // namespace detail

class StructureProp :
              public xlib::PropertyClass<detail::StructureEnum, StructureProp> {
    template<typename, typename>           friend class GraphBase;
    template<typename, typename>           friend class GraphStd;
    template<typename, typename, typename> friend class GraphWeight;
public:
    explicit StructureProp(const detail::StructureEnum& value) noexcept;
private:
    //enum WType   { NONE, INTEGER, REAL };
    //WType _wtype { NONE };
    bool is_directed()        const noexcept;
    bool is_undirected()      const noexcept;
    bool is_reverse()         const noexcept;
    bool is_coo()             const noexcept;
    bool is_direction_set()   const noexcept;
    bool is_weighted()        const noexcept;
    bool _is_non_compatible() const noexcept override;
};

namespace structure_prop {

const StructureProp NONE           ( detail::StructureEnum::NONE );
const StructureProp DIRECTED       ( detail::StructureEnum::DIRECTED );
const StructureProp UNDIRECTED     ( detail::StructureEnum::UNDIRECTED );
const StructureProp ENABLE_INGOING ( detail::StructureEnum::ENABLE_INGOING );
const StructureProp COO            ( detail::StructureEnum::COO );

} // namespace structure_prop

//==============================================================================

struct GInfo {
    size_t        num_vertices;
    size_t        num_edges;
    size_t        num_lines;
    StructureProp direction;
};

template<typename vid_t, typename eoff_t>
class GraphBase {
public:
    virtual vid_t  nV() const noexcept final;
    virtual eoff_t nE() const noexcept final;
    virtual const std::string& name() const noexcept final;

    virtual void read(const char* filename,
                      const ParsingProp& prop = parsing_prop::PRINT_INFO)
                      final;   //NOLINT

    virtual void print()     const noexcept = 0;
    virtual void print_raw() const noexcept = 0;

    GraphBase(const GraphBase&)      = delete;
    void operator=(const GraphBase&) = delete;
protected:
    StructureProp _structure  { structure_prop::NONE };
    ParsingProp   _prop       { parsing_prop::NONE };
    std::string   _graph_name { "" };
    vid_t         _nV         { 0 };
    eoff_t        _nE         { 0 };
    bool          _directed_to_undirected { false };
    bool          _undirected_to_directed { false };
    bool          _stored_undirected      { false };

    explicit GraphBase() = default;
    explicit GraphBase(StructureProp structure) noexcept;
    explicit GraphBase(vid_t nV, eoff_t nE, StructureProp structure)
                       noexcept;
    virtual ~GraphBase() noexcept = default;

    virtual void   set_structure(const StructureProp& structure) noexcept final;

    virtual void   readMarket     (std::ifstream& fin, bool print)   = 0;
    virtual void   readMarketLabel(std::ifstream& fin, bool print)   = 0;
    virtual void   readDimacs9    (std::ifstream& fin, bool print)   = 0;
    virtual void   readDimacs10   (std::ifstream& fin, bool print)   = 0;
    virtual void   readSnap       (std::ifstream& fin, bool print)   = 0;
    virtual void   readKonect     (std::ifstream& fin, bool print)   = 0;
    virtual void   readNetRepo    (std::ifstream& fin)               = 0;
    virtual void   readMPG        (std::ifstream& fin, bool print)   = 0;
    virtual void   readBinary     (const char* filename, bool print) = 0;

    virtual GInfo  getMarketLabelHeader(std::ifstream& fin) final;
    virtual GInfo  getMarketHeader     (std::ifstream& fin) final;
    virtual GInfo  getDimacs9Header    (std::ifstream& fin) final;
    virtual GInfo  getDimacs10Header   (std::ifstream& fin) final;
    virtual GInfo  getKonectHeader     (std::ifstream& fin) final;
    virtual void   getNetRepoHeader    (std::ifstream& fin) final;
    virtual GInfo  getSnapHeader       (std::ifstream& fin) final;
    virtual GInfo  getMPGHeader        (std::ifstream& fin) final;

    virtual void COOtoCSR() noexcept = 0;
    //virtual void CSRtoCOO() noexcept = 0;
};

template<typename vid_t, typename eoff_t>
inline vid_t GraphBase<vid_t, eoff_t>::nV() const noexcept {
    return _nV;
}

template<typename vid_t, typename eoff_t>
inline eoff_t GraphBase<vid_t, eoff_t>::nE() const noexcept {
    return _nE;
}

} // namespace graph
