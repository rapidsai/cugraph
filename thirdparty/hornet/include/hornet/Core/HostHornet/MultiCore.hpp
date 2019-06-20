/**
 * @brief Hornet, HornetInit, BatchUpdatem and BatchProperty classes
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
#pragma once

#include "BasicTypes.hpp"                       //vid_t
#include "Core/GPU/HornetDevice.cuh"            //HornetDevice
#include "Core/HornetInit.hpp"                  //HornetInit
#include "Core/BatchUpdate.cuh"                 //BatchUpdate
#include "Core/MemoryManager/MemoryManager.hpp" //MemoryManager
#include <cstddef>                              //size_t

namespace hornets_nest {

/**
 * @brief The namespace contanins all classes and methods related to the
 *        GPU Hornet data structure
 */
namespace mc {

struct ALIGN(16) VertexBasicData {
    edge_t*  __restrict__ neighbor_ptr;
    degree_t              degree;

    HOST_DEVICE
    VertexBasicData(degree_t degree, byte_t* neighbor_ptr) :
        degree(degree), neighbor_ptr(neighbor_ptr) {}
};

template<typename, typename> class Hornet;

/**
 * @brief list of types for additional vertex data
 * @details **Example**
 * @code{.cpp}
 *       using VertexTypes = TypeList<char, float>;
 * @endcode
 */

/**
 * @brief Main Hornet class
 */
template<typename... VertexTypes, typename... EdgeTypes>
class Hornet<TypeList<VertexTypes...>, TypeList<EdgeTypes...>> {
    using VertexTypesList = TypeList<VertexTypes...>;
    using   EdgeTypesList = TypeList<EdgeTypes...>;
    using   HornetDeviceT = HornetDevice<VertexTypesList, EdgeTypesList>;

    using vertex_t = typename xlib::TupleConcat<TypeList<VertexBasicData>,
                                                VertexTypesList>::type;

    using   edge_t = typename xlib::TupleConcat<TypeList<vid_t>,
                                                EdgeTypesList>::type;

    using      VTypeSizes = typename xlib::TupleToTypeSizeSeq<vertex_t>::type;
    using      ETypeSizes = typename xlib::TupleToTypeSizeSeq<edge_t>::type;
    using ExtraVTypeSizes = typename xlib::TupleToTypeSizeSeq<VertexTypesList>
                                                ::type;
    using ExtraETypeSizes = typename xlib::TupleToTypeSizeSeq<EdgeTypesList>
                                                ::type;
    using    VTypeSizesPS = typename xlib::ExcPrefixSum<VTypeSizes>::type;
    using    ETypeSizesPS = typename xlib::ExcPrefixSum<ETypeSizes>::type;

    //--------------------------------------------------------------------------
    ///@internal @brief Array of all vertex field (type) sizes
    static constexpr VTypeSizes      VTYPE_SIZES       {};
    ///@internal @brief Array of all edge field (type) sizes
    static constexpr ETypeSizes      ETYPE_SIZES       {};
    ///@internal @brief Array of extra vertex field (type) sizes
    static constexpr ExtraVTypeSizes EXTRA_VTYPE_SIZES {};
    ///@constexpr @brief Array of extra edge field (type) sizes
    static constexpr ExtraETypeSizes EXTRA_ETYPE_SIZES {};

    ///@internal @brief Array of exclusive prefix-sum of all vertex field (type)
    ///                 sizes
    static constexpr VTypeSizesPS    VTYPE_SIZES_PS    {};
    ///@internal @brief Array of exclusive prefix-sum of all edge field (type) sizes
    static constexpr ETypeSizesPS    ETYPE_SIZES_PS    {};

    ///@internal @brief Number of all vertex fields (types)
    static const unsigned       NUM_VTYPES = std::tuple_size<vertex_t>::value;
    ///@internal @brief Number of all edge fields (types)
    static const unsigned       NUM_ETYPES = std::tuple_size<edge_t>::value;
    ///@internal @brief Number of extra vertex fields (types)
    static const unsigned NUM_EXTRA_VTYPES = std::tuple_size<VertexTypesList>
                                                ::value;
    ///@internal @brief Number of extra vertex fields (types)
    static const unsigned NUM_EXTRA_ETYPES = std::tuple_size<EdgeTypesList>
                                                ::value;
public:
    /**
     * @brief default costructor
     * @param[in] hornet_init Hornet initilialization data structure
     * @param[in] traspose if `true` traspose the input graph, keep the initial
     *            representation otherwise
     */
    explicit Hornet(const HornetInit& hornet_init,
                    bool traspose = false) noexcept;

    /**
     * @brief Decostructor
     */
    ~Hornet() noexcept;

    /**
     * @brief **actual** number of vertices in the graph
     * @return actual number of vertices
     */
    size_t nV() const noexcept;

    /**
     * @brief **actual** number of edges in the graph
     * @return actual number of edges
     */
    size_t nE() const noexcept;

    /**
     * @brief **actual** csr offsets of the graph
     * @return pointer to csr offsets
     */
    const eoff_t* csr_offsets() noexcept;

    /**
     * @brief **actual** csr edges of the graph
     * @return pointer to csr edges
     */
    const vid_t* csr_edges() noexcept;

    /**
     * @internal
     * @brief device data to used the Hornet data structure on the device
     * @return device data associeted to the Hornet instance
     */
    HornetDeviceT device_side() const noexcept;

    vid_t max_degree_id() noexcept;

    degree_t max_degree() noexcept;

    /**
     * @brief print the graph directly from the device
     * @warning this function should be applied only on small graphs
     */
    void print() noexcept;
    //--------------------------------------------------------------------------

    void allocateEdgeDeletion(vid_t max_batch_size,
                              BatchProperty batch_prop) noexcept;

    void allocateEdgeInsertion(vid_t max_batch_size,
                               BatchProperty batch_prop) noexcept;

    void insertEdgeBatch(BatchUpdate& batch_update,
                         BatchProperty batch_prop) noexcept;

    void deleteEdgeBatch(BatchUpdate& batch_update,
                         BatchProperty batch_prop = BatchProperty()) noexcept;

    /**
     * @brief unique identifier of the Hornet instance among all created
     *        instances
     * @return unique identifier
     */
    int id() const noexcept;

private:
    static int global_id;

    MemoryManager<edge_t, vid_t, false> mem_manager;

    /**
     * @internal
     * @brief device pointer for *all* vertex data
     *        (degree and edge pointer included)
     */
    byte_t* _d_vertex_ptrs[NUM_VTYPES] = {};
    byte_t* _d_edge_ptrs[NUM_ETYPES]   = {};

    const HornetInit& _hornet_init;
    const eoff_t* _csr_offsets       { nullptr };
    const vid_t*  _csr_edges         { nullptr };
    byte_t*       _d_vertices        { nullptr };
    byte_t*       _d_edges           { nullptr };   //for CSR
    degree_t*     _d_degrees         { nullptr };
    eoff_t*       _d_csr_offsets     { nullptr };
    size_t        _nV                { 0 };
    size_t        _nE                { 0 };
    const int     _id                { 0 };
    bool          _internal_csr_data { false };
    bool          _is_sorted         { false };

    std::pair<degree_t, vid_t> max_degree_data { -1, -1 };

    void initialize() noexcept;

    /**
     * @internal
     * @brief traspose the Hornet graph directly on the device
     */
    void transpose() noexcept;

    void build_device_degrees() noexcept;

    //==========================================================================
    //BATCH

    void build_batch_csr(int num_uniques) noexcept;
    void batch_preprocessing() noexcept;
};

} // namespace mc
} // namespace hornets_nest

#define HORNET Hornet<TypeList<VertexTypes...>,TypeList<EdgeTypes...>>

#include "impl/Hornet.i.cuh"

#undef HORNET
