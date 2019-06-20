/**
 * @brief Hornet
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

#include "Core/DataLayout/DataLayout.cuh"
#include "BasicTypes.hpp"                       //vid_t
#include "Core/GPUCsr/CsrDevice.cuh"            //CsrDevice
#include "Core/HornetInit.hpp"                  //HornetInit

namespace hornets_nest {

/**
 * @brief The namespace contanins all classes and methods related to the
 *        GPU Hornet data structure
 */
namespace gpu {

/**
 * @brief Main Hornet class
 */
template<typename... VertexTypes, typename... EdgeTypes>
class Csr<TypeList<VertexTypes...>, TypeList<EdgeTypes...>> {
    using   EdgeAllocT = AoSData<vid_t, EdgeTypes...>;
    using VertexArrayT = BestLayoutAux< TypeList<off2_t, VertexTypes...> >;
    using   EdgeArrayT = BestLayoutAux< TypeList<vid_t, EdgeTypes...> >;
    using   CsrDeviceT = CsrDevice<TypeList<VertexTypes...>,
                                   TypeList<EdgeTypes...>>;

public:
    /**
     * @brief Default costructor
     * @param[in] hornet_init Hornet initilialization data structure
     * @param[in] traspose if `true` traspose the input graph, keep the initial
     *            representation otherwise
     */
    explicit Csr(const HornetInit& hornet_init,
                    bool traspose = false) noexcept;

    /**
     * @brief Decostructor
     */
    ~Csr() noexcept;

    /**
     * @brief **actual** number of vertices in the graph
     * @return actual number of vertices
     */
    vid_t nV() const noexcept;

    /**
     * @brief **actual** number of edges in the graph
     * @return actual number of edges
     */
    eoff_t nE() const noexcept;

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

    template<int INDEX>
    const typename xlib::SelectType<INDEX, VertexTypes...>::type*
    vertex_field() noexcept;

    template<int INDEX>
    const typename xlib::SelectType<INDEX, vid_t, EdgeTypes...>::type*
    edge_field() noexcept;

    /**
     * @brief **actual** device csr offsets of the graph
     * @return device pointer to csr offsets
     */
    const eoff_t* device_csr_offsets() const noexcept;

    const degree_t* device_degrees() const noexcept;

    /**
     * @brief device data to used the Hornet data structure on the device
     * @return device data associeted to the Hornet instance
     */
    CsrDeviceT device_side() const noexcept;

    /**
     * @brief
     */
    vid_t max_degree_id() noexcept;

    /**
     * @brief
     */
    degree_t max_degree() noexcept;

    /**
     * @brief print the graph directly from the device
     * @warning this function should be applied only on small graphs
     */
    void print() noexcept;

    /**
     * @brief unique identifier of the Hornet instance among all created
     *        instances
     * @return unique identifier
     */
    int id() const noexcept;

private:
    ///@internal @brief Number of all vertex fields (types)
    static const unsigned NUM_EXTRA_VTYPES = sizeof...(VertexTypes);
    static const unsigned NUM_EXTRA_ETYPES = sizeof...(EdgeTypes);

    static const unsigned       NUM_VTYPES = NUM_EXTRA_VTYPES + 1;

    ///@internal @brief Number of all edge fields (types)
    static const unsigned       NUM_ETYPES = NUM_EXTRA_ETYPES + 1;

    static int global_id;

    VertexArrayT      _vertex_array;
    EdgeArrayT        _edge_array;
    const HornetInit& _hornet_init;

    eoff_t*      _d_csr_offsets { nullptr };
    degree_t*    _d_degrees     { nullptr };
    vid_t        _nV            { 0 };
    eoff_t       _nE            { 0 };
    const int    _id            { 0 };
    std::pair<degree_t, vid_t> _max_degree_data { -1, -1 };

    void initialize() noexcept;

    void build_device_degrees() noexcept;
};

#define CSR Csr<TypeList<VertexTypes...>,TypeList<EdgeTypes...>>

} // namespace gpu
} // namespace hornets_nest

#include "impl/Csr.i.cuh"

#undef CSR
