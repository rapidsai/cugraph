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
 *
 * @file
 */
#pragma once

#include "BasicTypes.hpp"       //vid_t
#include <vector>               //std::vector

/**
 * @brief The namespace contanins all classes and methods related to the
 *        Hornet data structure
 */
namespace hornets_nest {

namespace gpu {
    template<typename, typename, bool> class Hornet;
    template<typename, typename> class Csr;
} //namespace gpu

/**
 * @brief Hornet initialization class
 */
class HornetInit {
    template<typename, typename, bool> friend class gpu::Hornet;
    template<typename, typename>       friend class gpu::Csr;
public:
    /**
     * @brief Default costructor
     * @param[in] num_vertices number of vertices
     * @param[in] num_edges number of edges
     * @param[in] csr_offsets csr offsets array
     * @param[in] csr_edges csr edges array
     */
    explicit HornetInit(vid_t num_vertices, eoff_t num_edges,
                        const eoff_t* csr_offsets, const vid_t* csr_edges,
                        bool is_sorted = false) noexcept;

    ~HornetInit() noexcept;

    /**
     * @brief Insert additional vertex data
     * @param[in] vertex_data list of vertex data array
     * @remark the types of the input arrays must be equal to the type List
     *         for vertices specified in the *config.inc* file
     * @details **Example**
     *         @code{.cpp}
     *             int* array1 = ...;
     *             float* array2 = ...;
     *             HornetInit Hornet_init(...);
     *             Hornet_init.insertVertexData(array1, array2);
     *         @endcode
     */
    template<typename... TArgs>
    void insertVertexData(const TArgs*... vertex_data) noexcept;

    /**
     * @brief Insert additional edge data
     * @param[in] edge_data list of vertex data array
     * @remark the types of the input arrays must be equal to the type List
     *         for edges specified in the *config.inc* file
     * @see ::insertVertexData
     */
    template<typename... TArgs>
    void insertEdgeData(const TArgs*... edge_data) noexcept;

    /**
     * @brief number of vertices in the graph passed to the costructor
     * @return number of vertices in the graph
     */
    vid_t nV() const noexcept;

    /**
     * @brief number of edges in the graph passed to the costructor
     * @return number of edges in the graph
     */
    eoff_t nE() const noexcept;

    /**
     * @brief CSR offsets of the graph passed to the costructor
     * @return constant pointer to csr offsets array
     */
    const eoff_t* csr_offsets() const noexcept;

    /**
     * @brief CSR edges of the graph passed to the costructor
     * @return constant pointer to csr edges array
     */
    const vid_t* csr_edges() const noexcept;

    template<typename T>
    void addEmptyVertexField() noexcept;

    template<typename T>
    void addEmptyEdgeField() noexcept;

    bool is_sorted() const noexcept;

private:
    static constexpr int MAX_TYPES = 32;

    std::vector<byte_t*> ptrs_to_delete;

    /**
     * @internal
     * @brief Array of pointers of the *all* vertex data
     */
    const byte_t* _vertex_data_ptrs[ MAX_TYPES ] = {};

    /**
     * @internal
     * @brief Array of pointers of the *all* edge data
     */
    const byte_t* _edge_data_ptrs[ MAX_TYPES ] = {};

    vid_t  _nV                 { 0 };
    eoff_t _nE                 { 0 };
    int    _vertex_field_count { 1 };
    int    _edge_field_count   { 1 };
    bool   _is_sorted          { false };
};

} // namespace hornets_nest

#include "HornetInit.i.hpp"
