/**
 * @brief Hornet
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
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

#include "BasicTypes.hpp"
#include "Core/DataLayout/DataLayout.cuh"
#include "Core/GPUHornet/BatchUpdate.cuh"             //BatchUpdate
#include "Core/GPUHornet/HornetDevice.cuh"            //HornetDevice
#include "Core/HornetInit.hpp"                  //HornetInit
#include "Core/MemoryManager/MemoryManager.hpp" //MemoryManager
#include <Device/Primitives/CubWrapper.cuh>

namespace hornets_nest {
namespace gpu {
/**
 * @brief The namespace contains all classes and methods related to the
 *        GPU Hornet data structure
 */

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
template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
class Hornet<TypeList<VertexTypes...>, TypeList<EdgeTypes...>, FORCE_SOA> {
    using  HornetDeviceT = HornetDevice<TypeList<VertexTypes...>,
                                        TypeList<EdgeTypes...>,
                                        FORCE_SOA>;

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

    /**
     * @brief **actual** device csr offsets of the graph
     * @return device pointer to csr offsets
     */
    const degree_t* device_degrees() const noexcept;

    /**
     * @internal
     * @brief device data to used the Hornet data structure on the device
     * @return device data associeted to the Hornet instance
     */
    HornetDeviceT device_side() const noexcept;

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
     * @brief
     */
    void mem_manager_info() const noexcept;

    //--------------------------------------------------------------------------

    /**
     * @brief
     */
    void reserveBatchOpResource(const size_t max_batch_size,
                         const BatchProperty batch_prop = batch_property::IN_PLACE |
                         batch_property::REMOVE_CROSS_DUPLICATE |
                         batch_property::REMOVE_BATCH_DUPLICATE) noexcept;

    /**
     * @brief
     */
    void allocateEdgeInsertion(size_t max_batch_size,
                               BatchProperty batch_prop = BatchProperty())
                               noexcept;

    /**
     * @brief
     */
    void insertEdgeBatch(BatchUpdate& batch_update,
                         const BatchProperty batch_prop = batch_property::IN_PLACE |
                         batch_property::REMOVE_CROSS_DUPLICATE |
                         batch_property::REMOVE_BATCH_DUPLICATE) noexcept;

    /**
     * @brief
     */
    void deleteEdgeBatch(BatchUpdate& batch_update,
                         const BatchProperty batch_prop = batch_property::IN_PLACE |
                         batch_property::REMOVE_BATCH_DUPLICATE) noexcept;

    void deleteOOPEdgeBatch(BatchUpdate& batch_update) noexcept;
    //--------------------------------------------------------------------------
    /**
     * @brief store the actual gpu representation to disk for future use
     * @param[in] filename name of the file where the graph
     */
    void store_snapshot(const std::string& filename) const noexcept;

    /**
     * @brief unique identifier of the Hornet instance among all created
     *        instances
     * @return unique identifier
     */
    int id() const noexcept;

    /**
     * @brief Check the consistency of the device data structure with the host
     *        data structure provided in the input
     * @details revert the initilization process to rebuild the device data
     *          structure on the host and then perform the comparison
     */
    void check_consistency(const HornetInit& hornet_init) const noexcept;

    void check_sorted_adjs() const noexcept;

private:
    using     EdgeAllocT = AoSData<vid_t, EdgeTypes...>;
    using   VertexArrayT = BestLayoutAux<
                                TypeList<size_t, void*, VertexTypes...>,
                                FORCE_SOA>;
    using        OffsetT = typename std::conditional<
                               !FORCE_SOA &&
                               xlib::IsVectorizable<vid_t, EdgeTypes...>::value,
                               EdgeAllocT, vid_t>::type;
    using MemoryManagerT = MemoryManager<EdgeAllocT, OffsetT, true>;

    using   ETypeSizes = typename xlib::Seq<sizeof(vid_t),sizeof(EdgeTypes)...>;
    using ETypeSizesPS = typename xlib::ExcPrefixSum<ETypeSizes>::type;

    ///@internal @brief Array of all edge field (type) sizes
    static const ETypeSizes ETYPE_SIZES;
    //--------------------------------------------------------------------------
    ///@internal @brief Number of all vertex fields (types)
    static const unsigned NUM_EXTRA_VTYPES = sizeof...(VertexTypes);
    static const unsigned NUM_EXTRA_ETYPES = sizeof...(EdgeTypes);

    static const unsigned       NUM_VTYPES = NUM_EXTRA_VTYPES + 1;

    ///@internal @brief Number of all edge fields (types)
    static const unsigned       NUM_ETYPES = NUM_EXTRA_ETYPES + 1;
    //--------------------------------------------------------------------------

    static int global_id;

    MemoryManagerT _mem_manager;
    VertexArrayT   _vertex_array;

    const HornetInit& _hornet_init;
    const eoff_t* _csr_offsets       { nullptr };
    const vid_t*  _csr_edges         { nullptr };
    degree_t*     _d_degrees         { nullptr };
    eoff_t*       _d_csr_offsets     { nullptr };
    vid_t         _nV                { 0 };
    eoff_t        _nE                { 0 };
    const int     _id                { 0 };
    bool          _internal_csr_data { false };
    bool          _is_sorted         { false };
    BatchProperty _batch_prop;

    std::pair<degree_t, vid_t> _max_degree_data { -1, -1 };


    void initialize() noexcept;

    /**
     * @internal
     * @brief traspose the Hornet graph directly on the device
     */
    [[deprecated]]
    void transpose() noexcept;

    /**
     * @internal
     * @brief convert the actual Hornet graph into csr offsets and csr edges
     * @param[out] csr_offsets csr offsets to build
     * @param[out] csr_offsets csr edges to build
     */
    [[deprecated]]
    void convert_to_csr(eoff_t* csr_offsets, vid_t* csr_edges)
                        const noexcept;

    void build_device_degrees() noexcept;

    //==========================================================================
    //==========================================================================
    ///////////
    // BATCH //
    ///////////

    size_t _max_batch_size {0};
    size_t _csr_size {0};

    //BatchProperty _batch_prop;
    vid_t* _d_batch_src { nullptr };
    vid_t* _d_batch_dst { nullptr };
    vid_t* _d_wide_csr  { nullptr };

    ///Batch common tmp variables
    //vid_t*    _d_src_array    { nullptr };
    //vid_t*    _d_dst_array    { nullptr };
    vid_t*    _d_unique       { nullptr };
    int*      _d_counts       { nullptr };
    int*      _d_counter      { nullptr };
    vid_t*    _d_tmp_sort_src { nullptr };
    vid_t*    _d_tmp_sort_dst { nullptr };
    degree_t* _d_degree_tmp   { nullptr };
    bool*     _d_flags        { nullptr };

    degree_t* _d_locations    { nullptr };
    degree_t* _d_batch_offset { nullptr };
    xlib::CubExclusiveSum<int>       cub_prefixsum;
    xlib::CubRunLengthEncode<vid_t>  cub_runlength;
    xlib::CubSelectFlagged<vid_t>    cub_select_flag;
    xlib::CubSortByKey<vid_t, vid_t> cub_sort;

    xlib::CubSortPairs2<vid_t, vid_t> cub_sort_pair;

    ///Batch delete tmp variables

    degree_t*   _d_degree_new   { nullptr };
    void*      *_d_ptrs_array   { nullptr };
    EdgeAllocT* _d_tmp          { nullptr };
    eoff_t*     _d_inverse_pos  { nullptr };

    ///Batch in-place tmp variables
    vid_t*    _d_queue_id         { nullptr };
    void**    _d_queue_old_ptr    { nullptr };
    void**    _d_queue_new_ptr    { nullptr };
    degree_t* _d_queue_old_degree { nullptr };
    degree_t* _d_queue_new_degree { nullptr };
    int*      _d_queue_size       { nullptr };
    //----------------------------------------
    void**    _h_queue_old_ptr    { nullptr };
    void**    _h_queue_new_ptr    { nullptr };
    degree_t* _h_queue_old_degree { nullptr };
    degree_t* _h_queue_new_degree { nullptr };

    /**
     * @brief
     */
    void allocatePrepocessing(const size_t max_batch_size) noexcept;
    void allocateInPlaceUpdate(const size_t csr_size) noexcept;
    void fixInternalRepresentation(int num_uniques, bool is_insert,
                                   bool get_old_degree) noexcept;

    void build_batch_csr(
            BatchUpdate& batch_update,
            const BatchProperty batch_prop,
            int num_uniques,
            bool require_prefix_sum = true) noexcept;

    void copyBatchUpdateData(
            const BatchUpdate& batch_update,
            const BatchProperty batch_prop,
            vid_t * const d_batch_src,
            vid_t * const d_batch_dst,
            size_t * const batch_size) noexcept;

    void sort_batch(
            BatchUpdate& batch_update,
            const BatchProperty batch_prop,
            const bool is_insert,
            vid_t ** d_batch_src_ptr,
            vid_t ** d_batch_dst_ptr,
            const size_t batch_size) noexcept;

    void remove_batch_duplicates(
            vid_t * d_batch_src,
            vid_t * d_batch_dst,
            size_t * const batch_size) noexcept;

    void remove_cross_duplicates(
            BatchUpdate& batch_update,
            vid_t * const d_batch_src,
            vid_t * const d_batch_dst,
            size_t * const batch_size) noexcept;

    int batch_preprocessing(BatchUpdate& batch_update, bool is_insert)
                            noexcept;
    int batch_preprocessing(
            BatchUpdate& batch_update,
            const BatchProperty batch_prop,
            bool is_insert) noexcept;

    void copySparseToContinuos(const degree_t* prefixsum,
                               int             prefixsum_size,
                               int             total_sum,
                               void**          sparse_ptrs,
                               void*           continuous_array) noexcept;

    void copySparseToContinuos(const degree_t* prefixsum,
                               int             prefixsum_size,
                               int             total_sum,
                               void**          sparse_ptrs,
                               const int*      continuos_offsets,
                               void*           continuous_array) noexcept;

    void copySparseToSparse(const degree_t* d_prefixsum,
                            int             work_size,
                            int             prefixsum_total,
                            void**          d_old_ptrs,
                            void**          d_new_ptrs) noexcept;

    void copyContinuosToSparse(const degree_t* prefixsum,
                               int             prefixsum_size,
                               int             total_sum,
                               void*           continuous_array,
                               void**          sparse_ptrs) noexcept;
};

#define HORNET Hornet<TypeList<VertexTypes...>, TypeList<EdgeTypes...>, \
                      FORCE_SOA>

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
const HORNET::ETypeSizes HORNET::ETYPE_SIZES;

} // namespace gpu
} // namespace hornets_nest

#include "impl/Hornet1.i.cuh"
#include "impl/Hornet2.i.cuh"
#include "Batch/BatchAllocateAndCopy.i.cuh"
#include "Batch/BatchCommon.i.cuh"
#include "Batch/BatchInsert.i.cuh"
#include "Batch/BatchDelete.i.cuh"
#include "Batch/BatchOOPDelete.i.cuh"

#undef HORNET
