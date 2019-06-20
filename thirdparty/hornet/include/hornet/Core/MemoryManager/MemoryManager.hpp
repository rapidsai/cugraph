/**
 * @internal
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

#include "BasicTypes.hpp"
#include "Core/MemoryManager/BitTree.hpp"
#include "Core/MemoryManager/MemoryManagerConf.hpp"

#if defined(B_PLUS_TREE)
    #include <btree_map.h>
    #define  SC_MACRO second.
#elif defined(RB_TREE)
    #include <map>
    #define  SC_MACRO second.
#else
    #include <vector>
    #define  SC_MACRO
#endif

//#define EXPERIMENTAL

namespace hornets_nest {

/**
 * @brief Container for a set of *BlockArrays* of the same sime
 */
#if defined(B_PLUS_TREE)
    template<typename T>
    using Container = btree::btree_map<byte_t*, T>;
#elif defined(RB_TREE)
    template<typename T>
    using Container = std::map<byte_t*, T>;
#else
    template<typename T>
    using Container = std::vector<T>;
#endif

/**
 * @brief The `MemoryManager` class provides the infrastructure to organize
 *        the set of *BlockArrays*
 */
template<typename block_t, typename offset_t, bool DEVICE>
class MemoryManager {

public:
    explicit MemoryManager() noexcept;

    /**
     * @brief Insert a new *block* of size
     *        \f$2^{\lceil \log_2(degree + 1) \rceil}\f$
     * @details If the actual *BlockArrays* of the correct size is full, it
     *          automatically allocates a new one.
     * @param[in] degree degree of the vertex to insert
     * @return Pair < `host_block_ptr`, `device_block_ptr` > of the
     *         corresponding *block*
     */
    std::pair<byte_t*, byte_t*> insert(degree_t degree) noexcept;

    /**
     * @brief Remove the *block* pointed by `ptr`
     * @param[in] ptr pointer to the *block* to remove
     * @param[in] degree degree of the vertex to remove
     * @warning the pointer must be previously inserted, otherwise an error is
     *          raised (debug mode)
     */
    void remove(void* device_ptr, degree_t degree) noexcept;

    /**
     * @brief Number of *BlockArrays*
     * @return the number of allocated *BlockArrays*
     */
    int num_blockarrays() const noexcept;

    /**
     * @brief Various statistics about the space efficency
     * @details It provides the number of *BlockArrays*, the total number of
     *          allocated edges, the total number of used edges, the space
     *          efficiency (used/allocated), and the number of *blocks* for each
     *          level
     */
    void statistics() const noexcept;

    /**
     * @brief Get the base pointer of a specific *BlockArrays*
     * @param[in] block_index index of the *BlockArrays*
     * @return Pair < `host_BlockArray_ptr`, `device_BlockArray_ptr` > of the
     *         corresponding *BlockArray*
     * @warning `block_index` must be
     *           \f$0 \le \text{block_index} < \text{total_block_arrays} \f$,
     *           otherwise an error is raised (debug mode)
     */
    std::pair<byte_t*, byte_t*> get_blockarray_ptr(int block_index) noexcept;

    /**
     * @brief Free the host memory reserved for all *BlockArrays*
     */
    void free_host_ptrs() noexcept;

    /**
     * @brief Deallocate all host and device *BlockArrays*
     */
    void clear() noexcept;

private:
    using ContainerT = Container<BitTree<block_t, offset_t, DEVICE>>;

    int _num_blockarrays    { 0 };

    int _num_inserted_edges { 0 };

    static constexpr int MM_LOG_LIMIT = 32;

    ContainerT bit_tree_set[MM_LOG_LIMIT];

#if defined(EXPERIMENTAL)
    ContainerT zero_container;
#endif

    int find_bin(degree_t degree) const noexcept;

    std::pair<byte_t*, byte_t*> insert_aux(ContainerT& container,
                                           int index = 0) noexcept;
};

} // namespace hornets_nest

#include "MemoryManager.i.hpp"

#undef EXPERIMENTAL
