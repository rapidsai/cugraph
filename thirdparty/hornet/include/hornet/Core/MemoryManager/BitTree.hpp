/**
 * @internal
 * @brief Vectorized Bit Tree (Vec-Tree) interface
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

#include "BasicTypes.hpp"                           //xlib::byte
#include "Core/MemoryManager/MemoryManagerConf.hpp" //EDGES_PER_BLOCKARRAY
#include <Host/Metaprogramming.hpp>                 //xlib::GeometricSerie
#include <utility>                                  //std::pair

namespace hornets_nest {

/**
 * @brief **Vectorized Bit Tree**
 * @details Internal representation                                         <br>
 * [1011]                                   // root       --> internal      <br>
 * [0001] [0000] [1001] [0110]              //            --> internal      <br>
 * [....] [....] [....] [....] [....] ...   // last_level --> external      <br>
 *
 * @remark 1 means *block* available, 0 *block* used
 */
template<typename block_t>
class BitTreeBase {
protected:
    /**
     * @brief Default Costrustor
     * @details Build a empty *BitTree* with `blockarray_items` bits.
     *         It allocates a *BlockArray* for the HOST and another one for the
     *         DEVICE
     * @param[in] block_items Number of edges per *block*. Maximum number of
                              items that  that can fit in a single *block*
     * @param[in] blockarray_items Number of edges per *BlockArray*.
     *                          Maximum number of items that that can fit in a
     *                          single *BlockArray*
     * @pre BLOCK_ITEMS \f$\le\f$ BLOCKARRAY_ITEMS
     */
    explicit BitTreeBase(int block_items, int blockarray_items) noexcept;

    /**
     * @brief Move constructor
     */
    explicit BitTreeBase(BitTreeBase&& obj) noexcept;

    /**
     * @brief Insert a new *block*
     * @details Find the first empty *block* within the *BlockArray*
     * @return pointers to the *BlockArray*
     *         < `host_block_ptr`, `device_block_ptr` >
     */
    int insert_aux() noexcept;

    void remove_aux(unsigned diff) noexcept;

    /**
     * @brief Size of the *BitTree*
     * @return number of used blocks within the *BlockArray*
     */
    int size() const noexcept;

    /**
     * @brief Check if the *BitTree* is full
     * @return `true` if *BitTree* is full, `false` otherwise
     */
    bool full() const noexcept;

    /**
     * @brief Print BitTree internal representation
     */
    void print() const noexcept;

    /**
     * @brief Print BitTree statistics
     */
    void statistics() const noexcept;

#if defined(B_PLUS_TREE)

    /**
     * @brief Default constructor
     */
    explicit BitTreeBase() noexcept;

    /**
     * @brief Copy constructor
     * @warning Internally replaced with the move constructor
     */
    explicit BitTreeBase(const BitTreeBase& obj) noexcept;

    /**
     * @brief Assignment operator
     */
    BitTreeBase& operator=(BitTreeBase&& obj) noexcept;

#elif defined(RB_TREE)

    explicit BitTreeBase(const BitTreeBase& obj) noexcept;

    BitTreeBase& operator=(const BitTreeBase& obj) = delete;
#else

    BitTreeBase& operator=(BitTreeBase&& obj) noexcept;

    BitTreeBase(BitTreeBase& obj) = delete;
#endif

    //--------------------------------------------------------------------------
    const int _blockarray_bytes;
    const int _log_block_items;
private:
    using word_t = unsigned;

    static const unsigned   WORD_SIZE = sizeof(word_t) * 8;
    static const unsigned  NUM_BLOCKS = EDGES_PER_BLOCKARRAY /
                                        MIN_EDGES_PER_BLOCK;
    static const auto      NUM_LEVELS = xlib::max(1u,
                                   xlib::CeilLog<NUM_BLOCKS, WORD_SIZE>::value);
    //  WORD_SIZE^1 + WORD_SIZE^2 + ... + WORD_SIZE^(NUM_LEVELS - 1)
    using GS = typename xlib::GeometricSerie<WORD_SIZE, NUM_LEVELS - 1>;
    static const auto   INTERNAL_BITS = GS::value - 1;           //-1 : for root
    static const auto  INTERNAL_WORDS = xlib::CeilDiv<INTERNAL_BITS,
                                                      WORD_SIZE>::value;
    static const auto EXTERNAL_WORLDS = xlib::CeilDiv<NUM_BLOCKS,
                                                      WORD_SIZE>::value;
    static const auto   MAX_NUM_WORDS = INTERNAL_WORDS + EXTERNAL_WORLDS;

    const int _block_items;
    const int _blockarray_items;
    const int _num_blocks;
    const int _num_levels;
    const int _internal_bits;
    const int _internal_words;
    const int _external_words;
    const int _num_words;
    const int _total_bits;

    word_t  _array[MAX_NUM_WORDS];
    word_t* _last_level { nullptr };
    size_t  _size       { 0 };

    template<typename Lambda>
    void parent_traverse(int index, const Lambda& lambda) noexcept;
};

//==============================================================================

template<typename block_t, typename offset_t, bool DEVICE>
class BitTree : public BitTreeBase<block_t> {};

template<typename block_t, typename offset_t>
class BitTree<block_t, offset_t, true> : public BitTreeBase<block_t> {
public:
     explicit BitTree(int block_items, int blockarray_items) noexcept;

     explicit BitTree(BitTree&& obj) noexcept;

    /**
     * @brief Decostructor
     * @details Deallocate HOST and DEVICE *BlockArrays*
     */
    ~BitTree() noexcept;

    /**
     * @brief Free host *BlockArray* pointer
     */
    void free_host_ptr() noexcept;

    /**
     * @brief Insert a new *block*
     * @details Find the first empty *block* within the *BlockArray*
     * @return pointers to the *BlockArray*
     *         < `host_block_ptr`, `device_block_ptr` >
     */
    std::pair<byte_t*, byte_t*> insert() noexcept;

    void remove(void* device_ptr) noexcept;

    using BitTreeBase<block_t>::full;
    using BitTreeBase<block_t>::size;
    using BitTreeBase<block_t>::statistics;
    using BitTreeBase<block_t>::print;

    /**
     * @brief Base address of the *BlockArray*
     * @return Pair < `host_block_ptr`, `device_block_ptr` > of *BlockArray*
     */
    std::pair<byte_t*, byte_t*> base_address() const noexcept;

    /**
     * @brief Check if a particular *block* device address belong to the actual
     *        *BlockArray*
     * @param[in] device_ptr pointer to check
     * @return `true` if `device_ptr` belong to the actual *BlockArray*,
     *         `false` otherwise
     */
    bool belong_to(void* device_ptr) const noexcept;

#if defined(B_PLUS_TREE)
    explicit BitTree() noexcept {}

    BitTree(const BitTree& obj) noexcept;   //no explicit for btree::btree_map

    BitTree& operator=(BitTree&& obj) noexcept;

#elif defined(RB_TREE)
    BitTree(const BitTree& obj) noexcept;   //no explicit for std::map

    BitTree& operator=(const BitTree& obj) = delete;
#else
    BitTree(BitTree& obj)                  = delete;

    BitTree& operator=(BitTree&& obj) noexcept; //for vector.erase
#endif

private:
    using BitTreeBase<block_t>::_blockarray_bytes;
    using BitTreeBase<block_t>::_log_block_items;
    byte_t* _h_ptr { nullptr };
    byte_t* _d_ptr { nullptr };
};

//------------------------------------------------------------------------------

template<typename block_t, typename offset_t>
class BitTree<block_t, offset_t, false> : public BitTreeBase<block_t> {
public:
    explicit BitTree(int block_items, int blockarray_items) noexcept;

    explicit BitTree(BitTree&& obj) noexcept;

    ~BitTree() noexcept;

    /**
     * @brief Insert a new *block*
     * @details Find the first empty *block* within the *BlockArray*
     * @return pointers to the *BlockArray*
     *         < `host_block_ptr`, `device_block_ptr` >
     */
    byte_t* insert() noexcept;

    void remove(void* device_ptr) noexcept;

    using BitTreeBase<block_t>::full;
    using BitTreeBase<block_t>::size;
    using BitTreeBase<block_t>::statistics;
    using BitTreeBase<block_t>::print;

    byte_t* base_address() const noexcept;

    bool belong_to(void* host_ptr) const noexcept;

private:
    using BitTreeBase<block_t>::_blockarray_bytes;
    using BitTreeBase<block_t>::_log_block_items;
    byte_t* _h_ptr { nullptr };
};

} // namespace hornets_nest

#include "BitTree.i.hpp"
