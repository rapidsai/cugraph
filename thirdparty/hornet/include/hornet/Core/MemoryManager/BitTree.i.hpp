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
 */
#pragma once

#include "StandardAPI.hpp"
#include <Host/Numeric.hpp>             //xlib::ceil_log
#include <iterator>                     //std::distance

namespace hornets_nest {

template<typename block_t>
BitTreeBase<block_t>
::BitTreeBase(int block_items, int blockarray_items) noexcept :
        _block_items(block_items),
        _blockarray_items(blockarray_items),
        _log_block_items(xlib::log2(block_items)),
        _blockarray_bytes(blockarray_items * sizeof(block_t)),
        _num_blocks(_blockarray_items / _block_items),
        _num_levels(xlib::max(xlib::ceil_log<WORD_SIZE>(_num_blocks), 1)),
        _internal_bits(xlib::geometric_serie<WORD_SIZE>(_num_levels - 1) - 1),
        _internal_words(xlib::ceil_div<WORD_SIZE>(_internal_bits)),
        _external_words(xlib::ceil_div<WORD_SIZE>(_num_blocks)),
        _num_words(_internal_words + _external_words),
        _total_bits(_num_words * WORD_SIZE) {

    assert(xlib::is_power2(block_items));
    assert(xlib::is_power2(blockarray_items));
    assert(block_items <= blockarray_items);

    const word_t EMPTY = static_cast<word_t>(-1);
    std::fill(_array, _array + _num_words, EMPTY);
    _last_level = _array + _internal_words;
}

template<typename block_t>
BitTreeBase<block_t>::BitTreeBase(BitTreeBase&& obj) noexcept :
                    _block_items(obj._block_items),
                    _blockarray_items(obj._blockarray_items),
                    _log_block_items(obj._log_block_items),
                    _blockarray_bytes(obj._blockarray_bytes),
                    _num_blocks(obj._num_blocks),
                    _num_levels(obj._num_levels),
                    _internal_bits(obj._internal_bits),
                    _internal_words(obj._internal_words),
                    _external_words(obj._external_words),
                    _num_words(obj._num_words),
                    _total_bits(obj._total_bits),
                    _last_level(_array + _internal_words),
                    _size(obj._size) {
    assert(_block_items == obj._block_items);
    assert(_blockarray_items == obj._blockarray_items);

    std::copy(obj._array, obj._array + _num_words, _array);
    obj._last_level = nullptr;
    obj._size       = 0;
}

#if defined(B_PLUS_TREE)

template<typename block_t>
BitTreeBase<block_t>::BitTreeBase() noexcept :
                _block_items(0),
                _blockarray_items(0),
                _log_block_items(0),
                _blockarray_bytes(0),
                _num_blocks(0),
                _num_levels(0),
                _internal_bits(0),
                _internal_words(0),
                _external_words(0),
                _num_words(0),
                _total_bits(0) {}

#endif

#if !defined(RB_TREE)

template<typename block_t>
BitTreeBase<block_t>& BitTreeBase<block_t>
::operator=(BitTreeBase<block_t>&& obj) noexcept {
    assert(_block_items == obj._block_items || _block_items == 0 ||
           obj._block_items == 0);
    assert(_blockarray_items == obj._blockarray_items ||
           _blockarray_items == 0 || obj._blockarray_items == 0);
    if (_block_items == 0) {
        const_cast<int&>(_block_items)      = obj._block_items;
        const_cast<int&>(_blockarray_items) = obj._blockarray_items;
        const_cast<int&>(_log_block_items)  = obj._log_block_items;
        const_cast<int&>(_blockarray_bytes) = obj._blockarray_bytes;
        const_cast<int&>(_num_blocks)       = obj._num_blocks;
        const_cast<int&>(_num_levels)       = obj._num_levels;
        const_cast<int&>(_internal_bits)    = obj._internal_bits;
        const_cast<int&>(_internal_words)   = obj._internal_words;
        const_cast<int&>(_external_words)   = obj._external_words;
        const_cast<int&>(_num_words)        = obj._num_words;
        const_cast<int&>(_total_bits)       = obj._total_bits;
    }
    std::copy(obj._array, obj._array + _num_words, _array);
    _last_level = _array + _internal_words;
    _size       = obj._size;

    obj._last_level = nullptr;
    obj._size       = 0;
    return *this;
}
#endif

#if defined(B_PLUS_TREE) || defined(RB_TREE)

template<typename block_t>
BitTreeBase<block_t>::BitTreeBase(const BitTreeBase& obj) noexcept :
                    _block_items(obj._block_items),
                    _blockarray_items(obj._blockarray_items),
                    _log_block_items(obj._log_block_items),
                    _blockarray_bytes(obj._blockarray_bytes),
                    _num_blocks(obj._num_blocks),
                    _num_levels(obj._num_levels),
                    _internal_bits(obj._internal_bits),
                    _internal_words(obj._internal_words),
                    _external_words(obj._external_words),
                    _num_words(obj._num_words),
                    _total_bits(obj._total_bits),
                    _last_level(_array + _internal_words),
                    _size(obj._size) {
    assert(_block_items == obj._block_items);
    assert(_blockarray_items == obj._blockarray_items);

    std::copy(obj._array, obj._array + _num_words, _array);
    _last_level = _array + _internal_words;
    _size       = obj._size;

    const_cast<BitTreeBase<block_t>&>(obj)._last_level = nullptr;
    const_cast<BitTreeBase<block_t>&>(obj)._size       = 0;
}

#endif

//------------------------------------------------------------------------------

template<typename block_t>
int BitTreeBase<block_t>::insert_aux() noexcept {
    assert(_size < _num_blocks && "tree is full");
    _size++;
    //find the first empty location
    int index = 0;
    for (int i = 0; i < _num_levels - 1; i++) {
        assert(index < _total_bits && _array[index / WORD_SIZE] != 0);
        int pos = __builtin_ctz(_array[index / WORD_SIZE]);
        index   = (index + pos + 1) * WORD_SIZE;
    }
    assert(index < _total_bits && _array[index / WORD_SIZE] != 0);
    index += __builtin_ctz(_array[index / WORD_SIZE]);
    assert(index < _total_bits);

    xlib::delete_bit(_array, index);
    if (_array[index / WORD_SIZE] == 0) {
        const auto& lambda = [&](int index) {
                                          xlib::delete_bit(_array, index);
                                          return _array[index / WORD_SIZE] != 0;
                                        };
        parent_traverse(index, lambda);
    }
    int block_index = index - _internal_bits;
    assert(block_index >= 0 && block_index < _blockarray_items);
    return block_index;
}

//------------------------------------------------------------------------------
template<typename block_t>
void BitTreeBase<block_t>::remove_aux(unsigned diff) noexcept {
    assert(_size != 0 && "tree is empty");
    _size--;
    int p_index = diff >> _log_block_items;   // diff / block_items
    assert(p_index < _external_words * sizeof(word_t) * 8u);
    assert(xlib::read_bit(_last_level, p_index) == 0 && "not found");
    xlib::write_bit(_last_level, p_index);
    p_index += _internal_bits;

    parent_traverse(p_index, [&](int index) {
                                bool ret = _array[index / WORD_SIZE] != 0;
                                xlib::write_bit(_array, index);
                                return ret;
                            });
}

template<typename block_t>
template<typename Lambda>
void BitTreeBase<block_t>
::parent_traverse(int index, const Lambda& lambda) noexcept {
    index /= WORD_SIZE;
    while (index != 0) {
        index--;
        if (lambda(index))
            return;
        index /= WORD_SIZE;
    }
}

template<typename block_t>
int BitTreeBase<block_t>::size() const noexcept {
    return _size;
}

template<typename block_t>
bool BitTreeBase<block_t>::full() const noexcept {
    return _size == static_cast<size_t>(_num_blocks);
}

template<typename block_t>
void BitTreeBase<block_t>::print() const noexcept {
    const int ROW_SIZE = 64;
    int          count = WORD_SIZE;
    auto       tmp_ptr = _array;
    std::cout << "BitTree:\n";

    for (int i = 0; i < _num_levels - 1; i++) {
        std::cout << "\nlevel " << i << " :\n";
        assert(count < ROW_SIZE || count % ROW_SIZE == 0);

        int size = std::min(count, ROW_SIZE);
        for (int j = 0; j < count; j += ROW_SIZE) {
            xlib::printBits(tmp_ptr, size);
            tmp_ptr += size / WORD_SIZE;
            if (tmp_ptr >= _array + _num_words)
                break;
        }
        count *= WORD_SIZE;
    }
    std::cout << "\nlevel " << _num_levels - 1 << " :\n";
    xlib::printBits(tmp_ptr, _external_words * WORD_SIZE);
    std::cout << std::endl;
}

template<typename block_t>
void BitTreeBase<block_t>::statistics() const noexcept {
    std::cout << "\nBitTree Statistics:\n"
              << "\n     BLOCK_ITEMS: " << _block_items
              << "\nBLOCKARRAY_ITEMS: " << _blockarray_items
              << "\n      NUM_BLOCKS: " << _num_blocks
              << "\n       sizeof(T): " << sizeof(block_t)
              << "\n   BLOCK_SIZE(b): " << _block_items * sizeof(block_t)
              << "\n"
              << "\n      NUM_LEVELS: " << _num_levels
              << "\n       WORD_SIZE: " << WORD_SIZE
              << "\n   INTERNAL_BITS: " << _internal_bits
              << "\n   EXTERNAL_BITS: " << _num_blocks
              << "\n      TOTAL_BITS: " << _total_bits
              << "\n  INTERNAL_WORDS: " << _internal_words
              << "\n EXTERNAL_WORLDS: " << _external_words
              << "\n       NUM_WORDS: " << _num_words << "\n\n";
}

//==============================================================================
//==============================================================================

template<typename block_t, typename offset_t>
BitTree<block_t, offset_t, true>
::BitTree(int block_items, int blockarray_items) noexcept :
                           BitTreeBase<block_t>(block_items, blockarray_items) {
    _h_ptr = new byte_t[_blockarray_bytes];
    gpu::allocate(_d_ptr, _blockarray_bytes);
}

template<typename block_t, typename offset_t>
BitTree<block_t, offset_t, true>
::BitTree(BitTree<block_t, offset_t, true>&& obj) noexcept :
                            BitTreeBase<block_t>(std::move(obj)),
                            _h_ptr(obj._h_ptr),
                            _d_ptr(obj._d_ptr) {
    obj._h_ptr = nullptr;
    obj._d_ptr = nullptr;
}

template<typename block_t, typename offset_t>
BitTree<block_t, offset_t, true>::~BitTree() noexcept {
    gpu::free(_d_ptr);
    delete[] _h_ptr;
}

template<typename block_t, typename offset_t>
std::pair<byte_t*, byte_t*> BitTree<block_t, offset_t, true>::insert() noexcept{
    int block_index = BitTreeBase<block_t>::insert_aux();
    auto     offset = (block_index * sizeof(offset_t)) << _log_block_items;
    return std::pair<byte_t*, byte_t*>(_h_ptr + offset, _d_ptr + offset);
}

template<typename block_t, typename offset_t>
void BitTree<block_t, offset_t, true>::remove(void* device_ptr) noexcept {
    unsigned diff = std::distance(reinterpret_cast<block_t*>(_d_ptr),
                                  static_cast<block_t*>(device_ptr));
    BitTreeBase<block_t>::remove_aux(diff);
}

template<typename block_t, typename offset_t>
std::pair<byte_t*, byte_t*>
BitTree<block_t, offset_t, true>::base_address() const noexcept {
    return std::pair<byte_t*, byte_t*>(_h_ptr, _d_ptr);
}

template<typename block_t, typename offset_t>
bool BitTree<block_t, offset_t, true>::belong_to(void* to_check) const noexcept{
    return to_check >= _d_ptr && to_check < _d_ptr + _blockarray_bytes;
}

template<typename block_t, typename offset_t>
void BitTree<block_t, offset_t, true>::free_host_ptr() noexcept {
    delete[] _h_ptr;
    _h_ptr = nullptr;
}

#if !defined(RB_TREE)

template<typename block_t, typename offset_t>
BitTree<block_t, offset_t, true>&
BitTree<block_t, offset_t, true>
::operator=(BitTree<block_t, offset_t, true>&& obj) noexcept {
    BitTreeBase<block_t>::operator=(std::move(obj));
    _h_ptr = obj._h_ptr;
    _d_ptr = obj._d_ptr;
    obj._h_ptr = nullptr;
    obj._d_ptr = nullptr;
    return *this;
}

#endif

#if defined(RB_TREE) || defined(B_PLUS_TREE)

template<typename block_t, typename offset_t>
BitTree<block_t, offset_t, true>::BitTree(const BitTree& obj) noexcept :
                                BitTreeBase<block_t>(obj),
                                _h_ptr(obj._h_ptr),
                                _d_ptr(obj._d_ptr) {}
#endif

//==============================================================================
//==============================================================================

template<typename block_t, typename offset_t>
BitTree<block_t, offset_t, false>
::BitTree(int block_items, int blockarray_items) noexcept :
                           BitTreeBase<block_t>(block_items, blockarray_items) {
    _h_ptr = new byte_t[_blockarray_bytes];
}

template<typename block_t, typename offset_t>
BitTree<block_t, offset_t, false>
::BitTree(BitTree<block_t, offset_t, false>&& obj) noexcept :
                                    BitTreeBase<block_t>(obj),
                                    _h_ptr(obj._h_ptr) {
    obj._h_ptr  = nullptr;
}

template<typename block_t, typename offset_t>
BitTree<block_t, offset_t, false>::~BitTree() noexcept {
    delete[] _h_ptr;
}

template<typename block_t, typename offset_t>
byte_t* BitTree<block_t, offset_t, false>::insert() noexcept {
    int block_index = BitTreeBase<block_t>::insert_aux();
    auto     offset = (block_index * sizeof(offset_t)) << _log_block_items;
    return _h_ptr + offset;
}

template<typename block_t, typename offset_t>
void BitTree<block_t, offset_t, false>::remove(void* host_ptr) noexcept {
    unsigned diff = std::distance(reinterpret_cast<block_t*>(_h_ptr),
                                  static_cast<block_t*>(host_ptr));
    BitTreeBase<block_t>::remove_aux(diff);
}

template<typename block_t, typename offset_t>
byte_t* BitTree<block_t, offset_t, false>::base_address() const noexcept {
    return _h_ptr;
}

template<typename block_t, typename offset_t>
bool BitTree<block_t, offset_t, false>
::belong_to(void* to_check) const noexcept {
    return to_check >= _h_ptr && to_check < _h_ptr + _blockarray_bytes;
}

} // namespace hornets_nest
