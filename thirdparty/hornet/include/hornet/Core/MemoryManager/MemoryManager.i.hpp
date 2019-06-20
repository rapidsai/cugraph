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
#include <iomanip>  //std::setw

namespace hornets_nest {

#define MEMORY_MANAGER MemoryManager<block_t,offset_t,DEVICE>

template<typename block_t, typename offset_t, bool DEVICE>
inline MEMORY_MANAGER::MemoryManager() noexcept {
#if !defined(B_PLUS_TREE) && !defined(RB_TREE)
    const auto  LOW = MIN_EDGES_PER_BLOCK;
    const auto HIGH = EDGES_PER_BLOCKARRAY;
    for (size_t size = LOW, i = 0; size <= HIGH; size *= 2, i++)
        bit_tree_set[i].reserve(512);
#if defined(EXPERIMENTAL)
    zero_container.reserve(512);
#endif
#endif
}

template<typename block_t, typename offset_t, bool DEVICE>
std::pair<byte_t*, byte_t*> MEMORY_MANAGER
::insert_aux(ContainerT& container, int index) noexcept {
    using BitTreeT = BitTree<block_t, offset_t, DEVICE>;
    using   pair_t = std::pair<byte_t*, BitTreeT>;
    const auto HIGH = EDGES_PER_BLOCKARRAY;

    if (container.size() == 0) {
#if defined(B_PLUS_TREE) || defined(RB_TREE)
        BitTreeT bit_tree(MIN_EDGES_PER_BLOCK * (1 << index), HIGH);
        container.insert(pair_t(bit_tree.base_address().second, bit_tree));
#else
        container.push_back(BitTreeT(MIN_EDGES_PER_BLOCK * (1 << index), HIGH));
#endif
        _num_blockarrays++;
    }
    for (auto& it : container) {
        if (!it.SC_MACRO full())
            return it.SC_MACRO insert();
    }
    _num_blockarrays++;
    auto      block_items = MIN_EDGES_PER_BLOCK * (1 << index);
    auto blockarray_items = block_items <= EDGES_PER_BLOCKARRAY ?
                            EDGES_PER_BLOCKARRAY : block_items;
#if defined(B_PLUS_TREE) || defined(RB_TREE)
    BitTreeT bit_tree(block_items, blockarray_items);
    auto ret = bit_tree.insert();
    container.insert(pair_t(bit_tree.base_address().second, bit_tree));
    return ret;
#else
    container.push_back(BitTreeT(block_items, blockarray_items));
    return container.back().insert();
#endif
}

template<typename block_t, typename offset_t, bool DEVICE>
std::pair<byte_t*, byte_t*> MEMORY_MANAGER::insert(degree_t degree) noexcept {
#if defined(EXPERIMENTAL)
    if (degree == 0)
        return insert_aux(zero_container, degree);
#endif
    int index = find_bin(degree);
    _num_inserted_edges += degree;
    return insert_aux(bit_tree_set[index], index);
}

template<typename block_t, typename offset_t, bool DEVICE>
void MEMORY_MANAGER::remove(void* device_ptr, degree_t degree) noexcept {
    //assert(degree > 0);
    if (device_ptr == nullptr)
        return;
    int index = find_bin(degree);
    _num_inserted_edges -= degree;

    auto& container = bit_tree_set[index];
#if defined(B_PLUS_TREE) || defined(RB_TREE)
    byte_t* low_address = reinterpret_cast<byte_t*>(device_ptr)
                          - EDGES_PER_BLOCKARRAY;
    auto& it = container.upper_bound(low_address);
#if defined(EXPERIMENTAL)
    if (index == 0 && it == container.end()) {
        it = zero_container.upper_bound(low_address);
        assert(it != zero_container.end() && "pointer not found");
        it->second.remove(device_ptr);
        if (it->second.size() == 0) {  //shrink
            _num_blockarrays--;
            zero_container.erase(it);
        }
        return;
    }
#endif
    assert(it != container.end() && "pointer not found");
    it->second.remove(device_ptr);
    if (it->second.size() == 0) {  //shrink
        _num_blockarrays--;
        container.erase(it);
    }
#else
    const auto& end_it = container.end();
    for (auto it = container.begin(); it != end_it; it++) {
        if (it->belong_to(device_ptr)) {
            it->remove(device_ptr);
            if (it->size() == 0) {  //shrink
                _num_blockarrays--;
                container.erase(it);
            }
            return;
        }
    }
#if defined(EXPERIMENTAL)
    if (index == 0) {
        const auto& end_it = zero_container.end();
        for (auto it = zero_container.begin(); it != end_it; it++) {
            if (it->belong_to(device_ptr)) {
                it->remove(device_ptr);
                if (it->size() == 0) {  //shrink
                    _num_blockarrays--;
                    zero_container.erase(it);
                }
                return;
            }
        }
    }
#endif
    assert(false && "pointer not found");
#endif
}

template<typename block_t, typename offset_t, bool DEVICE>
std::pair<byte_t*, byte_t*>
MEMORY_MANAGER::get_blockarray_ptr(int blockarray_index) noexcept {
    assert(blockarray_index >= 0 && blockarray_index < _num_blockarrays);
    for (int i = 0; i < MM_LOG_LIMIT; i++) {
        int container_size = bit_tree_set[i].size();

        if (blockarray_index < container_size) {
            const auto& it = std::next(bit_tree_set[i].begin(),
                                       blockarray_index);
            return it->SC_MACRO base_address();
        }
        blockarray_index -= container_size;
    }
    assert(false && "blockarray_index out-of-bounds");
    return std::pair<byte_t*, byte_t*>(nullptr, nullptr);
}

template<typename block_t, typename offset_t, bool DEVICE>
void MEMORY_MANAGER::free_host_ptrs() noexcept {
    for (int i = 0; i < MM_LOG_LIMIT; i++) {
        for (auto& it : bit_tree_set[i])
            it.SC_MACRO free_host_ptr();
    }
}

template<typename block_t, typename offset_t, bool DEVICE>
void MEMORY_MANAGER::clear() noexcept {
    for (int i = 0; i < MM_LOG_LIMIT; i++)
        bit_tree_set[i].clear();
}

//------------------------------------------------------------------------------

template<typename block_t, typename offset_t, bool DEVICE>
void MEMORY_MANAGER::statistics() const noexcept {
    std::cout << std::setw(5)  << "IDX"
              << std::setw(14) << "BLOCKS_EDGES"
              << std::setw(18) << "BLOCKARRAY_EDGES"
              << std::setw(16) << "N. BLOCKARRAYS"
              << std::setw(11) << "N. BLOCKS" << "\n";
    int max_index = 0;
    for (int i = 0; i < MM_LOG_LIMIT; i++) {
        if (bit_tree_set[i].size() > 0)
            max_index = i;
    }
    int allocated_items = 0, block_edges = 0;
    for (int index = 0; index <= max_index; index++) {
        const degree_t  block_items = MIN_EDGES_PER_BLOCK * (1 << index);
        const auto blockarray_items = block_items <= EDGES_PER_BLOCKARRAY ?
                                      EDGES_PER_BLOCKARRAY : block_items;
        const auto& container = bit_tree_set[index];
        int used_blocks = 0;
        for (const auto& it : container)
            used_blocks += it.SC_MACRO size();
        block_edges     += used_blocks * block_items;
        allocated_items += container.size() * blockarray_items;

        std::cout << std::setw(4)  << index
                  << std::setw(15) << block_items
                  << std::setw(18) << blockarray_items
                  << std::setw(16) << container.size()
                  << std::setw(11) << used_blocks << "\n";
    }
    auto efficiency1 = xlib::per_cent(_num_inserted_edges, block_edges);
    auto efficiency2 = xlib::per_cent(block_edges, allocated_items);
    auto efficiency3 = xlib::per_cent(_num_inserted_edges, allocated_items);

    std::cout << "\n         N. BlockArrays: " << xlib::format(_num_blockarrays)
              << "\n  Total Allocated Edges: " << xlib::format(allocated_items)
              << "\n      Total Block Edges: " << xlib::format(block_edges)
              << "\n        Allocated Space: " << (allocated_items / xlib::MB)
                                               << " MB"
              /*<< "\n             Used Space: " << (block_edges / xlib::MB)
                                               << " MB"*/
              << "\n  (Internal) Efficiency: " << xlib::format(efficiency1, 1)
                                               << " %"
              << "\n  (External) Efficiency: " << xlib::format(efficiency2, 1)
                                               << " %"
              << "\n   (Overall) Efficiency: " << xlib::format(efficiency3, 1)
                                               << " %\n" << std::endl;
}

template<typename block_t, typename offset_t, bool DEVICE>
int MEMORY_MANAGER::find_bin(degree_t degree) const noexcept {
    const unsigned LOG_EDGES_PER_BLOCK = xlib::Log2<MIN_EDGES_PER_BLOCK>::value;
    return PREFER_FASTER_UPDATE ?
        (static_cast<size_t>(degree) < MIN_EDGES_PER_BLOCK ? 0 :
             xlib::ceil_log2(degree + 1) - LOG_EDGES_PER_BLOCK) :
        (static_cast<size_t>(degree) <= MIN_EDGES_PER_BLOCK ? 0 :
            xlib::ceil_log2(degree) - LOG_EDGES_PER_BLOCK);
}

template<typename block_t, typename offset_t, bool DEVICE>
int MEMORY_MANAGER::num_blockarrays() const noexcept {
    return _num_blockarrays;
}

} // namespace hornets_nest
