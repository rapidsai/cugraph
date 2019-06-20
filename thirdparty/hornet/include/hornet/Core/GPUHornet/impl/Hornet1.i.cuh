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
#include "StandardAPI.hpp"
#include <Device/Util/Timer.cuh>        //timer::Timer
#include <cstring>                      //std::memcpy

namespace hornets_nest {
namespace gpu {

////////////////
// Hornet GPU //
////////////////

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
int HORNET::global_id = 0;

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
HORNET::Hornet(const HornetInit& hornet_init,
               bool transpose) noexcept :
                            _hornet_init(hornet_init),
                            _nV(hornet_init.nV()),
                            _nE(hornet_init.nE()),
                            _id(global_id++),
                            _is_sorted(hornet_init.is_sorted()),
                            cub_prefixsum(0),
                            cub_runlength(0),
                            cub_select_flag(0),
                            cub_sort(0),
                            cub_sort_pair(0, false) {
    initialize();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
HORNET::~Hornet() noexcept {
    gpu::free(_d_csr_offsets, _d_degrees);
    delete[] _csr_offsets;
    delete[] _csr_edges;
    gpu::free(_d_batch_src, _d_batch_dst, _d_tmp_sort_src, _d_tmp_sort_dst,
            _d_counts, _d_unique, _d_degree_tmp, _d_flags);
    gpu::free(_d_locations, _d_batch_offset, _d_counter,
            _d_queue_new_degree, _d_queue_new_ptr, _d_queue_old_ptr,
            _d_queue_old_degree, _d_queue_id, _d_queue_size);
    host::freePageLocked(_h_queue_new_ptr, _h_queue_new_degree, _h_queue_old_ptr,
               _h_queue_old_degree);
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::initialize() noexcept {
    using namespace timer;
    auto& vertex_init = _hornet_init._vertex_data_ptrs;
    auto&   edge_init = _hornet_init._edge_data_ptrs;
    auto  csr_offsets = _hornet_init.csr_offsets();

    const auto& lamba = [](const byte_t* ptr) { return ptr != nullptr; };
    bool vertex_check = std::all_of(vertex_init, vertex_init + NUM_VTYPES,
                                    lamba);
    bool   edge_check = std::all_of(edge_init, edge_init + NUM_ETYPES, lamba);
    if (!vertex_check)
        ERROR("Vertex data not initializated");
    if (!edge_check)
        ERROR("Edge data not initializated");
    //Timer<DEVICE> TM;
    //TM.start();
    //--------------------------------------------------------------------------
    ///////////////////////////////////
    // EDGES INITIALIZATION AND COPY //
    ///////////////////////////////////
    using R = AoSData<vid_t, EdgeTypes...>;
    auto degrees_array = new size_t[_nV];
    auto ptrs_array    = new void*[_nV];
    const size_t BLOCKARRAY_SIZE = PITCH<EdgeTypes...> * NUM_ETYPES;

    for (vid_t i = 0; i < _nV; i++) {
        auto      degree = csr_offsets[i + 1] - csr_offsets[i];
        degrees_array[i] = static_cast<size_t>(degree);

        /*if (degree == 0) {
            ptrs_array[i] = nullptr;
            continue;
        }*/
        if (static_cast<size_t>(degree) >= EDGES_PER_BLOCKARRAY)
            ERROR("degree >= EDGES_PER_BLOCKARRAY, (", degree, ")")

        const auto& mem_data = _mem_manager.insert(degree);
        ptrs_array[i]        = mem_data.second;
        if (degree == 0)
            continue;
        byte_t* h_blockarray = mem_data.first;
        size_t        offset = csr_offsets[i];

        if (FORCE_SOA || !xlib::IsVectorizable<vid_t, EdgeTypes...>::value ||
                sizeof...(EdgeTypes) == 0) {
            for (size_t j = 0; j < NUM_ETYPES; j++) {
                size_t    num_bytes = degree * ETYPE_SIZES[j];
                size_t offset_bytes = offset * ETYPE_SIZES[j];
                std::memcpy(h_blockarray + PITCH<EdgeTypes...> * j,
                            edge_init[j] + offset_bytes, num_bytes);
            }
        }
        else {
            for (auto k = 0; k < degree; k++)
                reinterpret_cast<R*>(h_blockarray)[k] = R(edge_init,offset + k);
        }
    }

    int num_blockarrays = _mem_manager.num_blockarrays();
    for (int i = 0; i < num_blockarrays; i++) {
        const auto& mem_data = _mem_manager.get_blockarray_ptr(i);
        host::copyToDevice(mem_data.first, BLOCKARRAY_SIZE, mem_data.second);
    }
    //--------------------------------------------------------------------------
    ////////////////////////
    // COPY VERTICES DATA //
    ////////////////////////
    const void* vertex_ptrs[NUM_VTYPES + 1] = { degrees_array, ptrs_array };
    std::copy(vertex_init + 1, vertex_init + NUM_VTYPES, vertex_ptrs + 2);

    _vertex_array.initialize(vertex_ptrs, _nV);

    delete[] degrees_array;
    delete[] ptrs_array;
    //--------------------------------------------------------------------------
    //TM.stop();
    //TM.print("Initilization Time:");

    //_mem_manager.free_host_ptr();
    build_device_degrees();

    gpu::allocate(_d_csr_offsets, _nV + 1);
    host::copyToDevice(csr_offsets, _nV + 1, _d_csr_offsets);
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
vid_t HORNET::nV() const noexcept {
    return _nV;
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
eoff_t HORNET::nE() const noexcept {
    return _nE;
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
const eoff_t* HORNET::csr_offsets() noexcept {
    return _hornet_init.csr_offsets();
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
const vid_t* HORNET::csr_edges() noexcept {
    return _hornet_init.csr_edges();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
template<int INDEX>
const typename xlib::SelectType<INDEX, VertexTypes...>::type*
HORNET::vertex_field() noexcept {
    using T = typename xlib::SelectType<INDEX, VertexTypes...>::type;
    return reinterpret_cast<const T*>(
                _hornet_init._vertex_data_ptrs[INDEX + 1]);
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
template<int INDEX>
const typename xlib::SelectType<INDEX, vid_t, EdgeTypes...>::type*
HORNET::edge_field() noexcept {
    using T = typename xlib::SelectType<INDEX, vid_t, EdgeTypes...>::type;
    return reinterpret_cast<const T*>(_hornet_init._edge_data_ptrs[INDEX]);
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
const eoff_t* HORNET::device_csr_offsets() const noexcept {
    /*if (_d_csr_offsets == nullptr) {
        gpu::allocate(_d_csr_offsets, _nV + 1);
        host::copyToDevice(csr_offsets(), _nV + 1, _d_csr_offsets);
    }*/
    return _d_csr_offsets;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
const degree_t* HORNET::device_degrees() const noexcept {
    return _d_degrees;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
HORNET::HornetDeviceT HORNET::device_side() const noexcept {
    using HornetDeviceT = HornetDevice<std::tuple<VertexTypes...>,
                                       std::tuple<EdgeTypes...>,
                                       FORCE_SOA>;
    return HornetDeviceT(_nV, _nE, _vertex_array.device_ptr(),
                         _vertex_array.pitch());
}

} // namespace gpu
} // namespace hornets_nest
