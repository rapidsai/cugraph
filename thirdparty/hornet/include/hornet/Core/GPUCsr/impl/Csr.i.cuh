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
#include <Device/Util/Timer.cuh>   //timer::Timer
#include <Device/Primitives/CubWrapper.cuh>
//#include "Core/GPUHornet/impl/HornetKernels.cuh"
#include "Core/GPUCsr/impl/CsrKernels.cuh"

namespace hornets_nest {
namespace gpu {

/////////////
// GPU Csr //
/////////////

template<typename... VertexTypes, typename... EdgeTypes>
int CSR::global_id = 0;

template<typename... VertexTypes, typename... EdgeTypes>
CSR::Csr(const HornetInit& hornet_init,
               bool traspose) noexcept :
                            _hornet_init(hornet_init),
                            _nV(hornet_init.nV()),
                            _nE(hornet_init.nE()),
                            _id(global_id++) {
    /*if (traspose)
        transpose();
    else*/
        initialize();
}

template<typename... VertexTypes, typename... EdgeTypes>
CSR::~Csr() noexcept {
    gpu::free(_d_csr_offsets, _d_degrees);
}

template<typename... VertexTypes, typename... EdgeTypes>
void CSR::initialize() noexcept {
    using namespace timer;
    const auto& vertex_init = _hornet_init._vertex_data_ptrs;
    const auto&   edge_init = _hornet_init._edge_data_ptrs;
    auto        csr_offsets = _hornet_init.csr_offsets();

    const auto& lamba = [](const byte_t* ptr) { return ptr != nullptr; };
    bool vertex_check = std::all_of(vertex_init, vertex_init + NUM_VTYPES - 1,
                                    lamba);
    bool   edge_check = std::all_of(edge_init, edge_init + NUM_ETYPES, lamba);
    if (!vertex_check)
        ERROR("Vertex data not initializated");
    if (!edge_check)
        ERROR("Edge data not initializated");
    Timer<DEVICE> TM;
    TM.start();
    //--------------------------------------------------------------------------
    ///////////////////////////
    // COPY VERTEX/EDGE DATA //
    ///////////////////////////
    auto csr2_offsets = new off2_t[_nV];
    for (vid_t i = 0; i < _nV; i++)
        csr2_offsets[i] = xlib::make2(csr_offsets[i], csr_offsets[i + 1]);

    const void* vertex_ptrs[NUM_VTYPES] = { csr2_offsets };
    const void* edge_ptrs[NUM_ETYPES];
    std::copy(vertex_init + 1, vertex_init + NUM_VTYPES, vertex_ptrs + 1);
    std::copy(edge_init, edge_init + NUM_ETYPES, edge_ptrs);

    _vertex_array.initialize(vertex_ptrs, _nV);
    _edge_array.initialize(edge_ptrs, _nE);

    delete[] csr2_offsets;
    //--------------------------------------------------------------------------
    TM.stop();
    TM.print("Initilization Time:");

    build_device_degrees();

    gpu::allocate(_d_csr_offsets, _nV + 1);
    host::copyToDevice(csr_offsets, _nV + 1, _d_csr_offsets);
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes>
vid_t CSR::nV() const noexcept {
    return _nV;
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes>
eoff_t CSR::nE() const noexcept {
    return _nE;
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes>
const eoff_t* CSR::csr_offsets() noexcept {
    return _hornet_init.csr_offsets();
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes>
const vid_t* CSR::csr_edges() noexcept {
    return _hornet_init.csr_edges();
}

template<typename... VertexTypes, typename... EdgeTypes>
template<int INDEX>
const typename xlib::SelectType<INDEX, VertexTypes...>::type*
CSR::vertex_field() noexcept {
    using T = typename xlib::SelectType<INDEX, VertexTypes...>::type;
    return reinterpret_cast<const T*>(
                _hornet_init._vertex_data_ptrs[INDEX + 1]);
}

template<typename... VertexTypes, typename... EdgeTypes>
template<int INDEX>
const typename xlib::SelectType<INDEX, vid_t, EdgeTypes...>::type*
CSR::edge_field() noexcept {
    using T = typename xlib::SelectType<INDEX, vid_t, EdgeTypes...>::type;
    return reinterpret_cast<const T*>(_hornet_init._edge_data_ptrs[INDEX]);
}

// TO IMPROVE !!!!
template<typename... VertexTypes, typename... EdgeTypes>
const eoff_t* CSR::device_csr_offsets() const noexcept {
    /*if (_d_csr_offsets == nullptr) {
        gpu::allocate(_d_csr_offsets, _nV + 1);
        host::copyToDevice(csr_offsets(), _nV + 1, _d_csr_offsets);
    }*/
    return _d_csr_offsets;
}

template<typename... VertexTypes, typename... EdgeTypes>
const degree_t* CSR::device_degrees() const noexcept {
    return _d_degrees;
}

template<typename... VertexTypes, typename... EdgeTypes>
CSR::CsrDeviceT CSR::device_side() const noexcept {
    using CsrDeviceT = CsrDevice<std::tuple<VertexTypes...>,
                                 std::tuple<EdgeTypes...>>;
    return CsrDeviceT(_nV, _nE,
                      _vertex_array.device_ptr(), _vertex_array.pitch(),
                      _edge_array.device_ptr(), _edge_array.pitch());
}

template<typename... VertexTypes, typename... EdgeTypes>
void CSR::print() noexcept {
    printCsrKernel<<<1, 1>>>(device_side());
    CHECK_CUDA_ERROR
}

template<typename... VertexTypes, typename... EdgeTypes>
void CSR::build_device_degrees() noexcept {
    gpu::allocate(_d_degrees, _nV);
    buildDegreeKernel <<< xlib::ceil_div(_nV, 256), 256 >>>
        (device_side(), _d_degrees);
}

template<typename... VertexTypes, typename... EdgeTypes>
vid_t CSR::max_degree_id() noexcept {
    if (_max_degree_data.first == -1) {
        xlib::CubArgMax<degree_t> arg_max(_d_degrees, _nV);
        _max_degree_data = arg_max.run();
    }
    return _max_degree_data.first;
}

template<typename... VertexTypes, typename... EdgeTypes>
degree_t CSR::max_degree() noexcept {
    if (_max_degree_data.first == -1) {
        xlib::CubArgMax<degree_t> arg_max(_d_degrees, _nV);
        _max_degree_data = arg_max.run();
    }
    return _max_degree_data.second;
}

} // namespace gpu
} // namespace hornets_nest
