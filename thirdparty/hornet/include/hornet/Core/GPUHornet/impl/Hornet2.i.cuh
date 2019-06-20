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
#include <Graph/GraphBase.hpp>               //graph::StructureProp
#include "HornetKernels.cuh"
#include <Device/Primitives/CubWrapper.cuh> //xlib::CubSortByKey
#include <Host/FileUtil.hpp>                //xlib::MemoryMapped

namespace hornets_nest {
namespace gpu {

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::print() noexcept {
    printKernel<<<1, 1>>>(device_side());
    CHECK_CUDA_ERROR
}

/*
 * !!!!! 4E + 2V
 */
template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
[[deprecated]]
void HORNET::transpose() noexcept {
    const unsigned BLOCK_SIZE = 256;
    _mem_manager.clear();

    eoff_t* d_csr_offsets, *d_counts_out;
    vid_t*  d_coo_src, *d_coo_dst, *d_coo_src_out, *d_coo_dst_out,*d_unique_out;

    gpu::allocate(d_csr_offsets, _nV + 1);
    gpu::allocate(d_coo_src, _nE);
    gpu::allocate(d_coo_dst, _nE);
    gpu::allocate(d_coo_src_out, _nE);
    gpu::allocate(d_coo_dst_out, _nE);
    gpu::allocate(d_counts_out, _nV + 1);
    gpu::allocate(d_unique_out, _nV);

    host::copyToDevice(_csr_offsets, _nV + 1, d_csr_offsets);
    host::copyToDevice(_csr_edges, _nE, d_coo_dst);
    host::copyToDevice(0, d_counts_out + _nV);

    CSRtoCOOKernel<BLOCK_SIZE>
        <<< xlib::ceil_div(_nV, BLOCK_SIZE), BLOCK_SIZE >>>
        (d_csr_offsets, _nV, d_coo_dst);

    xlib::CubSortByKey<vid_t, vid_t>::srun(d_coo_dst, d_coo_src, _nE,
                                           d_coo_dst_out, d_coo_src_out,
                                           _nV - 1);
    xlib::CubRunLengthEncode<vid_t>::srun(d_coo_dst_out, _nE,
                                          d_unique_out, d_counts_out);
    xlib::CubExclusiveSum<eoff_t>::srun(d_counts_out, _nV + 1);

    _csr_offsets = new eoff_t[_nV + 1];
    _csr_edges   = new vid_t[_nV + 1];
    gpu::copyToHost(d_counts_out, _nV + 1, const_cast<eoff_t*>(_csr_offsets));
    gpu::copyToHost(d_coo_src_out, _nE, const_cast<vid_t*>(_csr_edges));
    _internal_csr_data = true;

    gpu::free(d_unique_out, d_counts_out, d_coo_dst_out, d_coo_src_out, d_coo_dst, d_coo_src, d_csr_offsets);

    initialize();
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::check_sorted_adjs() const noexcept {
    checkSortedKernel <<< xlib::ceil_div<256>(_nV), 256 >>> (device_side());
}
//------------------------------------------------------------------------------

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::build_device_degrees() noexcept {
    gpu::allocate(_d_degrees, _nV);
    buildDegreeKernel <<< xlib::ceil_div(_nV, 256), 256 >>>
        (device_side(), _d_degrees);
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
vid_t HORNET::max_degree_id() noexcept {
    if (_max_degree_data.first == -1) {
        xlib::CubArgMax<degree_t> arg_max(_d_degrees, _nV);
        _max_degree_data = arg_max.run();
    }
    return _max_degree_data.first;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
vid_t HORNET::max_degree() noexcept {
    if (_max_degree_data.first == -1) {
        xlib::CubArgMax<degree_t> arg_max(_d_degrees, _nV);
        _max_degree_data = arg_max.run();
    }
    return _max_degree_data.second;
}

//==============================================================================

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
[[deprecated]]
void HORNET::convert_to_csr(eoff_t* csr_offsets, vid_t* csr_edges)
                               const noexcept {

    /*using pair_t = typename std::pair<vid_t*, degree_t>;
    auto d_vertex_basic_ptr = reinterpret_cast<pair_t*>(_d_vertices);

    auto h_vertex_basic_ptr = new pair_t[_nV];
    gpu::copyToHost(d_vertex_basic_ptr, _nV, h_vertex_basic_ptr);

    csr_offsets[0] = 0;
    for (vid_t i = 1; i <= _nV; i++)
        csr_offsets[i] = h_vertex_basic_ptr[i - 1].second + csr_offsets[i - 1];
    //--------------------------------------------------------------------------
    eoff_t offset = 0;
    for (vid_t i = 0; i < _nV; i++) {
        degree_t degree = h_vertex_basic_ptr[i].second;
        if (degree == 0) continue;
        cuMemcpyToHostAsync(h_vertex_basic_ptr[i].first,
                            h_vertex_basic_ptr[i].second, csr_edges + offset);
        offset += degree;
    }
    cudaDeviceSynchronize();
    delete[] h_vertex_basic_ptr;*/
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
[[deprecated]]
void HORNET::check_consistency(const HornetInit& hornet_init) const noexcept {
    auto csr_offsets = new eoff_t[_nV + 1];
    auto csr_edges   = new vid_t[_nE];
    convert_to_csr(csr_offsets, csr_edges);

    auto offsets_check = std::equal(csr_offsets, csr_offsets + _nV,
                                    hornet_init.csr_offsets());
    if (!offsets_check)
        ERROR("Vertex Array not consistent")
    auto edge_ref = hornet_init._edge_data_ptrs[0];
    auto neighbor_ptr = reinterpret_cast<const vid_t*>(edge_ref);
    if (!std::equal(csr_edges, csr_edges + _nE, neighbor_ptr))
        ERROR("Edge Array not consistent")
    delete[] csr_offsets;
    delete[] csr_edges;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
[[deprecated]]
void HORNET::store_snapshot(const std::string& filename) const noexcept {
    auto csr_offsets = new eoff_t[_nV + 1];
    auto csr_edges   = new vid_t[_nE];
    convert_to_csr(csr_offsets, csr_edges);

    graph::StructureProp structure(graph::structure_prop::DIRECTED);
    size_t  base_size = sizeof(_nV) + sizeof(_nE) + sizeof(structure);
    size_t file_size1 = (static_cast<size_t>(_nV) + 1) * sizeof(eoff_t) +
                        (static_cast<size_t>(_nE)) * sizeof(vid_t);

    size_t file_size  = base_size + file_size1;

    std::cout << "Graph To binary file: " << filename
              << " (" << (file_size >> 20) << ") MB" << std::endl;

    std::string class_id = xlib::type_name<vid_t>() + xlib::type_name<eoff_t>();
    file_size           += class_id.size();
    xlib::MemoryMapped memory_mapped(filename.c_str(), file_size,
                                     xlib::MemoryMapped::WRITE, true);

    memory_mapped.write(class_id.c_str(), class_id.size(),              //NOLINT
                        &_nV, 1, &_nE, 1,                               //NOLINT
                        csr_offsets, _nV + 1, csr_edges, _nE);          //NOLINT
    delete[] csr_offsets;
    delete[] csr_edges;
}

template<typename... VertexTypes, typename... EdgeTypes, bool FORCE_SOA>
void HORNET::mem_manager_info() const noexcept {
    _mem_manager.statistics();
}

} // namespace gpu
} // namespace hornets_nest
