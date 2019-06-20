/**
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
#include "VertexBasedKernel.cuh"

namespace hornets_nest {
namespace load_balancing {
/*
template<unsigned VW_SIZE>
template<typename HornetClass, typename Operator>
void VertexBased<VW_SIZE>::apply(const HornetClass& hornet,
                                 const vid_t*       d_input,
                                 int                num_vertices,
                                 const Operator&    op) const noexcept {
    static_assert(IsHornet<HornetClass>::value,
                  "VertexBased: paramenter is not an instance of Hornet Class");

    kernel::vertexBasedKernel<VW_SIZE>
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices) * VW_SIZE, BLOCK_SIZE >>>
        (hornet.device_side(), d_input, num_vertices, op);
    CHECK_CUDA_ERROR
}*/

template<unsigned VW_SIZE>
template<typename HornetClass, typename Operator>
void VertexBased<VW_SIZE>::apply(const HornetClass& hornet,
                                 const vid_t*       d_input,
                                 int                num_vertices,
                                 Operator&&         op) const noexcept {
  if (num_vertices == 0) { return; }
    static_assert(IsHornet<HornetClass>::value,
                  "VertexBased: paramenter is not an instance of Hornet Class");
    int dyn_smem_size = xlib::DeviceProperty::smem_per_block(BLOCK_SIZE);

    kernel::vertexBasedKernel<VW_SIZE>
        <<< xlib::ceil_div<BLOCK_SIZE>(num_vertices) * VW_SIZE, BLOCK_SIZE,
            dyn_smem_size >>>
        (hornet.device_side(), d_input, num_vertices, op);

    CHECK_CUDA_ERROR
}

template<unsigned VW_SIZE>
template<typename HornetClass, typename Operator>
void VertexBased<VW_SIZE>::apply(const HornetClass& hornet, Operator&& op)
                                 const noexcept {
    if (hornet.nV() == 0) { return; }
    static_assert(IsHornet<HornetClass>::value,
                 "VertexBased: paramenter is not an instance of Hornet Class");
    int dyn_smem_size = xlib::DeviceProperty::smem_per_block(BLOCK_SIZE);

    kernel::vertexBasedKernel<VW_SIZE>
        <<< xlib::ceil_div<BLOCK_SIZE>(hornet.nV()) * VW_SIZE, BLOCK_SIZE,
            dyn_smem_size >>>
        (hornet.device_side(), op);
    CHECK_CUDA_ERROR
}

template<unsigned VW_SIZE>
template<typename HornetClass, typename Operator>
void VertexBased<VW_SIZE>::applyVertexPairs(const HornetClass& hornet, Operator&& op)
                                       const noexcept {
    if (hornet.nV() == 0) { return; }
    static_assert(IsHornet<HornetClass>::value,
                 "VertexBased: paramenter is not an instance of Hornet Class");
    //const auto ITEMS_PER_BLOCK = xlib::SMemPerBlock<BLOCK_SIZE, vid_t>::value;
    //const auto   DYN_SMEM_SIZE = ITEMS_PER_BLOCK * sizeof(vid_t);

    int dyn_smem_size = xlib::DeviceProperty::smem_per_block(BLOCK_SIZE);
    kernel::vertexBasedVertexPairsKernel<VW_SIZE>
        <<< xlib::ceil_div<BLOCK_SIZE>(hornet.nV()) * VW_SIZE, BLOCK_SIZE,
            dyn_smem_size >>>
        (hornet.device_side(), op);
    CHECK_CUDA_ERROR
}

} // namespace load_balancing
} // namespace hornets_nest
