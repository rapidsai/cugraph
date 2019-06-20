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
#include <Device/Primitives/BinarySearchLB.cuh>

namespace hornets_nest {
/**
 * @brief
 */
namespace load_balancing {
namespace kernel {

template<bool = true>
__global__
void computeWorkKernel(const vid_t*    __restrict__ d_input,
                       const degree_t* __restrict__ d_degrees,
                       int                          num_vertices,
                       int*            __restrict__ d_work) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (auto i = id; i < num_vertices; i += stride)
        d_work[i] = d_degrees[ d_input[i] ];
}

template<unsigned BLOCK_SIZE,
         typename HornetDevice, typename Operator>
__global__
void binarySearchKernel(HornetDevice              hornet,
                        const vid_t* __restrict__ d_input,
                        const int*   __restrict__ d_work,
                        int                       work_size,
                        Operator                  op) {

    const int ITEMS_PER_BLOCK = xlib::smem_per_block<vid_t, BLOCK_SIZE>();
    const auto& lambda = [&](int pos, degree_t offset) {
                            const auto& vertex = hornet.vertex(d_input[pos]);
                            const auto&   edge = vertex.edge(offset);
                            op(vertex, edge);
                        };
    xlib::binarySearchLB<BLOCK_SIZE, ITEMS_PER_BLOCK / BLOCK_SIZE>
        (d_work, work_size, xlib::dyn_smem, lambda);
}

template<unsigned BLOCK_SIZE,
         typename HornetDevice, typename Operator>
__global__
void binarySearchKernel(HornetDevice              hornet,
                        const int*   __restrict__ d_work,
                        int                       work_size,
                        Operator                  op) {

    const auto& lambda = [&](int pos, degree_t offset) {
                            const auto& vertex = hornet.vertex(pos);
                            const auto&   edge = vertex.edge(offset);
                            op(vertex, edge);
                        };
    xlib::binarySearchLB<BLOCK_SIZE>(d_work, work_size, xlib::dyn_smem, lambda);
}

} // namespace kernel
} // namespace load_balancing
} // namespace hornets_nest
