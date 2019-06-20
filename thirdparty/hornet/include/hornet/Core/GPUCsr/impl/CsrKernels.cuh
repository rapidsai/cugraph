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
//#include "Device/DeviceProperties.cuh"        //xlib::SMemPerBlock
//#include "Device/BinarySearchLB.cuh"    //xlib::BinarySearchLB
#include "Device/Util/PrintExt.cuh"          //cu::Cout

namespace hornets_nest {
namespace gpu {

template<typename HornetDevice>
__global__
void printCsrKernel(HornetDevice hornet) {
    xlib::gpu::Cout cout;
    for (vid_t i = 0; i < hornet.nV(); i++) {
        auto vertex = hornet.vertex(i);
        cout << i << " [" << vertex.degree() << "]: ";

        for (degree_t j = 0; j < vertex.degree(); j++) {
            auto edge = vertex.edge(j);
            //auto weight = edge.weight();
            //cout << vertex.neighbor_id(j) << " ";
            cout << edge.dst_id() << " (" << edge.weight() << ")\t";
        }
        cout << "\n";
    }
}
/*
template<unsigned BLOCK_SIZE>
__global__
void CSRtoCOOKernel(const eoff_t* __restrict__ csr_offsets,
                    vid_t                      nV,
                    vid_t*        __restrict__ coo_src) {
    __shared__ int smem[xlib::SMemPerBlock<BLOCK_SIZE, int>::value];

    const auto& lambda = [&](int pos, eoff_t offset) {
                            eoff_t   index = csr_offsets[pos] + offset;
                            coo_src[index] = pos;
                        };
    xlib::binarySearchLB<BLOCK_SIZE>(csr_offsets, nV, smem, lambda);
}

template<typename HornetDevice>
__global__ void checkSortedKernel(HornetDevice hornet) {
    int    idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (vid_t i = idx; i < hornet.nV(); i += stride) {
        auto vertex = hornet.vertex(i);
        auto    ptr = vertex.neighbor_ptr();

        for (degree_t j = 0; j < vertex.degree() - 1; j++) {
            if (ptr[j] > ptr[j + 1]) {
                printf("Edge %d\t-> %d\t(d: %d)\t(value %d) not sorted \n",
                        i, j, vertex.degree(), ptr[j]);
            }
            else if (ptr[j] == ptr[j + 1]) {
                printf("Edge %d\t-> %d\t(d: %d)\t(value %d) duplicated\n",
                        i, j, vertex.degree(), ptr[j]);
            }
        }
    }
}

template<typename HornetDevice>
__global__
void buildDegreeKernel(HornetDevice           hornet,
                       degree_t* __restrict__ d_degrees) {
    int    idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < hornet.nV(); i += stride)
        d_degrees[i] = hornet.vertex(i).degree();
}*/

} // namespace gpu
} // namespace hornets_nest
