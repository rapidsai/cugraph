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
#include <Device/Util/Basic.cuh>
#include <Device/Util/DeviceProperties.cuh>
#include <Device/Primitives/WarpScan.cuh>

namespace hornets_nest {
namespace load_balancing {
namespace kernel {

template<unsigned BLOCK_SIZE, bool WARP_GATHER = false,
         bool BLOCK_GATHER = false, bool DEVICE_GATHER = false,
         typename HornetDevice, typename Operator>
__global__
void scanBasedKernel(HornetDevice              hornet,
                     const vid_t* __restrict__ d_input,
                     int                       num_vertices,
                     Operator                  op) {
    using  it_t = typename HornetDevice::edgeit_t;
    using EdgeT = typename HornetDevice::EdgeT;
    const unsigned      LOCAL_SIZE = xlib::smem_per_warp<it_t>();
    const unsigned ITEMS_PER_BLOCK = xlib::smem_per_block<it_t, BLOCK_SIZE>();
    __shared__ it_t smem[ITEMS_PER_BLOCK];

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int     lane_id = xlib::lane_id();
    it_t smem_local = smem + xlib::warp_id() * LOCAL_SIZE;
    int approx_size = xlib::upper_approx<xlib::WARP_SIZE>(num_vertices);

    for (auto i = id; i < approx_size; i += stride) {
        degree_t degree = 0;
        it_t start, end;
        if (i < num_vertices) {
            const auto& vertex = hornet.vertex(d_input[i]);
            degree = vertex.degree();
            start  = vertex.edge_begin();
            end    = vertex.edge_end();
        }
        else {
            degree = 0;
            end = std::numeric_limits<it_t>::lowest();
        }

        degree_t total;
        xlib::WarpExclusiveScan<>::add(degree, total);

        while (total > 0) {
            while ( start < end && degree < LOCAL_SIZE ) {
                smem_local[degree] = start++;
                degree++;
            }
            int limit = min(total, LOCAL_SIZE);

            for (auto index = lane_id; index < limit; index += xlib::WARP_SIZE){
                auto edge_offset = smem_local[index];
                const auto& edge = EdgeT(hornet, edge_offset);
                const auto& vertex = hornet.fake_vertex();
                op(vertex, edge);
            }
            total  -= LOCAL_SIZE;
            degree -= LOCAL_SIZE;
        }
    }
}

} // kernel
} // namespace load_balancing
} // namespace hornets_nest
