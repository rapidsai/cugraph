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
namespace hornets_nest {
namespace load_balancing {
namespace kernel {

/**
 * @brief
 */
template<unsigned VW_SIZE, typename HornetDevice, typename Operator>
__global__
void vertexBasedVertexPairsKernel(HornetDevice              hornet,
                                  const vid_t* __restrict__ d_input,
                                  int                       num_vertices,
                                  Operator                  op) {
    int   group_id = (blockIdx.x * blockDim.x + threadIdx.x) / VW_SIZE;
    int     stride = (gridDim.x * blockDim.x) / VW_SIZE;
    int group_lane = threadIdx.x % VW_SIZE;

    for (auto i = group_id; i < num_vertices; i += stride) {
        __syncthreads();
        const auto& src = hornet.vertex(d_input[i]);
        for (auto j = group_lane; j < src.degree(); j += VW_SIZE) {
            const auto& edge = src.edge(j);
            const auto& dst = hornet.vertex(edge.dst_id());
            op(src, dst);
        }
        __syncthreads();
    }
}

/**
 * @brief
 */
template<unsigned VW_SIZE, typename HornetDevice, typename Operator>
__global__
void vertexBasedKernel(HornetDevice              hornet,
                       const vid_t* __restrict__ d_input,
                       int                       num_vertices,
                       Operator                  op) {
    int   group_id = (blockIdx.x * blockDim.x + threadIdx.x) / VW_SIZE;
    int     stride = (gridDim.x * blockDim.x) / VW_SIZE;
    int group_lane = threadIdx.x % VW_SIZE;

    for (auto i = group_id; i < num_vertices; i += stride) {
        __syncthreads();
        const auto& vertex = hornet.vertex(d_input[i]);
        for (auto j = group_lane; j < vertex.degree(); j += VW_SIZE) {
            const auto& edge = vertex.edge(j);
            op(vertex, edge);
        }
        __syncthreads();
    }
}

/**
 * @brief
 */
template<unsigned VW_SIZE, typename HornetDevice, typename Operator>
__global__
void vertexBasedKernel(HornetDevice hornet, Operator op) {
    int   group_id = (blockIdx.x * blockDim.x + threadIdx.x) / VW_SIZE;
    int     stride = (gridDim.x * blockDim.x) / VW_SIZE;
    int group_lane = threadIdx.x % VW_SIZE;

    for (auto i = group_id; i < hornet.nV(); i += stride) {
        const auto& vertex = hornet.vertex(i);
        for (auto j = group_lane; j < vertex.degree();  j += VW_SIZE) {
            const auto& edge = vertex.edge(j);
            op(vertex, edge);
        }
    }
}

/**
 * @brief
 */
template<unsigned VW_SIZE, typename HornetDevice, typename Operator>
__global__
void vertexBasedVertexPairsKernel(HornetDevice hornet, Operator op) {
    int   group_id = (blockIdx.x * blockDim.x + threadIdx.x) / VW_SIZE;
    int     stride = (gridDim.x * blockDim.x) / VW_SIZE;
    int group_lane = threadIdx.x % VW_SIZE;

    for (auto i = group_id; i < hornet.nV(); i += stride) {
        const auto& src = hornet.vertex(i);

        for (auto j = group_lane; j < src.degree();  j += VW_SIZE) {
            const auto& edge = src.edge(j);
            const auto& dst = hornet.vertex(edge.dst_id());
            op(src, dst);
        }
    }
}

} // kernel
} // namespace load_balancing
} // namespace hornets_nest
