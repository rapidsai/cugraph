/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <thrust/device_vector.h>

#include "include/nvgraph_error.hxx"
#include "include/nvgraph_vector_kernels.hxx"
#include "include/pagerank_kernels.hxx"

namespace nvgraph
{

template <typename ValueType_>
__global__ void update_dn_kernel(int num_vertices, ValueType_* aa, ValueType_ beta)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int r = tidx; r < num_vertices; r += blockDim.x * gridDim.x)
    {
        // NOTE 1 : a = alpha*a + (1-alpha)e
        if (aa[r] == 0.0)
            aa[r] = beta; // NOTE 2 : alpha*0 + (1-alpha)*1 = (1-alpha)
    }
}

template <typename ValueType_>
void update_dangling_nodes(int num_vertices, ValueType_* dangling_nodes, ValueType_ damping_factor, cudaStream_t stream)
{
	
	int num_threads = 256;
    int max_grid_size = 4096;
    int num_blocks = std::min(max_grid_size, (num_vertices/num_threads)+1);
    ValueType_ beta = 1.0-damping_factor;
    update_dn_kernel<<<num_blocks, num_threads, 0, stream>>>(num_vertices, dangling_nodes,beta);
    cudaCheckError();
}

//Explicit

template void update_dangling_nodes<double> (int num_vertices, double* dangling_nodes, double damping_factor, cudaStream_t stream);
template void update_dangling_nodes<float> (int num_vertices, float* dangling_nodes, float damping_factor, cudaStream_t stream);
} // end namespace nvgraph

