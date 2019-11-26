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
/** ---------------------------------------------------------------------------*
 * @brief Functions for computing the two hop neighbor pairs of a graph
 *
 * @file two_hop_neighbors.cu
 * ---------------------------------------------------------------------------**/

#include "two_hop_neighbors.cuh"
#include "utilities/error_utils.h"
#include <rmm_utils.h>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

namespace cugraph{
namespace detail{
template<typename IndexType>
void get_two_hop_neighbors_impl(IndexType num_verts,
                                         IndexType* offsets,
                                         IndexType* indices,
                                         IndexType** first,
                                         IndexType** second,
                                         IndexType& outputSize) {
    // Get the number of edges from the adjacency representation
    IndexType num_edges;
    cudaMemcpy(&num_edges, &offsets[num_verts], sizeof(IndexType), cudaMemcpyDefault);

    cudaStream_t stream {nullptr};

    // Allocate memory for temporary stuff
    IndexType *exsum_degree = nullptr;
    IndexType *first_pair = nullptr;
    IndexType *second_pair = nullptr;
    IndexType *block_bucket_offsets = nullptr;

    ALLOC_TRY(&exsum_degree, sizeof(IndexType) * (num_edges + 1), stream);

    // Find the degree of the out vertex of each edge
    degree_iterator<IndexType> deg_it(offsets);
    deref_functor<degree_iterator<IndexType>, IndexType> deref(deg_it);
    thrust::fill(rmm::exec_policy(stream)->on(stream), exsum_degree, exsum_degree + 1, 0);
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                                        indices,
                                        indices + num_edges,
                                        exsum_degree + 1,
                                        deref);

    // Take the inclusive sum of the degrees
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                                                    exsum_degree + 1,
                                                    exsum_degree + num_edges + 1,
                                                    exsum_degree + 1);

    // Copy out the last value to get the size of scattered output
    IndexType output_size;
    cudaMemcpy(&output_size, &exsum_degree[num_edges], sizeof(IndexType), cudaMemcpyDefault);

    // Allocate memory for the scattered output
    ALLOC_TRY(&second_pair, sizeof(IndexType) * output_size, stream);
    ALLOC_TRY(&first_pair, sizeof(IndexType) * output_size, stream);

    // Figure out number of blocks and allocate memory for block bucket offsets
    IndexType num_blocks = (output_size + TWO_HOP_BLOCK_SIZE - 1) / TWO_HOP_BLOCK_SIZE;
    ALLOC_TRY(&block_bucket_offsets, sizeof(IndexType) * (num_blocks + 1), stream);

    // Compute the block bucket offsets
    dim3 grid, block;
    block.x = 512;
    grid.x = min((IndexType) MAXBLOCKS, (num_blocks / 512) + 1);
    compute_bucket_offsets_kernel<<<grid, block, 0, nullptr>>>(exsum_degree,
                                                               block_bucket_offsets,
                                                               num_edges,
                                                               output_size);
    cudaMemcpy(&block_bucket_offsets[num_blocks], &num_edges, sizeof(IndexType), cudaMemcpyDefault);

    // Scatter the expanded edge lists into temp space
    grid.x = min((IndexType) MAXBLOCKS, num_blocks);
    scatter_expand_kernel<<<grid, block, 0, nullptr>>>(exsum_degree,
                                                       indices,
                                                       offsets,
                                                       block_bucket_offsets,
                                                       num_verts,
                                                       output_size,
                                                       num_blocks,
                                                       first_pair,
                                                       second_pair);

    // Remove duplicates and self pairings
    auto tuple_start = thrust::make_zip_iterator(thrust::make_tuple(first_pair, second_pair));
    auto tuple_end = tuple_start + output_size;
    thrust::sort(rmm::exec_policy(stream)->on(stream), tuple_start, tuple_end);
    tuple_end = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                                                            tuple_start,
                                                            tuple_end,
                                                            tuple_start,
                                                            self_loop_flagger<IndexType>());
    tuple_end = thrust::unique(rmm::exec_policy(stream)->on(stream), tuple_start, tuple_end);

    // Get things ready to return
    outputSize = tuple_end - tuple_start;
    ALLOC_TRY(first, sizeof(IndexType) * outputSize, nullptr);
    ALLOC_TRY(second, sizeof(IndexType) * outputSize, nullptr);
    cudaMemcpy(*first, first_pair, sizeof(IndexType) * outputSize, cudaMemcpyDefault);
    cudaMemcpy(*second, second_pair, sizeof(IndexType) * outputSize, cudaMemcpyDefault);

    // Free up temporary stuff
    ALLOC_FREE_TRY(exsum_degree, nullptr);
    ALLOC_FREE_TRY(first_pair, nullptr);
    ALLOC_FREE_TRY(second_pair, nullptr);
    ALLOC_FREE_TRY(block_bucket_offsets, nullptr);

    
}

} //namespace

void get_two_hop_neighbors(Graph* graph, gdf_column* first, gdf_column* second) {
    CUGRAPH_EXPECTS(graph != nullptr, "Invalid API parameter");
    CUGRAPH_EXPECTS(first != nullptr, "Invalid API parameter");
    CUGRAPH_EXPECTS(second != nullptr, "Invalid API parameter");

    size_t num_verts = graph->adjList->offsets->size - 1;
    switch (graph->adjList->offsets->dtype) {
        case GDF_INT32: {
            int32_t* first_ptr;
            int32_t* second_ptr;
            int32_t outputSize;
            detail::get_two_hop_neighbors_impl((int32_t) num_verts,
                                           (int32_t*) graph->adjList->offsets->data,
                                           (int32_t*) graph->adjList->indices->data,
                                           &first_ptr,
                                           &second_ptr,
                                           outputSize);
            first->data = first_ptr;
            first->dtype = GDF_INT32;
            first->size = outputSize;
            second->data = second_ptr;
            second->dtype = GDF_INT32;
            second->size = outputSize;
            break;
        }
        case GDF_INT64: {
            int64_t* first_ptr;
            int64_t* second_ptr;
            int64_t outputSize;
            detail::get_two_hop_neighbors_impl((int64_t) num_verts,
                                           (int64_t*) graph->adjList->offsets->data,
                                           (int64_t*) graph->adjList->indices->data,
                                           &first_ptr,
                                           &second_ptr,
                                           outputSize);
            first->data = first_ptr;
            first->dtype = GDF_INT64;
            first->size = outputSize;
            second->data = second_ptr;
            second->dtype = GDF_INT64;
            second->size = outputSize;
            break;
        }
        default:
            CUGRAPH_FAIL("Unsupported data type");
    }

    
}
} //namespace
