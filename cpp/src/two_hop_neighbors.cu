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

#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

template<typename IndexType>
gdf_error gdf_get_two_hop_neighbors_impl(IndexType num_verts,
																					IndexType* offsets,
																					IndexType* indices,
																					IndexType** first,
																					IndexType** second,
																					IndexType& outputSize) {
	// Get the number of edges from the adjacency representation
	IndexType num_edges;
	cudaMemcpy(&num_edges, &offsets[num_verts], sizeof(IndexType), cudaMemcpyDefault);

	// Allocate memory for temporary stuff
	IndexType *exsum_degree;
	void *cub_storage;
	IndexType *first_pair;
	IndexType *second_pair;
	IndexType *block_bucket_offsets;

	ALLOC_MANAGED_TRY(&exsum_degree, sizeof(IndexType) * (num_edges + 1), nullptr);

	// Find the degree of the out vertex of each edge
	degree_iterator<IndexType> deg_it(offsets);
	deref_functor<degree_iterator<IndexType>, IndexType> deref(deg_it);
	rmm_temp_allocator allocator(nullptr);
	thrust::fill(exsum_degree, exsum_degree + 1, 0);
	thrust::transform(thrust::cuda::par(allocator).on(nullptr),
										indices,
										indices + num_edges,
										exsum_degree + 1,
										deref);

	// Take the inclusive sum of the degrees
	thrust::inclusive_scan(thrust::cuda::par(allocator).on(nullptr),
	                       exsum_degree + 1,
	                       exsum_degree + num_edges + 1,
	                       exsum_degree + 1);

	// Copy out the last value to get the size of scattered output
	IndexType output_size;
	cudaMemcpy(&output_size, &exsum_degree[num_edges], sizeof(IndexType), cudaMemcpyDefault);

	// Allocate memory for the scattered output
	ALLOC_MANAGED_TRY(&second_pair, sizeof(IndexType) * output_size, nullptr);
	ALLOC_MANAGED_TRY(&first_pair, sizeof(IndexType) * output_size, nullptr);

	// Figure out number of blocks and allocate memory for block bucket offsets
	IndexType num_blocks = (output_size + TWO_HOP_BLOCK_SIZE - 1) / TWO_HOP_BLOCK_SIZE;
	ALLOC_MANAGED_TRY(&block_bucket_offsets, sizeof(IndexType) * (num_blocks + 1), nullptr);

	// Compute the block bucket offsets
	dim3 grid, block;
	block.x = 512;
	grid.x = min((IndexType)MAXBLOCKS, num_blocks);
	compute_bucket_offsets_kernel<<<grid, block, 0, nullptr>>>(exsum_degree, block_bucket_offsets, num_edges, output_size);

	// Scatter the expanded edge lists into temp space


	// Remove duplicates and self pairings

	// Get things ready to return
	return GDF_SUCCESS;
}

gdf_error gdf_get_two_hop_neighbors(gdf_graph* graph, gdf_column* first, gdf_column* second) {
	GDF_REQUIRE(graph != nullptr, GDF_INVALID_API_CALL);
	GDF_REQUIRE(first != nullptr, GDF_INVALID_API_CALL);
	GDF_REQUIRE(second != nullptr, GDF_INVALID_API_CALL);
	GDF_TRY(gdf_add_adj_list(graph));

	size_t num_verts = graph->adjList->offsets->size - 1;
	switch (graph->adjList->offsets->dtype) {
		case GDF_INT32: {
			int32_t* first_ptr;
			int32_t* second_ptr;
			int32_t outputSize;
			gdf_get_two_hop_neighbors_impl((int32_t) num_verts,
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
			gdf_get_two_hop_neighbors_impl((int64_t) num_verts,
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
			return GDF_UNSUPPORTED_DTYPE;
	}

	return GDF_SUCCESS;
}
