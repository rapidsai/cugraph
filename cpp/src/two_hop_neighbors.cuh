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
 * @file two_hop_neighbors.cuh
 * ---------------------------------------------------------------------------**/

#include <cugraph.h>
#include <thrust/tuple.h>

#define MAXBLOCKS 65535
#define TWO_HOP_BLOCK_SIZE 512

template<typename IndexType>
struct degree_iterator {
	IndexType* offsets;
	degree_iterator(IndexType* _offsets) :
			offsets(_offsets) {
	}

	__host__    __device__
	IndexType operator[](IndexType place) {
		return offsets[place + 1] - offsets[place];
	}
};

template<typename It, typename IndexType>
struct deref_functor {
	It iterator;
	deref_functor(It it) :
			iterator(it) {
	}

	__host__    __device__
	IndexType operator()(IndexType in) {
		return iterator[in];
	}
};

template<typename IndexType>
struct self_loop_flagger {
	__host__ __device__
	bool operator()(const thrust::tuple<IndexType, IndexType> pair) {
		if (thrust::get<0>(pair) == thrust::get<1>(pair))
			return false;
		return true;
	}
};

template<typename IndexType>
__device__ IndexType binsearch_maxle(const IndexType *vec,
																			const IndexType val,
																			IndexType low,
																			IndexType high) {
	while (true) {
		if (low == high)
			return low; //we know it exists
		if ((low + 1) == high)
			return (vec[high] <= val) ? high : low;

		IndexType mid = low + (high - low) / 2;

		if (vec[mid] > val)
			high = mid - 1;
		else
			low = mid;
	}
}

template<typename IndexType>
__global__ void compute_bucket_offsets_kernel(const IndexType *frontier_degrees_exclusive_sum,
																							IndexType *bucket_offsets,
																							const IndexType frontier_size,
																							IndexType total_degree) {
	IndexType end = ((total_degree - 1 + TWO_HOP_BLOCK_SIZE) / TWO_HOP_BLOCK_SIZE);

	for (IndexType bid = blockIdx.x * blockDim.x + threadIdx.x;
			bid <= end;
			bid += gridDim.x * blockDim.x) {

		IndexType eid = min(bid * TWO_HOP_BLOCK_SIZE, total_degree - 1);

		bucket_offsets[bid] = binsearch_maxle(frontier_degrees_exclusive_sum,
																					eid,
																					(IndexType) 0,
																					frontier_size - 1);

	}
}

template<typename IndexType>
__global__ void scatter_expand_kernel(const IndexType *exsum_degree,
																			const IndexType *indices,
																			const IndexType *offsets,
																			const IndexType *bucket_offsets,
																			IndexType num_verts,
																			IndexType max_item,
																			IndexType max_block,
																			IndexType *output_first,
																			IndexType *output_second) {
	__shared__ IndexType blockRange[2];
	for (IndexType bid = blockIdx.x; bid < max_block; bid += gridDim.x) {
		// Copy the start and end of the buckets range into shared memory
		if (threadIdx.x == 0) {
			blockRange[0] = bucket_offsets[bid];
			blockRange[1] = bucket_offsets[bid + 1];
		}
		__syncthreads();

		// Get the global thread id (for this virtual block)
		IndexType tid = bid * blockDim.x + threadIdx.x;
		if (tid < max_item) {
			IndexType sourceIdx = binsearch_maxle(exsum_degree, tid, blockRange[0], blockRange[1]);
			IndexType sourceId = indices[sourceIdx];
			IndexType itemRank = tid - exsum_degree[sourceIdx];
			output_second[tid] = indices[offsets[sourceId] + itemRank];
			IndexType baseSourceId = binsearch_maxle(offsets, sourceIdx, (IndexType)0, num_verts);
			output_first[tid] = baseSourceId;
		}
	}
}
