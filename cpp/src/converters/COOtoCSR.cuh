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
/*
 * COOtoCSR_kernels.cuh
 *
 *  Created on: Mar 8, 2018
 *      Author: jwyles
 */

#pragma once

#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>

#include <rmm_utils.h>

template <typename T>
struct CSR_Result {
    std::int64_t size;
    std::int64_t nnz;
    T* rowOffsets;
    T* colIndices;

    CSR_Result() : size(0), nnz(0), rowOffsets(nullptr), colIndices(nullptr){}

};

template <typename T, typename W>
struct CSR_Result_Weighted {
    std::int64_t size;
    std::int64_t nnz;
    T* rowOffsets;
    T* colIndices;
    W* edgeWeights;

    CSR_Result_Weighted() : size(0), nnz(0), rowOffsets(nullptr), colIndices(nullptr), edgeWeights(nullptr){}

};

// Define kernel for copying run length encoded values into offset slots.
template <typename T>
__global__ void offsetsKernel(T runCounts, T* unique, T* counts, T* offsets) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < runCounts)
        offsets[unique[tid]] = counts[tid];
}

// Method for constructing CSR from COO
template <typename T>
void ConvertCOOtoCSR(T* sources, T* destinations, int64_t nnz, CSR_Result<T>& result) {
    // Sort source and destination columns by source
    //   Allocate local memory for operating on
    T* srcs{nullptr}, *dests{nullptr};

    cudaStream_t stream {nullptr};
    
    ALLOC_TRY((void**)&srcs, sizeof(T) * nnz, stream);
    ALLOC_TRY((void**)&dests, sizeof(T) * nnz, stream);
    
    CUDA_TRY(cudaMemcpy(srcs, sources, sizeof(T) * nnz, cudaMemcpyDefault));
    CUDA_TRY(cudaMemcpy(dests, destinations, sizeof(T) * nnz, cudaMemcpyDefault));

    //   Call CUB SortPairs to sort using srcs as the keys
    void* tmpStorage = nullptr;
    size_t tmpBytes = 0;

    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream), dests, dests + nnz, srcs);
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream), srcs, srcs + nnz, dests);

    // Find max id (since this may be in the dests array but not the srcs array we need to check both)
    T maxId = -1;
    //   Max from srcs after sorting is just the last element
    CUDA_TRY(cudaMemcpy(&maxId, &(srcs[nnz-1]), sizeof(T), cudaMemcpyDefault));
    auto maxId_it = thrust::max_element(rmm::exec_policy(stream)->on(stream), dests, dests + nnz);
    T maxId2;
    CUDA_TRY(cudaMemcpy(&maxId2, maxId_it, sizeof(T), cudaMemcpyDefault));
    maxId = maxId > maxId2 ? maxId : maxId2;
    result.size = maxId + 1;
    // Sending a warning rather than an error here as this may be intended and suported.
    if (result.size > nnz ) {
        std::cerr<< "WARNING: there are more vertices than edges in the graph ";
        std::cerr<< ": V=" << result.size <<", E="<<nnz <<". ";
        std::cerr<< "Sometime this is not intended and may cause performace and stability issues. ";
        std::cerr<< "Vertex identifieres must be in the range [0, V) where V is the number of vertices. ";
        std::cerr<< "Please refer to cuGraph's renumbering feature ";
        std::cerr<< "if some identifiers are larger than your actual number of vertices." << std::endl;
    }
    // Allocate offsets array
    ALLOC_TRY((void**)&result.rowOffsets, (maxId + 2) * sizeof(T), stream);

    // Set all values in offsets array to zeros
    CUDA_TRY(cudaMemset(result.rowOffsets, 0,(maxId + 2) * sizeof(int)));

    // Allocate temporary arrays same size as sources array, and single value to get run counts
    T* unique{nullptr}, *counts{nullptr}, *runCount{nullptr};
    ALLOC_TRY((void**)&unique, (maxId + 1) * sizeof(T), stream);
    ALLOC_TRY((void**)&counts, (maxId + 1) * sizeof(T), stream);
    ALLOC_TRY((void**)&runCount, sizeof(T), stream);

    // Use CUB run length encoding to get unique values and run lengths
    tmpStorage = nullptr;
    CUDA_TRY(cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz));
    ALLOC_TRY((void**)&tmpStorage, tmpBytes, stream);
    CUDA_TRY(cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz));
    ALLOC_FREE_TRY(tmpStorage, stream);

    // Set offsets to run sizes for each index
    T runCount_h;
    CUDA_TRY(cudaMemcpy(&runCount_h, runCount, sizeof(T), cudaMemcpyDefault));
    int threadsPerBlock = 1024;
    int numBlocks = (runCount_h + threadsPerBlock - 1) / threadsPerBlock;
    offsetsKernel<<<numBlocks, threadsPerBlock>>>(runCount_h, unique, counts, result.rowOffsets);

    // Scan offsets to get final offsets
    thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream), result.rowOffsets, result.rowOffsets + maxId + 2, result.rowOffsets);

    // Clean up temporary allocations
    result.nnz = nnz;
    result.colIndices = dests;
    ALLOC_FREE_TRY(srcs, stream);
    ALLOC_FREE_TRY(unique, stream);
    ALLOC_FREE_TRY(counts, stream);
    ALLOC_FREE_TRY(runCount, stream);
    
}

// Method for constructing CSR from COO
template <typename T, typename W>
void ConvertCOOtoCSR_weighted(T* sources, T* destinations, W* edgeWeights, int64_t nnz, CSR_Result_Weighted<T, W>& result) {
    // Sort source and destination columns by source
    //   Allocate local memory for operating on
    T* srcs{nullptr};
    T* dests{nullptr};
    W* weights{nullptr};
    
    cudaStream_t stream {nullptr};

    ALLOC_TRY((void**)&srcs, sizeof(T) * nnz, stream);
    ALLOC_TRY((void**)&dests, sizeof(T) * nnz, stream);
    ALLOC_TRY((void**)&weights, sizeof(W) * nnz, stream);
    CUDA_TRY(cudaMemcpy(srcs, sources, sizeof(T) * nnz, cudaMemcpyDefault));
    CUDA_TRY(cudaMemcpy(dests, destinations, sizeof(T) * nnz, cudaMemcpyDefault));
    CUDA_TRY(cudaMemcpy(weights, edgeWeights, sizeof(W) * nnz, cudaMemcpyDefault));

    // Call Thrust::sort_by_key to sort the arrays with srcs as keys:
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream), dests, dests + nnz, thrust::make_zip_iterator(thrust::make_tuple(srcs, weights)));
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream), srcs, srcs + nnz, thrust::make_zip_iterator(thrust::make_tuple(dests, weights)));

    // Find max id (since this may be in the dests array but not the srcs array we need to check both)
    T maxId = -1;
    //   Max from srcs after sorting is just the last element
    CUDA_TRY(cudaMemcpy(&maxId, &(srcs[nnz-1]), sizeof(T), cudaMemcpyDefault));
    auto maxId_it = thrust::max_element(rmm::exec_policy(stream)->on(stream), dests, dests + nnz);
    //   Max from dests requires a scan to find
    T maxId2;
    CUDA_TRY(cudaMemcpy(&maxId2, maxId_it, sizeof(T), cudaMemcpyDefault));
    maxId = maxId > maxId2 ? maxId : maxId2;
    result.size = maxId + 1;

    // Allocate offsets array
    ALLOC_TRY((void**)&result.rowOffsets, (maxId + 2) * sizeof(T), stream);

    // Set all values in offsets array to zeros
    // /CUDA_TRY(
    //    cudaMemset(result.rowOffsets, 0, (maxId + 2) * sizeof(T));

    CUDA_TRY(cudaMemset(result.rowOffsets, 0,(maxId + 2) * sizeof(int)));

    // Allocate temporary arrays same size as sources array, and single value to get run counts
    T* unique, *counts, *runCount;
    ALLOC_TRY((void**)&unique, (maxId + 1) * sizeof(T), stream);
    ALLOC_TRY((void**)&counts, (maxId + 1) * sizeof(T), stream);
    ALLOC_TRY((void**)&runCount, sizeof(T), stream);

    // Use CUB run length encoding to get unique values and run lengths
    void *tmpStorage = nullptr;
    size_t tmpBytes = 0;
    CUDA_TRY(cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz));
    ALLOC_TRY(&tmpStorage, tmpBytes, stream);
    CUDA_TRY(cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz));
    ALLOC_FREE_TRY(tmpStorage, stream);

    // Set offsets to run sizes for each index
    T runCount_h;
    CUDA_TRY(cudaMemcpy(&runCount_h, runCount, sizeof(T), cudaMemcpyDefault));
    int threadsPerBlock = 1024;
    int numBlocks = (runCount_h + threadsPerBlock - 1) / threadsPerBlock;
    offsetsKernel<<<numBlocks, threadsPerBlock>>>(runCount_h, unique, counts, result.rowOffsets);

    // Scan offsets to get final offsets
    thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream), result.rowOffsets, result.rowOffsets + maxId + 2, result.rowOffsets);

    // Clean up temporary allocations
    result.nnz = nnz;
    result.colIndices = dests;
    result.edgeWeights = weights;
    ALLOC_FREE_TRY(srcs, stream);
    ALLOC_FREE_TRY(unique, stream);
    ALLOC_FREE_TRY(counts, stream);
    ALLOC_FREE_TRY(runCount, stream);
    
}
