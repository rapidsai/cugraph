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

#include "utilities/error_utils.h"

template <typename T>
struct CSR_Result {
	std::int64_t size;
	std::int64_t nnz;
	T* rowOffsets;
	T* colIndices;

	CSR_Result() : size(0), nnz(0), rowOffsets(nullptr), colIndices(nullptr){}
    /*
	~CSR_Result() {
		if (rowOffsets)
			cudaFree(rowOffsets);
		if (colIndices)
			cudaFree(colIndices);
		//return GDF_SUCCESS;
	}
    */
};

template <typename T, typename W>
struct CSR_Result_Weighted {
	std::int64_t size;
	std::int64_t nnz;
	T* rowOffsets;
	T* colIndices;
	W* edgeWeights;

	CSR_Result_Weighted() : size(0), nnz(0), rowOffsets(nullptr), colIndices(nullptr), edgeWeights(nullptr){}
    /*
	~CSR_Result_Weighted() {
		if (rowOffsets)
			cudaFree(rowOffsets);
		if (colIndices)
			cudaFree(colIndices);
		if (edgeWeights)
			cudaFree(edgeWeights);
		//return GDF_SUCCESS;
	}
    */
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
gdf_error ConvertCOOtoCSR(T* sources, T* destinations, int64_t nnz, CSR_Result<T>& result) {
    // Sort source and destination columns by source
    //   Allocate local memory for operating on
    T* srcs, *dests;
    CUDA_TRY(cudaMallocManaged(&srcs, sizeof(T) * nnz));
    CUDA_TRY(cudaMallocManaged(&dests, sizeof(T) * nnz));

    CUDA_TRY(cudaMemcpy(srcs, sources, sizeof(T) * nnz, cudaMemcpyDefault));
    CUDA_TRY(cudaMemcpy(dests, destinations, sizeof(T) * nnz, cudaMemcpyDefault));

    //   Call CUB SortPairs to sort using srcs as the keys
    void* tmpStorage = nullptr;
    size_t tmpBytes = 0;
    thrust::stable_sort_by_key(thrust::device, dests, dests + nnz, srcs);
    thrust::stable_sort_by_key(thrust::device, srcs, srcs + nnz, dests);

	// Find max id (since this may be in the dests array but not the srcs array we need to check both)
    T maxId = -1;
    //   Max from srcs after sorting is just the last element
    CUDA_TRY(cudaMemcpy(&maxId, &(srcs[nnz-1]), sizeof(T), cudaMemcpyDefault));
    auto maxId_it = thrust::max_element(thrust::device, dests, dests + nnz);
    T maxId2;
    CUDA_TRY(cudaMemcpy(&maxId2, maxId_it, sizeof(T), cudaMemcpyDefault));
    maxId = maxId > maxId2 ? maxId : maxId2;
    result.size = maxId + 1;

    // Allocate offsets array
    CUDA_TRY(cudaMallocManaged(&result.rowOffsets, (maxId + 2) * sizeof(T)));

    // Set all values in offsets array to zeros
    CUDA_TRY(cudaMemset(result.rowOffsets, 0,(maxId + 2) * sizeof(int)));

    // Allocate temporary arrays same size as sources array, and single value to get run counts
    T* unique, *counts, *runCount;
    CUDA_TRY(cudaMallocManaged(&unique, (maxId + 1) * sizeof(T)));
    CUDA_TRY(cudaMallocManaged(&counts, (maxId + 1) * sizeof(T)));
    CUDA_TRY(cudaMallocManaged(&runCount, sizeof(T)));

    // Use CUB run length encoding to get unique values and run lengths
    tmpStorage = nullptr;
    cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz);
    CUDA_TRY(cudaMallocManaged(&tmpStorage, tmpBytes));
    cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz);
    CUDA_TRY(cudaFree(tmpStorage));

    // Set offsets to run sizes for each index
    T runCount_h;
    CUDA_TRY(cudaMemcpy(&runCount_h, runCount, sizeof(T), cudaMemcpyDefault));
    int threadsPerBlock = 1024;
    int numBlocks = (runCount_h + threadsPerBlock - 1) / threadsPerBlock;
    offsetsKernel<<<numBlocks, threadsPerBlock>>>(runCount_h, unique, counts, result.rowOffsets);

    // Scan offsets to get final offsets
    thrust::exclusive_scan(thrust::device, result.rowOffsets, result.rowOffsets + maxId + 2, result.rowOffsets);

    // Clean up temporary allocations
    result.nnz = nnz;
    result.colIndices = dests;
    CUDA_TRY(cudaFree(srcs));
    CUDA_TRY(cudaFree(unique));
    CUDA_TRY(cudaFree(counts));
    CUDA_TRY(cudaFree(runCount));
    return GDF_SUCCESS;
}

// Method for constructing CSR from COO
template <typename T, typename W>
gdf_error ConvertCOOtoCSR_weighted(T* sources, T* destinations, W* edgeWeights, int64_t nnz, CSR_Result_Weighted<T, W>& result) {
    // Sort source and destination columns by source
    //   Allocate local memory for operating on
	T* srcs, *dests;
	W* weights;
    CUDA_TRY(cudaMallocManaged(&srcs, sizeof(T) * nnz));
    CUDA_TRY(cudaMallocManaged(&dests, sizeof(T) * nnz));
    CUDA_TRY(cudaMallocManaged(&weights, sizeof(W) * nnz));
    CUDA_TRY(cudaMemcpy(srcs, sources, sizeof(T) * nnz, cudaMemcpyDefault));
    CUDA_TRY(cudaMemcpy(dests, destinations, sizeof(T) * nnz, cudaMemcpyDefault));
    CUDA_TRY(cudaMemcpy(weights, edgeWeights, sizeof(W) * nnz, cudaMemcpyDefault));

    // Call Thrust::sort_by_key to sort the arrays with srcs as keys:
    thrust::stable_sort_by_key(thrust::device, dests, dests + nnz, thrust::make_zip_iterator(thrust::make_tuple(srcs, weights)));
    thrust::stable_sort_by_key(thrust::device, srcs, srcs + nnz, thrust::make_zip_iterator(thrust::make_tuple(dests, weights)));

	// Find max id (since this may be in the dests array but not the srcs array we need to check both)
    T maxId = -1;
    //   Max from srcs after sorting is just the last element
    CUDA_TRY(cudaMemcpy(&maxId, &(srcs[nnz-1]), sizeof(T), cudaMemcpyDefault));
    auto maxId_it = thrust::max_element(thrust::device, dests, dests + nnz);
    //   Max from dests requires a scan to find
    T maxId2;
    CUDA_TRY(cudaMemcpy(&maxId2, maxId_it, sizeof(T), cudaMemcpyDefault));
    maxId = maxId > maxId2 ? maxId : maxId2;
    result.size = maxId + 1;

    // Allocate offsets array
    CUDA_TRY(cudaMallocManaged(&result.rowOffsets, (maxId + 2) * sizeof(T)));

    // Set all values in offsets array to zeros
    // /CUDA_TRY(
    //    cudaMemset(result.rowOffsets, 0, (maxId + 2) * sizeof(T));

    CUDA_TRY(cudaMemset(result.rowOffsets, 0,(maxId + 2) * sizeof(int)));

    // Allocate temporary arrays same size as sources array, and single value to get run counts
    T* unique, *counts, *runCount;
    CUDA_TRY(cudaMallocManaged(&unique, (maxId + 1) * sizeof(T)));
    CUDA_TRY(cudaMallocManaged(&counts, (maxId + 1) * sizeof(T)));
    CUDA_TRY(cudaMallocManaged(&runCount, sizeof(T)));

    // Use CUB run length encoding to get unique values and run lengths
    void *tmpStorage = nullptr;
    size_t tmpBytes = 0;
    cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz);
    CUDA_TRY(cudaMallocManaged(&tmpStorage, tmpBytes));
    cub::DeviceRunLengthEncode::Encode(tmpStorage, tmpBytes, srcs, unique, counts, runCount, nnz);
    CUDA_TRY(cudaFree(tmpStorage));

    // Set offsets to run sizes for each index
    T runCount_h;
    CUDA_TRY(cudaMemcpy(&runCount_h, runCount, sizeof(T), cudaMemcpyDefault));
    int threadsPerBlock = 1024;
    int numBlocks = (runCount_h + threadsPerBlock - 1) / threadsPerBlock;
    offsetsKernel<<<numBlocks, threadsPerBlock>>>(runCount_h, unique, counts, result.rowOffsets);

    // Scan offsets to get final offsets
    thrust::exclusive_scan(thrust::device, result.rowOffsets, result.rowOffsets + maxId + 2, result.rowOffsets);

    // Clean up temporary allocations
    result.nnz = nnz;
    result.colIndices = dests;
    result.edgeWeights = weights;
    CUDA_TRY(cudaFree(srcs));
    CUDA_TRY(cudaFree(unique));
    CUDA_TRY(cudaFree(counts));
    CUDA_TRY(cudaFree(runCount));
    return GDF_SUCCESS;
}
