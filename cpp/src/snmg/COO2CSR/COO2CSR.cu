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

#include <omp.h>
#include <vector>
#include <sstream>
#include <string>
#include "utilities/graph_utils.cuh"
#include "snmg/utils.cuh"
#include "rmm_utils.h"
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <cub/device/device_run_length_encode.cuh>

namespace cugraph { 
namespace snmg {

template<typename idx_t, typename val_t>
class communicator {
public:
  idx_t* maxIds;
  idx_t* rowCounts;
  idx_t** rowPtrs;
  idx_t** colPtrs;
  unsigned long long int** reductionSpace;
  val_t** valPtrs;
  communicator(idx_t p) {
    maxIds = reinterpret_cast<idx_t*>(malloc(sizeof(idx_t) * p));
    rowCounts = reinterpret_cast<idx_t*>(malloc(sizeof(idx_t) * p * p));
    rowPtrs = reinterpret_cast<idx_t**>(malloc(sizeof(idx_t*) * p));
    colPtrs = reinterpret_cast<idx_t**>(malloc(sizeof(idx_t*) * p));
    valPtrs = reinterpret_cast<val_t**>(malloc(sizeof(val_t*) * p));
    reductionSpace = reinterpret_cast<unsigned long long int**>(malloc(sizeof(unsigned long long int*) * p));
  }
  ~communicator() {
    free(maxIds);
    free(rowCounts);
    free(rowPtrs);
    free(colPtrs);
    free(reductionSpace);
    free(valPtrs);
  }
};

void serializeMessage(cugraph::snmg::SNMGinfo& env, std::string message){
  auto i = env.get_thread_num();
  auto p = env.get_num_threads();
  for (int j = 0; j < p; j++){
    if (i == j)
      std::cout << "Thread " << i << ": " << message << "\n";
#pragma omp barrier
  }
}

template<typename idx_t, typename val_t>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
findStartRange(idx_t n, idx_t* result, val_t edgeCount, val_t* scanned) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
    if (scanned[i] < edgeCount && scanned[i + 1] >= edgeCount)
      *result = i + 1;
}

// Define kernel for copying run length encoded values into offset slots.
template <typename T>
__global__ void offsetsKernel(T runCounts, T* unique, T* counts, T* offsets) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < runCounts)
        offsets[unique[tid]] = counts[tid];
}

template <typename T>
__global__ void writeSingleValue(T* ptr, T val) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0)
    *ptr = val;
}

template<typename idx_t, typename val_t>
void snmg_coo2csr_impl(size_t* part_offsets,
                            bool free_input,
                            void** comm1,
                            gdf_column* cooRow,
                            gdf_column* cooCol,
                            gdf_column* cooVal,
                            gdf_column* csrOff,
                            gdf_column* csrInd,
                            gdf_column* csrVal) {
  cugraph::snmg::SNMGinfo env;
  auto i = env.get_thread_num();
  auto p = env.get_num_threads();

  // First thread allocates communicator object
  if (i == 0) {
    cugraph::snmg::communicator<idx_t, val_t>* comm = new cugraph::snmg::communicator<idx_t, val_t>(p);
    *comm1 = reinterpret_cast<void*>(comm);
  }
#pragma omp barrier

  cugraph::snmg::communicator<idx_t, val_t>* comm = reinterpret_cast<cugraph::snmg::communicator<idx_t, val_t>*>(*comm1);

  // Each thread scans its cooRow and cooCol for the greatest ID
  idx_t size = cooRow->size;
  idx_t* max_ptr = thrust::max_element(rmm::exec_policy(nullptr)->on(nullptr),
                                       reinterpret_cast<idx_t*>(cooRow->data),
                                       reinterpret_cast<idx_t*>(cooRow->data) + size);
  idx_t rowID;
  cudaMemcpy(&rowID, max_ptr, sizeof(idx_t), cudaMemcpyDefault);
  max_ptr = thrust::max_element(rmm::exec_policy(nullptr)->on(nullptr),
                                reinterpret_cast<idx_t*>(cooCol->data),
                                reinterpret_cast<idx_t*>(cooCol->data) + size);
  idx_t colID;
  cudaMemcpy(&colID, max_ptr, sizeof(idx_t), cudaMemcpyDefault);
  comm->maxIds[i] = max(rowID, colID);

#pragma omp barrier

  // First thread finds maximum global ID
  if (i == 0) {
    idx_t best_id = comm->maxIds[0];
    for (int j = 0; j < p; j++)
      best_id = max(best_id, comm->maxIds[j]);
    comm->maxIds[0] = best_id;
  }
#pragma omp barrier

  // Each thread allocates space for the source node counts
  idx_t maxId = comm->maxIds[0];
  idx_t offsetsSize = maxId + 2;
  unsigned long long int* sourceCounts;
  ALLOC_TRY(&sourceCounts, sizeof(unsigned long long int) * offsetsSize, nullptr);
  cudaMemset(sourceCounts, 0, sizeof(unsigned long long int) * offsetsSize);


  // Each thread computes the source node counts for its owned rows
  dim3 nthreads, nblocks;
  nthreads.x = min(size, static_cast<idx_t>(CUDA_MAX_KERNEL_THREADS));
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x = min(static_cast<idx_t>((size + nthreads.x - 1) / nthreads.x),
                  static_cast<idx_t>(env.get_num_sm() * 32));
  nblocks.y = 1;
  nblocks.z = 1;
  cugraph::detail::degree_coo<idx_t, unsigned long long int><<<nblocks, nthreads>>>(size,
                                                                            size,
                                                                            reinterpret_cast<idx_t*>(cooRow->data),
                                                                            sourceCounts);
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST();

  // Threads globally reduce their local source node counts to get the global ones
  unsigned long long int* sourceCountsTemp;
  ALLOC_TRY(&sourceCountsTemp, sizeof(unsigned long long int) * offsetsSize, nullptr);
  comm->reductionSpace[i] = sourceCountsTemp;
#pragma omp barrier

  cugraph::snmg::treeReduce<unsigned long long int, thrust::plus<unsigned long long int>>(env,
                                                                                    offsetsSize,
                                                                                    sourceCounts,
                                                                                    comm->reductionSpace);
  cugraph::snmg::treeBroadcast(env, offsetsSize, sourceCounts, comm->reductionSpace);

  // Each thread takes the exclusive scan of the global counts
  thrust::exclusive_scan(rmm::exec_policy(nullptr)->on(nullptr),
                         sourceCountsTemp,
                         sourceCountsTemp + offsetsSize,
                         sourceCountsTemp);
  ALLOC_FREE_TRY(sourceCounts, nullptr);
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST();

  // Each thread reads the global edgecount
  unsigned long long int globalEdgeCount;
  cudaMemcpy(&globalEdgeCount, sourceCountsTemp + maxId + 1, sizeof(unsigned long long int), cudaMemcpyDefault);
  CUDA_CHECK_LAST();

  // Each thread searches the global source node counts prefix sum to find the start of its vertex ID range
  idx_t myStartVertex = 0;
  if (i != 0) {
    unsigned long long int edgeCount = (globalEdgeCount / p) * i;
    idx_t* vertexRangeStart;
    ALLOC_TRY(&vertexRangeStart, sizeof(idx_t), nullptr);
    dim3 nthreads, nblocks;
    nthreads.x = min(offsetsSize, static_cast<idx_t>(CUDA_MAX_KERNEL_THREADS));
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((offsetsSize + nthreads.x - 1) / nthreads.x, static_cast<idx_t>(env.get_num_sm() * 32));
    nblocks.y = 1;
    nblocks.z = 1;
    cugraph::snmg::findStartRange<<<nblocks, nthreads>>>(maxId, vertexRangeStart, edgeCount, sourceCountsTemp);
    cudaDeviceSynchronize();
    cudaMemcpy(&myStartVertex, vertexRangeStart, sizeof(idx_t), cudaMemcpyDefault);
    part_offsets[i] = myStartVertex;
    ALLOC_FREE_TRY(vertexRangeStart, nullptr);
  }
  else {
    part_offsets[0] = 0;
    part_offsets[p] = maxId + 1;
  }
  CUDA_CHECK_LAST();
#pragma omp barrier

  // Each thread determines how many edges it will have in its partition
  idx_t myEndVertex = part_offsets[i + 1];
  unsigned long long int startEdge;
  unsigned long long int endEdge;
  cudaMemcpy(&startEdge, sourceCountsTemp + myStartVertex, sizeof(unsigned long long int), cudaMemcpyDefault);
  cudaMemcpy(&endEdge, sourceCountsTemp + myEndVertex, sizeof(unsigned long long int), cudaMemcpyDefault);
  ALLOC_FREE_TRY(sourceCountsTemp, nullptr);
  idx_t myEdgeCount = endEdge - startEdge;

  // Each thread sorts its cooRow, cooCol, and cooVal
  idx_t *cooRowTemp, *cooColTemp;
  val_t *cooValTemp;
  ALLOC_TRY(&cooRowTemp, sizeof(idx_t) * size, nullptr);
  ALLOC_TRY(&cooColTemp, sizeof(idx_t) * size, nullptr);
  cudaMemcpy(cooRowTemp, cooRow->data, sizeof(idx_t) * size, cudaMemcpyDefault);
  cudaMemcpy(cooColTemp, cooCol->data, sizeof(idx_t) * size, cudaMemcpyDefault);
  if (cooVal != nullptr) {
    ALLOC_TRY(&cooValTemp, sizeof(val_t) * size, nullptr);
    cudaMemcpy(cooValTemp, cooVal->data, sizeof(val_t) * size, cudaMemcpyDefault);
  }
  else
    cooValTemp = nullptr;
  CUDA_CHECK_LAST();

  if (cooValTemp != nullptr){
    auto zippy = thrust::make_zip_iterator(thrust::make_tuple(cooRowTemp, cooColTemp));
    thrust::sort_by_key(rmm::exec_policy(nullptr)->on(nullptr), zippy, zippy + size, cooValTemp);
  }
  else {
    auto zippy = thrust::make_zip_iterator(thrust::make_tuple(cooRowTemp, cooColTemp));
    thrust::sort(rmm::exec_policy(nullptr)->on(nullptr), zippy, zippy + size);
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST();

  // Each thread determines the count of rows it needs to transfer to each other thread
  idx_t localMinId, localMaxId;
  cudaMemcpy(&localMinId, cooRowTemp, sizeof(idx_t), cudaMemcpyDefault);
  cudaMemcpy(&localMaxId, cooRowTemp + size - 1, sizeof(idx_t), cudaMemcpyDefault);
  idx_t *endPositions;
  ALLOC_TRY(&endPositions, sizeof(idx_t) * (p - 1), nullptr);
  for (int j = 0; j < p - 1; j++) {
    idx_t endVertexId = part_offsets[j + 1];
    if (endVertexId <= localMinId) {
      // Write out zero for this position
      cugraph::snmg::writeSingleValue<<<1, 256>>>(endPositions + j, static_cast<idx_t>(0));
    }
    else if (endVertexId >= localMaxId) {
      // Write out size for this position
      cugraph::snmg::writeSingleValue<<<1, 256>>>(endPositions + j, size);
    }
    else if (endVertexId > localMinId && endVertexId < localMaxId) {
      dim3 nthreads, nblocks;
      nthreads.x = min(size, static_cast<idx_t>(CUDA_MAX_KERNEL_THREADS));
      nthreads.y = 1;
      nthreads.z = 1;
      nblocks.x = min((size + nthreads.x - 1) / nthreads.x,
                      static_cast<idx_t>(env.get_num_sm() * 32));
      nblocks.y = 1;
      nblocks.z = 1;
      cugraph::snmg::findStartRange<<<nblocks, nthreads>>>(size, endPositions + j, endVertexId, cooRowTemp);
    }
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST();
  std::vector<idx_t> positions(p + 1);
  cudaMemcpy(&positions[1], endPositions, sizeof(idx_t) * (p - 1), cudaMemcpyDefault);
  ALLOC_FREE_TRY(endPositions, nullptr);
  CUDA_CHECK_LAST();
  positions[0] = 0;
  positions[p] = size;
  idx_t* myRowCounts = comm->rowCounts + (i * p);
  for (int j = 0; j < p; j++){
    myRowCounts[j] = positions[j + 1] - positions[j];
  }

#pragma omp barrier

  int myRowCount = 0;
  for (int j = 0; j < p; j++){
    idx_t* otherRowCounts = comm->rowCounts + (j * p);
    myRowCount += otherRowCounts[i];
  }

  // Each thread allocates space to receive their rows from others
  idx_t *cooRowNew, *cooColNew;
  val_t *cooValNew;
  ALLOC_TRY(&cooRowNew, sizeof(idx_t) * myRowCount, nullptr);
  ALLOC_TRY(&cooColNew, sizeof(idx_t) * myRowCount, nullptr);
  if (cooValTemp != nullptr) {
    ALLOC_TRY(&cooValNew, sizeof(val_t) * myRowCount, nullptr);
  }
  else {
    cooValNew = nullptr;
  }
  comm->rowPtrs[i] = cooRowNew;
  comm->colPtrs[i] = cooColNew;
  comm->valPtrs[i] = cooValNew;
  CUDA_CHECK_LAST();
  cudaDeviceSynchronize();
#pragma omp barrier

  // Each thread copies the rows needed by other threads to them
  for (int other = 0; other < p; other++) {
    idx_t offset = 0;
    idx_t rowCount = myRowCounts[other];
    for (int prev = 0; prev < i; prev++) {
      idx_t* prevRowCounts = comm->rowCounts + (prev * p);
      offset += prevRowCounts[other];
    }

    if (rowCount > 0) {
      cudaMemcpy(comm->rowPtrs[other] + offset,
                 cooRowTemp + positions[other],
                 rowCount * sizeof(idx_t),
                 cudaMemcpyDefault);
      cudaMemcpy(comm->colPtrs[other] + offset,
                 cooColTemp + positions[other],
                 rowCount * sizeof(idx_t),
                 cudaMemcpyDefault);
      if (cooValTemp != nullptr) {
        cudaMemcpy(comm->valPtrs[other],
                   cooValTemp + positions[other],
                   rowCount * sizeof(idx_t),
                   cudaMemcpyDefault);
      }
    }
  }
  CUDA_CHECK_LAST();
  cugraph::snmg::sync_all();

  // Each thread frees up the input if allowed
  ALLOC_FREE_TRY(cooRowTemp, nullptr);
  ALLOC_FREE_TRY(cooColTemp, nullptr);
  if (cooValTemp != nullptr){
    ALLOC_FREE_TRY(cooValTemp, nullptr);
  }
  if (free_input) {
    ALLOC_FREE_TRY(cooRow->data, nullptr);
    ALLOC_FREE_TRY(cooCol->data, nullptr);
    if (cooVal != nullptr){
      ALLOC_FREE_TRY(cooVal->data, nullptr);
    }
  }

  // Each thread applies the offset to it's row column to get locally zero-based
  idx_t myOffset = part_offsets[i];
  thrust::transform(rmm::exec_policy(nullptr)->on(nullptr),
                    cooRowNew,
                    cooRowNew + myRowCount,
                    thrust::make_constant_iterator(myOffset * -1),
                    cooRowNew,
                    thrust::plus<idx_t>());

  // Each thread does a local coo2csr on its rows
  if (cooValNew != nullptr) {
    auto zippy = thrust::make_zip_iterator(thrust::make_tuple(cooRowNew, cooColNew));
    thrust::sort_by_key(rmm::exec_policy(nullptr)->on(nullptr),
                        zippy,
                        zippy + myRowCount,
                        cooValNew);
  }
  else {
    auto zippy = thrust::make_zip_iterator(thrust::make_tuple(cooRowNew, cooColNew));
    thrust::sort(rmm::exec_policy(nullptr)->on(nullptr), zippy, zippy + myEdgeCount);
  }

  CUDA_CHECK_LAST();

  localMaxId = part_offsets[i + 1] - part_offsets[i] - 1;
  idx_t* offsets;
  ALLOC_TRY(&offsets, (localMaxId + 2) * sizeof(idx_t), nullptr);
  cudaMemset(offsets, 0, (localMaxId + 2) * sizeof(idx_t));
  idx_t *unique, *counts, *runcount;
  ALLOC_TRY(&unique, (localMaxId + 1) * sizeof(idx_t), nullptr);
  ALLOC_TRY(&counts, (localMaxId + 1) * sizeof(idx_t), nullptr);
  ALLOC_TRY(&runcount, sizeof(idx_t), nullptr);
  void* tmpStorage = nullptr;
  size_t tmpBytes = 0;
  cub::DeviceRunLengthEncode::Encode(tmpStorage,
                                     tmpBytes,
                                     cooRowNew,
                                     unique,
                                     counts,
                                     runcount,
                                     myRowCount);
  ALLOC_TRY(&tmpStorage, tmpBytes, nullptr);
  cub::DeviceRunLengthEncode::Encode(tmpStorage,
                                     tmpBytes,
                                     cooRowNew,
                                     unique,
                                     counts,
                                     runcount,
                                     myRowCount);
  ALLOC_FREE_TRY(tmpStorage, nullptr);

  cudaDeviceSynchronize();
  idx_t runCount_h;
  cudaMemcpy(&runCount_h, runcount, sizeof(idx_t), cudaMemcpyDefault);
  int threadsPerBlock = 1024;
  int numBlocks = (runCount_h + threadsPerBlock - 1) / threadsPerBlock;

  CUDA_CHECK_LAST();

  cugraph::snmg::offsetsKernel<<<numBlocks, threadsPerBlock>>>(runCount_h, unique, counts, offsets);

  CUDA_CHECK_LAST();

  thrust::exclusive_scan(rmm::exec_policy(nullptr)->on(nullptr),
                         offsets,
                         offsets + localMaxId + 2,
                         offsets);
  ALLOC_FREE_TRY(cooRowNew, nullptr);
  ALLOC_FREE_TRY(unique, nullptr);
  ALLOC_FREE_TRY(counts, nullptr);
  ALLOC_FREE_TRY(runcount, nullptr);

  // Each thread sets up the results into the provided gdf_columns
  cugraph::detail::gdf_col_set_defaults(csrOff);
  csrOff->dtype = cooRow->dtype;
  csrOff->size = localMaxId + 2;
  csrOff->data = offsets;
  cugraph::detail::gdf_col_set_defaults(csrInd);
  csrInd->dtype = cooRow->dtype;
  csrInd->size = myRowCount;
  csrInd->data = cooColNew;
  if (cooValNew != nullptr) {
    cugraph::detail::gdf_col_set_defaults(cooVal);
    csrVal->dtype = cooVal->dtype;
    csrVal->size = myRowCount;
    csrVal->data = cooValNew;
  }
#pragma omp barrier

  // First thread deletes communicator object
  if (i == 0) {
    delete comm;
  }

  
}

} //namespace snmg

void snmg_coo2csr(size_t* part_offsets,
                           bool free_input,
                           void** comm1,
                           gdf_column* cooRow,
                           gdf_column* cooCol,
                           gdf_column* cooVal,
                           gdf_column* csrOff,
                           gdf_column* csrInd,
                           gdf_column* csrVal) {
  CUGRAPH_EXPECTS(part_offsets != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(cooRow != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(cooCol != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(csrOff != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(csrInd != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(comm1 != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(cooRow->size > 0, "Invalid API parameter");
  CUGRAPH_EXPECTS(cooCol->size > 0, "Invalid API parameter");
  CUGRAPH_EXPECTS(cooCol->dtype == cooRow->dtype, "Invalid API parameter");

  if (cooVal == nullptr) {
    if (cooRow->dtype == GDF_INT32) {
      return snmg::snmg_coo2csr_impl<int32_t, float>(part_offsets,
                                               free_input,
                                               comm1,
                                               cooRow,
                                               cooCol,
                                               cooVal,
                                               csrOff,
                                               csrInd,
                                               csrVal);
    }
    else if (cooRow->dtype == GDF_INT64) {
      return snmg::snmg_coo2csr_impl<int64_t, float>(part_offsets,
                                               free_input,
                                               comm1,
                                               cooRow,
                                               cooCol,
                                               cooVal,
                                               csrOff,
                                               csrInd,
                                               csrVal);
    }
    else
      CUGRAPH_FAIL("Unsupported data type");
  }
  else {
    if (cooRow->dtype == GDF_INT32 && cooVal->dtype == GDF_FLOAT32) {
      return snmg::snmg_coo2csr_impl<int32_t, float>(part_offsets,
                                               free_input,
                                               comm1,
                                               cooRow,
                                               cooCol,
                                               cooVal,
                                               csrOff,
                                               csrInd,
                                               csrVal);
    }
    else if (cooRow->dtype == GDF_INT32 && cooVal->dtype == GDF_FLOAT64) {
      return snmg::snmg_coo2csr_impl<int32_t, double>(part_offsets,
                                                free_input,
                                                comm1,
                                                cooRow,
                                                cooCol,
                                                cooVal,
                                                csrOff,
                                                csrInd,
                                                csrVal);
    }
    else if (cooRow->dtype == GDF_INT64 && cooVal->dtype == GDF_FLOAT32) {
      return snmg::snmg_coo2csr_impl<int64_t, float>(part_offsets,
                                               free_input,
                                               comm1,
                                               cooRow,
                                               cooCol,
                                               cooVal,
                                               csrOff,
                                               csrInd,
                                               csrVal);
    }
    else if (cooRow->dtype == GDF_INT64 && cooVal->dtype == GDF_FLOAT64) {
      return snmg::snmg_coo2csr_impl<int64_t, double>(part_offsets,
                                                free_input,
                                                comm1,
                                                cooRow,
                                                cooCol,
                                                cooVal,
                                                csrOff,
                                                csrInd,
                                                csrVal);
    }
    else
      CUGRAPH_FAIL("Unsupported data type");
  }
}

} // namespace cugraph 