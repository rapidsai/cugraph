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
 * @brief The cugraph Jaccard core functionality
 *
 * @file jaccard.cu
 * ---------------------------------------------------------------------------**/

#include "utilities/graph_utils.cuh"
#include "cugraph.h"
#include "rmm_utils.h"
#include "utilities/error_utils.h"

namespace cugraph {
  // Volume of neighboors (*weight_s)
  template<bool weighted, typename IdxType, typename ValType>
  __global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
  overlap_row_sum(IdxType n,
                  IdxType *csrPtr,
                  IdxType *csrInd,
                  ValType *v,
                  ValType *work) {
    IdxType row, start, end, length;
    ValType sum;
    for (row = threadIdx.y + blockIdx.y * blockDim.y;
        row < n;
        row += gridDim.y * blockDim.y) {
      start = csrPtr[row];
      end = csrPtr[row + 1];
      length = end - start;
      //compute row sums
      if (weighted) {
        sum = parallel_prefix_sum(length, csrInd + start, v);
        if (threadIdx.x == 0)
          work[row] = sum;
      }
      else {
        work[row] = (ValType) length;
      }
    }
  }

  // Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
  template<bool weighted, typename IdxType, typename ValType>
  __global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
  overlap_is(IdxType n,
             IdxType *csrPtr,
             IdxType *csrInd,
             ValType *v,
             ValType *work,
             ValType *weight_i,
             ValType *weight_s) {
    IdxType i, j, row, col, Ni, Nj;
    IdxType ref, cur, ref_col, cur_col, match;
    ValType ref_val;

    for (row = threadIdx.z + blockIdx.z * blockDim.z;
        row < n;
        row += gridDim.z * blockDim.z) {
      for (j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y;
          j < csrPtr[row + 1];
          j += gridDim.y * blockDim.y) {
        col = csrInd[j];
        //find which row has least elements (and call it reference row)
        Ni = csrPtr[row + 1] - csrPtr[row];
        Nj = csrPtr[col + 1] - csrPtr[col];
        ref = (Ni < Nj) ? row : col;
        cur = (Ni < Nj) ? col : row;

        //compute new sum weights
        weight_s[j] = min(work[row], work[col]);

        //compute new intersection weights
        //search for the element with the same column index in the reference row
        for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x; i < csrPtr[ref + 1];
            i += gridDim.x * blockDim.x) {
          match = -1;
          ref_col = csrInd[i];
          if (weighted) {
            ref_val = v[ref_col];
          }
          else {
            ref_val = 1.0;
          }

          //binary search (column indices are sorted within each row)
          IdxType left = csrPtr[cur];
          IdxType right = csrPtr[cur + 1] - 1;
          while (left <= right) {
            IdxType middle = (left + right) >> 1;
            cur_col = csrInd[middle];
            if (cur_col > ref_col) {
              right = middle - 1;
            }
            else if (cur_col < ref_col) {
              left = middle + 1;
            }
            else {
              match = middle;
              break;
            }
          }

          //if the element with the same column index in the reference row has been found
          if (match != -1) {
            atomicAdd(&weight_i[j], ref_val);
          }
        }
      }
    }
  }

  // Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
  // Using list of node pairs
  template<bool weighted, typename IdxType, typename ValType>
  __global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
  overlap_is_pairs(IdxType num_pairs,
                   IdxType *csrPtr,
                   IdxType *csrInd,
                   IdxType *first_pair,
                   IdxType *second_pair,
                   ValType *v,
                   ValType *work,
                   ValType *weight_i,
                   ValType *weight_s) {
    IdxType i, idx, row, col, Ni, Nj;
    IdxType ref, cur, ref_col, cur_col, match;
    ValType ref_val;

    for (idx = threadIdx.z + blockIdx.z * blockDim.z;
        idx < num_pairs;
        idx += gridDim.z * blockDim.z) {
      row = first_pair[idx];
      col = second_pair[idx];
      //find which row has least elements (and call it reference row)
      Ni = csrPtr[row + 1] - csrPtr[row];
      Nj = csrPtr[col + 1] - csrPtr[col];
      ref = (Ni < Nj) ? row : col;
      cur = (Ni < Nj) ? col : row;

      //compute new sum weights
      weight_s[idx] = min(work[row], work[col]);

      //compute new intersection weights
      //search for the element with the same column index in the reference row
      for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x;
          i < csrPtr[ref + 1];
          i += gridDim.x * blockDim.x) {
        match = -1;
        ref_col = csrInd[i];
        if (weighted) {
          ref_val = v[ref_col];
        }
        else {
          ref_val = 1.0;
        }

        //binary search (column indices are sorted within each row)
        IdxType left = csrPtr[cur];
        IdxType right = csrPtr[cur + 1] - 1;
        while (left <= right) {
          IdxType middle = (left + right) >> 1;
          cur_col = csrInd[middle];
          if (cur_col > ref_col) {
            right = middle - 1;
          }
          else if (cur_col < ref_col) {
            left = middle + 1;
          }
          else {
            match = middle;
            break;
          }
        }

        //if the element with the same column index in the reference row has been found
        if (match != -1) {
          atomicAdd(&weight_i[idx], ref_val);
        }
      }
    }
  }

  //Jaccard  weights (*weight)
  template<bool weighted, typename IdxType, typename ValType>
  __global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
  overlap_jw(IdxType e,
             IdxType *csrPtr,
             IdxType *csrInd,
             ValType *weight_i,
             ValType *weight_s,
             ValType *weight_j) {
    IdxType j;
    ValType Wi, Wu;

    for (j = threadIdx.x + blockIdx.x * blockDim.x;
        j < e;
        j += gridDim.x * blockDim.x) {
      Wi = weight_i[j];
      Wu = weight_s[j];
      weight_j[j] = (Wi / Wu);
    }
  }

  template<bool weighted, typename IdxType, typename ValType>
  int overlap(IdxType n,
              IdxType e,
              IdxType *csrPtr,
              IdxType *csrInd,
              ValType *weight_in,
              ValType *work,
              ValType *weight_i,
              ValType *weight_s,
              ValType *weight_j) {
    dim3 nthreads, nblocks;
    int y = 4;

    //setup launch configuration
    nthreads.x = 32;
    nthreads.y = y;
    nthreads.z = 1;
    nblocks.x = 1;
    nblocks.y = min((n + nthreads.y - 1) / nthreads.y, (IdxType) CUDA_MAX_BLOCKS);
    nblocks.z = 1;
    //launch kernel
    overlap_row_sum<weighted, IdxType, ValType> <<<nblocks, nthreads>>>(n,
                                                                        csrPtr,
                                                                        csrInd,
                                                                        weight_in,
                                                                        work);
    cudaDeviceSynchronize();
    fill(e, weight_i, (ValType) 0.0);
    //setup launch configuration
    nthreads.x = 32 / y;
    nthreads.y = y;
    nthreads.z = 8;
    nblocks.x = 1;
    nblocks.y = 1;
    nblocks.z = min((n + nthreads.z - 1) / nthreads.z, (IdxType) CUDA_MAX_BLOCKS); //1;
    //launch kernel
    overlap_is<weighted, IdxType, ValType> <<<nblocks, nthreads>>>(n,
                                                                   csrPtr,
                                                                   csrInd,
                                                                   weight_in,
                                                                   work,
                                                                   weight_i,
                                                                   weight_s);

    //setup launch configuration
    nthreads.x = min(e, (IdxType) CUDA_MAX_KERNEL_THREADS);
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((e + nthreads.x - 1) / nthreads.x, (IdxType) CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;
    //launch kernel
    overlap_jw<weighted, IdxType, ValType> <<<nblocks, nthreads>>>(e,
                                                                   csrPtr,
                                                                   csrInd,
                                                                   weight_i,
                                                                   weight_s,
                                                                   weight_j);

    return 0;
  }

  template<bool weighted, typename IdxType, typename ValType>
  int overlap_pairs(IdxType n,
                    IdxType num_pairs,
                    IdxType *csrPtr,
                    IdxType *csrInd,
                    IdxType *first_pair,
                    IdxType *second_pair,
                    ValType *weight_in,
                    ValType *work,
                    ValType *weight_i,
                    ValType *weight_s,
                    ValType *weight_j) {
    dim3 nthreads, nblocks;
    int y = 4;

    //setup launch configuration
    nthreads.x = 32;
    nthreads.y = y;
    nthreads.z = 1;
    nblocks.x = 1;
    nblocks.y = min((n + nthreads.y - 1) / nthreads.y, (IdxType) CUDA_MAX_BLOCKS);
    nblocks.z = 1;
    //launch kernel
    overlap_row_sum<weighted, IdxType, ValType> <<<nblocks, nthreads>>>(n,
                                                                        csrPtr,
                                                                        csrInd,
                                                                        weight_in,
                                                                        work);
    cudaDeviceSynchronize();
    fill(num_pairs, weight_i, (ValType) 0.0);
    //setup launch configuration
    nthreads.x = 32;
    nthreads.y = 1;
    nthreads.z = 8;
    nblocks.x = 1;
    nblocks.y = 1;
    nblocks.z = min((n + nthreads.z - 1) / nthreads.z, (IdxType) CUDA_MAX_BLOCKS); //1;
    //launch kernel
    overlap_is_pairs<weighted, IdxType, ValType> <<<nblocks, nthreads>>>(num_pairs,
                                                                         csrPtr,
                                                                         csrInd,
                                                                         first_pair,
                                                                         second_pair,
                                                                         weight_in,
                                                                         work,
                                                                         weight_i,
                                                                         weight_s);

    //setup launch configuration
    nthreads.x = min(num_pairs, (IdxType) CUDA_MAX_KERNEL_THREADS);
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((num_pairs + nthreads.x - 1) / nthreads.x, (IdxType) CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;
    //launch kernel
    overlap_jw<weighted, IdxType, ValType> <<<nblocks, nthreads>>>(num_pairs,
                                                                   csrPtr,
                                                                   csrInd,
                                                                   weight_i,
                                                                   weight_s,
                                                                   weight_j);

    return 0;
  }
} // End cugraph namespace

gdf_error gdf_overlap(gdf_graph *graph, gdf_column *weights, gdf_column *result) {
  GDF_REQUIRE(graph != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(graph->adjList != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(result != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(result->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!result->valid, GDF_VALIDITY_UNSUPPORTED);

  bool weighted = (weights != nullptr);

  gdf_dtype ValueType = result->dtype;
  gdf_dtype IndexType = graph->adjList->offsets->dtype;

  void *csrPtr = graph->adjList->offsets->data;
  void *csrInd = graph->adjList->indices->data;
  void *weight_i = nullptr;
  void *weight_s = nullptr;
  void *weight_j = result->data;
  void *work = nullptr;
  void *weight_in = nullptr;
  if (weighted)
    weight_in = weights->data;

  if (ValueType == GDF_FLOAT32 && IndexType == GDF_INT32 && weighted) {
    int32_t n = graph->adjList->offsets->size - 1;
    int32_t e = graph->adjList->indices->size;
    ALLOC_TRY(&weight_i, sizeof(float) * e, nullptr);
    ALLOC_TRY(&weight_s, sizeof(float) * e, nullptr);
    ALLOC_TRY(&work, sizeof(float) * n, nullptr);
    cugraph::overlap<true, int32_t, float>(n,
                                           e,
                                           (int32_t*) csrPtr,
                                           (int32_t*) csrInd,
                                           (float*) weight_in,
                                           (float*) work,
                                           (float*) weight_i,
                                           (float*) weight_s,
                                           (float*) weight_j);
  }
  if (ValueType == GDF_FLOAT32 && IndexType == GDF_INT32 && !weighted) {
    int32_t n = graph->adjList->offsets->size - 1;
    int32_t e = graph->adjList->indices->size;
    ALLOC_TRY(&weight_i, sizeof(float) * e, nullptr);
    ALLOC_TRY(&weight_s, sizeof(float) * e, nullptr);
    ALLOC_TRY(&work, sizeof(float) * n, nullptr);
    cugraph::overlap<false, int32_t, float>(n,
                                            e,
                                            (int32_t*) csrPtr,
                                            (int32_t*) csrInd,
                                            (float*) weight_in,
                                            (float*) work,
                                            (float*) weight_i,
                                            (float*) weight_s,
                                            (float*) weight_j);
  }
  if (ValueType == GDF_FLOAT64 && IndexType == GDF_INT32 && weighted) {
    int32_t n = graph->adjList->offsets->size - 1;
    int32_t e = graph->adjList->indices->size;
    ALLOC_TRY(&weight_i, sizeof(double) * e, nullptr);
    ALLOC_TRY(&weight_s, sizeof(double) * e, nullptr);
    ALLOC_TRY(&work, sizeof(double) * n, nullptr);
    cugraph::overlap<true, int32_t, double>(n,
                                            e,
                                            (int32_t*) csrPtr,
                                            (int32_t*) csrInd,
                                            (double*) weight_in,
                                            (double*) work,
                                            (double*) weight_i,
                                            (double*) weight_s,
                                            (double*) weight_j);
  }
  if (ValueType == GDF_FLOAT64 && IndexType == GDF_INT32 && !weighted) {
    int32_t n = graph->adjList->offsets->size - 1;
    int32_t e = graph->adjList->indices->size;
    ALLOC_TRY(&weight_i, sizeof(double) * e, nullptr);
    ALLOC_TRY(&weight_s, sizeof(double) * e, nullptr);
    ALLOC_TRY(&work, sizeof(double) * n, nullptr);
    cugraph::overlap<false, int32_t, double>(n,
                                             e,
                                             (int32_t*) csrPtr,
                                             (int32_t*) csrInd,
                                             (double*) weight_in,
                                             (double*) work,
                                             (double*) weight_i,
                                             (double*) weight_s,
                                             (double*) weight_j);
  }
  if (ValueType == GDF_FLOAT32 && IndexType == GDF_INT64 && weighted) {
    int64_t n = graph->adjList->offsets->size - 1;
    int64_t e = graph->adjList->indices->size;
    ALLOC_TRY(&weight_i, sizeof(float) * e, nullptr);
    ALLOC_TRY(&weight_s, sizeof(float) * e, nullptr);
    ALLOC_TRY(&work, sizeof(float) * n, nullptr);
    cugraph::overlap<true, int64_t, float>(n,
                                           e,
                                           (int64_t*) csrPtr,
                                           (int64_t*) csrInd,
                                           (float*) weight_in,
                                           (float*) work,
                                           (float*) weight_i,
                                           (float*) weight_s,
                                           (float*) weight_j);
  }
  if (ValueType == GDF_FLOAT32 && IndexType == GDF_INT64 && !weighted) {
    int64_t n = graph->adjList->offsets->size - 1;
    int64_t e = graph->adjList->indices->size;
    ALLOC_TRY(&weight_i, sizeof(float) * e, nullptr);
    ALLOC_TRY(&weight_s, sizeof(float) * e, nullptr);
    ALLOC_TRY(&work, sizeof(float) * n, nullptr);
    cugraph::overlap<false, int64_t, float>(n,
                                            e,
                                            (int64_t*) csrPtr,
                                            (int64_t*) csrInd,
                                            (float*) weight_in,
                                            (float*) work,
                                            (float*) weight_i,
                                            (float*) weight_s,
                                            (float*) weight_j);
  }
  if (ValueType == GDF_FLOAT64 && IndexType == GDF_INT64 && weighted) {
    int64_t n = graph->adjList->offsets->size - 1;
    int64_t e = graph->adjList->indices->size;
    ALLOC_TRY(&weight_i, sizeof(double) * e, nullptr);
    ALLOC_TRY(&weight_s, sizeof(double) * e, nullptr);
    ALLOC_TRY(&work, sizeof(double) * n, nullptr);
    cugraph::overlap<true, int64_t, double>(n,
                                            e,
                                            (int64_t*) csrPtr,
                                            (int64_t*) csrInd,
                                            (double*) weight_in,
                                            (double*) work,
                                            (double*) weight_i,
                                            (double*) weight_s,
                                            (double*) weight_j);
  }
  if (ValueType == GDF_FLOAT64 && IndexType == GDF_INT64 && !weighted) {
    int64_t n = graph->adjList->offsets->size - 1;
    int64_t e = graph->adjList->indices->size;
    ALLOC_TRY(&weight_i, sizeof(double) * e, nullptr);
    ALLOC_TRY(&weight_s, sizeof(double) * e, nullptr);
    ALLOC_TRY(&work, sizeof(double) * n, nullptr);
    cugraph::overlap<false, int64_t, double>(n,
                                             e,
                                             (int64_t*) csrPtr,
                                             (int64_t*) csrInd,
                                             (double*) weight_in,
                                             (double*) work,
                                             (double*) weight_i,
                                             (double*) weight_s,
                                             (double*) weight_j);
  }

// Clean up temp arrays
  ALLOC_FREE_TRY(weight_i, nullptr);
  ALLOC_FREE_TRY(weight_s, nullptr);
  ALLOC_FREE_TRY(work, nullptr);

  return GDF_SUCCESS;
}

gdf_error gdf_overlap_list(gdf_graph* graph,
                           gdf_column* weights,
                           gdf_column* first,
                           gdf_column* second,
                           gdf_column* result) {
  GDF_REQUIRE(graph != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(graph->adjList != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(result != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(result->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!result->valid, GDF_VALIDITY_UNSUPPORTED);

  GDF_REQUIRE(first != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(first->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!first->valid, GDF_VALIDITY_UNSUPPORTED);

  GDF_REQUIRE(second != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(second->data != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(!second->valid, GDF_VALIDITY_UNSUPPORTED);

  bool weighted = (weights != nullptr);

  gdf_dtype ValueType = result->dtype;
  gdf_dtype IndexType = graph->adjList->offsets->dtype;
  GDF_REQUIRE(first->dtype == IndexType, GDF_INVALID_API_CALL);
  GDF_REQUIRE(second->dtype == IndexType, GDF_INVALID_API_CALL);

  void *first_pair = first->data;
  void *second_pair = second->data;
  void *csrPtr = graph->adjList->offsets->data;
  void *csrInd = graph->adjList->indices->data;
  void *weight_i = nullptr;
  void *weight_s = nullptr;
  void *weight_j = result->data;
  void *work = nullptr;
  void *weight_in = nullptr;
  if (weighted)
    weight_in = weights->data;

  if (ValueType == GDF_FLOAT32 && IndexType == GDF_INT32 && weighted) {
    int32_t n = graph->adjList->offsets->size - 1;
    int32_t num_pairs = first->size;
    ALLOC_TRY(&weight_i, sizeof(float) * num_pairs, nullptr);
    ALLOC_TRY(&weight_s, sizeof(float) * num_pairs, nullptr);
    ALLOC_TRY(&work, sizeof(float) * n, nullptr);
    cugraph::overlap_pairs<true, int32_t, float>(n,
                                                 num_pairs,
                                                 (int32_t*) csrPtr,
                                                 (int32_t*) csrInd,
                                                 (int32_t*) first_pair,
                                                 (int32_t*) second_pair,
                                                 (float*) weight_in,
                                                 (float*) work,
                                                 (float*) weight_i,
                                                 (float*) weight_s,
                                                 (float*) weight_j);
  }

  if (ValueType == GDF_FLOAT32 && IndexType == GDF_INT32 && !weighted) {
    int32_t n = graph->adjList->offsets->size - 1;
    int32_t num_pairs = first->size;
    ALLOC_TRY(&weight_i, sizeof(float) * num_pairs, nullptr);
    ALLOC_TRY(&weight_s, sizeof(float) * num_pairs, nullptr);
    ALLOC_TRY(&work, sizeof(float) * n, nullptr);
    cugraph::overlap_pairs<false, int32_t, float>(n,
                                                  num_pairs,
                                                  (int32_t*) csrPtr,
                                                  (int32_t*) csrInd,
                                                  (int32_t*) first_pair,
                                                  (int32_t*) second_pair,
                                                  (float*) weight_in,
                                                  (float*) work,
                                                  (float*) weight_i,
                                                  (float*) weight_s,
                                                  (float*) weight_j);
  }

  if (ValueType == GDF_FLOAT64 && IndexType == GDF_INT32 && weighted) {
    int32_t n = graph->adjList->offsets->size - 1;
    int32_t num_pairs = first->size;
    ALLOC_TRY(&weight_i, sizeof(double) * num_pairs, nullptr);
    ALLOC_TRY(&weight_s, sizeof(double) * num_pairs, nullptr);
    ALLOC_TRY(&work, sizeof(double) * n, nullptr);
    cugraph::overlap_pairs<true, int32_t, double>(n,
                                                  num_pairs,
                                                  (int32_t*) csrPtr,
                                                  (int32_t*) csrInd,
                                                  (int32_t*) first_pair,
                                                  (int32_t*) second_pair,
                                                  (double*) weight_in,
                                                  (double*) work,
                                                  (double*) weight_i,
                                                  (double*) weight_s,
                                                  (double*) weight_j);
  }

  if (ValueType == GDF_FLOAT64 && IndexType == GDF_INT32 && !weighted) {
    int32_t n = graph->adjList->offsets->size - 1;
    int32_t num_pairs = first->size;
    ALLOC_TRY(&weight_i, sizeof(double) * num_pairs, nullptr);
    ALLOC_TRY(&weight_s, sizeof(double) * num_pairs, nullptr);
    ALLOC_TRY(&work, sizeof(double) * n, nullptr);
    cugraph::overlap_pairs<false, int32_t, double>(n,
                                                   num_pairs,
                                                   (int32_t*) csrPtr,
                                                   (int32_t*) csrInd,
                                                   (int32_t*) first_pair,
                                                   (int32_t*) second_pair,
                                                   (double*) weight_in,
                                                   (double*) work,
                                                   (double*) weight_i,
                                                   (double*) weight_s,
                                                   (double*) weight_j);
  }

  if (ValueType == GDF_FLOAT32 && IndexType == GDF_INT64 && weighted) {
    int64_t n = graph->adjList->offsets->size - 1;
    int64_t num_pairs = first->size;
    ALLOC_TRY(&weight_i, sizeof(float) * num_pairs, nullptr);
    ALLOC_TRY(&weight_s, sizeof(float) * num_pairs, nullptr);
    ALLOC_TRY(&work, sizeof(float) * n, nullptr);
    cugraph::overlap_pairs<true, int64_t, float>(n,
                                                 num_pairs,
                                                 (int64_t*) csrPtr,
                                                 (int64_t*) csrInd,
                                                 (int64_t*) first_pair,
                                                 (int64_t*) second_pair,
                                                 (float*) weight_in,
                                                 (float*) work,
                                                 (float*) weight_i,
                                                 (float*) weight_s,
                                                 (float*) weight_j);
  }

  if (ValueType == GDF_FLOAT32 && IndexType == GDF_INT64 && !weighted) {
    int64_t n = graph->adjList->offsets->size - 1;
    int64_t num_pairs = first->size;
    ALLOC_TRY(&weight_i, sizeof(float) * num_pairs, nullptr);
    ALLOC_TRY(&weight_s, sizeof(float) * num_pairs, nullptr);
    ALLOC_TRY(&work, sizeof(float) * n, nullptr);
    cugraph::overlap_pairs<false, int64_t, float>(n,
                                                  num_pairs,
                                                  (int64_t*) csrPtr,
                                                  (int64_t*) csrInd,
                                                  (int64_t*) first_pair,
                                                  (int64_t*) second_pair,
                                                  (float*) weight_in,
                                                  (float*) work,
                                                  (float*) weight_i,
                                                  (float*) weight_s,
                                                  (float*) weight_j);
  }

  if (ValueType == GDF_FLOAT64 && IndexType == GDF_INT64 && weighted) {
    int64_t n = graph->adjList->offsets->size - 1;
    int64_t num_pairs = first->size;
    ALLOC_TRY(&weight_i, sizeof(double) * num_pairs, nullptr);
    ALLOC_TRY(&weight_s, sizeof(double) * num_pairs, nullptr);
    ALLOC_TRY(&work, sizeof(double) * n, nullptr);
    cugraph::overlap_pairs<true, int64_t, double>(n,
                                                  num_pairs,
                                                  (int64_t*) csrPtr,
                                                  (int64_t*) csrInd,
                                                  (int64_t*) first_pair,
                                                  (int64_t*) second_pair,
                                                  (double*) weight_in,
                                                  (double*) work,
                                                  (double*) weight_i,
                                                  (double*) weight_s,
                                                  (double*) weight_j);
  }

  if (ValueType == GDF_FLOAT64 && IndexType == GDF_INT64 && !weighted) {
    int64_t n = graph->adjList->offsets->size - 1;
    int64_t num_pairs = first->size;
    ALLOC_TRY(&weight_i, sizeof(double) * num_pairs, nullptr);
    ALLOC_TRY(&weight_s, sizeof(double) * num_pairs, nullptr);
    ALLOC_TRY(&work, sizeof(double) * n, nullptr);
    cugraph::overlap_pairs<false, int64_t, double>(n,
                                                   num_pairs,
                                                   (int64_t*) csrPtr,
                                                   (int64_t*) csrInd,
                                                   (int64_t*) first_pair,
                                                   (int64_t*) second_pair,
                                                   (double*) weight_in,
                                                   (double*) work,
                                                   (double*) weight_i,
                                                   (double*) weight_s,
                                                   (double*) weight_j);
  }

  // Clean up temp arrays
  ALLOC_FREE_TRY(weight_i, nullptr);
  ALLOC_FREE_TRY(weight_s, nullptr);
  ALLOC_FREE_TRY(work, nullptr);

  return GDF_SUCCESS;
}

