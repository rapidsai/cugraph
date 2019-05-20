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
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <assert.h>

#include "include/triangles_counting_defines.hxx"
#include "include/triangles_counting_kernels.hxx"

#include "include/nvgraph_error.hxx"

#include "cub/cub.cuh"
#include <thrust/iterator/counting_iterator.h>
#include "include/sm_utils.h"
using namespace cub;

#include "rmm/rmm.h"

#define TH_CENT_K_LOCLEN (34)
#define WP_LEN_TH1 (24)
#define WP_LEN_TH2 (2)

#if WP_LEN_TH1 > 32
#error WP_LEN_TH1 must be <= 32!
#endif

template<typename T>
__device__  __forceinline__ T LDG(const T* x)
                                 {
#if __CUDA_ARCH__ < 350
  return *x;
#else
  return __ldg(x);
#endif
}

namespace nvgraph
{

  namespace triangles_counting
  {
    // Better return std::unique_ptr than a raw pointer, but we haven't decide
    // whether to create our own unique_ptr with RMM's deleter or to implement
    // this in librmm. So, we may wait till this decision is made.
    void* get_temp_storage(size_t size, cudaStream_t stream) {
      auto t = static_cast<void*>(nullptr);
      auto status = RMM_ALLOC(&t, size, stream);
      if (status == RMM_ERROR_OUT_OF_MEMORY) {
        FatalError("Not enough memory", NVGRAPH_ERR_NO_MEMORY);
      }
      else if (status != RMM_SUCCESS) {
        FatalError("Memory manager internal error (alloc)", NVGRAPH_ERR_UNKNOWN);
      }

      return t;
    }

    void free_temp_storage(void* ptr, cudaStream_t stream) {
      auto status = RMM_FREE(ptr, stream);
      if (status != RMM_SUCCESS) {
        FatalError("Memory manager internal error (release)", NVGRAPH_ERR_UNKNOWN);
      }
    }

// cub utility wrappers ////////////////////////////////////////////////////////
    template<typename InputIteratorT,
        typename OutputIteratorT,
        typename ReductionOpT,
        typename T>
    static inline void cubReduce(InputIteratorT d_in, OutputIteratorT d_out,
                                 int num_items,
                                 ReductionOpT reduction_op,
                                 T init,
                                 cudaStream_t stream = 0,
                                 bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
                                d_in,
                                d_out, num_items, reduction_op,
                                init,
                                stream, debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
                                d_in,
                                d_out, num_items, reduction_op,
                                init,
                                stream, debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT, typename OutputIteratorT>
    static inline void cubSum(InputIteratorT d_in, OutputIteratorT d_out,
                              int num_items,
                              cudaStream_t stream = 0,
                              bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                             d_in,
                             d_out, num_items, stream,
                             debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                             d_in,
                             d_out, num_items, stream,
                             debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename KeyT>
    static inline void cubSortKeys(KeyT *d_keys_in, KeyT *d_keys_out, int num_items,
                                   int begin_bit = 0,
                                   int end_bit = sizeof(KeyT) * 8,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                     d_keys_in,
                                     d_keys_out, num_items,
                                     begin_bit,
                                     end_bit, stream,
                                     debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                     d_keys_in,
                                     d_keys_out, num_items,
                                     begin_bit,
                                     end_bit, stream,
                                     debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename KeyT, typename ValueT>
    static inline void cubSortPairs(KeyT *d_keys_in, KeyT *d_keys_out,
                                    ValueT *d_values_in,
                                    ValueT *d_values_out,
                                    int num_items,
                                    int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                    cudaStream_t stream = 0,
                                    bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                      d_keys_in,
                                      d_keys_out, d_values_in,
                                      d_values_out,
                                      num_items, begin_bit,
                                      end_bit,
                                      stream, debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                      d_keys_in,
                                      d_keys_out, d_values_in,
                                      d_values_out,
                                      num_items, begin_bit,
                                      end_bit,
                                      stream, debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename KeyT, typename ValueT>
    static inline void cubSortPairsDescending(KeyT *d_keys_in, KeyT *d_keys_out,
                                              ValueT *d_values_in,
                                              ValueT *d_values_out,
                                              int num_items,
                                              int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                              cudaStream_t stream = 0,
                                              bool debug_synchronous = false) {
      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                                d_keys_in,
                                                d_keys_out, d_values_in,
                                                d_values_out,
                                                num_items, begin_bit,
                                                end_bit,
                                                stream, debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                                d_keys_in,
                                                d_keys_out, d_values_in,
                                                d_values_out,
                                                num_items, begin_bit,
                                                end_bit,
                                                stream, debug_synchronous);
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT,
        typename OutputIteratorT,
        typename NumSelectedIteratorT>
    static inline void cubUnique(InputIteratorT d_in, OutputIteratorT d_out,
                                 NumSelectedIteratorT d_num_selected_out,
                                 int num_items,
                                 cudaStream_t stream = 0,
                                 bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                                d_in,
                                d_out, d_num_selected_out,
                                num_items,
                                stream, debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                                d_in,
                                d_out, d_num_selected_out,
                                num_items,
                                stream, debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT,
        typename UniqueOutputIteratorT,
        typename LengthsOutputIteratorT,
        typename NumRunsOutputIteratorT>
    static inline void cubEncode(InputIteratorT d_in, UniqueOutputIteratorT d_unique_out,
                                 LengthsOutputIteratorT d_counts_out,
                                 NumRunsOutputIteratorT d_num_runs_out,
                                 int num_items,
                                 cudaStream_t stream = 0, bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes,
                                         d_in,
                                         d_unique_out, d_counts_out,
                                         d_num_runs_out,
                                         num_items, stream,
                                         debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes,
                                         d_in,
                                         d_unique_out, d_counts_out,
                                         d_num_runs_out,
                                         num_items, stream,
                                         debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT,
        typename OutputIteratorT>
    static inline void cubMin(InputIteratorT d_in, OutputIteratorT d_out,
                              int num_items,
                              cudaStream_t stream = 0,
                              bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                             d_in,
                             d_out, num_items, stream,
                             debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                             d_in,
                             d_out, num_items, stream,
                             debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT,
        typename OutputIteratorT>
    static inline void cubMax(InputIteratorT d_in, OutputIteratorT d_out,
                              int num_items,
                              cudaStream_t stream = 0,
                              bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                             d_in,
                             d_out, num_items, stream,
                             debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                             d_in,
                             d_out, num_items, stream,
                             debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT,
        typename OutputIteratorT,
        typename NumSelectedIteratorT,
        typename SelectOp>
    static inline void cubIf(InputIteratorT d_in, OutputIteratorT d_out,
                             NumSelectedIteratorT d_num_selected_out,
                             int num_items, SelectOp select_op,
                             cudaStream_t stream = 0,
                             bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                            d_in,
                            d_out, d_num_selected_out,
                            num_items,
                            select_op, stream,
                            debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                            d_in,
                            d_out, d_num_selected_out,
                            num_items,
                            select_op, stream,
                            debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT,
        typename FlagIterator,
        typename OutputIteratorT,
        typename NumSelectedIteratorT>
    static inline void cubFlagged(InputIteratorT d_in, FlagIterator d_flags,
                                  OutputIteratorT d_out,
                                  NumSelectedIteratorT d_num_selected_out,
                                  int num_items,
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                                 d_in,
                                 d_flags, d_out, d_num_selected_out,
                                 num_items,
                                 stream, debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                                 d_in,
                                 d_flags, d_out, d_num_selected_out,
                                 num_items,
                                 stream, debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT,
        typename OutputIteratorT>
    static inline void cubExclusiveSum(InputIteratorT d_in, OutputIteratorT d_out,
                                       int num_items,
                                       cudaStream_t stream = 0,
                                       bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                    d_in,
                                    d_out, num_items, stream,
                                    debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                    d_in,
                                    d_out, num_items, stream,
                                    debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename InputIteratorT,
        typename OutputIteratorT>
    static inline void cubInclusiveSum(InputIteratorT d_in, OutputIteratorT d_out,
                                       int num_items,
                                       cudaStream_t stream = 0,
                                       bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                    d_in,
                                    d_out, num_items, stream,
                                    debug_synchronous);
      cudaCheckError()
      ;
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                    d_in,
                                    d_out, num_items, stream,
                                    debug_synchronous);
      cudaCheckError()
      ;
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename KeysInputIteratorT,
        typename UniqueOutputIteratorT,
        typename ValuesInputIteratorT,
        typename AggregatesOutputIteratorT,
        typename NumRunsOutputIteratorT,
        typename ReductionOpT>
    static inline void cubReduceByKey(KeysInputIteratorT d_keys_in,
                                      UniqueOutputIteratorT d_unique_out,
                                      ValuesInputIteratorT d_values_in,
                                      AggregatesOutputIteratorT d_aggregates_out,
                                      NumRunsOutputIteratorT d_num_runs_out,
                                      ReductionOpT reduction_op,
                                      int num_items,
                                      cudaStream_t stream = 0,
                                      bool debug_synchronous = false) {

      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;

      cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                     d_keys_in,
                                     d_unique_out,
                                     d_values_in,
                                     d_aggregates_out,
                                     d_num_runs_out,
                                     reduction_op,
                                     num_items,
                                     stream, debug_synchronous);
      cudaCheckError();
      d_temp_storage = get_temp_storage(temp_storage_bytes, stream);
      cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                     d_keys_in,
                                     d_unique_out,
                                     d_values_in,
                                     d_aggregates_out,
                                     d_num_runs_out,
                                     reduction_op,
                                     num_items,
                                     stream, debug_synchronous);
      cudaCheckError();
      free_temp_storage(d_temp_storage, stream);

      return;
    }

    template<typename T2>
    __device__ __host__ inline bool operator==(const T2 &lhs, const T2 &rhs) {
      return (lhs.x == rhs.x && lhs.y == rhs.y);
    }

//////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    __device__ T __block_bcast(const T v, const int x) {

      __shared__ T shv;

      __syncthreads();
      if (threadIdx.x == x)
        shv = v;
      __syncthreads();

      return shv;
    }

    template<int BDIM_X,
        int BDIM_Y,
        int WSIZE,
        typename T>
    __device__  __forceinline__ T block_sum(T v) {

      __shared__ T sh[BDIM_X * BDIM_Y / WSIZE];

      const int lid = threadIdx.x % 32;
      const int wid = threadIdx.x / 32 + ((BDIM_Y > 1) ? threadIdx.y * (BDIM_X / 32) : 0);

      #pragma unroll
      for (int i = WSIZE / 2; i; i >>= 1) {
        v += utils::shfl_down(v, i);
      }
      if (lid == 0)
        sh[wid] = v;

      __syncthreads();
      if (wid == 0) {
        v = (lid < (BDIM_X * BDIM_Y / WSIZE)) ? sh[lid] : 0;

        #pragma unroll
        for (int i = (BDIM_X * BDIM_Y / WSIZE) / 2; i; i >>= 1) {
          v += utils::shfl_down(v, i);
        }
      }
      return v;
    }

//////////////////////////////////////////////////////////////////////////////////////////
    template<int BDIM,
        int WSIZE,
        int BWL0,
        typename ROW_T,
        typename OFF_T,
        typename CNT_T,
        typename MAP_T>
    __global__ void tricnt_b2b_k(const ROW_T ner,
                                 const ROW_T *__restrict__ rows,
                                 const OFF_T *__restrict__ roff,
                                 const ROW_T *__restrict__ cols,
                                 CNT_T *__restrict__ ocnt,
                                 MAP_T *__restrict__ bmapL0,
                                 const size_t bmldL0,
                                 MAP_T *__restrict__ bmapL1,
                                 const size_t bmldL1) {
      CNT_T __cnt = 0;

      bmapL1 += bmldL1 * blockIdx.x;
      bmapL0 += bmldL0 * blockIdx.x;
      for (ROW_T bid = blockIdx.x; bid < ner; bid += gridDim.x) {

        const OFF_T rbeg = roff[rows[bid]];
        const OFF_T rend = roff[rows[bid] + 1];

        ROW_T firstcol = 0;
        ROW_T lastcol = 0;

        for (OFF_T i = rbeg; i < rend; i += BDIM) {
          const ROW_T c = (i + threadIdx.x < rend) ? cols[i + threadIdx.x] : -1;

          __syncthreads();
          if (c > -1) {
            atomicOr(bmapL1 + c / BITSOF(bmapL1), ((MAP_T) 1) << (c % BITSOF(bmapL1)));
            atomicOr(bmapL0 + c / BWL0 / BITSOF(bmapL0),
                     ((MAP_T) 1) << ((c / BWL0) % BITSOF(bmapL0)));
          }
          __syncthreads();

#pragma unroll
          for (int j = 0; j < BDIM; j++) {

            const ROW_T curc = __block_bcast(c, j);
            if (curc == -1)
              break;

            lastcol = curc;
            if ((i == rbeg) && !j) {
              firstcol = curc;
              continue;
            }
            const OFF_T soff = roff[curc];
            const OFF_T eoff = roff[curc + 1];

            for (OFF_T k = eoff - 1; k >= soff; k -= BDIM) {
              if (k - (int) threadIdx.x < soff)
                break;

              const ROW_T cc = LDG(cols + k - threadIdx.x);
              if (cc < firstcol)
                break;

              MAP_T mm = ((MAP_T) 1) << ((cc / BWL0) % BITSOF(bmapL0));
              if (0 == (bmapL0[cc / BWL0 / BITSOF(bmapL0)] & mm))
                continue;

              mm = ((MAP_T) 1) << (cc % BITSOF(bmapL1));
              if (bmapL1[cc / BITSOF(bmapL1)] & mm) {
                __cnt++;
              }
            }
          }
        }

        lastcol /= 64;
        firstcol /= 64;

        __syncthreads();
        for (int i = rbeg; i < rend; i += BDIM) {
          if (i + threadIdx.x < rend) {
            ROW_T c = cols[i + threadIdx.x];
            bmapL1[c / BITSOF(bmapL1)] = 0;
            bmapL0[c / BWL0 / BITSOF(bmapL0)] = 0;
          }
        }
        __syncthreads();
      }

      __cnt = block_sum<BDIM, 1, WSIZE>(__cnt);
      if (threadIdx.x == 0)
        ocnt[blockIdx.x] = __cnt;

      return;
    }

    template<typename T>
    void tricnt_b2b(T nblock,
                    spmat_t<T> *m,
                    uint64_t *ocnt_d,
                    unsigned int *bmapL0_d,
                    size_t bmldL0,
                    unsigned int *bmapL1_d,
                    size_t bmldL1,
                    cudaStream_t stream) {

      // still best overall (with no psum)
      tricnt_b2b_k<THREADS, 32, BLK_BWL0> <<<nblock, THREADS, 0, stream>>>(m->nrows, m->rows_d,
                                                                           m->roff_d,
                                                                           m->cols_d, ocnt_d,
                                                                           bmapL0_d,
                                                                           bmldL0,
                                                                           bmapL1_d,
                                                                           bmldL1);
      cudaCheckError()
      ;
      return;
    }
//////////////////////////////////////////////////////////////////////////////////////////
    template<int BDIM_X,
        int BDIM_Y,
        int WSIZE,
        typename T>
    __device__  __forceinline__ T block_sum_sh(T v, T *sh) {

      const int lid = threadIdx.x % 32;
      const int wid = threadIdx.x / 32 + ((BDIM_Y > 1) ? threadIdx.y * (BDIM_X / 32) : 0);

#pragma unroll
      for (int i = WSIZE / 2; i; i >>= 1) {
        v += utils::shfl_down(v, i);
      }
      if (lid == 0)
        sh[wid] = v;

      __syncthreads();
      if (wid == 0) {
        v = (lid < (BDIM_X * BDIM_Y / WSIZE)) ? sh[lid] : 0;

#pragma unroll
        for (int i = (BDIM_X * BDIM_Y / WSIZE) / 2; i; i >>= 1) {
          v += utils::shfl_down(v, i);
        }
      }
      return v;
    }

    template<int BDIM,
        int WSIZE,
        typename ROW_T,
        typename OFF_T,
        typename CNT_T>
    __global__ void tricnt_bsh_k(const ROW_T ner,
                                 const ROW_T *__restrict__ rows,
                                 const OFF_T *__restrict__ roff,
                                 const ROW_T *__restrict__ cols,
                                 CNT_T *__restrict__ ocnt,
                                 const size_t bmld) {
      CNT_T __cnt = 0;
      extern __shared__ unsigned int shm[];

      for (int i = 0; i < bmld; i += BDIM) {
        if (i + threadIdx.x < bmld) {
          shm[i + threadIdx.x] = 0;
        }
      }

      for (ROW_T bid = blockIdx.x; bid < ner; bid += gridDim.x) {

        const OFF_T rbeg = roff[rows[bid]];
        const OFF_T rend = roff[rows[bid] + 1];

        ROW_T firstcol = 0;
        ROW_T lastcol = 0;

        for (OFF_T i = rbeg; i < rend; i += BDIM) {
          const ROW_T c = (i + threadIdx.x < rend) ? cols[i + threadIdx.x] : -1;

          __syncthreads();
          if (c > -1)
            atomicOr(shm + c / BITSOF(shm), 1u << (c % BITSOF(shm)));
          __syncthreads();

#pragma unroll
          for (int j = 0; j < BDIM; j++) {

            const ROW_T curc = __block_bcast(c, j);
            if (curc == -1)
              break;

            lastcol = curc;
            if ((i == rbeg) && !j) {
              firstcol = curc;
              continue;
            }

            const OFF_T soff = roff[curc];
            const OFF_T eoff = roff[curc + 1];
            for (OFF_T k = eoff - 1; k >= soff; k -= BDIM) {
              if (k - (int) threadIdx.x < soff)
                break;

              const ROW_T cc = LDG(cols + k - threadIdx.x);
              if (cc < firstcol)
                break;

              const unsigned int mm = 1u << (cc % BITSOF(shm));
              if (shm[cc / BITSOF(shm)] & mm) {
                __cnt++;
              }
            }
          }
        }
        lastcol /= 64;
        firstcol /= 64;

        __syncthreads();
        if (lastcol - firstcol < rend - rbeg) {
          for (int i = firstcol; i <= lastcol; i += BDIM) {
            if (i + threadIdx.x <= lastcol) {
              ((unsigned long long *) shm)[i + threadIdx.x] = 0ull;
            }
          }
        } else {
          for (int i = rbeg; i < rend; i += BDIM) {
            if (i + threadIdx.x < rend) {
              shm[cols[i + threadIdx.x] / BITSOF(shm)] = 0;
            }
          }
        }
        __syncthreads();
      }
      __cnt = block_sum_sh<BDIM, 1, WSIZE>(__cnt, (uint64_t *) shm);
      if (threadIdx.x == 0)
        ocnt[blockIdx.x] = __cnt;

      return;
    }

    template<typename T>
    void tricnt_bsh(T nblock, spmat_t<T> *m, uint64_t *ocnt_d, size_t bmld, cudaStream_t stream) {

      tricnt_bsh_k<THREADS, 32> <<<nblock, THREADS, sizeof(unsigned int) * bmld, stream>>>(m->nrows,
                                                                                           m->rows_d,
                                                                                           m->roff_d,
                                                                                           m->cols_d,
                                                                                           ocnt_d,
                                                                                           bmld);
      cudaCheckError()
      ;
      return;
    }

////////////////////////////////////////////////////////////////////////////////////////
    template<int WSIZE,
        int NWARP,
        int RLEN_THR1,
        int RLEN_THR2,
        typename ROW_T,
        typename OFF_T,
        typename CNT_T,
        typename MAP_T>
    __global__ void tricnt_wrp_ps_k(const ROW_T ner,
                                    const ROW_T *__restrict__ rows,
                                    const OFF_T *__restrict__ roff,
                                    const ROW_T *__restrict__ cols,
                                    CNT_T *__restrict__ ocnt,
                                    MAP_T *__restrict__ bmap,
                                    const size_t bmld) {

      __shared__ OFF_T sho[NWARP][WSIZE];
      __shared__ ROW_T shs[NWARP][WSIZE];
      __shared__ ROW_T shc[NWARP][WSIZE];

      CNT_T __cnt = 0;
      ROW_T wid = blockIdx.x * blockDim.y + threadIdx.y;

      bmap += bmld * wid;
      for (; wid < ner; wid += gridDim.x * blockDim.y) {

        const OFF_T rbeg = roff[rows[wid]];
        const OFF_T rend = roff[rows[wid] + 1];

        //RLEN_THR1 <= 32
        if (rend - rbeg <= RLEN_THR1) {
          const int nloc = rend - rbeg;

          OFF_T soff;
          OFF_T eoff;
          if (threadIdx.x < nloc) {
            const ROW_T c = cols[rbeg + threadIdx.x];
            shc[threadIdx.y][threadIdx.x] = c;
            soff = roff[c];
            eoff = roff[c + 1];
          }

          int mysm = -1;

          #pragma unroll
          for (int i = 1; i < RLEN_THR1; i++) {

            if (i == nloc)
              break;

            const OFF_T csoff = utils::shfl(soff, i);
            const OFF_T ceoff = utils::shfl(eoff, i);

            if (ceoff - csoff < RLEN_THR2) {
              if (threadIdx.x == i)
                mysm = i;
              continue;
            }
            for (OFF_T k = ceoff - 1; k >= csoff; k -= WSIZE) {
              if (k - (int) threadIdx.x < csoff)
                break;

              const ROW_T cc = cols[k - threadIdx.x];
              if (cc < shc[threadIdx.y][0])
                break;
              for (int j = i - 1; j >= 0; j--) {
                if (cc == shc[threadIdx.y][j]) {
                  __cnt++;
                }
              }
            }
          }
          if (mysm > -1) {
            for (OFF_T k = eoff - 1; k >= soff; k--) {
              const ROW_T cc = cols[k];
              if (cc < shc[threadIdx.y][0])
                break;
              for (int j = mysm - 1; j >= 0; j--) {
                if (cc == shc[threadIdx.y][j]) {
                  __cnt++;
                }
              }
            }
          }
        } else {
          ROW_T firstcol = cols[rbeg];
          ROW_T lastcol = cols[rend - 1];
          for (OFF_T i = rbeg; i < rend; i += 32) {

            const ROW_T c = (i + threadIdx.x < rend) ? cols[i + threadIdx.x] : -1;

            if (c > -1)
              atomicOr(bmap + c / BITSOF(bmap), ((MAP_T) 1) << (c % BITSOF(bmap)));
          }

          for (OFF_T i = rbeg; i < rend; i+= 32) {
            const ROW_T c = (i + threadIdx.x < rend) ? cols[i + threadIdx.x] : -1;
            sho[threadIdx.y][threadIdx.x] = (c > -1) ? roff[c] : 0;
            shc[threadIdx.y][threadIdx.x] = c;

            ROW_T len = (c > -1) ? roff[c + 1] - sho[threadIdx.y][threadIdx.x] : 0;
            ROW_T lensum = len;

            #pragma unroll
            for (int j = 1; j < 32; j <<= 1) {
              lensum += (threadIdx.x >= j) * (utils::shfl_up(lensum, j));
            }
            shs[threadIdx.y][threadIdx.x] = lensum - len;

            lensum = utils::shfl(lensum, 31);

            int k = WSIZE - 1;
            for (int j = lensum - 1; j >= 0; j -= WSIZE) {

              if (j < threadIdx.x)
                break;

              // bisect-right
              for (; k >= 0; k--) {
                if (shs[threadIdx.y][k] <= j - threadIdx.x)
                  break;
              }

              const ROW_T cc = LDG(cols
                  + (sho[threadIdx.y][k] + j - threadIdx.x - shs[threadIdx.y][k]));

              if (cc < shc[threadIdx.y][k])
                continue;
//              if (cc < firstcol)
//                continue;

              const MAP_T mm = ((MAP_T) 1) << (cc % BITSOF(bmap));
              if (bmap[cc / BITSOF(bmap)] & mm) {
                __cnt++;
              }
            }
          }
          lastcol /= 64;
          firstcol /= 64;

          if (lastcol - firstcol < rend - rbeg) {
            for (int i = firstcol; i <= lastcol; i += WSIZE) {
              if (i + threadIdx.x <= lastcol) {
                ((unsigned long long *) bmap)[i + threadIdx.x] = 0ull;
              }
            }
          } else {
            for (int i = rbeg; i < rend; i += WSIZE) {
              if (i + threadIdx.x < rend) {
                bmap[cols[i + threadIdx.x] / BITSOF(bmap)] = 0;
              }
            }
          }
        }
      }
      __syncthreads();
      __cnt = block_sum<WSIZE, NWARP, WSIZE>(__cnt);
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ocnt[blockIdx.x] = __cnt;
      }
      return;
    }

    template<typename T>
    void tricnt_wrp(T nblock,
                    spmat_t<T> *m,
                    uint64_t *ocnt_d,
                    unsigned int *bmap_d,
                    size_t bmld,
                    cudaStream_t stream) {

      dim3 block(32, THREADS / 32);
      tricnt_wrp_ps_k<32, THREADS / 32, WP_LEN_TH1, WP_LEN_TH2> <<<nblock, block, 0, stream>>>(m->nrows,
                                                                                               m->rows_d,
                                                                                               m->roff_d,
                                                                                               m->cols_d,
                                                                                               ocnt_d,
                                                                                               bmap_d,
                                                                                               bmld);
      cudaCheckError();
      return;
    }

//////////////////////////////////////////////////////////////////////////////////////////
    template<int BDIM,
        int LOCLEN,
        typename ROW_T,
        typename OFF_T,
        typename CNT_T>
    __global__ void tricnt_thr_k(const ROW_T ner,
                                 const ROW_T *__restrict__ rows,
                                 const OFF_T *__restrict__ roff,
                                 const ROW_T *__restrict__ cols,
                                 CNT_T *__restrict__ ocnt) {
      CNT_T __cnt = 0;
      const ROW_T tid = blockIdx.x * BDIM + threadIdx.x;

      for (ROW_T rid = tid; rid < ner; rid += gridDim.x * BDIM) {

        const ROW_T r = rows[rid];

        const OFF_T rbeg = roff[r];
        const OFF_T rend = roff[r + 1];
        const ROW_T rlen = rend - rbeg;

        if (!rlen)
          continue;
        if (rlen <= LOCLEN) {
          int nloc = 0;
          ROW_T loc[LOCLEN];

#pragma unroll
          for (nloc = 0; nloc < LOCLEN; nloc++) {
            if (rbeg + nloc >= rend)
              break;
            loc[nloc] = LDG(cols + rbeg + nloc);
          }

#pragma unroll
          for (int i = 1; i < LOCLEN; i++) {

            if (i == nloc)
              break;

            const ROW_T c = loc[i];
            const OFF_T soff = roff[c];
            const OFF_T eoff = roff[c + 1];

            for (OFF_T k = eoff - 1; k >= soff; k--) {

              const ROW_T cc = LDG(cols + k);
              if (cc < loc[0])
                break;

              for (int j = i - 1; j >= 0; j--) {
                if (cc == loc[j])
                  __cnt++;
              }
            }
          }
        } else {
          const ROW_T minc = cols[rbeg];
          for (int i = 1; i < rlen; i++) {

            const ROW_T c = LDG(cols + rbeg + i);
            const OFF_T soff = roff[c];
            const OFF_T eoff = roff[c + 1];

            for (OFF_T k = eoff - 1; k >= soff; k--) {

              const ROW_T cc = LDG(cols + k);
              if (cc < minc)
                break;

              for (int j = i - 1; j >= 0; j--) {
                if (cc == LDG(cols + rbeg + j))
                  __cnt++;
              }
            }
          }
        }
      }

      __syncthreads();
      __cnt = block_sum<BDIM, 1, 32>(__cnt);
      if (threadIdx.x == 0)
        ocnt[blockIdx.x] = __cnt;

      return;
    }

    template<typename T>
    void tricnt_thr(T nblock, spmat_t<T> *m, uint64_t *ocnt_d, cudaStream_t stream) {

      cudaFuncSetCacheConfig(tricnt_thr_k<THREADS, TH_CENT_K_LOCLEN, typename type_utils<T>::LOCINT,
                                 typename type_utils<T>::LOCINT, uint64_t>,
                             cudaFuncCachePreferL1);

      tricnt_thr_k<THREADS, TH_CENT_K_LOCLEN> <<<nblock, THREADS, 0, stream>>>(m->nrows, m->rows_d,
                                                                               m->roff_d,
                                                                               m->cols_d,
                                                                               ocnt_d);
      cudaCheckError()
      ;
      return;
    }

/////////////////////////////////////////////////////////////////
    __global__ void myset(unsigned long long *p, unsigned long long v, long long n) {
      const long long tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) {
        p[tid] = v;
      }
      return;
    }

    void myCudaMemset(unsigned long long *p,
                      unsigned long long v,
                      long long n,
                      cudaStream_t stream) {
      if (n <= 0)
        return;
      myset<<<DIV_UP(n, THREADS), THREADS, 0, stream>>>(p, v, n);
      cudaCheckError();
    }

    template<typename IndexType>
    struct NonEmptyRow
    {
      const IndexType* p_roff;
      __host__ __device__ NonEmptyRow(const IndexType* roff) :
          p_roff(roff) {
      }
      __host__ __device__ __forceinline__
      bool operator()(const IndexType &a) const
                      {
        return (p_roff[a] < p_roff[a + 1]);
      }
    };

    template<typename T>
    void create_nondangling_vector(const T* roff,
                                   T *p_nonempty,
                                   T *n_nonempty,
                                   size_t n,
                                   cudaStream_t stream)
                                   {
      if (n <= 0)
        return;
      thrust::counting_iterator<T> it(0);
      NonEmptyRow<T> temp_func(roff);
      T* d_out_num = (T*) get_temp_storage(sizeof(*n_nonempty), stream);

      cubIf(it, p_nonempty, d_out_num, n, temp_func, stream);
      cudaMemcpy(n_nonempty, d_out_num, sizeof(*n_nonempty), cudaMemcpyDeviceToHost);
      cudaCheckError();
      free_temp_storage(d_out_num, stream);
      cudaCheckError();
    }

    template<typename T>
    uint64_t reduce(uint64_t *v_d, T n, cudaStream_t stream) {

      uint64_t n_h;
      uint64_t *n_d = (uint64_t *) get_temp_storage(sizeof(*n_d), stream);

      cubSum(v_d, n_d, n, stream);
      cudaCheckError();
      cudaMemcpy(&n_h, n_d, sizeof(*n_d), cudaMemcpyDeviceToHost);
      cudaCheckError();
      free_temp_storage(n_d, stream);

      return n_h;
    }

// instantiate for int
    template void tricnt_bsh<int>(int nblock,
                                  spmat_t<int> *m,
                                  uint64_t *ocnt_d,
                                  size_t bmld,
                                  cudaStream_t stream);
    template void tricnt_wrp<int>(int nblock,
                                  spmat_t<int> *m,
                                  uint64_t *ocnt_d,
                                  unsigned int *bmap_d,
                                  size_t bmld,
                                  cudaStream_t stream);
    template void tricnt_thr<int>(int nblock,
                                  spmat_t<int> *m,
                                  uint64_t *ocnt_d,
                                  cudaStream_t stream);
    template void tricnt_b2b<int>(int nblock,
                                  spmat_t<int> *m,
                                  uint64_t *ocnt_d,
                                  unsigned int *bmapL0_d,
                                  size_t bmldL0,
                                  unsigned int *bmapL1_d,
                                  size_t bmldL1,
                                  cudaStream_t stream);

    template uint64_t reduce<int>(uint64_t *v_d, int n, cudaStream_t stream);
    template void create_nondangling_vector<int>(const int *roff,
                                                 int *p_nonempty,
                                                 int *n_nonempty,
                                                 size_t n,
                                                 cudaStream_t stream);

  } // end namespace triangle counting

} // end namespace nvgraph
