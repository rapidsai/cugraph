// -*-c++-*-

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

#pragma once

#include <stdlib.h>
#include <omp.h>

#include "binning.cuh"
#include <cub/cub.cuh>

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include "utilities/error_utils.h"
#include "rmm_utils.h"

namespace cusort {

  namespace detail {
    //
    //  Define a device function to count leading zeros, since
    //  the intrinsic is different for each type.
    //
    //  Note, C++ doesn't currently support partial template
    //  specialization, so this is done with a function object.
    //
    template <typename Key_t, int size>
    struct CountLeadingZeros {
      __inline__ __device__ int operator()(Key_t k) {
        return __clz(k);
      }
    };

    template <typename Key_t>
    struct CountLeadingZeros<Key_t, 8> {
      __inline__ __device__ int operator()(Key_t k) {
        return __clzll(k);
      }
    };
  }
  
  template <typename Key_t,typename Value_t, typename Length_t,
            int MAX_NUM_GPUS = 16, int BIN_SCALE = 16,
            int BLOCK_DIM = 128, int MEM_ALIGN = 512>
  class Cusort {
  public:
    Cusort() {
      memset(h_max_key, 0, sizeof(Key_t) * MAX_NUM_GPUS);
      memset(h_readPositions, 0, sizeof(Length_t) * (MAX_NUM_GPUS + 1) * (MAX_NUM_GPUS + 1));
      memset(h_writePositions, 0, sizeof(Length_t) * (MAX_NUM_GPUS + 1) * (MAX_NUM_GPUS + 1));
      memset(h_writePositionsTransposed, 0, sizeof(Length_t) * (MAX_NUM_GPUS + 1) * (MAX_NUM_GPUS + 1));
      memset(h_binMap, 0, sizeof(unsigned char) * (1 << BIN_SCALE));
    }
    
    // This structure is used for allocating memory once for CUB's sorting function. 
    class BufferData {
    public:
      Key_t         *d_keys;
      Value_t       *d_vals;
      Length_t       h_length;
      unsigned char *buffer;
      unsigned char *cubBuffer;

      BufferData(): d_keys(nullptr), d_vals(nullptr), h_length(0), buffer(nullptr), cubBuffer(nullptr) {}

      void allocate(Length_t len, Length_t cubData) {
        Length_t cubDataSize = ((cubData + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        Length_t sdSize = ((len + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        Length_t startingPoint = sdSize * sizeof(Key_t);         
        Length_t sdSize2 =  startingPoint + sdSize * sizeof(Value_t);

        ALLOC_TRY(&buffer, cubDataSize + sdSize2, nullptr);

        d_keys = (Key_t *) buffer;
        d_vals = (Value_t *) (buffer + startingPoint);
        cubBuffer = buffer + sdSize2;
        h_length = len;        
      }

      void allocate_keys_only(Length_t len, Length_t cubData) {
        Length_t cubDataSize = ((cubData + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        Length_t sdSize = ((len + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        Length_t startingPoint = sdSize * sizeof(Key_t);         

        ALLOC_TRY(&buffer, cubDataSize + startingPoint, nullptr);

        d_keys = (Key_t *) buffer;
        cubBuffer = buffer + startingPoint;
        h_length = len;

        
      }

      void free() {
        if (buffer != nullptr)
          ALLOC_FREE_TRY(buffer, nullptr);

        
      }
    };

    // template <typename Key_t, typename Value_t, typename Length_t>
    struct ThreadData {
      Key_t          *d_input_keys;
      Value_t        *d_input_values;
      Length_t        h_input_length;
      Key_t          *d_output_keys;
      Value_t        *d_output_values;
      Length_t        h_output_length;
      BufferData      bdReorder;

      // Device data -- accessible to a specific GPU\Device
      unsigned char  *buffer;
      Length_t       *binSizes;
      Length_t       *binPrefix;
      Length_t       *tempPrefix;
      unsigned char  *binMap;
      Key_t          *binSplitters;
      unsigned char  *cubSmallBuffer;

      size_t          cubSortBufferSize;

      // Host data -- accessible to all threads on the CPU
      Length_t       *h_binSizes;
      Length_t       *h_binPrefix;

      ThreadData(): d_input_keys(nullptr), d_input_values(nullptr), h_input_length(0),
                    d_output_keys(nullptr), d_output_values(nullptr), h_output_length(0),
                    bdReorder(), buffer(nullptr), binSizes(nullptr), binPrefix(nullptr),
                    tempPrefix(nullptr), binMap(nullptr), binSplitters(nullptr),
                    cubSmallBuffer(nullptr), cubSortBufferSize(0), h_binSizes(nullptr),
                    h_binPrefix(nullptr) {}

      void allocate(int32_t num_bins, int num_gpus) {
        Length_t binsAligned = ((num_bins + 1 + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        Length_t gpusAligned = ((num_gpus + 1 + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;

        Length_t mallocSizeBytes = 
          (binsAligned + binsAligned + gpusAligned) * sizeof(Length_t) + 
          gpusAligned * sizeof(Key_t) + 
          binsAligned +
          (1L << BIN_SCALE); // cubSmallBuffer;

        ALLOC_TRY(&buffer, mallocSizeBytes, nullptr);

        int64_t pos = 0;

        binSizes = (Length_t*) (buffer + pos);
        pos += (sizeof(Length_t) * binsAligned);

        binPrefix = (Length_t*) (buffer + pos);
        pos += (sizeof(Length_t) * binsAligned);

        tempPrefix = (Length_t*) (buffer + pos);
        pos += (sizeof(Length_t) * gpusAligned);

        binSplitters = (Key_t*) (buffer + pos);
        pos += (sizeof(Key_t) * gpusAligned);

        binMap = buffer + pos;
        pos += binsAligned;

        cubSmallBuffer = buffer + pos;

        CUDA_TRY(cudaMemset(binSizes, 0, (num_bins + 1) * sizeof(Key_t)));

        bdReorder.buffer = nullptr;
        bdReorder.d_keys = nullptr;
        bdReorder.d_vals = nullptr;
        bdReorder.h_length = 0;

        // Host memory allocations
        h_binSizes  = new Length_t[num_bins + 1];
        h_binPrefix = new Length_t[num_bins + 1];

        
      }

      void free() {
        ALLOC_FREE_TRY(buffer, nullptr);

        delete [] h_binSizes;
        delete [] h_binPrefix;

        
      }
    };

    void sort_one(ThreadData *tData, Length_t average_array_size, int cpu_tid, int num_gpus, bool keys_only) {
      Key_t * d_max = nullptr;
      void * d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      int num_bins = (1 << BIN_SCALE);
      Length_t blocks = (tData[cpu_tid].h_input_length + BLOCK_DIM - 1) / BLOCK_DIM;

      //
      //  First order of business is to compute the range
      //  of values.  Binning and load balancing will be
      //  suboptimal if the data is skewed, so let's find
      //  the maximum value of our data (actually, we want
      //  the number of leading zeros in the maximum value).
      //

      //
      //  Use binSplitters (not needed until later) to compute the max
      //
      d_max = tData[cpu_tid].binSplitters;

      cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tData[cpu_tid].d_input_keys, d_max, tData[cpu_tid].h_input_length);

      ALLOC_TRY(&d_temp_storage, temp_storage_bytes, nullptr);
      cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tData[cpu_tid].d_input_keys, d_max, tData[cpu_tid].h_input_length);

      thrust::for_each_n(thrust::device,
                         d_max, 1,
                         [d_max] __device__ (Key_t &val) {
                           d_max[0] = detail::CountLeadingZeros<Key_t, sizeof(Key_t)>()(d_max[0]);
                         });

      CUDA_TRY(cudaMemcpy(h_max_key + cpu_tid, d_max, sizeof(Key_t), cudaMemcpyDeviceToHost));

      ALLOC_FREE_TRY(d_temp_storage, nullptr);

#pragma omp barrier

#pragma omp master
      {
        //
        //  Reduce across parallel regions and share
        //  the number of leading zeros of the global
        //  maximum
        //
        Key_t local_max = h_max_key[0];

        for (int i = 1 ; i < num_gpus ; ++i)
          local_max = max(local_max, h_max_key[i]);

        for (int i = 0 ; i < num_gpus ; ++i)
          h_max_key[i] = local_max;
      }

      //
      //  SkipNBits will skip the leading zeros
      //
      SkipNBits<Key_t, Length_t> computeBin(BIN_SCALE, h_max_key[cpu_tid]);
      
      binCounting<<<blocks,BLOCK_DIM>>>(tData[cpu_tid].d_input_keys,
                                        tData[cpu_tid].h_input_length, 
                                        tData[cpu_tid].binSizes,
                                        computeBin);

      //
      //  NOTE: this assumes 2^16 bins
      //
      temp_storage_bytes = 2047;

      cub::DeviceScan::ExclusiveSum(tData[cpu_tid].cubSmallBuffer, temp_storage_bytes,
                                    tData[cpu_tid].binSizes, tData[cpu_tid].binPrefix, num_bins + 1);

      CUDA_TRY(cudaMemcpy(tData[cpu_tid].h_binPrefix, tData[cpu_tid].binPrefix, (num_bins+1)*sizeof(Length_t), cudaMemcpyDeviceToHost));

#pragma omp barrier

#pragma omp master
      {
        //
        //  Rewrote this logic.  This could move to the masters'
        //  GPU, perhaps that would speed things up (we have
        //  several loops over num_bins that could be parallelized).
        //
        //  At the moment, this section seems fast enough.
        //
        memset(h_readPositions, 0, (num_gpus + 1) * (num_gpus + 1) * sizeof(Length_t));
        memset(h_writePositions, 0, (num_gpus + 1) * (num_gpus + 1) * sizeof(Length_t));

        Length_t binSplits[num_gpus + 1] = { 0 };
        Length_t globalPrefix[num_bins + 1];


        // Computing global prefix sum array to find partition points.
        globalPrefix[0] = 0;
        
        for (int b = 0 ; b < num_bins ; ++b) {
          globalPrefix[b+1] = globalPrefix[b];

          for (int g = 0 ; g < num_gpus ; ++g) {
            globalPrefix[b+1] += (tData[g].h_binPrefix[b+1]
                                  - tData[g].h_binPrefix[b]);
          }
        }

        for (int b = 0 ; b < num_bins ; ++b) {
          unsigned char ttt = globalPrefix[b] / average_array_size;
          h_binMap[b] = ttt;

          if (binSplits[h_binMap[b]] == 0)
            binSplits[h_binMap[b]] = b;
        }

        //
        //  Overwrite binSplits[0] with 0 again
        //
        binSplits[0] = 0;

        //
        //  It's possible we had a large bin near the
        //  end, we want to make sure that all entries
        //  after h_binMap[num_bins-1] point to the last
        //  entry
        //
        for (int i = h_binMap[num_bins-1] ; i < num_gpus ; ++i)
          binSplits[i+1] = num_bins;

        // Each thread (row) knows the length of the partitions it needs to write to the other threads
        for (int r = 0 ; r < num_gpus ; ++r) {
          for (int c = 0 ; c < num_gpus ; ++c) {
            h_readPositions[r+1][c+1] = tData[r].h_binPrefix[binSplits[c+1]];
          }
        }

        // Each thread learns the position in the array other threads inputKey that it will copy its data into
        for (int r = 0 ; r < num_gpus ; ++r) {
          for (int c = 0 ; c < num_gpus ; ++c) {
            h_writePositions[r+1][c] = h_writePositions[r][c] + (h_readPositions[r+1][c+1] - h_readPositions[r+1][c]);
          }
        }

        for (int r = 0 ; r < num_gpus ; ++r) {
          for (int c = 0 ; c <= num_gpus ; ++c) {
            h_writePositionsTransposed[r][c] = h_writePositions[c][r];
          }
        }

        for (int r = 0 ; r < num_gpus ; ++r) {
          for (int c = 0 ; c <= num_gpus ; ++c) {
            h_writePositionsTransposed[r][c] = h_writePositions[c][r];
          }
        }
      }

#pragma omp barrier

      CUDA_TRY(cudaMemcpy(tData[cpu_tid].binMap, h_binMap, num_bins * sizeof(unsigned char), cudaMemcpyHostToDevice));
      CUDA_TRY(cudaMemcpy(tData[cpu_tid].tempPrefix, h_readPositions[cpu_tid+1], (num_gpus + 1) * sizeof(Length_t), cudaMemcpyHostToDevice));

      //
      // Creating a temporary buffer that will be used for both reordering the input in the binning phase
      // and possibly in the sorting phase if CUB's sort is used. 
      // Therefore, the maximal buffer size is taken in this phase, where max=(array size of input, array size of output) 
      //
      Length_t elements = std::max(tData[cpu_tid].h_input_length, h_writePositionsTransposed[cpu_tid][num_gpus]);

      if (elements > (1L << 31)) {
        CUGRAPH_FAIL("input column is too big");
      }

      tData[cpu_tid].cubSortBufferSize = 0;

      if (keys_only) {
        cub::DeviceRadixSort::SortKeys<Key_t>(nullptr, tData[cpu_tid].cubSortBufferSize,
                                              nullptr, nullptr, elements);

        tData[cpu_tid].bdReorder.allocate_keys_only(h_writePositionsTransposed[cpu_tid][num_gpus], tData[cpu_tid].cubSortBufferSize);
      } else {
        cub::DeviceRadixSort::SortPairs<Key_t,Value_t>(nullptr, tData[cpu_tid].cubSortBufferSize,
                                                       nullptr, nullptr, nullptr, nullptr, elements);

        tData[cpu_tid].bdReorder.allocate(h_writePositionsTransposed[cpu_tid][num_gpus], tData[cpu_tid].cubSortBufferSize);
      }

      tData[cpu_tid].h_output_length = h_writePositionsTransposed[cpu_tid][num_gpus];
      cudaDeviceSynchronize();
      CUDA_CHECK_LAST();

#pragma omp barrier

      if (keys_only) {
        partitionRelabel<32, BLOCK_DIM> <<<blocks,BLOCK_DIM>>>
          (tData[cpu_tid].d_input_keys,
           tData[cpu_tid].bdReorder.d_keys,
           tData[cpu_tid].h_input_length,
           tData[cpu_tid].tempPrefix,
           computeBin,
           tData[cpu_tid].binMap,
           num_gpus);
      } else {
        partitionRelabel<32, BLOCK_DIM> <<<blocks,BLOCK_DIM>>>
          (tData[cpu_tid].d_input_keys,
           tData[cpu_tid].bdReorder.d_keys,
           tData[cpu_tid].d_input_values,
           tData[cpu_tid].bdReorder.d_vals,
           tData[cpu_tid].h_input_length,
           tData[cpu_tid].tempPrefix,
           computeBin,
           tData[cpu_tid].binMap,
           num_gpus);
      }

      CUDA_CHECK_LAST();

      ALLOC_TRY(&(tData[cpu_tid].d_output_keys), tData[cpu_tid].h_output_length * sizeof(Key_t), nullptr);

      if (!keys_only)
        ALLOC_TRY(&(tData[cpu_tid].d_output_values), tData[cpu_tid].h_output_length * sizeof(Value_t), nullptr);

      CUDA_CHECK_LAST();

      //
      //  Need all partition labeling to complete before we start copying data
      //
#pragma omp barrier

      for (int other = 0 ; other < num_gpus ; ++other) {
        int from_id = (cpu_tid + other) % num_gpus;

        CUDA_TRY(cudaMemcpyAsync(tData[cpu_tid].d_output_keys + h_writePositionsTransposed[cpu_tid][from_id],
                                 tData[from_id].bdReorder.d_keys + h_readPositions[from_id+1][cpu_tid],
                                 (h_readPositions[from_id+1][cpu_tid+1] - h_readPositions[from_id+1][cpu_tid]) * sizeof(Key_t),
                                 cudaMemcpyDeviceToDevice));

        if (!keys_only)
          CUDA_TRY(cudaMemcpyAsync(tData[cpu_tid].d_output_values + h_writePositionsTransposed[cpu_tid][from_id],
                                   tData[from_id].bdReorder.d_vals + h_readPositions[from_id+1][cpu_tid],
                                   (h_readPositions[from_id+1][cpu_tid+1] - h_readPositions[from_id+1][cpu_tid]) * sizeof(Value_t),
                                   cudaMemcpyDeviceToDevice));

      }
      cudaDeviceSynchronize();

#pragma omp barrier

      if (keys_only) {
        d_temp_storage = (void*) tData[cpu_tid].bdReorder.cubBuffer;
        cub::DeviceRadixSort::SortKeys<Key_t>(d_temp_storage,
                                              tData[cpu_tid].cubSortBufferSize,
                                              tData[cpu_tid].d_output_keys,
                                              tData[cpu_tid].bdReorder.d_keys, 
                                              tData[cpu_tid].h_output_length);
      } else {
        d_temp_storage = (void*) tData[cpu_tid].bdReorder.cubBuffer;
        cub::DeviceRadixSort::SortPairs<Key_t,Value_t>(d_temp_storage,
                                                       tData[cpu_tid].cubSortBufferSize,
                                                       tData[cpu_tid].d_output_keys,
                                                       tData[cpu_tid].bdReorder.d_keys, 
                                                       tData[cpu_tid].d_output_values,
                                                       tData[cpu_tid].bdReorder.d_vals,
                                                       tData[cpu_tid].h_output_length);
      }

      CUDA_CHECK_LAST();
      cudaDeviceSynchronize();

      CUDA_TRY(cudaMemcpy(tData[cpu_tid].d_output_keys, tData[cpu_tid].bdReorder.d_keys, tData[cpu_tid].h_output_length * sizeof(Key_t), cudaMemcpyDeviceToDevice));

      if (!keys_only)
        CUDA_TRY(cudaMemcpy(tData[cpu_tid].d_output_values, tData[cpu_tid].bdReorder.d_vals, tData[cpu_tid].h_output_length * sizeof(Value_t), cudaMemcpyDeviceToDevice));

      cudaDeviceSynchronize();

      
    }

    void sort(Key_t **d_input_keys,
                   Value_t **d_input_values,
                   Length_t *h_input_partition_offsets,
                   Key_t **d_output_keys,
                   Value_t **d_output_values,
                   Length_t *h_output_partition_offsets,
                   int num_gpus = 1) {

      if (num_gpus > MAX_NUM_GPUS) {
        CUGRAPH_FAIL("num_gpus > MAX_NUM_GPUS");
      }

      if ((sizeof(Key_t) != 8) && (sizeof(Key_t) != 4)) {
        CUGRAPH_FAIL("Unsupported data type");
      }

      ThreadData tData[num_gpus];

      Length_t keyCount = h_input_partition_offsets[num_gpus];

      // Used for partitioning the output and ensuring that each GPU sorts a near equal number of elements.
      Length_t average_array_size = (keyCount + num_gpus - 1) / num_gpus;

      int original_number_threads = 0;
#pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          original_number_threads = omp_get_num_threads();
      }

      omp_set_num_threads(num_gpus);

#pragma omp parallel
      {
        int cpu_tid = omp_get_thread_num();
        cudaSetDevice(cpu_tid);

        tData[cpu_tid].h_input_length = h_input_partition_offsets[cpu_tid+1] - h_input_partition_offsets[cpu_tid];
        tData[cpu_tid].d_input_keys = d_input_keys[cpu_tid];
        tData[cpu_tid].d_input_values = d_input_values[cpu_tid];

        tData[cpu_tid].allocate(1 << BIN_SCALE, num_gpus);

        sort_one(tData, average_array_size, cpu_tid, num_gpus, false);

        tData[cpu_tid].bdReorder.free();
        tData[cpu_tid].free();

        d_output_keys[cpu_tid] = tData[cpu_tid].d_output_keys;
        d_output_values[cpu_tid] = tData[cpu_tid].d_output_values;
      }

      //
      //  Restore the OpenMP configuration
      //
      omp_set_num_threads(original_number_threads);

      h_output_partition_offsets[0] = Length_t{0};
      for (int i = 0 ; i < num_gpus ; ++i)
        h_output_partition_offsets[i+1] = h_output_partition_offsets[i] + tData[i].h_output_length;
    }

    void sort(Key_t **d_input_keys,
                   Length_t *h_input_partition_offsets,
                   Key_t **d_output_keys,
                   Length_t *h_output_partition_offsets,
                   int num_gpus = 1) {

      if (num_gpus > MAX_NUM_GPUS) {
        CUGRAPH_FAIL("num_gpus > MAX_NUM_GPUS in sort");
      }

      if ((sizeof(Key_t) != 8) && (sizeof(Key_t) != 4)) {
        CUGRAPH_FAIL("Unsupported data type");
      }

      ThreadData tData[num_gpus];

      Length_t keyCount = h_input_partition_offsets[num_gpus];

      // Used for partitioning the output and ensuring that each GPU sorts a near equal number of elements.
      Length_t average_array_size = (keyCount + num_gpus - 1) / num_gpus;

      int original_number_threads = 0;
#pragma omp parallel
      {
        if (omp_get_thread_num() == 0)
          original_number_threads = omp_get_num_threads();
      }

      omp_set_num_threads(num_gpus);

#pragma omp parallel
      {
        int cpu_tid = omp_get_thread_num();
        cudaSetDevice(cpu_tid);

        tData[cpu_tid].h_input_length = h_input_partition_offsets[cpu_tid+1] - h_input_partition_offsets[cpu_tid];
        tData[cpu_tid].d_input_keys = d_input_keys[cpu_tid];

        tData[cpu_tid].allocate(1 << BIN_SCALE, num_gpus);

        sort_one(tData, average_array_size, cpu_tid, num_gpus, true);

        tData[cpu_tid].bdReorder.free();
        tData[cpu_tid].free();

        d_output_keys[cpu_tid] = tData[cpu_tid].d_output_keys;
      }

      //
      //  Restore the OpenMP configuration
      //
      omp_set_num_threads(original_number_threads);

      h_output_partition_offsets[0] = Length_t{0};
      for (int i = 0 ; i < num_gpus ; ++i)
        h_output_partition_offsets[i+1] = h_output_partition_offsets[i] + tData[i].h_output_length;

    }

  private:
    Key_t         h_max_key[MAX_NUM_GPUS];
    Length_t      h_readPositions[MAX_NUM_GPUS + 1][MAX_NUM_GPUS + 1];
    Length_t      h_writePositions[MAX_NUM_GPUS + 1][MAX_NUM_GPUS + 1];
    Length_t      h_writePositionsTransposed[MAX_NUM_GPUS + 1][MAX_NUM_GPUS + 1];
    unsigned char h_binMap[1 << BIN_SCALE];
  };
}
