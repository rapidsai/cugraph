// -*-c++-*-

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

  template <typename Key_t, int size>
  __inline__ __device__ int count_leading_zeros(Key_t k) {
    return __clz(k);
  }

  template <typename Key_t>
  __inline__ __device__ int count_leading_zeros<Key_t, 8>(Key_t k) {
    return __clzll(k);
  }
  
  template <typename Key_t,typename Value_t, typename Length_t,
            int MAX_NUM_GPUS = 16, int BIN_SCALE = 16,
            int BLOCK_DIM = 128, int MEM_ALIGN = 512>
  class Cusort {
  public:
    // This structure is used for allocating memory once for CUB's sorting function. 
    class BufferData {
    public:
      Key_t         *d_keys;
      Value_t       *d_vals;
      Length_t       h_length;
      unsigned char *buffer;
      unsigned char *cubBuffer;

      gdf_error allocate(Length_t len, Length_t cubData) {
        Length_t cubDataSize = ((cubData + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        Length_t sdSize = ((len + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        Length_t startingPoint = sdSize * sizeof(Key_t);         
        Length_t sdSize2 =  startingPoint + sdSize * sizeof(Value_t);

        ALLOC_TRY(&buffer, cubDataSize + sdSize2, nullptr);

        d_keys = (Key_t *) buffer;
        d_vals = (Value_t *) (buffer + startingPoint);
        cubBuffer = buffer + sdSize2;
        h_length = len;

        return GDF_SUCCESS;
      }

      gdf_error free() {
        if (buffer != nullptr)
          ALLOC_FREE_TRY(buffer, nullptr);

        return GDF_SUCCESS;
      }
    };

    // template <typename Key_t, typename Value_t, typename Length_t>
    struct ThreadData {
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

      gdf_error allocate(int32_t numBins, int numGPUs) {
        Length_t binsAligned = ((numBins + 1 + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        Length_t gpusAligned = ((numGPUs + 1 + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;

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

        CUDA_TRY(cudaMemset(binSizes, 0, (numBins + 1) * sizeof(Key_t)));

        bdReorder.buffer = nullptr;
        bdReorder.d_keys = nullptr;
        bdReorder.d_vals = nullptr;
        bdReorder.h_length = 0;

        // Host memory allocations
        h_binSizes  = new Length_t[numBins + 1];
        h_binPrefix = new Length_t[numBins + 1];

        return GDF_SUCCESS;
      }

      gdf_error free() {
        ALLOC_FREE_TRY(buffer, nullptr);

        delete [] h_binSizes;
        delete [] h_binPrefix;

        return GDF_SUCCESS;
      }
    };

    gdf_error sort(Key_t **d_input_keys,
                   Value_t **d_input_values,
                   Length_t *h_input_partition_offsets,
                   Key_t **d_output_keys,
                   Value_t **d_output_values,
                   Length_t *h_output_partition_offsets,
                   int numGPUs = 1,
                   Length_t binScale = BIN_SCALE,
                   bool useThrust = false) {

      if (numGPUs > MAX_NUM_GPUS) {
        return GDF_C_ERROR;  // TODO: There are no existing SNMG errors, should be its own error, I think
      }

      if ((sizeof(Key_t) != 8) && (sizeof(Key_t) != 4)) {
        return GDF_UNSUPPORTED_DTYPE;
      }

      int numBins = 1 << BIN_SCALE;
      Length_t keyCount = h_input_partition_offsets[numGPUs];

      // Used for partitioning the output and ensuring that each GPU sorts a near equal number of elements.
      Length_t avgArraySize = (keyCount + numGPUs - 1) / numGPUs;

      ThreadData tData[numGPUs];
      Key_t  max_key[numGPUs];

      Length_t h_readPositions[numGPUs+1][numGPUs+1];
      Length_t h_writePositions[numGPUs+1][numGPUs+1];
      Length_t h_writePositionsTransposed[numGPUs+1][numGPUs+1];
      Length_t h_lengths[numGPUs];

      unsigned char h_binMap[numBins] = {0};

      omp_set_num_threads(numGPUs);

#pragma omp parallel
      {         
        int cpu_tid = omp_get_thread_num();

        cudaSetDevice(cpu_tid);
        tData[cpu_tid].allocate(numBins, numGPUs);

        Length_t arraySize = h_input_partition_offsets[cpu_tid+1] - h_input_partition_offsets[cpu_tid];
        Length_t blocks = (arraySize + BLOCK_DIM - 1) / BLOCK_DIM;

        Key_t * d_max = nullptr;
        void * d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        //
        //  First order of business is to compute the range
        //  of values.  Binning and load balancing will be
        //  suboptimal if the data is skewed, so let's find
        //  the maximum value of our data (actually, we want
        //  the number of leading zeros in the maximum value).
        //
        ALLOC_TRY(&d_max, sizeof(Key_t), nullptr);

        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input_keys[cpu_tid], d_max, arraySize);

        ALLOC_TRY(&d_temp_storage, temp_storage_bytes, nullptr);
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_input_keys[cpu_tid], d_max, arraySize);

        thrust::for_each_n(thrust::device,
                           d_max, 1,
                           [d_max] __device__ (Key_t &val) {
                             d_max[0] = count_leading_zeros<Key_t, sizeof(Key_t)>(d_max[0]);
                           });

        cudaMemcpy(max_key + cpu_tid, d_max, sizeof(Key_t), cudaMemcpyDeviceToHost);

        ALLOC_FREE_TRY(d_max, nullptr);
        ALLOC_FREE_TRY(d_temp_storage, nullptr);

#pragma omp barrier

#pragma omp master
        {
          //
          //  Reduce across parallel regions and share
          //  the number of leading zeros of the global
          //  maximum
          //
          Key_t local_max = max_key[0];

          for (int i = 1 ; i < numGPUs ; ++i)
            local_max = max(local_max, max_key[i]);

          for (int i = 0 ; i < numGPUs ; ++i)
            max_key[i] = local_max;
        }

        //
        //  SkipNBits will skip the leading zeros
        //
        SkipNBits<Key_t, Length_t> computeBin(binScale, max_key[cpu_tid]);
      
        binCounting<<<blocks,BLOCK_DIM>>>(d_input_keys[cpu_tid],
                                          arraySize, 
                                          tData[cpu_tid].binSizes,
                                          computeBin);

        //
        //  NOTE: this assumes 2^16 bins
        //
        temp_storage_bytes = 2047;

        cub::DeviceScan::ExclusiveSum(tData[cpu_tid].cubSmallBuffer, temp_storage_bytes,
                                      tData[cpu_tid].binSizes, tData[cpu_tid].binPrefix, numBins + 1);

        cudaMemcpy(tData[cpu_tid].h_binPrefix, tData[cpu_tid].binPrefix, (numBins+1)*sizeof(Length_t), cudaMemcpyDeviceToHost);

#pragma omp barrier

#pragma omp master
        {
          //
          //  Rewrote this logic.  This could move to the masters'
          //  GPU, perhaps that would speed things up (we have
          //  several loops over numBins that could be parallelized).
          //
          //  At the moment, this section seems fast enough.
          //
          memset(h_readPositions, 0, (numGPUs + 1) * (numGPUs + 1) * sizeof(Length_t));
          memset(h_writePositions, 0, (numGPUs + 1) * (numGPUs + 1) * sizeof(Length_t));

          Length_t binSplits[numGPUs + 1] = { 0 };
          Length_t globalPrefix[numBins + 1];


          // Computing global prefix sum array to find partition points.
          globalPrefix[0] = 0;
        
          for (int b = 0 ; b < numBins ; ++b) {
            globalPrefix[b+1] = globalPrefix[b];

            for (int g = 0 ; g < numGPUs ; ++g) {
              globalPrefix[b+1] += (tData[g].h_binPrefix[b+1]
                                    - tData[g].h_binPrefix[b]);
            }
          }

          for (int b = 0 ; b < numBins ; ++b) {
            unsigned char ttt = globalPrefix[b] / avgArraySize;
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
          //  after h_binMap[numBins-1] point to the last
          //  entry
          //
          for (int i = h_binMap[numBins-1] ; i < numGPUs ; ++i)
            binSplits[i+1] = numBins;

          // Each thread (row) knows the length of the partitions it needs to write to the other threads
          for (int r = 0 ; r < numGPUs ; ++r) {
            for (int c = 0 ; c < numGPUs ; ++c) {
              h_readPositions[r+1][c+1] = tData[r].h_binPrefix[binSplits[c+1]];
            }
          }

          // Each thread learns the position in the array other threads inputKey that it will copy its data into
          for (int r = 0 ; r < numGPUs ; ++r) {
            for (int c = 0 ; c < numGPUs ; ++c) {
              h_writePositions[r+1][c] = h_writePositions[r][c] + (h_readPositions[r+1][c+1] - h_readPositions[r+1][c]);
            }
          }

          for (int r = 0 ; r < numGPUs ; ++r) {
            for (int c = 0 ; c <= numGPUs ; ++c) {
              h_writePositionsTransposed[r][c] = h_writePositions[c][r];
            }
          }
        }

#pragma omp barrier

        cudaMemcpy(tData[cpu_tid].binMap, h_binMap, numBins * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(tData[cpu_tid].tempPrefix, h_readPositions[cpu_tid+1], (numGPUs + 1) * sizeof(Length_t), cudaMemcpyHostToDevice);

        //
        // Creating a temporary buffer that will be used for both reordering the input in the binning phase
        // and possibly in the sorting phase if CUB's sort is used. 
        // Therefore, the maximal buffer size is taken in this phase, where max=(array size of input, array size of output) 
        //
        Length_t elements = std::max(arraySize, h_writePositionsTransposed[cpu_tid][numGPUs]);

        if (elements > (1L << 31)) {
          // TODO:  Need to clean up and return an error
          printf("The size of the array, after sampling\\binning is too large to fit on a single GPU\n");
          fflush(stdout);
          exit(0);
        }

        tData[cpu_tid].cubSortBufferSize = 0;
        cub::DeviceRadixSort::SortPairs<Key_t,Value_t>(nullptr, tData[cpu_tid].cubSortBufferSize,
                                                       nullptr, nullptr, nullptr, nullptr, elements);

        tData[cpu_tid].bdReorder.allocate(h_writePositionsTransposed[cpu_tid][numGPUs], tData[cpu_tid].cubSortBufferSize);
        h_lengths[cpu_tid] = h_writePositionsTransposed[cpu_tid][numGPUs];
        cudaDeviceSynchronize();
        cudaCheckError();

#pragma omp barrier

        partitionRelabel<32, BLOCK_DIM> <<<blocks,BLOCK_DIM>>>
          (d_input_keys[cpu_tid],
           tData[cpu_tid].bdReorder.d_keys,
           d_input_values[cpu_tid],
           tData[cpu_tid].bdReorder.d_vals,
           arraySize,
           tData[cpu_tid].tempPrefix,
           computeBin,
           tData[cpu_tid].binMap,
           numGPUs);

        cudaCheckError();

        ALLOC_TRY(d_output_keys + cpu_tid, h_lengths[cpu_tid] * sizeof(Key_t), nullptr);
        ALLOC_TRY(d_output_values + cpu_tid, h_lengths[cpu_tid] * sizeof(Value_t), nullptr);

        cudaCheckError();

        //
        //  Need all partition labeling to complete before we start copying data
        //
#pragma omp barrier

        for (int dest = 0 ; dest < numGPUs ; ++dest) {
          int dest_id = (cpu_tid + dest) % numGPUs;

          cudaMemcpyAsync(d_output_keys[cpu_tid] + h_writePositionsTransposed[cpu_tid][dest_id],
                          tData[dest_id].bdReorder.d_keys + h_readPositions[dest_id+1][cpu_tid],
                          (h_readPositions[dest_id+1][cpu_tid+1] - h_readPositions[dest_id+1][cpu_tid]) * sizeof(Key_t),
                          cudaMemcpyDeviceToDevice);

          cudaMemcpyAsync(d_output_values[cpu_tid] + h_writePositionsTransposed[cpu_tid][dest_id],
                          tData[dest_id].bdReorder.d_vals + h_readPositions[dest_id+1][cpu_tid],
                          (h_readPositions[dest_id+1][cpu_tid+1] - h_readPositions[dest_id+1][cpu_tid]) * sizeof(Value_t),
                          cudaMemcpyDeviceToDevice);

        }
        cudaDeviceSynchronize();

#pragma omp barrier

        if (useThrust) {
          tData[cpu_tid].bdReorder.free();

          thrust::sort_by_key(thrust::device,
                              d_output_keys[cpu_tid],
                              d_output_keys[cpu_tid] + h_lengths[cpu_tid],
                              d_output_values[cpu_tid]);

          cudaCheckError();
          cudaDeviceSynchronize();
        } else {
          void *d_temp_storage = nullptr;

          d_temp_storage = (void*) tData[cpu_tid].bdReorder.cubBuffer;
          cub::DeviceRadixSort::SortPairs<Key_t,Value_t>(d_temp_storage,
                                                         tData[cpu_tid].cubSortBufferSize,
                                                         d_output_keys[cpu_tid],
                                                         tData[cpu_tid].bdReorder.d_keys, 
                                                         d_output_values[cpu_tid],
                                                         tData[cpu_tid].bdReorder.d_vals,
                                                         h_lengths[cpu_tid]);

          cudaCheckError();
          cudaDeviceSynchronize();

          cudaMemcpy(d_output_keys[cpu_tid], tData[cpu_tid].bdReorder.d_keys, h_lengths[cpu_tid] * sizeof(Key_t), cudaMemcpyDeviceToDevice);
          cudaMemcpy(d_output_values[cpu_tid], tData[cpu_tid].bdReorder.d_vals, h_lengths[cpu_tid] * sizeof(Value_t), cudaMemcpyDeviceToDevice);
          cudaDeviceSynchronize();

          tData[cpu_tid].bdReorder.free();
        }

        tData[cpu_tid].free();
      }

      h_output_partition_offsets[0] = Length_t{0};
      for (int i = 0 ; i < numGPUs ; ++i)
        h_output_partition_offsets[i+1] = h_output_partition_offsets[i] + h_lengths[i];

      return GDF_SUCCESS;
    }
  };

}
