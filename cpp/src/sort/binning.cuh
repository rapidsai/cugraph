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

#include <stdint.h>
#include <cuda_profiler_api.h>

template <typename Key_t, typename Len_t>
struct LeftmostBits {
  /**
   * @brief Extract the leftmost bits from the key
   *
   *    This functor will return the leftmost numBits from the key, they
   *    are returned right justified so they can be used as an array index.
   *
   * @param[in]  numBits  The number of bits to gather from the left of the key
   */
  LeftmostBits(int numBits) {
    shiftRight_ = 8 * sizeof(Key_t) - numBits;
  }

  /**
   * @brief This is the () operator used by the functor
   *
   * @return  The leftmost bits in the key
   */
  Len_t __device__ operator() (const Key_t &v) const {
    return (v >> shiftRight_);
  }

  int shiftRight_;
};

template <typename Key_t, typename Len_t>
struct SkipNBits {
  /**
   * @brief Extract bits after skipping some number of bits on the left
   *
   *  This functor will skip skipBits in the key and then return the next
   *  numBits, they are returned right justified so they can be used
   *  as an array index.
   *
   * @param[in]  numBits   The number of bits to gather from the key
   * @param[in]  skipBits  The number of bits to skip from the left of the key
   *
   */
  SkipNBits(int numBits, int skipBits) {
    shiftRight_ = 8 * sizeof(Key_t) - (numBits + skipBits);
    if (shiftRight_ < 0)
      shiftRight_ = 0;

    bitMask_ = (Key_t{1} << numBits) - 1;
  }

  /**
   * @brief This is the () operator used by the functor
   *
   * @return  The desired bits in the key, right justified
   */
  Len_t __device__ operator() (const Key_t &v) const {
    return (v >> shiftRight_) & bitMask_;
  }

  int shiftRight_;
  Key_t bitMask_;
};

/**
 * @brief This global function iterates over the key array, applies the computeBin functor
 *        and counts how many keys fall into each bin.
 *
 * @param[in]     array       The array of keys
 * @param[in]     numKeys     The number of keys in array
 * @param[in/out] binSizes    The output histogram, note that the histogram is assumed to
 *                            be initialized properly.
 * @param[in]     computeBin  A functor that computes a bin number from a key
 */
template <typename Key_t, typename Len_t, typename ComputeBin_t>
__global__ void binCounting(Key_t* array, Len_t numKeys, Len_t* binSizes, ComputeBin_t computeBin)
{
  Len_t pos = blockIdx.x*blockDim.x + threadIdx.x;
  if(pos>=numKeys)
    return;

  Len_t myBin = computeBin(array[pos]);

  atomicAdd((Len_t*) binSizes+myBin,(Len_t)1L);
}

/**
 * @brief This global function partitions the data across
 *        the different GPUs.
 *
 * @param[in]     array         The array of keys
 * @param[out]    reorgArray    The output array of keys
 * @param[in]     vals          The array of corresponding values
 * @param[in]     reorgVals     The output array of corresponding values
 * @param[in]     numKeys       The number of keys in array
 * @param[in]     binOffsets    The (starting) offsets for each bin
 * @param[in]     computeBin    Functor to convert a key to a bin number
 * @param[in]     binMap        Maps each bin to a partition id
 * @param[in]     numPartitions Number of partitions
 */
template <int NUMGPUS, int THREADS,
          typename Key_t, typename Val_t, typename Len_t,
          typename ComputeBin_t>
__global__ void partitionRelabel(Key_t *array,
                                 Key_t *reorgArray,
                                 Val_t *vals,
                                 Val_t *reorgVals,
                                 Len_t  numKeys, 
                                 Len_t *binOffsets,
                                 ComputeBin_t computeBin,
                                 unsigned char *binMap,
                                 int numPartitions) {

  Len_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  Len_t tid = threadIdx.x;

  //  Need to see what of these we actually need
  //    NOTE:  These dimensions are NUMGPUS+1?  I think this is
  //           to reduce the number of bank collisions
  //
  __shared__ Len_t counter[2][NUMGPUS+1];
  __shared__ Len_t counter2[NUMGPUS+1];
  __shared__ Len_t prefix[NUMGPUS+1];
  __shared__ Len_t globalPositions[NUMGPUS+1];

  __shared__ Key_t reOrderedLocalKey[THREADS];
  __shared__ Val_t reOrderedLocalVal[THREADS];
  __shared__ Len_t reOrderedPositions[THREADS];

  //
  //  First we initialize the shared data structures
  //
  if (tid < numPartitions) {
    counter[0][tid] = 0L;
    counter[1][tid] = 0L;
    counter2[tid] = 0L;
  }

  __syncthreads();

  //
  //  Now we get the key/value used by this thread,
  //  which gpu bin they map to, and increment the
  //  count for how many elements go into each GPU.
  //
  Key_t key;
  Val_t val;
  Len_t gpuBin = 0L;

  if (pos < numKeys) {
    key    =  array[pos];
    val    =  vals[pos];

    gpuBin =  binMap[computeBin(key)];

    //
    // TODO:  Would % 2 be also efficient?
    //        Would 4 be better than 2?
    //
    Len_t tidBin =  tid / (THREADS / 2);
    //Len_t tidBin =  tid % 2;

    atomicAdd(counter[tidBin] + gpuBin, Len_t{1});
  }

  __syncthreads();

  //
  //  Now we compute globalPosition and prefix
  //  which will help us move the data to the
  //  right place.
  //
  if (tid < numPartitions) {
    globalPositions[tid] = atomicAdd(binOffsets + tid,
                                     counter[0][tid] + counter[1][tid]);
  }

  if (tid == 0) {
    prefix[0] = 0L;
    for (int p = 0 ; p < numPartitions ; ++p) {
      prefix[p+1] = prefix[p] + counter[0][p] + counter[1][p];
    }
  }

  __syncthreads();

  //
  //  Populate the key and value buffer with atomics,
  //  more efficient in shared memory.
  //
  Len_t posWithinBin;
  if (pos < numKeys) {
    posWithinBin = atomicAdd(counter2 + gpuBin, Len_t{1});
    
    reOrderedLocalKey[prefix[gpuBin] + posWithinBin] = key;
    reOrderedLocalVal[prefix[gpuBin] + posWithinBin] = val;

    reOrderedPositions[prefix[gpuBin] + posWithinBin] = posWithinBin + globalPositions[gpuBin];
  }
  __syncthreads();

  //
  //  Now do serial memory accesses to populate the output.
  //
  if (pos < numKeys) {
    reorgArray[reOrderedPositions[tid]] = reOrderedLocalKey[tid];
    reorgVals[reOrderedPositions[tid]] = reOrderedLocalVal[tid];
  }  
  __syncthreads();
}

/**
 * @brief This global function partitions the data across
 *        the different GPUs.
 *
 * @param[in]     array         The array of keys
 * @param[out]    reorgArray    The output array of keys
 * @param[in]     numKeys       The number of keys in array
 * @param[in]     binOffsets    The (starting) offsets for each bin
 * @param[in]     computeBin    Functor to convert a key to a bin number
 * @param[in]     binMap        Maps each bin to a partition id
 * @param[in]     numPartitions Number of partitions
 */
template <int NUMGPUS, int THREADS,
          typename Key_t, typename Len_t,
          typename ComputeBin_t>
__global__ void partitionRelabel(Key_t *array,
                                 Key_t *reorgArray,
                                 Len_t  numKeys, 
                                 Len_t *binOffsets,
                                 ComputeBin_t computeBin,
                                 unsigned char *binMap,
                                 int numPartitions) {

  Len_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  Len_t tid = threadIdx.x;

  //  Need to see what of these we actually need
  //    NOTE:  These dimensions are NUMGPUS+1?  I think this is
  //           to reduce the number of bank collisions
  //
  __shared__ Len_t counter[2][NUMGPUS+1];
  __shared__ Len_t counter2[NUMGPUS+1];
  __shared__ Len_t prefix[NUMGPUS+1];
  __shared__ Len_t globalPositions[NUMGPUS+1];

  __shared__ Key_t reOrderedLocalKey[THREADS];
  __shared__ Len_t reOrderedPositions[THREADS];

  //
  //  First we initialize the shared data structures
  //
  if (tid < numPartitions) {
    counter[0][tid] = 0L;
    counter[1][tid] = 0L;
    counter2[tid] = 0L;
  }

  __syncthreads();

  //
  //  Now we get the key used by this thread,
  //  which gpu bin they map to, and increment the
  //  count for how many elements go into each GPU.
  //
  Key_t key;
  Len_t gpuBin = 0L;

  if (pos < numKeys) {
    key    =  array[pos];
    gpuBin =  binMap[computeBin(key)];

    //
    // TODO:  Would % 2 be also efficient?
    //        Would 4 be better than 2?
    //
    Len_t tidBin =  tid / (THREADS / 2);
    //Len_t tidBin =  tid % 2;

    atomicAdd(counter[tidBin] + gpuBin, Len_t{1});
  }

  __syncthreads();

  //
  //  Now we compute globalPosition and prefix
  //  which will help us move the data to the
  //  right place.
  //
  if (tid < numPartitions) {
    globalPositions[tid] = atomicAdd(binOffsets + tid,
                                     counter[0][tid] + counter[1][tid]);
  }

  if (tid == 0) {
    prefix[0] = 0L;
    for (int p = 0 ; p < numPartitions ; ++p) {
      prefix[p+1] = prefix[p] + counter[0][p] + counter[1][p];
    }
  }

  __syncthreads();

  //
  //  Populate the key buffer with atomics,
  //  more efficient in shared memory.
  //
  Len_t posWithinBin;
  if (pos < numKeys) {
    posWithinBin = atomicAdd(counter2 + gpuBin, Len_t{1});
    reOrderedLocalKey[prefix[gpuBin] + posWithinBin] = key;
    reOrderedPositions[prefix[gpuBin] + posWithinBin] = posWithinBin + globalPositions[gpuBin];
  }
  __syncthreads();

  //
  //  Now do serial memory accesses to populate the output.
  //
  if (pos < numKeys) {
    reorgArray[reOrderedPositions[tid]] = reOrderedLocalKey[tid];
  }  
  __syncthreads();
}
