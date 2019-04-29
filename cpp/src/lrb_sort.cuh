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

/* 
   The segmented sort in this code use the LRB method designed for
   triangle counting.

   Additional details can be found:
   * J. Fox, O. Green, K. Gabert, X. An, D. Bader, “Fast and Adaptive List Intersections on the GPU”, 
   IEEE High Performance Extreme Computing Conference (HPEC), 
   Waltham, Massachusetts, 2018
   * O. Green, J. Fox, A. Tripathy, A. Watkins, K. Gabert, E. Kim, X. An, K. Aatish, D. Bader, 
   “Logarithmic Radix Binning and Vectorized Triangle Counting”, 
   IEEE High Performance Extreme Computing Conference (HPEC), 
   Waltham, Massachusetts, 2018
*/

#ifndef LRB_SORT_H
#define LRB_SORT_H

#include <rmm_utils.h>
#include <cub/cub.cuh>
#include "utilities/error_utils.h"
#include "graph_utils.cuh"
#include "timer.h"

template <typename T>
__global__ void binCountKernel(const T *offset, int32_t *bins, int N){
  int i = threadIdx.x + blockIdx.x *blockDim.x;
  if(i>=N)
    return;

  __shared__ int32_t localBins[33];
  int id = threadIdx.x;
  if(id<33){
    localBins[id]=0;
  }
  __syncthreads();

  int32_t adjSize=offset[i+1]-offset[i];
  int myBin  = __clz(adjSize);

  atomicAdd(localBins+myBin, 1);

  __syncthreads();
  if(id<33){
    atomicAdd(bins+id, localBins[id]);
  }
}

__global__ void  binPrefixKernel(const int32_t *bins, int32_t *d_binsPrefix) {
  int i = threadIdx.x + blockIdx.x *blockDim.x;
  if(i>=1)
    return;
  d_binsPrefix[0]=0;
  for(int b=0; b<33; b++){
    d_binsPrefix[b+1]=d_binsPrefix[b]+bins[b];
    printf("%d ",bins[b]);
  }
}

template <typename T, bool writeStop,bool writeStart,int THREADS>
__global__ void  rebinKernel(
  const T   *offset,
  int32_t   *d_binsPrefix,
  T         *d_reOrg,
  T         *d_start,
  T         *d_stop,
  int N) {

    int i = threadIdx.x + blockIdx.x *blockDim.x;
    if(i>=N)
      return;

    __shared__ int32_t localBins[33];
    __shared__ int32_t localPos[33];

    int id = threadIdx.x;
    if(id<33){
      localBins[id]=0;
      localPos[id]=0;
    }
    __syncthreads();

    int32_t adjSize=offset[i+1]-offset[i];
    int myBin  = __clz(adjSize);

    int my_pos = atomicAdd(localBins+myBin, 1);

  __syncthreads();
    if(id<33){
      localPos[id]=atomicAdd(d_binsPrefix+id, localBins[id]);
    }
  __syncthreads();

    int pos = localPos[myBin]+my_pos;
    d_reOrg[pos]=i;
    if(writeStart)
      d_start[pos]=offset[i];
    if(writeStop)
      d_stop[pos] =offset[i+1];
}

template <typename T>
__device__ void bubbleSort(int32_t size, T *edges){
  T temp; 
  for(int32_t i=0; i<(size-1); i++){
    int32_t min_idx=i;
    for(int32_t j=i+1; j<(size); j++){
       if(edges[j]<edges[min_idx])
          min_idx=j;
    }
    temp          = edges[min_idx];
    edges[min_idx]  = edges[i];
    edges[i]        = temp;
  }
}

template <typename T>
__global__ void sortSmallKernel(const T *edges,
				T *newEdges,
				const T *d_reOrg,
				const T *offset,
				int32_t pos,
				int32_t N) {

    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=N)
      return;
    T v=d_reOrg[i+pos];

    int32_t adjSize=offset[v+1]-offset[v];
    if(adjSize==0){
      return;
    } else if (adjSize==1){
      newEdges[offset[v]]=edges[offset[v]]; 
      return;
    }

    int32_t temp1[32];

    for(int32_t d=0; d<adjSize;d++){
     temp1[d]=edges[offset[v]+d];
    }

    bubbleSort(adjSize,temp1);

    for(int32_t d=0; d<adjSize;d++){
     newEdges[offset[v]+d]=temp1[d];
    }
}

template <typename T, int threads, int elements_per_thread,
	  int total_elements>
__global__ void sortOneSize(
    int32_t  posReorg,
    T       *d_reOrg,
    const T *offset,
    const T *edges,
    T       *newEdges,
    T       *d_start,
    T       *d_stop) {
  typedef cub::BlockRadixSort<T, threads, elements_per_thread> BlockRadixSort;
  
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  __shared__ T sharedEdges[total_elements];

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  T v = d_reOrg[posReorg+bid];

  int32_t adjSize=offset[v+1]-offset[v];

  for(int i=tid ; i < total_elements ; i += threads) {
    if(i<adjSize)
      sharedEdges[i]=edges[offset[v]+i];
    else
      sharedEdges[i]=cub::Traits<T>::MAX_KEY;
  }

  __syncthreads();

  T mine[elements_per_thread];


  int pos = tid * elements_per_thread;
  for(int s = 0; s < elements_per_thread; s++) {
    mine[s] = sharedEdges[pos+s];
  }

  __syncthreads();

  BlockRadixSort(temp_storage).Sort(mine);
  __syncthreads();

  pos=tid*elements_per_thread;
  for(int s = 0 ; s < elements_per_thread ; s++){
    sharedEdges[pos+s] = mine[s];
  }
  __syncthreads();

  for(int i = tid; i < adjSize; i += threads){
    newEdges[offset[v] + i] = sharedEdges[i];
  }
}

#ifndef HEAP_SEGMENTED_SORT
  /**
   * @brief Perform an in-place segmented sort
   *
   * This method uses the heap to perform an in-place segmented sort
   * of an array of values.
   *
   * @param [in] size - the number of segments to sort
   * @param [in] startOffset - array of start offsets
   * @param [in] endOffset - array of end offsets
   * @param [in, out] data - the data to sort.  Data will
   *                         be sorted in place.
   *
   * @return error code
   */
namespace cugraph {
  template <typename T>
  gdf_error segmentedSort(T size, const T *startOffset, const T *endOffset,
			  T * data) {
    cudaStream_t stream{nullptr};

    // TODO:  right now startOffset is assumed to be rowOffsets,
    //    we can't handle non-contiguous data.

    int32_t *d_bins;
    int32_t *d_binsPrefix;
    int32_t *d_binsPrefixTemp;
    T *d_reOrg;
    T *d_newOffset;
    T *d_newSize;
    T *d_start;
    T *d_stop;

    cugraph::timer::Timer timer;

    ALLOC_TRY(&d_bins, 33 * sizeof(int32_t), stream);
    ALLOC_TRY(&d_binsPrefix, 34 * sizeof(int32_t), stream);
    ALLOC_TRY(&d_binsPrefixTemp, 34 * sizeof(int32_t), stream);
    ALLOC_TRY(&d_reOrg, size * sizeof(T), stream);
    ALLOC_TRY(&d_newSize, (size + 2) * sizeof(T), stream);
    ALLOC_TRY(&d_newOffset, (size + 2) * sizeof(T), stream);
    ALLOC_TRY(&d_start, (size + 1) * sizeof(T), stream);
    ALLOC_TRY(&d_stop, (size + 1) * sizeof(T), stream);

    int sortRadix = 20;
    int32_t pos = 0;
    int sortRadixSmall = 28;    
    int32_t posSmall = 0;
    int32_t h_binsPrefix[34];

    CUDA_TRY(cudaMemset(d_bins, 0, 33 * sizeof(T)));
    CUDA_TRY(cudaMemset(d_newSize, 0, (size + 2) * sizeof(T)));

    int binCountBlocks = (size + 255) / 256;
    if (binCountBlocks) {
      binCountKernel<T> <<<binCountBlocks,256>>> (startOffset, d_bins, size);
    }

    binPrefixKernel <<<1,32>>> (d_bins, d_binsPrefix);     
    CUDA_TRY(cudaMemcpy(d_binsPrefixTemp, d_binsPrefix,
			sizeof(int32_t)*34, cudaMemcpyDeviceToDevice));

    const int RB_BLOCK_SIZE = 1024;
    int rebinblocks = (size + RB_BLOCK_SIZE - 1) / RB_BLOCK_SIZE;

    if(rebinblocks){
      rebinKernel<T, true,true,RB_BLOCK_SIZE><<<rebinblocks,RB_BLOCK_SIZE>>>(startOffset, d_binsPrefixTemp, d_reOrg, d_start, d_stop, size);
      cudaCheckError();
    }

    CUDA_TRY(cudaMemcpy(&pos, d_binsPrefix + sortRadix, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(&posSmall, d_binsPrefix + sortRadixSmall, sizeof(int32_t), cudaMemcpyDeviceToHost));

    T nnz;
    CUDA_TRY(cudaMemcpy(&nnz, startOffset + size, sizeof(T), cudaMemcpyDeviceToHost));
    
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    // TODO: is this safe?
    //T *extra_storage = data;
    T *extra_storage;
    ALLOC_TRY(&extra_storage, nnz * sizeof(T), stream);
    CUDA_TRY(cudaMemcpy(extra_storage, data, nnz * sizeof(T), cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();

    if (pos > 0) {
      timer.start();

      printf("calling SortKeys\n");
      cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, extra_storage, data, nnz, pos, d_start, d_stop);

      std::cout << "temp_storage_bytes = " << temp_storage_bytes << std::endl;
      ALLOC_TRY(&d_temp_storage, temp_storage_bytes, stream);
      cudaCheckError();
      cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, extra_storage, data, nnz, pos, d_start, d_stop);
      ALLOC_FREE_TRY(d_temp_storage, stream);
      cudaCheckError();
      printf("done calling SortKeys\n");

      timer.end();
      timer.print("cub radix sort");
    }

    cudaStream_t streams[9];
    for(int i=0;i<9; i++)
      cudaStreamCreate ( &(streams[i]));

    cudaMemcpy(h_binsPrefix,d_binsPrefix,sizeof(int32_t)*34, cudaMemcpyDeviceToHost);

    timer.start();

    if (h_binsPrefix[21] - h_binsPrefix[20] > 0)
      sortOneSize<T, 128,32,4096>  <<<h_binsPrefix[21]-h_binsPrefix[20],128, 0, streams[8]>>>    (h_binsPrefix[20],d_reOrg, startOffset, extra_storage, data,d_start,d_stop);
    if (h_binsPrefix[22] - h_binsPrefix[21] > 0)
      sortOneSize<T, 32,64,2048>  <<<h_binsPrefix[22]-h_binsPrefix[21],32, 0, streams[7]>>>    (h_binsPrefix[21],d_reOrg, startOffset, extra_storage, data, d_start,d_stop);
    if (h_binsPrefix[23] - h_binsPrefix[22] > 0)
      sortOneSize<T, 32,32,1024>   <<<h_binsPrefix[23]-h_binsPrefix[22],32, 0, streams[6]>>>    (h_binsPrefix[22],d_reOrg, startOffset, extra_storage, data, d_start,d_stop);
    if (h_binsPrefix[24] - h_binsPrefix[23] > 0)
      sortOneSize<T, 32,16,512>    <<<h_binsPrefix[24]-h_binsPrefix[23],32, 0, streams[5]>>>    (h_binsPrefix[23],d_reOrg, startOffset, extra_storage, data, d_start,d_stop);
    if (h_binsPrefix[25] - h_binsPrefix[24] > 0)
      sortOneSize<T, 32,8,256>   <<<h_binsPrefix[25]-h_binsPrefix[24],32, 0, streams[4]>>>   (h_binsPrefix[24],d_reOrg, startOffset, extra_storage, data, d_start,d_stop);
    if (h_binsPrefix[26] - h_binsPrefix[25] > 0)
      sortOneSize<T, 32,4,128>   <<<h_binsPrefix[26]-h_binsPrefix[25],32, 0, streams[3]>>>   (h_binsPrefix[25],d_reOrg, startOffset, extra_storage, data, d_start,d_stop);
    if (h_binsPrefix[27] - h_binsPrefix[26] > 0)
      sortOneSize<T, 32,2,64>    <<<h_binsPrefix[27]-h_binsPrefix[26],32, 0, streams[2]>>>   (h_binsPrefix[26],d_reOrg, startOffset, extra_storage, data, d_start,d_stop);
    if (h_binsPrefix[28] - h_binsPrefix[27] > 0)
      sortOneSize<T, 32,1,32>    <<<h_binsPrefix[28]-h_binsPrefix[27],32, 0, streams[1]>>>    (h_binsPrefix[27],d_reOrg, startOffset, extra_storage, data, d_start,d_stop);

    if(h_binsPrefix[31]-posSmall>0){
      int blocks = (h_binsPrefix[31]-posSmall)/32 + (((h_binsPrefix[31]-posSmall)%32)?1:0);
      sortSmallKernel<T> <<<blocks,32, 0, streams[0]>>>(extra_storage, data, d_reOrg, startOffset, posSmall,h_binsPrefix[31]-posSmall);
    }

    timer.end();
    timer.print("CUB Device Radix");

    fflush(stdout);

    ALLOC_FREE_TRY(extra_storage, stream);
    ALLOC_FREE_TRY(d_start, stream);
    ALLOC_FREE_TRY(d_stop, stream);
    ALLOC_FREE_TRY(d_newOffset, stream);
    ALLOC_FREE_TRY(d_newSize, stream);
    ALLOC_FREE_TRY(d_bins, stream);
    ALLOC_FREE_TRY(d_binsPrefix, stream);
    ALLOC_FREE_TRY(d_binsPrefixTemp, stream);
    ALLOC_FREE_TRY(d_reOrg, stream);

    return GDF_SUCCESS;
  }
}

#endif

#endif
