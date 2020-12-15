/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

# pragma once

/* CPU side validation */
#define MIN( A, B) ( (A) < (B) ) ?  (A) : (B)
#define cpudist(a, b) (sqrtf((pos[a] - pos[b]) * (pos[a] - pos[b]) + (pos[a+nodes+1] - pos[b+nodes+1]) * (pos[a+nodes+1] - pos[b+nodes+1])))
#define coo_dist(a, b) (sqrtf((xcoo[a] - xcoo[b]) * (xcoo[a] - xcoo[b]) + (ycoo[a] - ycoo[b]) * (ycoo[a] - ycoo[b])))


#define mallocOnGPU(addr, size) if (cudaSuccess != cudaMalloc((void **)&addr, size)) fprintf(stderr, "could not allocate GPU memory\n");  CudaTest("couldn't allocate GPU memory");
#define copyToGPU(to, from, size) if (cudaSuccess != cudaMemcpy(to, from, size, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of data to device failed\n");  CudaTest("data copy to device failed");
#define copyFromGPU(to, from, size) if (cudaSuccess != cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of data from device failed\n");  CudaTest("data copy from device failed");
#define copyFromGPUSymbol(to, from, size) if (cudaSuccess != cudaMemcpyFromSymbol(to, from, size)) fprintf(stderr, "copying of symbol from device failed\n");  CudaTest("symbol copy from device failed");
#define copyToGPUSymbol(to, from, size) if (cudaSuccess != cudaMemcpyToSymbol(to, from, size)) fprintf(stderr, "copying of symbol to device failed\n");  CudaTest("symbol copy to device failed");


namespace cugraph {
  namespace detail {

/* Affine map routine

   Transforms a vector of numbers by the affine map y = A*x + b
   Operates in-place and overwrite x
   Inverse operations happen in reverse order
*/
void affineTrans( int n, bool forward, float *x, float A, float b) {
   if (forward)
   for (int i = 0; i< n; i++){
      x[i] += b;
      x[i] *= A;
   }
   else
   for (int i = 0; i< n; i++){
      x[i] *= A;
      x[i] += b;
   }
}

/******************************************************************************/
/*** find best thread count ***************************************************/
/******************************************************************************/

int best_thread_count(int nodes)
{
  int max, best, threads, smem, blocks, thr, perf, bthr;
  int sm_count = 84;

  max = nodes - 2;
  if (max > 1024) max = 1024;
  best = 0;
  bthr = 4;
  for (threads = 1; threads <= max; threads++) {
    smem = sizeof(int) * threads + 2 * sizeof(float) * tilesize + sizeof(int) * tilesize;
    blocks = (16384 * 2) / smem;
    if (blocks > sm_count) blocks = sm_count;
    thr = (threads + 31) / 32 * 32;
    while (blocks * thr > 2048) blocks--;
    perf = threads * blocks;
    if (perf > best) {
      best = perf;
      bthr = threads;
    }
  }

  return bthr;
}

} // detail
} // cugraph
