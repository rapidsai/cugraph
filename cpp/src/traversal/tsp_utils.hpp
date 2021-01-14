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

#pragma once

/* CPU side validation */
#define MIN(A, B) ((A) < (B)) ? (A) : (B)
#define cpudist(a, b)                                                                          \
  (sqrtf((pos[a] - pos[b]) * (pos[a] - pos[b]) + (pos[a + nodes_ + 1] - pos[b + nodes_ + 1]) * \
                                                   (pos[a + nodes_ + 1] - pos[b + nodes_ + 1])))
#define coo_dist(a, b) \
  (sqrtf((xcoo[a] - xcoo[b]) * (xcoo[a] - xcoo[b]) + (ycoo[a] - ycoo[b]) * (ycoo[a] - ycoo[b])))

/******************************************************************************/
/*** Simulatenous Hill-climb Opt with beam-search restarts ********************/
/******************************************************************************/
// Round all distances to the nearest integer, which is good enough for realistic distances in ft
#define beamwidth 4
#define tilesize 128
#define kswaps 4
#define dist(a, b) \
  __float2int_rn(sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])))
#define acudist(a, b) (sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])))

/*only works for floats and int types with +- defined, and if a and b are distinct memory locations
#define swap(a, b) \
  {                \
    a = a + b;     \
    b = a - b;     \
    a = a - b;     \
  }
 */

namespace cugraph {
namespace detail {

/******************************************************************************/
/*** helper code **************************************************************/
/******************************************************************************/

static void CudaTest(char *msg)
{
  cudaError_t e;

  cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}

/* Affine map routine

   Transforms a vector of numbers by the affine map y = A*x + b
   Operates in-place and overwrite x
   Inverse operations happen in reverse order
*/
void affineTrans(int n, bool forward, float *x, float A, float b)
{
  if (forward)
    for (int i = 0; i < n; i++) {
      x[i] += b;
      x[i] *= A;
    }
  else
    for (int i = 0; i < n; i++) {
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
    smem   = sizeof(int) * threads + 2 * sizeof(float) * tilesize + sizeof(int) * tilesize;
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

}  // namespace detail
}  // namespace cugraph
