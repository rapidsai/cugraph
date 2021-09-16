/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 * Copyright (c) 2014-2020, Texas State University. All rights reserved.
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

#include <cuda.h>
#include <curand_kernel.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <raft/cuda_utils.cuh>

#include "tsp_utils.hpp"

namespace cugraph {
namespace detail {

__device__ float *best_soln;
__device__ int *best_route;
extern __shared__ int shbuf[];

__global__ void reset(int *mylock, int *best_tour, int *climbs)
{
  *mylock    = 0;
  *best_tour = INT_MAX;
  *climbs    = 0;
  best_soln  = nullptr;
  best_route = nullptr;
}

// random permutation kernel
__device__ void random_init(float const *posx,
                            float const *posy,
                            int const *vtx_ptr,
                            int *path,
                            float *px,
                            float *py,
                            int const nstart,
                            int const nodes)
{
  // Fill values
  for (int i = threadIdx.x; i <= nodes; i += blockDim.x) {
    px[i]   = posx[i];
    py[i]   = posy[i];
    path[i] = vtx_ptr[i];
  }

  __syncthreads();

  if (threadIdx.x == 0) { /* serial permutation as starting point */
    // swap to start at nstart node
    raft::swapVals(px[0], px[nstart]);
    raft::swapVals(py[0], py[nstart]);
    raft::swapVals(path[0], path[nstart]);

    curandState rndstate;
    curand_init(blockIdx.x, 0, 0, &rndstate);
    for (int i = 1; i < nodes; i++) {
      int j = curand(&rndstate) % (nodes - 1 - i) + i;
      if (i == j) continue;
      raft::swapVals(px[i], px[j]);
      raft::swapVals(py[i], py[j]);
      raft::swapVals(path[i], path[j]);
    }
    px[nodes]   = px[0]; /* close the loop now, avoid special cases later */
    py[nodes]   = py[0];
    path[nodes] = path[0];
  }
}

// Use KNN as a starting solution
__device__ void knn_init(float const *posx,
                         float const *posy,
                         int const *vtx_ptr,
                         int64_t const *neighbors,
                         int *buf,
                         int *path,
                         float *px,
                         float *py,
                         int const nstart,
                         int const nodes,
                         int const K)
{
  for (int i = threadIdx.x; i < nodes; i += blockDim.x) buf[i] = 0;

  __syncthreads();

  if (threadIdx.x == 0) {
    curandState rndstate;
    curand_init(blockIdx.x, 0, 0, &rndstate);
    int progress = 0;
    int initlen  = 0;

    px[0]     = posx[nstart];
    py[0]     = posy[nstart];
    path[0]   = vtx_ptr[nstart];
    int head  = nstart;
    int v     = 0;
    buf[head] = 1;
    while (progress < nodes - 1) {  // beam search as starting point
      for (int i = 1; i <= progress; i++) buf[i] = 0;
      progress      = 0;  // reset current location in path and visited array
      initlen       = 0;
      int randjumps = 0;
      while (progress < nodes - 1) {
        int nj     = curand(&rndstate) % K;
        int linked = 0;
        for (int nh = 0; nh < K; ++nh) {
          // offset (idx / K) + 1 filters the points as their own nearest neighbors.
          int offset = (K * head + nj) / K + 1;
          v          = neighbors[K * head + nj + offset];
          if (v < nodes && buf[v] == 0) {
            head = v;
            progress += 1;
            buf[head] = 1;
            linked    = 1;
            break;
          }
          nj = (nj + 1) % K;
        }
        if (linked == 0) {
          if (randjumps > nodes - 1)
            break;  // give up on this traversal, we failed to find a next link
          randjumps += 1;
          int nr = (head + 1) % nodes;  // jump to next node
          while (buf[nr] == 1) { nr = (nr + 1) % nodes; }
          head = nr;
          progress += 1;
          buf[head] = 1;
        }
        // copy from input into beam-search order, update len
        px[progress]   = posx[head];
        py[progress]   = posy[head];
        path[progress] = vtx_ptr[head];
        initlen += __float2int_rn(euclidean_dist(px, py, progress, progress - 1));
      }
    }
    px[nodes]   = px[nstart];
    py[nodes]   = py[nstart];
    path[nodes] = path[nstart];
    initlen += __float2int_rn(euclidean_dist(px, py, nodes, nstart));
  }
}

__device__ void two_opt_search(
  int *buf, float *px, float *py, int *shbuf, int *minchange, int *mini, int *minj, int const nodes)
{
  __shared__ float shmem_x[tilesize];
  __shared__ float shmem_y[tilesize];

  for (int ii = 0; ii < nodes - 2; ii += blockDim.x) {
    int i = ii + threadIdx.x;
    float pxi0, pyi0, pxi1, pyi1, pxj1, pyj1;
    if (i < nodes - 2) {
      minchange[0] -= buf[i];
      pxi0 = px[i];
      pyi0 = py[i];
      pxi1 = px[i + 1];
      pyi1 = py[i + 1];
      pxj1 = px[nodes];
      pyj1 = py[nodes];
    }
    for (int jj = nodes - 1; jj >= ii + 2; jj -= tilesize) {
      int bound = jj - tilesize + 1;
      for (int k = threadIdx.x; k < tilesize; k += blockDim.x) {
        if (k + bound >= ii + 2) {
          shmem_x[k] = px[k + bound];
          shmem_y[k] = py[k + bound];
          shbuf[k]   = buf[k + bound];
        }
      }
      __syncthreads();

      int lower = bound;
      if (lower < (i + 2)) lower = i + 2;
      for (int j = jj; j >= lower; j--) {
        int jm     = j - bound;
        float pxj0 = shmem_x[jm];
        float pyj0 = shmem_y[jm];
        int delta =
          shbuf[jm] +
          __float2int_rn(sqrtf((pxi0 - pxj0) * (pxi0 - pxj0) + (pyi0 - pyj0) * (pyi0 - pyj0))) +
          __float2int_rn(sqrtf((pxi1 - pxj1) * (pxi1 - pxj1) + (pyi1 - pyj1) * (pyi1 - pyj1)));
        pxj1 = pxj0;
        pyj1 = pyj0;

        if (delta < minchange[0]) {
          minchange[0] = delta;
          mini[0]      = i;
          minj[0]      = j;
        }
      }
      __syncthreads();
    }

    if (i < nodes - 2) { minchange[0] += buf[i]; }
  }
}

// This function being runned for each block
__device__ void hill_climbing(
  float *px, float *py, int *buf, int *path, int *shbuf, int const nodes, int *climbs)
{
  __shared__ int best_change[kswaps];
  __shared__ int best_i[kswaps];
  __shared__ int best_j[kswaps];

  int minchange;
  int mini;
  int minj;
  int kswaps_active = kswaps;
  int myswaps       = 0;

  // Hill climbing, iteratively improve from the starting guess
  do {
    if (threadIdx.x == 0) {
      for (int k = 0; k < kswaps; k++) {
        best_change[k] = 0;
        best_i[k]      = 0;
        best_j[k]      = 0;
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nodes; i += blockDim.x) {
      buf[i] = -__float2int_rn(euclidean_dist(px, py, i, i + 1));
    }
    __syncthreads();

    // Reset
    minchange = 0;
    mini      = 0;
    minj      = 0;

    // Find best indices
    two_opt_search(buf, px, py, shbuf, &minchange, &mini, &minj, nodes);
    __syncthreads();

    // Stats only
    if (threadIdx.x == 0) atomicAdd(climbs, 1);

    shbuf[threadIdx.x] = minchange;

    int j = blockDim.x;  // warp reduction to find best thread results
    do {
      int k = (j + 1) / 2;
      if ((threadIdx.x + k) < j) {
        shbuf[threadIdx.x] = min(shbuf[threadIdx.x + k], shbuf[threadIdx.x]);
      }
      j = k;
      __syncthreads();
    } while (j > 1);  // thread winner for this k is in shbuf[0]

    if (threadIdx.x == 0) {
      best_change[0] = shbuf[0];  // sort best result in shared
    }
    __syncthreads();

    if (minchange == shbuf[0]) {  // My thread is as good as the winner
      shbuf[1] = threadIdx.x;     // store thread ID in shbuf[1]
    }
    __syncthreads();

    if (threadIdx.x == shbuf[1]) {  // move from thread local to shared
      best_i[0] = mini;             // shared best indices for compatibility checks
      best_j[0] = minj;
    }
    __syncthreads();

    // look for more compatible swaps
    for (int kmin = 1; kmin < kswaps_active; kmin++) {
      // disallow swaps that conflict with ones already picked
      for (int kchk = kmin - 1; kchk >= 0; --kchk) {
        if ((mini < (best_j[kchk] + 1)) && (minj > (best_i[kchk] - 1))) {
          minchange = shbuf[threadIdx.x] = 0;
        }
        __syncthreads();
      }
      shbuf[threadIdx.x] = minchange;

      j = blockDim.x;
      do {
        int k = (j + 1) / 2;
        if ((threadIdx.x + k) < j) {
          shbuf[threadIdx.x] = min(shbuf[threadIdx.x + k], shbuf[threadIdx.x]);
        }
        j = k;
        __syncthreads();
      } while (j > 1);  // thread winner for this k is in shbuf[0]

      if (threadIdx.x == 0) {
        best_change[kmin] = shbuf[0];  // store best result in shared
      }
      __syncthreads();

      if (minchange == shbuf[0]) {  // My thread is as good as the winner
        shbuf[1] = threadIdx.x;     // store thread ID in shbuf[1]
        __threadfence_block();
      }
      __syncthreads();

      if (threadIdx.x == shbuf[1]) {  // move from thread local to shared
        best_i[kmin] = mini;          // store swap targets
        best_j[kmin] = minj;
        __threadfence_block();
      }
      __syncthreads();
      // look for the best compatible move
    }  // end loop over kmin
    minchange = best_change[0];
    myswaps += 1;
    for (int kmin = 0; kmin < kswaps_active; kmin++) {
      int sum = best_i[kmin] + best_j[kmin] + 1;  // = mini + minj +1
      // this is a reversal of all nodes included in the range [ i+1, j ]
      for (int i = threadIdx.x; (i + i) < sum; i += blockDim.x) {
        if (best_i[kmin] < i) {
          int j = sum - i;
          raft::swapVals(px[i], px[j]);
          raft::swapVals(py[i], py[j]);
          raft::swapVals(path[i], path[j]);
        }
      }
      __syncthreads();
    }
  } while (minchange < 0 && myswaps < 2 * nodes);
}

__device__ void get_optimal_tour(
  int *mylock, int *best_tour, float *px, float *py, int *path, int *shbuf, int const nodes)
{
  // Now find actual length of the last tour, result of the climb
  int term = 0;
  for (int i = threadIdx.x; i < nodes; i += blockDim.x) {
    term += __float2int_rn(euclidean_dist(px, py, i, i + 1));
  }
  shbuf[threadIdx.x] = term;
  __syncthreads();

  int j = blockDim.x;  // block level reduction
  do {
    int k = (j + 1) / 2;
    if ((threadIdx.x + k) < j) { shbuf[threadIdx.x] += shbuf[threadIdx.x + k]; }
    j = k;  // divide active warp size in half
    __syncthreads();
  } while (j > 1);
  term = shbuf[0];

  if (threadIdx.x == 0) {
    atomicMin(best_tour, term);
    while (atomicExch(mylock, 1) != 0)
      ;  // acquire
    if (best_tour[0] == term) {
      best_soln  = px;
      best_route = path;
    }
    *mylock = 0;  // release
    __threadfence();
  }
}

__global__ __launch_bounds__(2048, 2) void search_solution(int *mylock,
                                                           int *best_tour,
                                                           int const *vtx_ptr,
                                                           bool beam_search,
                                                           int const K,
                                                           int nodes,
                                                           int64_t const *neighbors,
                                                           float const *posx,
                                                           float const *posy,
                                                           int *work,
                                                           int const nstart,
                                                           float *times,
                                                           int *climbs,
                                                           int threads)
{
  int *buf  = &work[blockIdx.x * ((4 * nodes + 3 + 31) / 32 * 32)];
  float *px = (float *)(&buf[nodes]);
  float *py = &px[nodes + 1];
  int *path = (int *)(&py[nodes + 1]);
  __shared__ int shbuf[tilesize];
  clock_t start;

  start = clock64();
  if (!beam_search)
    random_init(posx, posy, vtx_ptr, path, px, py, nstart, nodes);
  else
    knn_init(posx, posy, vtx_ptr, neighbors, buf, path, px, py, nstart, nodes, K);
  __syncthreads();
  times[threadIdx.x] = clock64() - start;

  start = clock64();
  hill_climbing(px, py, buf, path, shbuf, nodes, climbs);
  __syncthreads();
  times[threads + threadIdx.x + 1] = clock64() - start;

  start = clock64();
  get_optimal_tour(mylock, best_tour, px, py, path, shbuf, nodes);
  times[2 * threads + threadIdx.x + 1] = clock64() - start;
}
}  // namespace detail
}  // namespace cugraph
