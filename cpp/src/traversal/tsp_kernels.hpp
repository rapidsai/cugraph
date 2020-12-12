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

/******************************************************************************/
/*** Simulatenous Hill-climb Opt with beam-search restarts ********************/
/******************************************************************************/
/* CPU side validation */
#define MIN( A, B) ( (A) < (B) ) ?  (A) : (B)
#define cpudist(a, b) (sqrtf((pos[a] - pos[b]) * (pos[a] - pos[b]) + (pos[a+nodes+1] - pos[b+nodes+1]) * (pos[a+nodes+1] - pos[b+nodes+1])))
#define coo_dist(a, b) (sqrtf((xcoo[a] - xcoo[b]) * (xcoo[a] - xcoo[b]) + (ycoo[a] - ycoo[b]) * (ycoo[a] - ycoo[b])))

// Round all distances to the nearest integer, which is good enough for realistic distances in ft
#define beamwidth 4
#define tilesize 128
#define kswaps 4
#define dist(a, b) __float2int_rn(sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])))
#define acudist(a, b) (sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])))

/*only works for floats and int types with +- defined, and if a and b are distinct memory locations */
#define swap(a, b) { a = a + b; b = a - b;  a = a - b;}

__device__ int mylock;
__device__ int n_climbs;
__device__ int best_tour;
__device__ float *best_soln;
__device__ int bw_d;
extern __shared__ int shbuf[];

int bw = beamwidth;

static __global__ void Init()
{
  mylock = 0;
  n_climbs = 0;
  best_tour = INT_MAX;
  best_soln = NULL;
  bw_d = beamwidth;
}
