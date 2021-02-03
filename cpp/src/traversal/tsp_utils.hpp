/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#define cpudist(a, b)                                                                          \
  (sqrtf((pos[a] - pos[b]) * (pos[a] - pos[b]) + (pos[a + nodes_ + 1] - pos[b + nodes_ + 1]) * \
                                                   (pos[a + nodes_ + 1] - pos[b + nodes_ + 1])))

// Round all distances to the nearest integer, which is good enough for realistic distances in ft
#define tilesize 128
#define kswaps 4
#define dist(a, b) \
  __float2int_rn(sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])))

namespace cugraph {
namespace detail {

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
