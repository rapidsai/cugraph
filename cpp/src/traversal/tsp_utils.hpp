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

#define tilesize 128
#define kswaps 4

#include <sys/time.h>
#include <string>
#include <vector>

namespace cugraph {
namespace detail {

__host__ __device__ inline float euclidean_dist(float *px, float *py, int a, int b)
{
  return sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b]));
}

static std::vector<std::string> device_func = {"Find First", "Hill Climbing", "Retrieve Path"};

void print_times(std::vector<float> &h_times, int const n_timers, int device, int threads)
{
  int clock_rate;
  cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device);

  double total = 0;
  h_times[0] /= (float)clock_rate;
  total += h_times[0];
  for (int i = 1; i < n_timers; ++i) {
    h_times[i * threads + 1] /= (float)clock_rate;
    total += h_times[i * threads + 1];
  }
  std::cout << "Stats: \n";
  std::cout << device_func[0] << " time: " << h_times[0] * 1e-3 << " "
            << (h_times[0] / total) * 100.0 << "%\n";
  for (int i = 1; i < n_timers; ++i) {
    std::cout << device_func[i] << " time: " << h_times[i * threads + 1] * 1e-3 << " "
              << (h_times[i * threads + 1] / total) * 100.0 << "%\n";
  }
}

// Get maximum number of threads we can run on based on number of nodes,
// shared memory usage, max threads per block and SM, max blocks for SM and registers per SM.
int best_thread_count(int nodes, int max_threads, int sm_count, int warp_size)
{
  int smem, blocks, thr, perf;
  int const max_threads_sm = 2048;
  int max                  = nodes - 2;
  int best                 = 0;
  int bthr                 = 4;

  if (max > max_threads) max = max_threads;

  for (int threads = 1; threads <= max; ++threads) {
    smem   = sizeof(int) * threads + 2 * sizeof(float) * tilesize + sizeof(int) * tilesize;
    blocks = (16384 * 2) / smem;
    if (blocks > sm_count) blocks = sm_count;
    thr = (threads + warp_size - 1) / warp_size * warp_size;
    while (blocks * thr > max_threads_sm) blocks--;
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
