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

#include <thrust/random.h>
#include <sys/time.h>

#include <sys/time.h>
#include <unistd.h>
#include <chrono>


namespace cugraph {
namespace detail {

struct prg {
    __host__ __device__
        float operator()(int n){
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(-0.0001f, 0.0001f);
            rng.discard(n);
            return dist(rng);
        }
};

void random_vector(float *vec, int n, int seed) {
    thrust::counting_iterator<uint32_t> index(seed);
    thrust::transform(rmm::exec_policy(nullptr)->on(nullptr), index,
            index + n, vec, prg());
}

/** helper method to get multi-processor count parameter */
inline int getMultiProcessorCount() {
    int devId;
    CUDA_TRY(cudaGetDevice(&devId));
    int mpCount;
    CUDA_TRY(
            cudaDeviceGetAttribute(&mpCount, cudaDevAttrMultiProcessorCount, devId));
    return mpCount;
}

long start, end;
struct timeval timecheck;
double BoundingBoxKernel_time = 0, ClearKernel1_time = 0,
       TreeBuildingKernel_time = 0, ClearKernel2_time = 0,
       SummarizationKernel_time = 0, SortKernel_time = 0, RepulsionTime = 0,
       Reduction_time = 0, attractive_time = 0, AdaptSpeed_time = 0,
       IntegrationKernel_time = 0;

// To silence warnings

#define START_TIMER                                                         \
  if (verbose) {                                                            \
    gettimeofday(&timecheck, NULL);                                         \
    start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000; \
  }

#define END_TIMER(add_onto)                                               \
  if (verbose) {                                                          \
    gettimeofday(&timecheck, NULL);                                       \
    end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000; \
    add_onto += (end - start);                                            \
  }

#define PRINT_TIMES                                                           \
  if (verbose) {                                                              \
    double total =                                                            \
      (BoundingBoxKernel_time + ClearKernel1_time + TreeBuildingKernel_time + \
       ClearKernel2_time + SummarizationKernel_time + SortKernel_time +       \
       RepulsionTime + Reduction_time + attractive_time + AdaptSpeed_time +   \
       IntegrationKernel_time) /                                              \
      100.0;                                                                  \
    printf(                                                                   \
      "BoundingBoxKernel_time = %.lf (%.lf)\n"                                \
      "ClearKernel1_time  = %.lf (%.lf)\n"                                    \
      "TreeBuildingKernel_time  = %.lf (%.lf)\n"                              \
      "ClearKernel2_time  = %.lf (%.lf)\n"                                    \
      "SummarizationKernel_time  = %.lf (%.lf)\n"                             \
      "SortKernel_time  = %.lf (%.lf)\n"                                      \
      "RepulsionTime  = %.lf (%.lf)\n"                                        \
      "attractive_time  = %.lf (%.lf)\n"                                      \
      "Reduction_time  = %.lf (%.lf)\n"                                       \
      "AdaptSpeed_time  = %.lf (%.lf)\n"                                      \
      "IntegrationKernel_time = %.lf (%.lf)\n"                                \
      "TOTAL TIME = %.lf\n\n",                                                \
      BoundingBoxKernel_time,         \
      BoundingBoxKernel_time / total, ClearKernel1_time,                      \
      ClearKernel1_time / total, TreeBuildingKernel_time,                     \
      TreeBuildingKernel_time / total, ClearKernel2_time,                     \
      ClearKernel2_time / total, SummarizationKernel_time,                    \
      SummarizationKernel_time / total, SortKernel_time,                      \
      SortKernel_time / total, RepulsionTime, RepulsionTime / total,          \
      attractive_time,                \
      attractive_time / total, Reduction_time, Reduction_time / total, \
      AdaptSpeed_time, AdaptSpeed_time / total, \
      IntegrationKernel_time,                        \
      IntegrationKernel_time / total, total * 100.0);                         \
  }
} // namespace detail
} // namespace cugraph
