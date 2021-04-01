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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <raft/cudart_utils.h>
#include <thrust/transform.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include "gtest/gtest.h"
struct StreamTest : public ::testing::Test {
};
TEST_F(StreamTest, basic_test)
{
  int n_streams = 4;
  raft::handle_t handle(n_streams);

  const size_t intput_size = 4096;

#pragma omp parallel for
  for (int i = 0; i < n_streams; i++) {
    rmm::device_uvector<int> u(intput_size, handle.get_internal_stream_view(i)),
      v(intput_size, handle.get_internal_stream_view(i));
    thrust::transform(rmm::exec_policy(handle.get_internal_stream_view(i)),
                      u.begin(),
                      u.end(),
                      v.begin(),
                      v.begin(),
                      2 * thrust::placeholders::_1 + thrust::placeholders::_2);
  }
}