/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtest/gtest.h"

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <thrust/transform.h>

struct StreamTest : public ::testing::Test {};

TEST_F(StreamTest, basic_test)
{
  size_t n_streams = 4;
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(n_streams);
  raft::handle_t handle(rmm::cuda_stream_per_thread, stream_pool);

  const size_t input_size = 4096;

  std::vector<std::thread> threads(n_streams);

  for (size_t i = 0; i < n_streams; ++i) {
    threads[i] = std::thread(
      [&handle, input_size](size_t i) {
        rmm::device_uvector<int> u(input_size, handle.get_next_usable_stream(i));
        rmm::device_uvector<int> v(input_size, handle.get_next_usable_stream(i));
        thrust::transform(rmm::exec_policy(handle.get_next_usable_stream(i)),
                          u.begin(),
                          u.end(),
                          v.begin(),
                          v.begin(),
                          2 * thrust::placeholders::_1 + thrust::placeholders::_2);
        RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_next_usable_stream(i)));
      },
      i);
  }

  for (size_t i = 0; i < n_streams; ++i) {
    threads[i].join();
  }
}
