/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>

#include <curand_kernel.h>

#include <rmm/device_uvector.hpp>

#include "cuda_profiler_api.h"
#include "gtest/gtest.h"

#include <thrust/equal.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

__global__ void setup_generator(curandState* state)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(43, id, 0, &state[id]);
}

template <typename T>
__global__ void generate_random(curandState* state, int n, T* data, int32_t upper_bound)
{
  int first  = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  thrust::uniform_int_distribution<int> rnd(0, upper_bound);

  curandState local_state = state[first];
  for (int id = first; id < n; id += stride) {
    T temp = curand_uniform(&local_state);
    temp *= (upper_bound - T{1.0});
    temp = floor(temp);
    temp += T{1.0};
    data[id] = temp;
  }

  state[first] = local_state;
}

struct HungarianTest : public ::testing::Test {
};

TEST_F(HungarianTest, Bipartite4x4)
{
  raft::handle_t handle{};

  int32_t src_data[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  int32_t dst_data[] = {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
  float cost[]       = {
    5.0, 9.0, 3.0, 7.0, 8.0, 7.0, 8.0, 2.0, 6.0, 10.0, 12.0, 7.0, 3.0, 10.0, 8.0, 6.0};

  int32_t workers[] = {0, 1, 2, 3};

  float min_cost = 18.0;
  std::vector<int32_t> expected({6, 7, 5, 4});

  int32_t length         = sizeof(src_data) / sizeof(src_data[0]);
  int32_t length_workers = sizeof(workers) / sizeof(workers[0]);
  int32_t num_vertices   = 1 + std::max(*std::max_element(src_data, src_data + length),
                                      *std::max_element(dst_data, dst_data + length));

  rmm::device_uvector<int32_t> src_v(length, handle.get_stream());
  rmm::device_uvector<int32_t> dst_v(length, handle.get_stream());
  rmm::device_uvector<float> cost_v(length, handle.get_stream());
  rmm::device_uvector<int32_t> workers_v(length_workers, handle.get_stream());
  rmm::device_uvector<int32_t> assignment_v(length_workers, handle.get_stream());

  raft::update_device(src_v.begin(), src_data, length, handle.get_stream());
  raft::update_device(dst_v.begin(), dst_data, length, handle.get_stream());
  raft::update_device(cost_v.begin(), cost, length, handle.get_stream());
  raft::update_device(workers_v.begin(), workers, length_workers, handle.get_stream());

  cugraph::legacy::GraphCOOView<int32_t, int32_t, float> g(
    src_v.data(), dst_v.data(), cost_v.data(), num_vertices, length);

  float r = cugraph::hungarian(handle, g, length_workers, workers_v.data(), assignment_v.data());

  auto assignment = cugraph::test::to_host(handle, assignment_v);

  EXPECT_EQ(min_cost, r);
  EXPECT_EQ(assignment, expected);
}

TEST_F(HungarianTest, Bipartite5x5)
{
  raft::handle_t handle{};

  int32_t src_data[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4};
  int32_t dst_data[] = {5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9};
  float cost[] = {11.0, 7.0,  10.0, 17.0, 10.0, 13.0, 21.0, 7.0,  11.0, 13.0, 13.0, 13.0, 15.0,
                  13.0, 14.0, 18.0, 10.0, 13.0, 16.0, 14.0, 12.0, 8.0,  16.0, 19.0, 10.0};

  int32_t workers[] = {0, 1, 2, 3, 4};

  float min_cost = 51.0;
  std::vector<int32_t> expected({5, 7, 8, 6, 9});

  int32_t length         = sizeof(src_data) / sizeof(src_data[0]);
  int32_t length_workers = sizeof(workers) / sizeof(workers[0]);
  int32_t num_vertices   = 1 + std::max(*std::max_element(src_data, src_data + length),
                                      *std::max_element(dst_data, dst_data + length));

  rmm::device_uvector<int32_t> src_v(length, handle.get_stream());
  rmm::device_uvector<int32_t> dst_v(length, handle.get_stream());
  rmm::device_uvector<float> cost_v(length, handle.get_stream());
  rmm::device_uvector<int32_t> workers_v(length_workers, handle.get_stream());
  rmm::device_uvector<int32_t> assignment_v(length_workers, handle.get_stream());

  raft::update_device(src_v.begin(), src_data, length, handle.get_stream());
  raft::update_device(dst_v.begin(), dst_data, length, handle.get_stream());
  raft::update_device(cost_v.begin(), cost, length, handle.get_stream());
  raft::update_device(workers_v.begin(), workers, length_workers, handle.get_stream());

  cugraph::legacy::GraphCOOView<int32_t, int32_t, float> g(
    src_v.data(), dst_v.data(), cost_v.data(), num_vertices, length);

  float r = cugraph::hungarian(handle, g, length_workers, workers_v.data(), assignment_v.data());

  auto assignment = cugraph::test::to_host(handle, assignment_v);

  EXPECT_EQ(min_cost, r);
  EXPECT_EQ(assignment, expected);
}

TEST_F(HungarianTest, Bipartite4x4_multiple_answers)
{
  raft::handle_t handle{};

  int32_t src_data[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  int32_t dst_data[] = {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
  float cost[] = {3.0, 1.0, 1.0, 4.0, 4.0, 2.0, 2.0, 5.0, 5.0, 3.0, 4.0, 8.0, 4.0, 2.0, 5.0, 9.0};

  int32_t workers[] = {0, 1, 2, 3};

  float min_cost = 13.0;

  std::vector<int32_t> expected1({7, 6, 5, 4});
  std::vector<int32_t> expected2({6, 7, 5, 4});
  std::vector<int32_t> expected3({7, 6, 4, 5});
  std::vector<int32_t> expected4({6, 7, 4, 5});

  int32_t length         = sizeof(src_data) / sizeof(src_data[0]);
  int32_t length_workers = sizeof(workers) / sizeof(workers[0]);
  int32_t num_vertices   = 1 + std::max(*std::max_element(src_data, src_data + length),
                                      *std::max_element(dst_data, dst_data + length));

  rmm::device_uvector<int32_t> src_v(length, handle.get_stream());
  rmm::device_uvector<int32_t> dst_v(length, handle.get_stream());
  rmm::device_uvector<float> cost_v(length, handle.get_stream());
  rmm::device_uvector<int32_t> workers_v(length_workers, handle.get_stream());
  rmm::device_uvector<int32_t> assignment_v(length_workers, handle.get_stream());

  raft::update_device(src_v.begin(), src_data, length, handle.get_stream());
  raft::update_device(dst_v.begin(), dst_data, length, handle.get_stream());
  raft::update_device(cost_v.begin(), cost, length, handle.get_stream());
  raft::update_device(workers_v.begin(), workers, length_workers, handle.get_stream());

  cugraph::legacy::GraphCOOView<int32_t, int32_t, float> g(
    src_v.data(), dst_v.data(), cost_v.data(), num_vertices, length);

  float r = cugraph::hungarian(handle, g, length_workers, workers_v.data(), assignment_v.data());

  EXPECT_EQ(min_cost, r);

  auto assignment = cugraph::test::to_host(handle, assignment_v);

  EXPECT_TRUE(std::equal(assignment.begin(), assignment.end(), expected1.begin()) ||
              std::equal(assignment.begin(), assignment.end(), expected2.begin()) ||
              std::equal(assignment.begin(), assignment.end(), expected3.begin()) ||
              std::equal(assignment.begin(), assignment.end(), expected4.begin()));
}

TEST_F(HungarianTest, May29InfLoop)
{
  raft::handle_t handle{};

  int32_t num_rows = 4;
  int32_t num_cols = 4;
  float cost[]     = {0, 16, 1, 0, 33, 45, 0, 4, 22, 0, 1000, 2000, 2, 0, 3000, 4000};

  float min_cost = 2;

  std::vector<int32_t> expected({3, 2, 1, 0});

  rmm::device_uvector<float> cost_v(num_rows * num_cols, handle.get_stream());
  rmm::device_uvector<int32_t> assignment_v(num_rows, handle.get_stream());

  raft::update_device(cost_v.begin(), cost, num_rows * num_cols, handle.get_stream());

  float r =
    cugraph::dense::hungarian(handle, cost_v.data(), num_rows, num_cols, assignment_v.data());

  auto assignment = cugraph::test::to_host(handle, assignment_v);

  EXPECT_EQ(min_cost, r);
  EXPECT_EQ(assignment, expected);
}

TEST_F(HungarianTest, Dense4x6)
{
  raft::handle_t handle{};

  int32_t num_rows = 4;
  int32_t num_cols = 6;
  float cost[]     = {0,  16, 1,    0,    90, 100, 33, 45, 0,    4,    90, 100,
                  22, 0,  1000, 2000, 90, 100, 2,  0,  3000, 4000, 90, 100};

  float min_cost = 2;

  std::vector<int32_t> expected({3, 2, 1, 0});

  rmm::device_uvector<float> cost_v(num_rows * num_cols, handle.get_stream());
  rmm::device_uvector<int32_t> assignment_v(num_rows, handle.get_stream());

  raft::update_device(cost_v.begin(), cost, num_rows * num_cols, handle.get_stream());

  float r =
    cugraph::dense::hungarian(handle, cost_v.data(), num_rows, num_cols, assignment_v.data());

  auto assignment = cugraph::test::to_host(handle, assignment_v);

  EXPECT_EQ(min_cost, r);
  EXPECT_EQ(assignment, expected);
}

TEST_F(HungarianTest, Dense6x4)
{
  raft::handle_t handle{};

  int32_t num_rows = 6;
  int32_t num_cols = 4;
  float cost[]     = {0,  16, 1,    0,    33, 45,  0,   4,   90, 100, 110,  120,
                  22, 0,  1000, 2000, 90, 100, 110, 120, 2,  0,   3000, 4000};

  float min_cost = 2;

  std::vector<int32_t> expected1({3, 2, 4, 1, 5, 0});
  std::vector<int32_t> expected2({3, 2, 5, 1, 4, 0});

  rmm::device_uvector<float> cost_v(num_rows * num_cols, handle.get_stream());
  rmm::device_uvector<int32_t> assignment_v(num_rows, handle.get_stream());

  raft::update_device(cost_v.begin(), cost, num_rows * num_cols, handle.get_stream());

  float r =
    cugraph::dense::hungarian(handle, cost_v.data(), num_rows, num_cols, assignment_v.data());

  auto assignment = cugraph::test::to_host(handle, assignment_v);

  EXPECT_EQ(min_cost, r);
  EXPECT_TRUE(std::equal(assignment.begin(), assignment.end(), expected1.begin()) ||
              std::equal(assignment.begin(), assignment.end(), expected2.begin()));
}

TEST_F(HungarianTest, PythonTestFailure)
{
  raft::handle_t handle{};

#if 0
  int32_t num_rows = 20;
  int32_t num_cols = 20;
  float   cost[]     = {
    12,  4,  5, 19,  5,  4,  7, 17,  3, 19,  1, 14,  3, 17, 10, 15,  9, 19, 14,  8,
    16, 13, 19,  9,  9,  6, 16,  4,  4,  4, 17,  4, 16,  4,  1, 18,  6,  7,  1,  8,
    10, 11,  6, 18, 18,  1, 17,  9,  6, 10,  2,  3,  2, 17, 17, 19,  9,  3, 11,  2,
    11,  6,  5,  7,  5,  9,  6,  2,  9, 16,  1,  2, 19, 12,  1, 12, 17,  5,  6,  4,
    18,  3,  3, 11,  3, 13,  1,  1, 17, 12, 14,  2, 13, 13, 18, 10, 15,  3,  9, 15,
    14,  4,  2, 19, 11, 12, 14, 15,  6, 19, 10,  4, 18, 14, 18, 15, 18,  5, 15, 13,
     9,  9, 10,  6, 16, 17,  4, 18,  6, 16, 14, 14,  1, 19, 15, 19,  1,  3, 14, 19,
     3, 12, 19, 14,  7, 17,  2,  4, 17, 17, 16,  4, 14,  7,  7, 18, 14, 14, 11, 13,
    12,  2,  1,  8, 16,  1,  3, 13,  8,  8,  1,  3, 15,  8, 13, 12, 18,  3, 19, 13,
     7, 17,  7, 14, 10,  2,  3, 16,  7, 16, 15,  5,  6, 10, 15, 10,  6,  8, 17,  2,
    14,  7, 14,  5,  1, 19,  9,  9,  4, 15, 16, 17, 15, 18,  6, 19, 14, 13, 13,  8,
     4, 11, 12, 12,  3, 19, 13, 11, 19, 14, 11,  8, 18, 13, 18, 12,  6,  2,  3,  3,
    12,  3, 14, 15,  8, 13, 18,  1, 16,  7, 16, 13,  1,  6,  7, 17,  9, 10, 12,  3,
    13,  4,  3, 14, 15, 14,  3,  2,  5, 15, 12,  6, 12,  4,  3,  3, 14, 17,  7, 10,
    19, 19, 16, 13, 17,  9,  3,  1,  3, 13, 11,  5, 12, 17,  1, 14, 17, 13,  6, 12,
    13,  6, 12, 19, 12,  8,  9, 17, 10, 18, 19,  9, 17, 12, 17, 11, 12, 15, 15, 12,
    19, 13,  7, 10, 17,  7,  6,  9,  3, 16,  1,  3, 11, 14,  6, 12,  2, 13, 15,  5,
    11, 12, 17, 19, 16, 18,  7,  6,  4, 18, 19,  8,  8, 12, 13,  6,  5,  8,  5,  6,
    17, 11,  3, 16,  2, 14, 17,  1,  3, 16, 14,  7,  1, 12, 12, 16, 14, 15,  1,  3,
    14,  8, 19,  2, 11,  9,  4,  8,  4,  8,  3, 19,  1, 17, 16,  1,  9, 19,  9, 16,
  };
#else
  int32_t num_rows = 5;
  int32_t num_cols = 5;
  float cost[]     = {
    7, 6, 3, 6, 4, 6, 9, 2, 9, 9, 7, 5, 3, 8, 9, 5, 3, 8, 3, 1, 6, 2, 1, 1, 3,
  };
#endif

  float min_cost = 16;

  std::vector<int32_t> expected({0, 2, 1, 4, 3});

  rmm::device_uvector<float> cost_v(num_rows * num_cols, handle.get_stream());
  rmm::device_uvector<int32_t> assignment_v(num_rows, handle.get_stream());

  raft::update_device(cost_v.begin(), cost, num_rows * num_cols, handle.get_stream());

  float r =
    cugraph::dense::hungarian(handle, cost_v.data(), num_rows, num_cols, assignment_v.data());

  auto assignment = cugraph::test::to_host(handle, assignment_v);

  EXPECT_EQ(min_cost, r);
  EXPECT_EQ(assignment, expected);
}

// FIXME:  Need to have tests with nxm (e.g. 4x5 and 5x4) to test those conditions

#if 0
void random_test(int32_t num_rows, int32_t num_cols, int32_t upper_bound, int repetitions = 1)
{
  const int num_threads{64};

  printf("benchmark run:  %d, %d\n", num_rows, upper_bound);

  HighResTimer hr_timer;

  rmm::device_uvector<int32_t>  data_v(num_rows * num_cols, handle.get_stream());
  rmm::device_uvector<curandState> state_vals_v(num_threads, handle.get_stream());
  rmm::device_uvector<int32_t> assignment_v(num_rows, handle.get_stream());

  std::vector<int32_t> validate(num_cols);

  hr_timer.start("initialization");

  cudaStream_t stream{0};
  int32_t *d_data = data_v.data();
  //int64_t seed{85};
  int64_t seed{time(nullptr)};

  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<int32_t>(0),
                   thrust::make_counting_iterator<int32_t>(num_rows * num_cols),
                   [d_data, seed, upper_bound] __device__ (int32_t e) {
                     thrust::random::default_random_engine rng(seed);
                     rng.discard(2*e);
                     thrust::uniform_int_distribution<int> rnd(0, upper_bound);
                     d_data[e] = rnd(rng);
                   });

  cudaDeviceSynchronize();

  hr_timer.stop();

  int32_t r{0};

  for (int i = 0 ; i < repetitions ; ++i) {
    hr_timer.start("hungarian");
    r = cugraph::hungarian_dense(cost_v.data(), num_rows, num_cols, assignment_v.data());
    hr_timer.stop();
  }

  std::cout << "cost = " << r << std::endl;
  hr_timer.display_and_clear(std::cout);


  for (int i = 0 ; i < num_cols ; ++i)
    validate[i] = 0;

  int32_t assignment_out_of_range{0};

  for (int32_t i = 0 ; i < num_rows ; ++i) {
    if (assignment_v[i] < num_cols)
      validate[assignment_v[i]]++;
    else {
      ++assignment_out_of_range;
    }
  }

  EXPECT_EQ(assignment_out_of_range, 0);

  int32_t assignment_missed = 0;

  for (int32_t i = 0 ; i < num_cols ; ++i) {
    if (validate[i] != 1) {
      ++assignment_missed;
    }
  }

  EXPECT_EQ(assignment_missed, 0);
}

TEST_F(HungarianTest, Benchmark)
{
  random_test(5000, 5000, 500, 3);
  random_test(5000, 5000, 5000, 3);
  random_test(5000, 5000, 50000, 3);
  random_test(10000, 10000, 1000, 3);
  random_test(10000, 10000, 10000, 3);
  random_test(10000, 10000, 100000, 3);
  random_test(15000, 15000, 1500, 3);
  random_test(15000, 15000, 15000, 3);
  random_test(15000, 15000, 150000, 3);
  random_test(20000, 20000, 2000, 3);
  random_test(20000, 20000, 20000, 3);
  random_test(20000, 20000, 200000, 3);
}
#endif
