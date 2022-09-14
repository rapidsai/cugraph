/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/graph_generators.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <random>

struct GeneratorsTest : public ::testing::Test {
};

TEST_F(GeneratorsTest, PathGraphTest)
{
  using vertex_t = int32_t;

  std::vector<vertex_t> expected_src_v({0, 1, 2, 3});
  std::vector<vertex_t> expected_dst_v({1, 2, 3, 4});

  std::vector<std::tuple<vertex_t, vertex_t>> parameters({{5, 0}});

  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::tie(src_v, dst_v) = cugraph::generate_path_graph_edgelist<vertex_t>(handle, parameters);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, Mesh2DGraphTest)
{
  using vertex_t = int32_t;

  std::vector<vertex_t> expected_src_v({0,  1,  2,  4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
                                        20, 21, 22, 0, 1, 2, 3, 8, 9,  10, 11, 16, 17, 18, 19});
  std::vector<vertex_t> expected_dst_v({1,  2,  3,  5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19,
                                        21, 22, 23, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23});
  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t, vertex_t>> parameters(
    {{4, 2, 0}, {4, 2, 8}, {4, 2, 16}});

  std::tie(src_v, dst_v) = cugraph::generate_2d_mesh_graph_edgelist<vertex_t>(handle, parameters);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, Mesh3DGraphTest)
{
  using vertex_t = int32_t;

  std::vector<vertex_t> expected_src_v(
    {0,  1,  3,  4,  6,  7,  9,  10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34,
     36, 37, 39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70,
     72, 73, 75, 76, 78, 79, 0,  1,  2,  3,  4,  5,  9,  10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23,
     27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59,
     63, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
     12, 13, 14, 15, 16, 17, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
     54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71});

  std::vector<vertex_t> expected_dst_v(
    {1,  2,  4,  5,  7,  8,  10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35,
     37, 38, 40, 41, 43, 44, 46, 47, 49, 50, 52, 53, 55, 56, 58, 59, 61, 62, 64, 65, 67, 68, 70, 71,
     73, 74, 76, 77, 79, 80, 3,  4,  5,  6,  7,  8,  12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26,
     30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 43, 44, 48, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 62,
     66, 67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 80, 9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
     21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
     63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80});

  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t, vertex_t, vertex_t>> parameters(
    {{3, 3, 3, 0}, {3, 3, 3, 27}, {3, 3, 3, 54}});

  std::tie(src_v, dst_v) = cugraph::generate_3d_mesh_graph_edgelist<vertex_t>(handle, parameters);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, CompleteGraphTestTriangles)
{
  using vertex_t = int32_t;

  std::vector<vertex_t> expected_src_v({0, 0, 1, 3, 3, 4, 6, 6, 7});
  std::vector<vertex_t> expected_dst_v({1, 2, 2, 4, 5, 5, 7, 8, 8});

  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t>> parameters({{3, 0}, {3, 3}, {3, 6}});

  std::tie(src_v, dst_v) = cugraph::generate_complete_graph_edgelist<vertex_t>(handle, parameters);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, CompleteGraphTest5)
{
  using vertex_t = int32_t;

  size_t num_vertices{5};
  size_t num_graphs{3};

  std::vector<vertex_t> expected_src_v({0, 0, 0, 0, 1, 1,  1,  2,  2,  3,  5,  5,  5,  5,  6,
                                        6, 6, 7, 7, 8, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13});
  std::vector<vertex_t> expected_dst_v({1, 2, 3, 4, 2, 3,  4,  3,  4,  4,  6,  7,  8,  9,  7,
                                        8, 9, 8, 9, 9, 11, 12, 13, 14, 12, 13, 14, 13, 14, 14});
  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t>> parameters({{5, 0}, {5, 5}, {5, 10}});

  std::tie(src_v, dst_v) = cugraph::generate_complete_graph_edgelist<vertex_t>(handle, parameters);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, LineGraphTestSymmetric)
{
  using vertex_t = int32_t;

  size_t num_vertices{5};
  std::vector<vertex_t> expected_src_v({0, 1, 2, 3, 1, 2, 3, 4});
  std::vector<vertex_t> expected_dst_v({1, 2, 3, 4, 0, 1, 2, 3});

  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t>> parameters({{5, 0}});

  std::tie(src_v, dst_v) = cugraph::generate_path_graph_edgelist<vertex_t>(handle, parameters);
  std::tie(src_v, dst_v, std::ignore) =
    cugraph::symmetrize_edgelist_from_triangular<vertex_t, float>(
      handle, std::move(src_v), std::move(dst_v), std::nullopt);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, Mesh2DGraphTestSymmetric)
{
  using vertex_t = int32_t;

  size_t x{4};
  size_t y{2};
  size_t num_graphs{3};

  std::vector<vertex_t> expected_src_v({0,  1,  2,  4, 5, 6, 8, 9,  10, 12, 13, 14, 16, 17, 18,
                                        20, 21, 22, 0, 1, 2, 3, 8,  9,  10, 11, 16, 17, 18, 19,
                                        1,  2,  3,  5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19,
                                        21, 22, 23, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23});
  std::vector<vertex_t> expected_dst_v({1,  2,  3,  5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19,
                                        21, 22, 23, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23,
                                        0,  1,  2,  4, 5, 6, 8, 9,  10, 12, 13, 14, 16, 17, 18,
                                        20, 21, 22, 0, 1, 2, 3, 8,  9,  10, 11, 16, 17, 18, 19});

  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t, vertex_t>> parameters(
    {{4, 2, 0}, {4, 2, 8}, {4, 2, 16}});

  std::tie(src_v, dst_v) = cugraph::generate_2d_mesh_graph_edgelist<vertex_t>(handle, parameters);
  std::tie(src_v, dst_v, std::ignore) =
    cugraph::symmetrize_edgelist_from_triangular<vertex_t, float>(
      handle, std::move(src_v), std::move(dst_v), std::nullopt);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, Mesh3DGraphTestSymmetric)
{
  using vertex_t = int32_t;

  size_t x{3};
  size_t y{3};
  size_t z{3};
  size_t num_graphs{3};

  std::vector<vertex_t> expected_src_v(
    {0,  1,  3,  4,  6,  7,  9,  10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34,
     36, 37, 39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70,
     72, 73, 75, 76, 78, 79, 0,  1,  2,  3,  4,  5,  9,  10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23,
     27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59,
     63, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
     12, 13, 14, 15, 16, 17, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
     54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 1,  2,  4,  5,  7,  8,
     10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 40, 41, 43, 44,
     46, 47, 49, 50, 52, 53, 55, 56, 58, 59, 61, 62, 64, 65, 67, 68, 70, 71, 73, 74, 76, 77, 79, 80,
     3,  4,  5,  6,  7,  8,  12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35,
     39, 40, 41, 42, 43, 44, 48, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 62, 66, 67, 68, 69, 70, 71,
     75, 76, 77, 78, 79, 80, 9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
     36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 63, 64, 65, 66, 67, 68,
     69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80});

  std::vector<vertex_t> expected_dst_v(
    {1,  2,  4,  5,  7,  8,  10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35,
     37, 38, 40, 41, 43, 44, 46, 47, 49, 50, 52, 53, 55, 56, 58, 59, 61, 62, 64, 65, 67, 68, 70, 71,
     73, 74, 76, 77, 79, 80, 3,  4,  5,  6,  7,  8,  12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26,
     30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 43, 44, 48, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 62,
     66, 67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 80, 9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
     21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
     63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 0,  1,  3,  4,  6,  7,
     9,  10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43,
     45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70, 72, 73, 75, 76, 78, 79,
     0,  1,  2,  3,  4,  5,  9,  10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32,
     36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
     72, 73, 74, 75, 76, 77, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
     27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 54, 55, 56, 57, 58, 59,
     60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71});

  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t, vertex_t, vertex_t>> parameters(
    {{3, 3, 3, 0}, {3, 3, 3, 27}, {3, 3, 3, 54}});

  std::tie(src_v, dst_v) = cugraph::generate_3d_mesh_graph_edgelist<vertex_t>(handle, parameters);
  std::tie(src_v, dst_v, std::ignore) =
    cugraph::symmetrize_edgelist_from_triangular<vertex_t, float>(
      handle, std::move(src_v), std::move(dst_v), std::nullopt);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, CompleteGraphTestTrianglesSymmetric)
{
  using vertex_t = int32_t;

  size_t num_vertices{3};
  size_t num_graphs{3};

  std::vector<vertex_t> expected_src_v({0, 0, 1, 3, 3, 4, 6, 6, 7, 1, 2, 2, 4, 5, 5, 7, 8, 8});
  std::vector<vertex_t> expected_dst_v({1, 2, 2, 4, 5, 5, 7, 8, 8, 0, 0, 1, 3, 3, 4, 6, 6, 7});

  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t>> parameters({{3, 0}, {3, 3}, {3, 6}});

  std::tie(src_v, dst_v) = cugraph::generate_complete_graph_edgelist<vertex_t>(handle, parameters);
  std::tie(src_v, dst_v, std::ignore) =
    cugraph::symmetrize_edgelist_from_triangular<vertex_t, float>(
      handle, std::move(src_v), std::move(dst_v), std::nullopt);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, CompleteGraphTest5Symmetric)
{
  using vertex_t = int32_t;

  size_t num_vertices{5};
  size_t num_graphs{3};

  std::vector<vertex_t> expected_src_v({0, 0, 0, 0, 1, 1,  1,  2,  2,  3,  5,  5,  5,  5,  6,
                                        6, 6, 7, 7, 8, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13,
                                        1, 2, 3, 4, 2, 3,  4,  3,  4,  4,  6,  7,  8,  9,  7,
                                        8, 9, 8, 9, 9, 11, 12, 13, 14, 12, 13, 14, 13, 14, 14});
  std::vector<vertex_t> expected_dst_v({1, 2, 3, 4, 2, 3,  4,  3,  4,  4,  6,  7,  8,  9,  7,
                                        8, 9, 8, 9, 9, 11, 12, 13, 14, 12, 13, 14, 13, 14, 14,
                                        0, 0, 0, 0, 1, 1,  1,  2,  2,  3,  5,  5,  5,  5,  6,
                                        6, 6, 7, 7, 8, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13});

  raft::handle_t handle;

  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t>> parameters({{5, 0}, {5, 5}, {5, 10}});

  std::tie(src_v, dst_v) = cugraph::generate_complete_graph_edgelist<vertex_t>(handle, parameters);
  std::tie(src_v, dst_v, std::ignore) =
    cugraph::symmetrize_edgelist_from_triangular<vertex_t, float>(
      handle, std::move(src_v), std::move(dst_v), std::nullopt);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, CombineGraphsTest)
{
  using vertex_t = int32_t;
  using weight_t = float;

  raft::handle_t handle;

  size_t num_vertices{8};

  std::vector<vertex_t> expected_src_v({0,  1,  2,  3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
                                        20, 21, 22, 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19});
  std::vector<vertex_t> expected_dst_v({1,  2,  3,  4, 5, 6, 7, 9,  10, 11, 13, 14, 15, 17, 18, 19,
                                        21, 22, 23, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23});

  rmm::device_uvector<vertex_t> src_graph_1_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_graph_1_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> src_graph_2_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_graph_2_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t>> parameters1({{num_vertices, 0}});
  std::vector<std::tuple<vertex_t, vertex_t, vertex_t>> parameters2(
    {{4, 2, 0}, {4, 2, 8}, {4, 2, 16}});

  std::tie(src_graph_1_v, dst_graph_1_v) =
    cugraph::generate_path_graph_edgelist<vertex_t>(handle, parameters1);
  std::tie(src_graph_2_v, dst_graph_2_v) =
    cugraph::generate_2d_mesh_graph_edgelist<vertex_t>(handle, parameters2);

  std::vector<rmm::device_uvector<vertex_t>> sources;
  sources.push_back(std::move(src_graph_1_v));
  sources.push_back(std::move(src_graph_2_v));

  std::vector<rmm::device_uvector<vertex_t>> dests;
  dests.push_back(std::move(dst_graph_1_v));
  dests.push_back(std::move(dst_graph_2_v));

  std::tie(src_v, dst_v, std::ignore) = cugraph::combine_edgelists<vertex_t, weight_t>(
    handle, std::move(sources), std::move(dests), std::nullopt);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, CombineGraphsOffsetsTest)
{
  using vertex_t = int32_t;
  using weight_t = float;

  raft::handle_t handle;

  size_t num_vertices{8};
  vertex_t offset{10};

  std::vector<vertex_t> expected_src_v({0,  1,  2,  3,  4,  5,  6,  10, 11, 12, 14, 15, 16,
                                        18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 10,
                                        11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29});
  std::vector<vertex_t> expected_dst_v({1,  2,  3,  4,  5,  6,  7,  11, 12, 13, 15, 16, 17,
                                        19, 20, 21, 23, 24, 25, 27, 28, 29, 31, 32, 33, 14,
                                        15, 16, 17, 22, 23, 24, 25, 30, 31, 32, 33});

  rmm::device_uvector<vertex_t> src_graph_1_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_graph_1_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> src_graph_2_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_graph_2_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());

  std::vector<std::tuple<vertex_t, vertex_t>> parameters1({{num_vertices, 0}});
  std::vector<std::tuple<vertex_t, vertex_t, vertex_t>> parameters2(
    {{4, 2, 10}, {4, 2, 18}, {4, 2, 26}});

  std::tie(src_graph_1_v, dst_graph_1_v) =
    cugraph::generate_path_graph_edgelist<vertex_t>(handle, parameters1);
  std::tie(src_graph_2_v, dst_graph_2_v) =
    cugraph::generate_2d_mesh_graph_edgelist<vertex_t>(handle, parameters2);

  std::vector<rmm::device_uvector<vertex_t>> sources;
  sources.push_back(std::move(src_graph_1_v));
  sources.push_back(std::move(src_graph_2_v));

  std::vector<rmm::device_uvector<vertex_t>> dests;
  dests.push_back(std::move(dst_graph_1_v));
  dests.push_back(std::move(dst_graph_2_v));

  std::tie(src_v, dst_v, std::ignore) = cugraph::combine_edgelists<vertex_t, weight_t>(
    handle, std::move(sources), std::move(dests), std::nullopt);

  auto actual_src_v = cugraph::test::to_host(handle, src_v);
  auto actual_dst_v = cugraph::test::to_host(handle, dst_v);

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected_src_v.begin(), expected_dst_v.begin())) +
      expected_src_v.size());

  thrust::sort(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(actual_src_v.begin(), actual_dst_v.begin())) +
      actual_src_v.size());

  EXPECT_EQ(expected_src_v, actual_src_v);
  EXPECT_EQ(expected_dst_v, actual_dst_v);
}

TEST_F(GeneratorsTest, ScrambleTest)
{
  using vertex_t = int32_t;
  using edge_t   = int32_t;

  edge_t num_vertices{30};
  edge_t num_edges{100};

  raft::handle_t handle;

  std::vector<vertex_t> input_src_v(num_edges);
  std::vector<vertex_t> input_dst_v(num_edges);

  std::default_random_engine generator{};
  std::uniform_int_distribution<vertex_t> distribution{0, num_vertices - 1};

  std::generate(input_src_v.begin(), input_src_v.end(), [&distribution, &generator]() {
    return distribution(generator);
  });
  std::generate(input_dst_v.begin(), input_dst_v.end(), [&distribution, &generator]() {
    return distribution(generator);
  });

  rmm::device_uvector<vertex_t> d_src_v(input_src_v.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(input_src_v.size(), handle.get_stream());

  raft::update_device(d_src_v.data(), input_src_v.data(), input_src_v.size(), handle.get_stream());
  raft::update_device(d_dst_v.data(), input_dst_v.data(), input_dst_v.size(), handle.get_stream());

  cugraph::scramble_vertex_ids(handle, d_src_v, d_dst_v, 5, 0);

  auto output_src_v = cugraph::test::to_host(handle, d_src_v);
  auto output_dst_v = cugraph::test::to_host(handle, d_dst_v);

  EXPECT_TRUE(cugraph::test::renumbered_vectors_same(handle, input_src_v, output_src_v));
  EXPECT_TRUE(cugraph::test::renumbered_vectors_same(handle, input_dst_v, output_dst_v));
}

CUGRAPH_TEST_PROGRAM_MAIN()
