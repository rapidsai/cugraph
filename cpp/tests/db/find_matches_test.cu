/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include "db/db_operators.cuh"
#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "rmm/device_buffer.hpp"
#include "utilities/test_utilities.hpp"
#include "utilities/error_utils.h"
#include "utilities/graph_utils.cuh"

class Test_FindMatches : public ::testing::Test {
 public:
  Test_FindMatches() {}
  virtual void SetUp()
  {
    cugraph::db::db_pattern<int32_t> p;
    cugraph::db::db_pattern_entry<int32_t> p1(0);
    cugraph::db::db_pattern_entry<int32_t> p2(1);
    cugraph::db::db_pattern_entry<int32_t> p3(2);
    p.addEntry(p1);
    p.addEntry(p2);
    p.addEntry(p3);
    table.addColumn("Start");
    table.addColumn("EdgeType");
    table.addColumn("End");
    table.addEntry(p);
    table.flush_input();
  }
  virtual void TearDown() {}
  void insertConstantEntry(int32_t a, int32_t b, int32_t c)
  {
    cugraph::db::db_pattern<int32_t> p;
    cugraph::db::db_pattern_entry<int32_t> p1(a);
    cugraph::db::db_pattern_entry<int32_t> p2(b);
    cugraph::db::db_pattern_entry<int32_t> p3(c);
    p.addEntry(p1);
    p.addEntry(p2);
    p.addEntry(p3);
    table.addEntry(p);
  }
  cugraph::db::db_table<int32_t> table;
};

TEST_F(Test_FindMatches, verifyIndices)
{
  insertConstantEntry(0, 1, 1);
  insertConstantEntry(2, 0, 1);
  table.flush_input();

  std::cout << table.toString();
  std::cout << "Index[0]: " << table.getIndex(0).toString();
  std::cout << "Index[1]: " << table.getIndex(1).toString();
  std::cout << "Index[2]: " << table.getIndex(2).toString();
}

TEST_F(Test_FindMatches, firstTest)
{
  cugraph::db::db_pattern<int32_t> p;
  cugraph::db::db_pattern_entry<int32_t> p1(0);
  cugraph::db::db_pattern_entry<int32_t> p2("a");
  cugraph::db::db_pattern_entry<int32_t> p3("b");
  p.addEntry(p1);
  p.addEntry(p2);
  p.addEntry(p3);
  cugraph::db::db_result<int32_t> result =
    cugraph::db::findMatches<int32_t>(p, table, nullptr, 0, 1);
  ASSERT_EQ(result.getSize(), 1);
  std::vector<int32_t> resultA(result.getSize());
  std::vector<int32_t> resultB(result.getSize());
  CUDA_TRY(cudaMemcpy(
    resultA.data(), result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault));
  CUDA_TRY(cudaMemcpy(
    resultB.data(), result.getData("b"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault));
  ASSERT_EQ(resultA[0], 1);
  ASSERT_EQ(resultB[0], 2);
}

TEST_F(Test_FindMatches, secondTest)
{
  insertConstantEntry(0, 1, 1);
  insertConstantEntry(2, 0, 1);
  table.flush_input();

  std::cout << table.toString() << "\n\n";

  std::cout << table.getIndex(2).toString() << "\n";

  cugraph::db::db_pattern<int32_t> q;
  cugraph::db::db_pattern_entry<int32_t> q1(0);
  cugraph::db::db_pattern_entry<int32_t> q2("a");
  cugraph::db::db_pattern_entry<int32_t> q3("b");
  q.addEntry(q1);
  q.addEntry(q2);
  q.addEntry(q3);

  cugraph::db::db_result<int32_t> result =
    cugraph::db::findMatches<int32_t>(q, table, nullptr, 0, 2);

  std::cout << result.toString();

  ASSERT_EQ(result.getSize(), 2);
  std::vector<int32_t> resultA(result.getSize());
  std::vector<int32_t> resultB(result.getSize());
  CUDA_TRY(cudaMemcpy(
    resultA.data(), result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault));
  CUDA_TRY(cudaMemcpy(
    resultB.data(), result.getData("b"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault));

  ASSERT_EQ(resultA[0], 1);
  ASSERT_EQ(resultB[0], 1);
  ASSERT_EQ(resultA[1], 1);
  ASSERT_EQ(resultB[1], 2);
}

TEST_F(Test_FindMatches, thirdTest)
{
  insertConstantEntry(1, 1, 2);
  insertConstantEntry(2, 1, 2);
  table.flush_input();

  cugraph::db::db_pattern<int32_t> q;
  cugraph::db::db_pattern_entry<int32_t> q1("a");
  cugraph::db::db_pattern_entry<int32_t> q2(1);
  cugraph::db::db_pattern_entry<int32_t> q3(2);
  q.addEntry(q1);
  q.addEntry(q2);
  q.addEntry(q3);

  rmm::device_buffer frontier(sizeof(int32_t));
  int32_t* frontier_ptr = reinterpret_cast<int32_t*>(frontier.data());
  thrust::fill(thrust::device, frontier_ptr, frontier_ptr + 1, 0);

  cugraph::db::db_result<int32_t> result =
    cugraph::db::findMatches<int32_t>(q, table, frontier_ptr, 1, 0);

  ASSERT_EQ(result.getSize(), 1);
  std::vector<int32_t> resultA(result.getSize());
  CUDA_TRY(cudaMemcpy(
    resultA.data(), result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault));

  std::cout << result.toString();

  ASSERT_EQ(resultA[0], 0);
}

TEST_F(Test_FindMatches, fourthTest)
{
  insertConstantEntry(1, 1, 2);
  insertConstantEntry(2, 1, 2);
  table.flush_input();

  cugraph::db::db_pattern<int32_t> q;
  cugraph::db::db_pattern_entry<int32_t> q1("a");
  cugraph::db::db_pattern_entry<int32_t> q2(1);
  cugraph::db::db_pattern_entry<int32_t> q3(2);
  cugraph::db::db_pattern_entry<int32_t> q4("r");
  q.addEntry(q1);
  q.addEntry(q2);
  q.addEntry(q3);
  q.addEntry(q4);

  cugraph::db::db_result<int32_t> result =
    cugraph::db::findMatches<int32_t>(q, table, nullptr, 0, 0);
  std::cout << result.toString();
  ASSERT_EQ(result.getSize(), 3);

  std::vector<int32_t> resultA(result.getSize());
  std::vector<int32_t> resultR(result.getSize());

  CUDA_TRY(cudaMemcpy(
    resultA.data(), result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault));
  CUDA_TRY(cudaMemcpy(
    resultR.data(), result.getData("r"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault));

  ASSERT_EQ(resultA[0], 0);
  ASSERT_EQ(resultA[1], 1);
  ASSERT_EQ(resultA[2], 2);
  ASSERT_EQ(resultR[0], 0);
  ASSERT_EQ(resultR[1], 1);
  ASSERT_EQ(resultR[2], 2);
}

TEST_F(Test_FindMatches, fifthTest)
{
  insertConstantEntry(0, 1, 3);
  insertConstantEntry(0, 2, 1);
  insertConstantEntry(0, 2, 2);
  table.flush_input();

  cugraph::db::db_pattern<int32_t> q;
  cugraph::db::db_pattern_entry<int32_t> q1("a");
  cugraph::db::db_pattern_entry<int32_t> q2(1);
  cugraph::db::db_pattern_entry<int32_t> q3("b");
  q.addEntry(q1);
  q.addEntry(q2);
  q.addEntry(q3);

  cugraph::db::db_result<int32_t> result =
    cugraph::db::findMatches<int32_t>(q, table, nullptr, 0, 1);
  std::cout << result.toString();

  ASSERT_EQ(result.getSize(), 2);
  std::vector<int32_t> resultA(result.getSize());
  std::vector<int32_t> resultB(result.getSize());

  CUDA_TRY(cudaMemcpy(
    resultA.data(), result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault));
  CUDA_TRY(cudaMemcpy(
    resultB.data(), result.getData("b"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault));

  ASSERT_EQ(resultA[0], 0);
  ASSERT_EQ(resultA[1], 0);
  ASSERT_EQ(resultB[0], 2);
  ASSERT_EQ(resultB[1], 3);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}
