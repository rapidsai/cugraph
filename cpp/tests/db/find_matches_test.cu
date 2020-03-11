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

#include "gtest/gtest.h"
#include "high_res_clock.h"
#include <cugraph.h>
#include "test_utils.h"
#include "db/db_operators.cuh"
#include "utilities/graph_utils.cuh"

class Test_FindMatches: public ::testing::Test {
public:
  Test_FindMatches() {}
  virtual void SetUp() {
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
  void insertConstantEntry(int32_t a, int32_t b, int32_t c) {
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

TEST_F(Test_FindMatches, verifyIndices) {
  int32_t* offsets_d = reinterpret_cast<int32_t*>(table.getIndex(0).getOffsets()->data);
  int32_t offsetsSize = table.getIndex(0).getOffsets()->size;
  int32_t* indirection_d = reinterpret_cast<int32_t*>(table.getIndex(0).getIndirection()->data);
  int32_t indirectionSize = table.getIndex(0).getIndirection()->size;
  int32_t* offsets_h = new int32_t[offsetsSize];
  int32_t* indirection_h = new int32_t[indirectionSize];
  cudaMemcpy(offsets_h, offsets_d, sizeof(int32_t) * offsetsSize, cudaMemcpyDefault);
  cudaMemcpy(indirection_h, indirection_d, sizeof(int32_t) * indirectionSize, cudaMemcpyDefault);
  std::cout << "Offsets[0]: ";
  for (int i = 0; i < offsetsSize; i++)
    std::cout << offsets_h[i] << " ";
  std::cout << "\n";
  std::cout << "Indirection[0]: ";
  for (int i = 0; i < indirectionSize; i++)
    std::cout << indirection_h[i] << " ";
  std::cout << "\n";
  delete[] offsets_h;
  delete[] indirection_h;

  offsets_d = reinterpret_cast<int32_t*>(table.getIndex(1).getOffsets()->data);
  offsetsSize = table.getIndex(1).getOffsets()->size;
  indirection_d = reinterpret_cast<int32_t*>(table.getIndex(1).getIndirection()->data);
  indirectionSize = table.getIndex(1).getIndirection()->size;
  offsets_h = new int32_t[offsetsSize];
  indirection_h = new int32_t[indirectionSize];
  cudaMemcpy(offsets_h, offsets_d, sizeof(int32_t) * offsetsSize, cudaMemcpyDefault);
  cudaMemcpy(indirection_h, indirection_d, sizeof(int32_t) * indirectionSize, cudaMemcpyDefault);
  std::cout << "Offsets[1]: ";
  for (int i = 0; i < offsetsSize; i++)
    std::cout << offsets_h[i] << " ";
  std::cout << "\n";
  std::cout << "Indirection[1]: ";
  for (int i = 0; i < indirectionSize; i++)
    std::cout << indirection_h[i] << " ";
  std::cout << "\n";
  delete[] offsets_h;
  delete[] indirection_h;

  offsets_d = reinterpret_cast<int32_t*>(table.getIndex(2).getOffsets()->data);
  offsetsSize = table.getIndex(2).getOffsets()->size;
  indirection_d = reinterpret_cast<int32_t*>(table.getIndex(2).getIndirection()->data);
  indirectionSize = table.getIndex(2).getIndirection()->size;
  offsets_h = new int32_t[offsetsSize];
  indirection_h = new int32_t[indirectionSize];
  cudaMemcpy(offsets_h, offsets_d, sizeof(int32_t) * offsetsSize, cudaMemcpyDefault);
  cudaMemcpy(indirection_h, indirection_d, sizeof(int32_t) * indirectionSize, cudaMemcpyDefault);
  std::cout << "Offsets[2]: ";
  for (int i = 0; i < offsetsSize; i++)
    std::cout << offsets_h[i] << " ";
  std::cout << "\n";
  std::cout << "Indirection[2]: ";
  for (int i = 0; i < indirectionSize; i++)
    std::cout << indirection_h[i] << " ";
  std::cout << "\n";
  delete[] offsets_h;
  delete[] indirection_h;
}

TEST_F(Test_FindMatches, firstTest){
  cugraph::db::db_pattern<int32_t> p;
  cugraph::db::db_pattern_entry<int32_t> p1(0);
  cugraph::db::db_pattern_entry<int32_t> p2("a");
  cugraph::db::db_pattern_entry<int32_t> p3("b");
  p.addEntry(p1);
  p.addEntry(p2);
  p.addEntry(p3);
  cugraph::db::db_result<int32_t> result = cugraph::db::findMatches(p, table, nullptr, 1);
  ASSERT_EQ(result.getSize(), 1);
  int32_t* resultA = new int32_t[result.getSize()];
  int32_t* resultB = new int32_t[result.getSize()];
  cudaMemcpy(resultA, result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault);
  cudaMemcpy(resultB, result.getData("b"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault);
  ASSERT_EQ(resultA[0], 1);
  ASSERT_EQ(resultB[0], 2);

  delete[] resultA;
  delete[] resultB;
}

/*
TEST_F(Test_FindMatches, secondTest) {
  insertConstantEntry(0, 1, 1);
  insertConstantEntry(2, 0, 1);
  table.flush_input();

  cugraph::db::db_pattern<int32_t> q;
  cugraph::db::db_pattern_entry<int32_t> q1(0);
  cugraph::db::db_pattern_entry<int32_t> q2("a");
  cugraph::db::db_pattern_entry<int32_t> q3("b");
  q.addEntry(q1);
  q.addEntry(q2);
  q.addEntry(q3);

  cugraph::db::db_result<int32_t> result = cugraph::db::findMatches(q, table, nullptr, 2);

  ASSERT_EQ(result.getSize(), 2);
  int32_t* resultA = new int32_t[result.getSize()];
  int32_t* resultB = new int32_t[result.getSize()];
  cudaMemcpy(resultA, result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault);
  cudaMemcpy(resultB, result.getData("b"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault);

  std::cout << result.toString();

  ASSERT_EQ(resultA[0], 1);
  ASSERT_EQ(resultB[0], 1);
  ASSERT_EQ(resultA[1], 1);
  ASSERT_EQ(resultB[1], 2);

  delete[] resultA;
  delete[] resultB;
}
*/
TEST_F(Test_FindMatches, thirdTest) {
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

  int32_t* frontier_ptr;
  cudaMalloc(&frontier_ptr, sizeof(int32_t));
  thrust::fill(thrust::device, frontier_ptr, frontier_ptr + 1, 0);
  gdf_column* frontier = (gdf_column*)malloc(sizeof(gdf_column));
  cugraph::detail::gdf_col_set_defaults(frontier);
  gdf_column_view(frontier, frontier_ptr, nullptr, 1, GDF_INT32);

  cugraph::db::db_result<int32_t> result = cugraph::db::findMatches(q, table, frontier, 0);

  ASSERT_EQ(result.getSize(), 1);
  int32_t* resultA = new int32_t[result.getSize()];
  cudaMemcpy(resultA, result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault);

  std::cout << result.toString();

  ASSERT_EQ(resultA[0], 0);
  delete[] resultA;
}

TEST_F(Test_FindMatches, fourthTest) {
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

  cugraph::db::db_result<int32_t> result = cugraph::db::findMatches(q, table, nullptr, 0);
  std::cout << result.toString();
  ASSERT_EQ(result.getSize(), 3);

  int32_t* resultA = new int32_t[result.getSize()];
  cudaMemcpy(resultA, result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault);
  int32_t* resultR = new int32_t[result.getSize()];
  cudaMemcpy(resultR, result.getData("r"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault);
  ASSERT_EQ(resultA[0], 0);
  ASSERT_EQ(resultA[1], 1);
  ASSERT_EQ(resultA[2], 2);
  ASSERT_EQ(resultR[0], 0);
  ASSERT_EQ(resultR[1], 1);
  ASSERT_EQ(resultR[2], 2);
  delete[] resultA;
  delete[] resultR;
}
/*
TEST_F(Test_FindMatches, fifthTest) {
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

  cugraph::db::db_result<int32_t> result = cugraph::db::findMatches(q, table, nullptr, 1);
  std::cout << result.toString();

  ASSERT_EQ(result.getSize(), 2);
  int32_t* resultA = new int32_t[result.getSize()];
  int32_t* resultB = new int32_t[result.getSize()];
  cudaMemcpy(resultA, result.getData("a"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault);
  cudaMemcpy(resultB, result.getData("b"), sizeof(int32_t) * result.getSize(), cudaMemcpyDefault);

  ASSERT_EQ(resultA[0], 0);
  ASSERT_EQ(resultA[1], 0);
  ASSERT_EQ(resultB[0], 2);
  ASSERT_EQ(resultB[1], 3);

  delete[] resultA;
  delete[] resultB;
}
*/
int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}