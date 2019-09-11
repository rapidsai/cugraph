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

class Test_FindMatches: public ::testing::Test {
public:
  Test_FindMatches() {}
  virtual void SetUp() {
    cugraph::db_pattern<int32_t> p;
    cugraph::db_pattern_entry<int32_t> p1(0);
    cugraph::db_pattern_entry<int32_t> p2(1);
    cugraph::db_pattern_entry<int32_t> p3(2);
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
  cugraph::db_table<int32_t> table;
};

TEST_F(Test_FindMatches, firstTest){
  cugraph::db_pattern<int32_t> p;
  cugraph::db_pattern_entry<int32_t> p1(0);
  cugraph::db_pattern_entry<int32_t> p2("a");
  cugraph::db_pattern_entry<int32_t> p3("b");
  p.addEntry(p1);
  p.addEntry(p2);
  p.addEntry(p3);
  cugraph::db_result<int32_t> result = cugraph::db::findMatches(p, table, nullptr, "a");
}

int main(int argc, char**argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
