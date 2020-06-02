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

#include <cugraph.h>
#include "db/db_object.cuh"
#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "test_utils.h"
#include "utilities/graph_utils.cuh"

class Test_Parser : public ::testing::Test {
};

TEST_F(Test_Parser, printOut)
{
  cugraph::db::db_object<int32_t> obj;
  std::string input = "CREATE (:Person {name:'George'})-[:FriendsWith]->(:Person {name : 'Henry'})";
  obj.query(input);
  std::cout << obj.toString();
}
