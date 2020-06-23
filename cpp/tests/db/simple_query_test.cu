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

#include "db/db_object.cuh"
#include "gtest/gtest.h"
#include "utilities/graph_utils.cuh"

class Test_Parser : public ::testing::Test {
};

TEST_F(Test_Parser, checkToSTring)
{
  std::string expected =
    "Database:\nEncoder:\nEncoder object with 5 encoded values and 2 allocated ids:\n"
    "2 == Person\n3 == name\n4 == George\n5 == FriendsWith\n6 == Henry\n"
    "Relationships Table:\nTable with 3 columns of length 1\nbegin end type \n0 5 1 \n"
    "Node Labels Table:\nTable with 2 columns of length 2\nnodeId LabelId \n0 2 \n1 2 \n"
    "Node Properties Table:\nTable with 3 columns of length 2\nnodeId propertyLabel value \n"
    "0 3 4 \n1 3 6 \nRelationship Properties Table:\nTable with 3 columns of length 0\n"
    "id name value \n";

  cugraph::db::db_object<int32_t> obj;
  std::string input = "CREATE (:Person {name:'George'})-[:FriendsWith]->(:Person {name : 'Henry'})";
  obj.query(input);
  std::string actual = obj.toString();
  ASSERT_EQ(actual, expected);
}
