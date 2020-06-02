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
#include "db/db_operators.cuh"
#include "db/db_parser_integration_test.cuh"
#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "test_utils.h"
#include "utilities/graph_utils.cuh"

class Test_Parser : public ::testing::Test {
};

TEST_F(Test_Parser, printOut)
{
  std::string input =
    "LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS csvLine\nCREATE (p:Person {id: "
    "toInteger(csvLine.id), name: csvLine.name})";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";

  input =
    "LOAD CSV WITH HEADERS FROM 'file:///movies.csv' AS csvLine\nMERGE (country:Country {name: "
    "csvLine.country})\nCREATE (movie:Movie {id: toInteger(csvLine.id), title: csvLine.title, "
    "year:toInteger(csvLine.year)})\nCREATE (movie)-[:MADE_IN]->(country)";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";

  input =
    "LOAD CSV WITH HEADERS FROM 'file:///roles.csv' AS csvLine\nMATCH (person:Person {id: "
    "toInteger(csvLine.personId)}),(movie:Movie {id: toInteger(csvLine.movieId)})\nCREATE "
    "(person)-[:PLAYED {role: csvLine.role}]->(movie)";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";

  input =
    "LOAD CSV FROM 'file:///roles.csv' AS csvLine FIELDTERMINATOR ';'\nCREATE (person:Person "
    "{name: csvLine[0]})";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";

  //  std::string input = "LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS csvLine";
  //  std::cout << input << "\n";
  //  cugraph::db::printOutAst(input);
  //  std::cout << "\n";
  //
  //  input = "MATCH (p:Person {name: 'James'})-[:HasPet]->(z:Pet)\nRETURN z.name";
  //  std::cout << input << "\n";
  //  cugraph::db::printOutAst(input);
  //  std::cout << "\n";
  //
  //  input = "CREATE (p:Person {name: 'James'})";
  //  std::cout << input << "\n";
  //  cugraph::db::printOutAst(input);
  //  std::cout << "\n";
  //
  //  input = "LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS csvLine\nCREATE (n:Person {name:
  //  csvLine.name})"; std::cout << input << "\n"; cugraph::db::printOutAst(input); std::cout <<
  //  "\n";
  //
  //  input = "LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS csvLine\nMATCH (person:Person
  //  {id: csvLine.personId}),(movie:Movie {id: csvLine.movieId})\nCREATE (person)-[:Played {role:
  //  csvLine.role}]->(movie)"; std::cout << input << "\n"; cugraph::db::printOutAst(input);
  //  std::cout << "\n";
  //
  input =
    "MATCH (p:Person)-[:FriendsWith]-(i:Person)-[:FriendsWith]-(q:Person)\nRETURN p.name, q.name";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";

  input =
    "MATCH (p:Person:Human)-[r:FriendsWith | AssociateOf "
    "{KnowsFrom:'School',KnowsFrom:'Work'}]->(q:Person)\n"
    "RETURN p.name, q.name";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";

  input = "CREATE (:Person)-[:FriendsWith]->(:Person)";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";
}
