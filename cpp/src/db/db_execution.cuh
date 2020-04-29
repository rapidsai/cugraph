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

#include <string>
#include <cypher-parser.h>
#include <db_object.cuh>

namespace cugraph {
namespace db {

/**
 * Class which encapsulates a table of strings, such as that used to store the
 * result of a LOAD CSV command.
 */
class string_table {
  std::vector<std::vector<std::string>>columns;
  std::vector<std::string>names;

public:
  string_table();
  string_table(std::string csvFile, bool with_headers);
  std::vector<std::string>& operator[](std::string colName);
};

/**
 * Super class from which all execution node sub-types inherit
 */
template <typename idx_t>
class execution_node {
public:
  virtual void ~execution_node() = 0;
  virtual void execute() = 0;
  virtual string_table& getStringResult() = 0;
  virtual db_result<idx_t>& getGPUResult() = 0;
  virtual std::string getResultIdentifier() = 0;
};

/**
 * Class which encapsulates a load csv node in the execution tree
 */
class load_csv_node {
  bool with_headers;
  std::string filename;
  std::string identifier;
  string_table result;
public:
  load_csv_node();
  load_csv_node(std::string filename, bool with_headers, std::string identifier);
  load_csv_node(const cypher_astnode_t* astNode);
  string_table& getResult();
  std::string getIdentifier();
};

/**
 * Class which encapsulates a match node in the execution tree
 */
class match_node {

};

/**
 * Class which  encapsulates a create node in the execution tree
 */
class create_node {

};

/**
 * Class which encapsulates a projection of the result set in the
 * execution tree.
 */
class return_node {

};

template <typename IdxT>
class query_plan {
  load_csv_node loadCsv;
  match_node match;
  create_node create;
  return_node project;
public:
  query_plan();
  query_plan(const cypher_parse_result_t* parseResult);

};


} // namespace db
} // namespace cugraph
