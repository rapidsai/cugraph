/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#pragma once

#include <cypher-parser.h>
#include <db/db_context.cuh>
#include <db/db_pattern.cuh>
#include <db/db_results.cuh>
#include <map>
#include <string>
#include <vector>

namespace cugraph {
namespace db {
enum class execution_type { LoadCsv, Match, Create, Merge, Return };

/**
 * Super class from which all execution node sub-types inherit
 */
template <typename idx_t>
class execution_node {
 public:
  virtual ~execution_node()                 = default;
  virtual void execute()                    = 0;
  virtual string_table& getStringResult()   = 0;
  virtual db_result<idx_t>& getGPUResult()  = 0;
  virtual std::string getResultIdentifier() = 0;
  virtual execution_type type()             = 0;
};

/**
 * Class which encapsulates a load csv node in the execution tree
 */
template <typename idx_t>
class load_csv_node : public execution_node<idx_t> {
  bool with_headers = false;
  std::string filename;
  std::string identifier;
  std::string delimiter;
  string_table result;
  bool executed = false;

 public:
  load_csv_node() = default;
  load_csv_node(const cypher_astnode_t* astNode);
  ~load_csv_node() = default;
  string_table& getStringResult() override;
  db_result<idx_t>& getGPUResult() override;
  std::string getResultIdentifier() override;
  void execute() override;
  execution_type type() override;
};

/**
 * Class which encapsulates a match node in the execution tree
 */
template <typename idx_t>
class match_node : public execution_node<idx_t> {
  std::vector<pattern_path<idx_t>> paths;
  db_result<idx_t> result;
  bool executed = false;

 public:
  match_node() = default;
  match_node(const cypher_astnode_t* astNode, context<idx_t>& ctx);
  match_node(const match_node& other) = delete;
  match_node(match_node&& other)      = default;
  match_node& operator=(const match_node& other) = delete;
  match_node& operator=(match_node&& other) = default;
  string_table& getStringResult() override;
  db_result<idx_t>& getGPUResult() override;
  std::string getResultIdentifier() override;
  void execute() override;
  execution_type type() override;
};

/**
 * Class which  encapsulates a create node in the execution tree
 */
template <typename idx_t>
class create_node : public execution_node<idx_t> {
  std::vector<pattern_path<idx_t>> paths;
  bool executed = false;

 public:
  create_node() = default;
  create_node(const cypher_astnode_t* astNode, context<idx_t>& ctx);
  create_node(const create_node& other) = delete;
  create_node(create_node&& other)      = default;
  create_node& operator=(const create_node& other) = delete;
  create_node& operator=(create_node&& other) = default;
  string_table& getStringResult() override;
  db_result<idx_t>& getGPUResult() override;
  std::string getResultIdentifier() override;
  void execute() override;
  execution_type type() override;
};

/**
 * Class which encapsulates a merge node in the execution tree
 */
template <typename idx_t>
class merge_node : public execution_node<idx_t> {
};

/**
 * Class which encapsulates a projection of the result set in the
 * execution tree.
 */
template <typename idx_t>
class return_node : public execution_node<idx_t> {
};

/**
 * Class which represents a full query plan
 */
template <typename idx_t>
class query_plan {
  std::vector<execution_node<idx_t>*> plan_nodes;
  context<idx_t> ctx;

 public:
  query_plan() = default;
  query_plan(const cypher_parse_result_t* parseResult, context<idx_t> ctx);
  query_plan(const query_plan& other) = delete;
  query_plan(query_plan&& other);
  ~query_plan();
  query_plan& operator=(const query_plan& other) = delete;
  query_plan& operator                           =(query_plan&& other);
  std::string execute();
};

}  // namespace db
}  // namespace cugraph
