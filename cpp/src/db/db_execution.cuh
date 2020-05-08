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

#include <cypher-parser.h>
#include <db/db_object.cuh>
#include <string>
#include <vector>

namespace cugraph {
namespace db {

/**
 * Class which encapsulates a table of strings, such as that used to store the
 * result of a LOAD CSV command.
 */
class string_table {
  std::vector<std::vector<std::string>> columns;
  std::vector<std::string> names;

 public:
  string_table();
  string_table(const string_table& other) = delete;
  string_table(string_table&& other);
  string_table(std::string csvFile, bool with_headers, std::string delim = ",");
  string_table& operator=(const string_table& other) = delete;
  string_table& operator                             =(string_table&& other);
  std::vector<std::string>& operator[](std::string colName);
  std::vector<std::string>& operator[](int colIdx);
};

/**
 * Super class from which all execution node sub-types inherit
 */
template <typename idx_t>
class execution_node {
 public:
  virtual ~execution_node()            = default;
  virtual void execute()                    = 0;
  virtual string_table& getStringResult()   = 0;
  virtual db_result<idx_t>& getGPUResult()  = 0;
  virtual std::string getResultIdentifier() = 0;
};

/**
 * Class which encapsulates a load csv node in the execution tree
 */
template <typename idx_t>
class load_csv_node : public execution_node<idx_t> {
  bool with_headers;
  std::string filename;
  std::string identifier;
  std::string delimiter;
  string_table result;

 public:
  load_csv_node() = default;
  load_csv_node(const cypher_astnode_t* astNode);
  ~load_csv_node() = default;
  string_table& getStringResult() override;
  db_result<idx_t>& getGPUResult() override;
  std::string getResultIdentifier() override;
  void execute() override;
};

enum class pattern_type {
  Node,
  Relationship
};

class pattern_element {
public:
  virtual ~pattern_element() = default;
  virtual std::string getIdentifier() = 0;
  virtual pattern_type type() = 0;
};

class node_pattern: public pattern_element {
  std::string identifier;
  std::vector<std::string> labels;
  std::map<std::string, std::string> properties;
public:
  node_pattern() = default;
  node_pattern(std::string id);
  ~node_pattern() = default;
  void setIdentifier(std::string id);
  void addLabel(std::string label);
  void addProperty(std::string name, std::string value);
  std::string getIdentifier() override;
  pattern_type type() override;
  std::vector<std::string>& getLabels();
  std::map<std::string, std::string>& getProperties();
};

class relationship_pattern: public pattern_element {
  std::string identifier;
  uint32_t direction;
  std::string startId;
  std::string endId;
  std::vector<std::string> relationshipTypes;
  std::map<std::string, std::string> properties;
public:
  relationship_pattern();
  relationship_pattern(const cypher_astnode_t* astNode);
  void addProperty(std::string name, std::string value);
  void setStart(std::string start);
  void setEnd(std::string end);
  void addType(std::string type);
  void setDirection(uint32_t dir);
  std::string getStart();
  std::string getEnd();
  uint32_t getDirection();
  std::vector<std::string>& getTypes();
  std::map<std::string, std::string>& getProperties();
  std::string getIdentifier() override;
  pattern_type type() override;
};

class pattern_path {
  std::vector<pattern_element*> path;
public:
  pattern_path() = default;
  pattern_path(const cypher_astnode_t* astNode);
  pattern_path(const pattern_path& other) = delete;
  pattern_path(pattern_path&& other);
  ~pattern_path();
  pattern_path& operator=(const pattern_path& other) = delete;
  pattern_path& operator=(pattern_path&& other);
  std::vector<pattern_element*>& getPathNodes();
};

/**
 * Class which encapsulates a match node in the execution tree
 */
template <typename idx_t>
class match_node : public execution_node<idx_t> {
  std::vector<pattern_path> paths;
  db_result<idx_t> result;
public:
  match_node() = default;
  match_node(const cypher_astnode_t* astNode);
  string_table& getStringResult() override;
  db_result<idx_t>& getGPUResult() override;
  std::string getResultIdentifier() override;
  void execute() override;
};

/**
 * Class which  encapsulates a create node in the execution tree
 */
template <typename idx_t>
class create_node : public execution_node<idx_t> {
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

template <typename IdxT>
class query_plan {

 public:
  query_plan();
  query_plan(const cypher_parse_result_t* parseResult);
};

}  // namespace db
}  // namespace cugraph
