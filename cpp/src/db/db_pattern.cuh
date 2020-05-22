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
#include <map>
#include <string>
#include <vector>

namespace cugraph {
namespace db {

enum class value_type { String, Identifier, Array_Access, Property_Access, Uninitialized };

class value {
  value_type valType;
  std::string identifier;
  std::string prop_identifier;
  uint32_t array_index;

 public:
  value();
  value(value_type type, std::string id);
  value(std::string id, std::string prop);
  value(std::string id, uint32_t idx);
  value(const value& other) = default;
  value(value&& other)      = default;
  value& operator=(const value& other) = default;
  value& operator=(value&& other) = default;
  value_type type();
  std::string getIdentifier();
  std::string getPropertyName();
  uint32_t getArrayIndex();
};

enum class pattern_type { Node, Relationship };

template <typename idx_t>
class pattern_element {
 public:
  virtual ~pattern_element()          = default;
  virtual std::string getIdentifier() = 0;
  virtual pattern_type type()         = 0;
};

template <typename idx_t>
class node_pattern : public pattern_element<idx_t> {
  std::string identifier;
  std::vector<std::string> labels;
  std::map<std::string, value> properties;

 public:
  node_pattern() = default;
  node_pattern(std::string id);
  node_pattern(const cypher_astnode_t* astNode, context<idx_t>& ctx);
  ~node_pattern() = default;
  void setIdentifier(std::string id);
  void addLabel(std::string label);
  void addProperty(std::string name, value val);
  std::string getIdentifier() override;
  pattern_type type() override;
  std::vector<std::string>& getLabels();
  std::map<std::string, value>& getProperties();
};

template <typename idx_t>
class relationship_pattern : public pattern_element<idx_t> {
  std::string identifier;
  uint32_t direction;
  std::string startId;
  std::string endId;
  std::vector<value> relationshipTypes;
  std::map<std::string, value> properties;

 public:
  relationship_pattern();
  relationship_pattern(const cypher_astnode_t* astNode, context<idx_t>& ctx);
  void addProperty(std::string name, value value);
  void setStart(std::string start);
  void setEnd(std::string end);
  void addType(value type);
  void setDirection(uint32_t dir);
  std::string getStart();
  std::string getEnd();
  uint32_t getDirection();
  std::vector<value>& getTypes();
  std::map<std::string, value>& getProperties();
  std::string getIdentifier() override;
  pattern_type type() override;
};

template <typename idx_t>
class pattern_path {
  std::vector<pattern_element<idx_t>*> path;

 public:
  pattern_path() = default;
  pattern_path(const cypher_astnode_t* astNode, context<idx_t>& ctx);
  pattern_path(const pattern_path& other) = delete;
  pattern_path(pattern_path&& other);
  ~pattern_path();
  pattern_path& operator=(const pattern_path& other) = delete;
  pattern_path& operator                             =(pattern_path&& other);
  std::vector<pattern_element<idx_t>*>& getPathNodes();
};

}  // namespace db
}  // namespace cugraph
