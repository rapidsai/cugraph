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

#include <utilities/error_utils.h>
#include <db/db_pattern.cuh>

namespace cugraph {
namespace db {

value::value()
{
  valType     = value_type::Uninitialized;
  array_index = 0;
}

value::value(value_type type, std::string id)
{
  CUGRAPH_EXPECTS(type == value_type::String || type == value_type::Identifier,
                  "Invalid value_type given");
  valType     = type;
  identifier  = id;
  array_index = 0;
}

value::value(std::string id, std::string prop)
{
  valType         = value_type::Property_Access;
  identifier      = id;
  prop_identifier = prop;
  array_index     = 0;
}

value::value(std::string id, uint32_t idx)
{
  valType     = value_type::Array_Access;
  identifier  = id;
  array_index = idx;
}

value_type value::type() { return valType; }

std::string value::getIdentifier()
{
  CUGRAPH_EXPECTS(valType != value_type::Uninitialized, "Can't access an unitialized value");
  return identifier;
}

std::string value::getPropertyName()
{
  CUGRAPH_EXPECTS(valType != value_type::Uninitialized, "Can't access an unitialized value");
  CUGRAPH_EXPECTS(valType == value_type::Property_Access,
                  "Value type inconsistent with attempted access");
  return prop_identifier;
}

uint32_t value::getArrayIndex()
{
  CUGRAPH_EXPECTS(valType != value_type::Uninitialized, "Can't access an unitialized value");
  CUGRAPH_EXPECTS(valType == value_type::Array_Access,
                  "Value type inconsistent with attempted access");
  return array_index;
}

node_pattern::node_pattern(std::string id) { identifier = id; }

node_pattern::node_pattern(const cypher_astnode_t* astNode)
{
  // Check that the given astnode is the right type:
  cypher_astnode_type_t astNodeType = cypher_astnode_type(astNode);
  std::string typeStr               = cypher_astnode_typestr(astNodeType);
  CUGRAPH_EXPECTS(typeStr == "node pattern", "Wrong type of astnode supplied");
}

void node_pattern::setIdentifier(std::string id) { identifier = id; }

void node_pattern::addLabel(value label) { labels.push_back(label); }

void node_pattern::addProperty(std::string name, value val) { properties[name] = val; }

std::string node_pattern::getIdentifier() { return identifier; }

pattern_type node_pattern::type() { return pattern_type::Node; }

std::vector<value>& node_pattern::getLabels() { return labels; }

std::map<std::string, value>& node_pattern::getProperties() { return properties; }

relationship_pattern::relationship_pattern() { direction = 1; }

relationship_pattern::relationship_pattern(const cypher_astnode_t* astNode)
{
  // Check that the given astnode is the right type:
  cypher_astnode_type_t astNodeType = cypher_astnode_type(astNode);
  std::string typeStr               = cypher_astnode_typestr(astNodeType);
  CUGRAPH_EXPECTS(typeStr == "rel pattern", "Wrong type of astnode supplied");

  // Extract the direction of the relation
  direction = cypher_ast_rel_pattern_get_direction(astNode);

  // Extract the identifier or if null assign a unique identifier
  const cypher_astnode_t* id = cypher_ast_rel_pattern_get_identifier(astNode);
  if (id != nullptr) {
  } else {
  }
}

void relationship_pattern::addProperty(std::string name, value val) { properties[name] = val; }

void relationship_pattern::setStart(std::string start) { startId = start; }

void relationship_pattern::setEnd(std::string end) { endId = end; }

void relationship_pattern::addType(value type) { relationshipTypes.push_back(type); }

void relationship_pattern::setDirection(uint32_t dir) { direction = dir; }

std::string relationship_pattern::getStart() { return startId; }

std::string relationship_pattern::getEnd() { return endId; }

std::vector<value>& relationship_pattern::getTypes() { return relationshipTypes; }

std::map<std::string, value>& relationship_pattern::getProperties() { return properties; }

std::string relationship_pattern::getIdentifier() { return identifier; }

pattern_type relationship_pattern::type() { return pattern_type::Relationship; }

pattern_path::pattern_path(const cypher_astnode_t* astNode)
{
  // Check that the given astnode is the right type:
  cypher_astnode_type_t astNodeType = cypher_astnode_type(astNode);
  std::string typeStr               = cypher_astnode_typestr(astNodeType);
  CUGRAPH_EXPECTS(typeStr == "pattern path", "Wrong type of astnode supplied");
}

pattern_path::pattern_path(pattern_path&& other)
{
  path = std::move(other.path);
  other.path.clear();
}

pattern_path::~pattern_path()
{
  for (size_t i = 0; i < path.size(); i++) delete path[i];
}

pattern_path& pattern_path::operator=(pattern_path&& other)
{
  if (this != &other) {
    path = std::move(other.path);
    other.path.clear();
  }
  return *this;
}

}  // namespace db
}  // namespace cugraph
