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

template <typename idx_t>
node_pattern<idx_t>::node_pattern(std::string id)
{
  identifier = id;
}

template <typename idx_t>
node_pattern<idx_t>::node_pattern(const cypher_astnode_t* astNode, context<idx_t> ctx)
{
  // Check that the given astnode is the right type:
  cypher_astnode_type_t astNodeType = cypher_astnode_type(astNode);
  std::string typeStr               = cypher_astnode_typestr(astNodeType);
  CUGRAPH_EXPECTS(typeStr == "node pattern", "Wrong type of astnode supplied");
}

template <typename idx_t>
void node_pattern<idx_t>::setIdentifier(std::string id)
{
  identifier = id;
}

template <typename idx_t>
void node_pattern<idx_t>::addLabel(value label)
{
  labels.push_back(label);
}

template <typename idx_t>
void node_pattern<idx_t>::addProperty(std::string name, value val)
{
  properties[name] = val;
}

template <typename idx_t>
std::string node_pattern<idx_t>::getIdentifier()
{
  return identifier;
}

template <typename idx_t>
pattern_type node_pattern<idx_t>::type()
{
  return pattern_type::Node;
}

template <typename idx_t>
std::vector<value>& node_pattern<idx_t>::getLabels()
{
  return labels;
}

template <typename idx_t>
std::map<std::string, value>& node_pattern<idx_t>::getProperties()
{
  return properties;
}

template <typename idx_t>
relationship_pattern<idx_t>::relationship_pattern()
{
  direction = 1;
}

template <typename idx_t>
relationship_pattern<idx_t>::relationship_pattern(const cypher_astnode_t* astNode,
                                                  context<idx_t> ctx)
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

template <typename idx_t>
void relationship_pattern<idx_t>::addProperty(std::string name, value val)
{
  properties[name] = val;
}

template <typename idx_t>
void relationship_pattern<idx_t>::setStart(std::string start)
{
  startId = start;
}

template <typename idx_t>
void relationship_pattern<idx_t>::setEnd(std::string end)
{
  endId = end;
}

template <typename idx_t>
void relationship_pattern<idx_t>::addType(value type)
{
  relationshipTypes.push_back(type);
}

template <typename idx_t>
void relationship_pattern<idx_t>::setDirection(uint32_t dir)
{
  direction = dir;
}

template <typename idx_t>
std::string relationship_pattern<idx_t>::getStart()
{
  return startId;
}

template <typename idx_t>
std::string relationship_pattern<idx_t>::getEnd()
{
  return endId;
}

template <typename idx_t>
std::vector<value>& relationship_pattern<idx_t>::getTypes()
{
  return relationshipTypes;
}

template <typename idx_t>
std::map<std::string, value>& relationship_pattern<idx_t>::getProperties()
{
  return properties;
}

template <typename idx_t>
std::string relationship_pattern<idx_t>::getIdentifier()
{
  return identifier;
}

template <typename idx_t>
pattern_type relationship_pattern<idx_t>::type()
{
  return pattern_type::Relationship;
}

template <typename idx_t>
pattern_path<idx_t>::pattern_path(const cypher_astnode_t* astNode, context<idx_t> ctx)
{
  // Check that the given astnode is the right type:
  cypher_astnode_type_t astNodeType = cypher_astnode_type(astNode);
  std::string typeStr               = cypher_astnode_typestr(astNodeType);
  CUGRAPH_EXPECTS(typeStr == "pattern path", "Wrong type of astnode supplied");
}

template <typename idx_t>
pattern_path<idx_t>::pattern_path(pattern_path<idx_t>&& other)
{
  path = std::move(other.path);
  other.path.clear();
}

template <typename idx_t>
pattern_path<idx_t>::~pattern_path()
{
  for (size_t i = 0; i < path.size(); i++) delete path[i];
}

template <typename idx_t>
pattern_path<idx_t>& pattern_path<idx_t>::operator=(pattern_path<idx_t>&& other)
{
  if (this != &other) {
    path = std::move(other.path);
    other.path.clear();
  }
  return *this;
}

}  // namespace db
}  // namespace cugraph
