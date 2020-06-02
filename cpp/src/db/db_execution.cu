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
#include <db/db_execution.cuh>
#include <db/parser_helpers.cuh>
#include <fstream>
#include <sstream>
#include <string>

namespace cugraph {
namespace db {

std::string execution_typeToString(execution_type type)
{
  switch (type) {
    case execution_type::LoadCsv: {
      return "LOAD CSV";
    }
    case execution_type::Match: {
      return "MATCH";
    }
    case execution_type::Create: {
      return "CREATE";
    }
    case execution_type::Merge: {
      return "MERGE";
    }
    case execution_type::Return: {
      return "RETURN";
    }
  }
  return "ERROR";
}

template <typename idx_t>
load_csv_node<idx_t>::load_csv_node(const cypher_astnode_t* astNode)
{
  cypher_astnode_type_t type = cypher_astnode_type(astNode);
  const char* type_desc      = cypher_astnode_typestr(type);
  CUGRAPH_EXPECTS(type_desc == std::string("LOAD CSV"),
                  "Load Csv requires a LOAD CSV node to construct.");
  const cypher_astnode_t* identifierNode = cypher_ast_load_csv_get_identifier(astNode);
  identifier                             = cypher_ast_identifier_get_name(identifierNode);
  const cypher_astnode_t* urlNode        = cypher_ast_load_csv_get_url(astNode);
  filename                               = cypher_ast_string_get_value(urlNode);
  with_headers                           = cypher_ast_load_csv_has_with_headers(astNode);
  const cypher_astnode_t* delimiterNode  = cypher_ast_load_csv_get_field_terminator(astNode);
  if (delimiterNode != nullptr)
    delimiter = cypher_ast_string_get_value(delimiterNode);
  else
    delimiter = ",";
}

template <typename idx_t>
void load_csv_node<idx_t>::execute(context<idx_t>& ctx)
{
  std::string toErase("file:///");
  std::string file = filename;
  size_t pos       = file.find(toErase);
  if (pos != std::string::npos) file.erase(pos, toErase.length());
  string_table temp(file, with_headers, delimiter);
  ctx.registerNamedResult(identifier, std::move(temp));
}

template <typename idx_t>
string_table& load_csv_node<idx_t>::getStringResult()
{
  CUGRAPH_EXPECTS(executed, "Can't get result before execution");
  return result;
}

template <typename idx_t>
db_result<idx_t>& load_csv_node<idx_t>::getGPUResult()
{
  CUGRAPH_FAIL("Load CSV node does not support GPU result");
}

template <typename idx_t>
std::string load_csv_node<idx_t>::getResultIdentifier()
{
  return identifier;
}

template <typename idx_t>
execution_type load_csv_node<idx_t>::type()
{
  return execution_type::LoadCsv;
}

template class load_csv_node<int32_t>;
template class load_csv_node<int64_t>;

template <typename idx_t>
match_node<idx_t>::match_node(const cypher_astnode_t* astNode, context<idx_t>& ctx)
{
  // Checking the type of the given astNode is correct
  std::string typeStr = getTypeString(astNode);
  CUGRAPH_EXPECTS(typeStr == "MATCH", "Need a MATCH node to construct a match_node");

  // A match node should have just one child which is a pattern node:
  uint32_t num_children = cypher_astnode_nchildren(astNode);
  CUGRAPH_EXPECTS(num_children == 1, "Unexpected child count on match node");
  const cypher_astnode_t* p_node = cypher_astnode_get_child(astNode, 0);
  typeStr                        = getTypeString(p_node);
  CUGRAPH_EXPECTS(typeStr == "pattern", "Unexpected child type on match node");

  // Pattern node should have some children that are pattern path nodes
  num_children = cypher_astnode_nchildren(p_node);
  for (uint32_t i = 0; i < num_children; i++) {
    const cypher_astnode_t* child = cypher_astnode_get_child(p_node, i);
    paths.emplace_back(child, ctx);
  }
}

template <typename idx_t>
string_table& match_node<idx_t>::getStringResult()
{
  CUGRAPH_FAIL("Match node does not have a string table result");
}

template <typename idx_t>
db_result<idx_t>& match_node<idx_t>::getGPUResult()
{
  return result;
}

template <typename idx_t>
std::string match_node<idx_t>::getResultIdentifier()
{
  CUGRAPH_EXPECTS(executed, "Must execute node before getting the result identifier.");
  return result.getIdentifier();
}

template <typename idx_t>
void match_node<idx_t>::execute(context<idx_t>& ctx)
{
  // Todo implement match node execution
  executed = true;
}

template <typename idx_t>
execution_type match_node<idx_t>::type()
{
  return execution_type::Match;
}

template class match_node<int32_t>;
template class match_node<int64_t>;

template <typename idx_t>
create_node<idx_t>::create_node(const cypher_astnode_t* astNode, context<idx_t>& ctx)
{
  // Checking the type of the given astNode is correct
  std::string typeStr = getTypeString(astNode);
  CUGRAPH_EXPECTS(typeStr == "CREATE", "Need a CREATE node to construct a create_node");

  // A create node should have just one child which is a pattern node:
  uint32_t num_children = cypher_astnode_nchildren(astNode);
  CUGRAPH_EXPECTS(num_children == 1, "Unexpected child count on match node");
  const cypher_astnode_t* p_node = cypher_astnode_get_child(astNode, 0);
  typeStr                        = getTypeString(p_node);
  CUGRAPH_EXPECTS(typeStr == "pattern", "Unexpected child type on match node");

  // Pattern node should have some children that are pattern path nodes
  num_children = cypher_astnode_nchildren(p_node);
  for (uint32_t i = 0; i < num_children; i++) {
    const cypher_astnode_t* child = cypher_astnode_get_child(p_node, i);
    paths.emplace_back(child, ctx);
  }
}

template <typename idx_t>
string_table& create_node<idx_t>::getStringResult()
{
  CUGRAPH_FAIL("Create node has no string result.");
}

template <typename idx_t>
db_result<idx_t>& create_node<idx_t>::getGPUResult()
{
  CUGRAPH_FAIL("Create node has no GPU result.");
}

template <typename idx_t>
std::string create_node<idx_t>::getResultIdentifier()
{
  CUGRAPH_FAIL("Create node has no result.");
}

template <typename idx_t>
void create_node<idx_t>::execute(context<idx_t>& ctx)
{
  // For each pattern path add the needed entries to each table
  for (size_t i = 0; i < paths.size(); i++) {
    pattern_path<idx_t>& path = paths[i];
    // Figure out if we are doing row by row or single create
    bool hasNamed    = path.hasNamedValues(ctx);
    bool hasVariable = path.hasBoundVariables(ctx);

    // Case where there are no variables or named values (simple insert)
    if (!hasNamed && !hasVariable) {
      // Cycle through all nodes on the path and get encoded ids for them
      std::map<std::string, idx_t> translated;
      for (size_t j = 0; j < path.getPathNodes().size(); j++) {
        if (path.getPathNodes()[j]->type() == pattern_type::Node) {
          std::string id = path.getPathNodes()[j]->getIdentifier();
          idx_t eid      = ctx.getEncoder()->getId();
          translated[id] = eid;
        }
      }

      // Cycle through all entries in the pattern and do the inserts into db tables
      for (size_t i = 0; i < path.getPathNodes().size(); i++) {
        if (path.getPathNodes()[i]->type() == pattern_type::Node) {
          node_pattern<idx_t>* node =
            reinterpret_cast<node_pattern<idx_t>*>(path.getPathNodes()[i]);
          std::string nodeId = node->getIdentifier();
          idx_t eNodeId      = translated[nodeId];

          // Insert entries into the node labels table
          for (size_t j = 0; j < node->getLabels().size(); j++) {
            std::string label = node->getLabels()[j];
            idx_t elabel      = ctx.getEncoder()->encode(label);
            db_pattern_entry<idx_t> p1(eNodeId);
            db_pattern_entry<idx_t> p2(elabel);
            db_pattern<idx_t> pat;
            pat.addEntry(p1);
            pat.addEntry(p2);
            auto table = ctx.getNodeLabelsTable();
            table->addEntry(pat);
          }

          // Insert entries into the node properties table
          for (auto it = node->getProperties().begin(); it != node->getProperties().end(); it++) {
            std::string propName  = it->first;
            idx_t ePropName       = ctx.getEncoder()->encode(propName);
            std::string propValue = it->second.getIdentifier();
            idx_t ePropValue      = ctx.getEncoder()->encode(propValue);
            db_pattern_entry<idx_t> p1(eNodeId);
            db_pattern_entry<idx_t> p2(ePropName);
            db_pattern_entry<idx_t> p3(ePropValue);
            db_pattern<idx_t> pat;
            pat.addEntry(p1);
            pat.addEntry(p2);
            pat.addEntry(p3);
            auto table = ctx.getNodePropertiesTable();
            table->addEntry(pat);
          }
        } else {
          relationship_pattern<idx_t>* rel =
            reinterpret_cast<relationship_pattern<idx_t>*>(path.getPathNodes()[i]);
          idx_t eStart = translated[rel->getStart()];
          idx_t eEnd   = translated[rel->getEnd()];

          idx_t numTypes = rel->getTypes().size();
          CUGRAPH_EXPECTS(numTypes == 1, "Relationships must have one type and one type only");
          idx_t eType = ctx.getEncoder()->encode(rel->getTypes()[0].getIdentifier());
          db_pattern_entry<idx_t> p1(eStart);
          db_pattern_entry<idx_t> p2(eType);
          db_pattern_entry<idx_t> p3(eEnd);
          db_pattern<idx_t> pat;
          pat.addEntry(p1);
          pat.addEntry(p2);
          pat.addEntry(p3);
          auto table = ctx.getRelationshipsTable();
          table->addEntry(pat);

          // Insert entries into relationship properties table
          idx_t relId = table->getLastRowId();
          for (auto it = rel->getProperties().begin(); it != rel->getProperties().end(); it++) {
            idx_t ePropName  = ctx.getEncoder()->encode(it->first);
            idx_t ePropValue = ctx.getEncoder()->encode(it->second.getIdentifier());
            db_pattern_entry<idx_t> t1(relId);
            db_pattern_entry<idx_t> t2(ePropName);
            db_pattern_entry<idx_t> t3(ePropValue);
            db_pattern<idx_t> tat;
            tat.addEntry(t1);
            tat.addEntry(t2);
            tat.addEntry(t3);
            auto relProp = ctx.getRelationshipPropertiesTable();
            relProp->addEntry(tat);
          }
        }
      }
    }

    // Case where there are bound variables but no named values
    else if (!hasNamed && hasVariable) {
    }

    // Case where there are named values, but no bound variables
    else if (hasNamed && !hasVariable) {
    }

    // Case where there are both named values and bound variables
    else if (hasNamed && hasVariable) {
    }
  }
}

template <typename idx_t>
execution_type create_node<idx_t>::type()
{
  return execution_type::Create;
}

template class create_node<int32_t>;
template class create_node<int64_t>;

template <typename idx_t>
query_plan<idx_t>::query_plan(const cypher_parse_result_t* parseResult, context<idx_t>&& c)
{
  ctx                = std::move(c);
  uint32_t numErrors = cypher_parse_result_nerrors(parseResult);
  CUGRAPH_EXPECTS(numErrors == 0, "There were errors in the parsed query");
  uint32_t numRoots = cypher_parse_result_nroots(parseResult);
  for (uint32_t i = 0; i < numRoots; i++) {
    // Get the root node and verify it is a 'statement' node with one child
    const cypher_astnode_t* root = cypher_parse_result_get_root(parseResult, i);
    std::string rootType         = getTypeString(root);
    CUGRAPH_EXPECTS(rootType == "statement", "Unexpected root type in parse result.");
    uint32_t rootChildren = cypher_astnode_nchildren(root);
    CUGRAPH_EXPECTS(rootChildren == 1, "Parse result root has more than one child");

    // Get the first child of the root node, expected to be a 'query'
    const cypher_astnode_t* queryNode = cypher_astnode_get_child(root, 0);
    std::string queryType             = getTypeString(queryNode);
    CUGRAPH_EXPECTS(queryType == "query", "Expected query node, got something else");

    // For each child of the 'query' node, construct an execution_node for it
    uint32_t queryChildren = cypher_astnode_nchildren(queryNode);
    for (uint32_t j = 0; j < queryChildren; j++) {
      const cypher_astnode_t* child = cypher_astnode_get_child(queryNode, j);
      std::string childType         = getTypeString(child);
      if (childType == "LOAD CSV") {
        execution_node<idx_t>* childPtr = new load_csv_node<idx_t>(child);
        plan_nodes.push_back(childPtr);
      } else if (childType == "MATCH") {
        execution_node<idx_t>* childPtr = new match_node<idx_t>(child, ctx);
        plan_nodes.push_back(childPtr);
      } else if (childType == "CREATE") {
        execution_node<idx_t>* childPtr = new create_node<idx_t>(child, ctx);
        plan_nodes.push_back(childPtr);
      } else {
        CUGRAPH_FAIL("Unsupported query clause given");
      }
    }
  }
}

template <typename idx_t>
query_plan<idx_t>::query_plan(query_plan&& other)
{
  ctx        = std::move(other.ctx);
  plan_nodes = std::move(other.plan_nodes);
  other.plan_nodes.clear();
}

template <typename idx_t>
query_plan<idx_t>::~query_plan()
{
  for (size_t i = 0; i < plan_nodes.size(); i++) delete plan_nodes[i];
  plan_nodes.clear();
}

template <typename idx_t>
query_plan<idx_t>& query_plan<idx_t>::operator=(query_plan<idx_t>&& other)
{
  if (this != &other) {
    ctx = std::move(other.ctx);
    for (size_t i = 0; i < plan_nodes.size(); i++) delete plan_nodes[i];
    plan_nodes = std::move(other.plan_nodes);
    other.plan_nodes.clear();
  }
  return *this;
}

template <typename idx_t>
std::string query_plan<idx_t>::execute()
{
  for (size_t i = 0; i < plan_nodes.size(); i++) {
    execution_type type = plan_nodes[i]->type();
    switch (type) {
      case execution_type::LoadCsv: {
        load_csv_node<idx_t>* csv_node = reinterpret_cast<load_csv_node<idx_t>*>(plan_nodes[i]);
        csv_node->execute(ctx);
        break;
      }
      case execution_type::Match: {
        break;
      }
      case execution_type::Merge: {
        break;
      }
      case execution_type::Create: {
        std::cout << "Calling create_node<idx_t>::execute()\n";
        create_node<idx_t>* c_node = reinterpret_cast<create_node<idx_t>*>(plan_nodes[i]);
        c_node->execute(ctx);
        break;
      }
      case execution_type::Return: {
        break;
      }
    }
  }
  std::stringstream ss;
  return ss.str();
}

template class query_plan<int32_t>;
template class query_plan<int64_t>;

}  // namespace db
}  // namespace cugraph
