/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <db/db_execution.cuh>
#include <fstream>
#include <sstream>
#include <string>

namespace cugraph {
namespace db {
string_table& string_table::operator=(string_table&& other)
{
  this->names   = std::move(other.names);
  this->columns = std::move(other.columns);
  return *this;
}

string_table::string_table() {}

string_table::string_table(string_table&& other)
{
  this->names   = std::move(other.names);
  this->columns = std::move(other.columns);
}

string_table::string_table(std::string csvFile, bool with_headers, std::string delim)
{
  std::ifstream myFileStream(csvFile);
  CUGRAPH_EXPECTS(myFileStream.is_open(), "Could not open CSV file!");
  std::string line, col;
  int colCount = 0;
  if (myFileStream.good()) {
    std::getline(myFileStream, line);
    std::stringstream ss(line);
    while (std::getline(ss, col, delim[0])) { names.push_back(col); }
  }
  colCount = names.size();
  columns.resize(colCount);
  if (!with_headers) {
    for (size_t i = 0; i < names.size(); i++) columns[i].push_back(names[i]);
  }
  while (std::getline(myFileStream, line)) {
    std::stringstream ss(line);
    int colId = 0;
    while (std::getline(ss, col, delim[0])) {
      columns[colId].push_back(col);
      colId++;
    }
  }
  myFileStream.close();
}

std::vector<std::string>& string_table::operator[](std::string colName)
{
  int colId = -1;
  for (size_t i = 0; i < names.size(); i++) {
    if (colName == names[i]) colId = i;
  }
  CUGRAPH_EXPECTS(colId != -1, "Given column name not found");
  return columns[colId];
}

std::vector<std::string>& string_table::operator[](int colIdx)
{
  CUGRAPH_EXPECTS(colIdx >= 0 && static_cast<size_t>(colIdx) < columns.size(),
                  "Index out of range");
  return columns[colIdx];
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
void load_csv_node<idx_t>::execute()
{
  std::string toErase("file:///");
  std::string file = filename;
  size_t pos       = file.find(toErase);
  if (pos != std::string::npos) file.erase(pos, toErase.length());
  string_table temp(file, with_headers, delimiter);
  this->result = std::move(temp);
}

template <typename idx_t>
string_table& load_csv_node<idx_t>::getStringResult() {
  return result;
}

template <typename idx_t>
db_result<idx_t>& load_csv_node<idx_t>::getGPUResult() {
  CUGRAPH_FAIL("Load CSV node does not support GPU result");
}

template <typename idx_t>
std::string load_csv_node<idx_t>::getResultIdentifier() {
  return identifier;
}

template class load_csv_node<int32_t>;
template class load_csv_node<int64_t>;

node_pattern::node_pattern(std::string id) {
  identifier = id;
}

void node_pattern::setIdentifier(std::string id) {
  identifier = id;
}

void node_pattern::addLabel(std::string label) {
  labels.push_back(label);
}

void node_pattern::addProperty(std::string name, std::string value) {
  properties[name] = value;
}

std::string node_pattern::getIdentifier() {
  return identifier;
}

pattern_type node_pattern::type() {
  return pattern_type::Node;
}

std::vector<std::string>& node_pattern::getLabels() {
  return labels;
}

std::map<std::string, std::string>& node_pattern::getProperties() {
  return properties;
}

relationship_pattern::relationship_pattern() {
  direction = 1;
}

relationship_pattern::relationship_pattern(const cypher_astnode_t* astNode) {
  //Check that the given astnode is the right type:
  cypher_astnode_type_t astNodeType = cypher_astnode_type(astNode);
  std::string typeStr = cypher_astnode_typestr(astNodeType);
  CUGRAPH_EXPECTS(typeStr == "rel pattern", "Wrong type of astnode supplied");

  //Extract the direction of the relation
  direction = cypher_ast_rel_pattern_get_direction(astNode);

  //Extract the identifier or if null assign a unique identifier
  const cypher_astnode_t* id = cypher_ast_rel_pattern_get_identifier(astNode);
  if (id != nullptr){

  }
  else {

  }
}

void relationship_pattern::addProperty(std::string name, std::string value) {
  properties[name] = value;
}

void relationship_pattern::setStart(std::string start) {
  startId = start;
}

void relationship_pattern::setEnd(std::string end) {
  endId = end;
}

void relationship_pattern::addType(std::string type) {
  relationshipTypes.push_back(type);
}

void relationship_pattern::setDirection(uint32_t dir) {
  direction = dir;
}

std::string relationship_pattern::getStart() {
  return startId;
}

std::string relationship_pattern::getEnd() {
  return endId;
}

std::vector<std::string>& relationship_pattern::getTypes() {
  return relationshipTypes;
}

std::map<std::string, std::string>& relationship_pattern::getProperties() {
  return properties;
}

std::string relationship_pattern::getIdentifier() {
  return identifier;
}

pattern_type relationship_pattern::type() {
  return pattern_type::Relationship;
}

pattern_path::pattern_path(const cypher_astnode_t* astNode) {

}

pattern_path::pattern_path(pattern_path&& other) {
  path = std::move(other.path);
  other.path.clear();
}

pattern_path::~pattern_path() {
  for (size_t i = 0; i < path.size(); i++)
    delete path[i];
}

pattern_path& pattern_path::operator=(pattern_path&& other) {
  if(this != &other) {
    path = std::move(other.path);
    other.path.clear();
  }
  return *this;
}

template <typename idx_t>
match_node<idx_t>::match_node(const cypher_astnode_t* astNode) {

}

}  // namespace db
}  // namespace cugraph
