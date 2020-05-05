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
    colId = 0;
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

}  // namespace db
}  // namespace cugraph
