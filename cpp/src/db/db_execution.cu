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
#include <fstream>
#include <sstream>
#include <string>

namespace cugraph {
namespace db {

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
string_table& load_csv_node<idx_t>::getStringResult()
{
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

template class load_csv_node<int32_t>;
template class load_csv_node<int64_t>;

template <typename idx_t>
match_node<idx_t>::match_node(const cypher_astnode_t* astNode)
{
}

}  // namespace db
}  // namespace cugraph
