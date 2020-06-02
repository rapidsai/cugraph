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
#include <db/db_context.cuh>
#include <sstream>

namespace cugraph {
namespace db {

template <typename idx_t>
encoder<idx_t>::encoder()
{
  nextId = 0;
}

template <typename idx_t>
idx_t encoder<idx_t>::encode(std::string val)
{
  if (stringToId.count(val) == 1) {
    return stringToId[val];
  } else {
    idx_t myId = nextId;
    ++nextId;
    idToString[myId] = val;
    stringToId[val]  = myId;
    return myId;
  }
}

template <typename idx_t>
std::string encoder<idx_t>::decode(idx_t val)
{
  if (idToString.count(val) == 1) {
    return idToString[val];
  } else {
    return std::string("");
  }
}

template <typename idx_t>
idx_t encoder<idx_t>::getId()
{
  idx_t myId = nextId;
  ++nextId;
  return myId;
}

template <typename idx_t>
std::string encoder<idx_t>::toString()
{
  std::stringstream ss;
  idx_t encoded = idToString.size();
  idx_t ids     = nextId - encoded;
  ss << "Encoder object with " << encoded << " encoded values and " << ids << " allocated ids:\n";
  for (auto it = idToString.begin(); it != idToString.end(); it++) {
    ss << it->first << " == " << it->second << "\n";
  }
  return ss.str();
}

template class encoder<int32_t>;
template class encoder<int64_t>;

template <typename idx_t>
context<idx_t>::context()
{
  coder                       = nullptr;
  relationshipsTable          = nullptr;
  nodeLabelsTable             = nullptr;
  nodePropertiesTable         = nullptr;
  relationshipPropertiesTable = nullptr;
  nextId                      = 0;
}

template <typename idx_t>
context<idx_t>::context(encoder<idx_t>* e,
                        db_table<idx_t>* rt,
                        db_table<idx_t>* nlt,
                        db_table<idx_t>* npt,
                        db_table<idx_t>* rpt)
{
  coder                       = e;
  relationshipsTable          = rt;
  nodeLabelsTable             = nlt;
  nodePropertiesTable         = npt;
  relationshipPropertiesTable = rpt;
  nextId                      = 0;
}

template <typename idx_t>
context<idx_t>::context(context<idx_t>&& other)
{
  coder                             = other.coder;
  other.coder                       = nullptr;
  relationshipsTable                = other.relationshipsTable;
  other.relationshipsTable          = nullptr;
  nodeLabelsTable                   = other.nodeLabelsTable;
  other.nodeLabelsTable             = nullptr;
  nodePropertiesTable               = other.nodePropertiesTable;
  other.nodePropertiesTable         = nullptr;
  relationshipPropertiesTable       = other.relationshipPropertiesTable;
  other.relationshipPropertiesTable = nullptr;
  nextId                            = other.nextId;
  namedResults                      = std::move(other.namedResults);
  resultNames                       = std::move(other.resultNames);
  variables                         = std::move(other.variables);
}

template <typename idx_t>
context<idx_t>& context<idx_t>::operator=(context<idx_t>&& other)
{
  if (this != &other) {
    coder                             = other.coder;
    other.coder                       = nullptr;
    relationshipsTable                = other.relationshipsTable;
    other.relationshipsTable          = nullptr;
    nodeLabelsTable                   = other.nodeLabelsTable;
    other.nodeLabelsTable             = nullptr;
    nodePropertiesTable               = other.nodePropertiesTable;
    other.nodePropertiesTable         = nullptr;
    relationshipPropertiesTable       = other.relationshipPropertiesTable;
    other.relationshipPropertiesTable = nullptr;
    nextId                            = other.nextId;
    namedResults                      = std::move(other.namedResults);
    resultNames                       = std::move(other.resultNames);
    variables                         = std::move(other.variables);
  }
  return *this;
}

template <typename idx_t>
std::string context<idx_t>::getUniqueId()
{
  std::stringstream ss;
  ss << "UniqueIdentifier_" << nextId++;
  return ss.str();
}

template <typename idx_t>
db_table<idx_t>* context<idx_t>::getRelationshipsTable()
{
  return relationshipsTable;
}

template <typename idx_t>
db_table<idx_t>* context<idx_t>::getNodeLabelsTable()
{
  return nodeLabelsTable;
}

template <typename idx_t>
db_table<idx_t>* context<idx_t>::getNodePropertiesTable()
{
  return nodePropertiesTable;
}

template <typename idx_t>
db_table<idx_t>* context<idx_t>::getRelationshipPropertiesTable()
{
  return relationshipPropertiesTable;
}

template <typename idx_t>
encoder<idx_t>* context<idx_t>::getEncoder()
{
  return coder;
}

template <typename idx_t>
void context<idx_t>::registerNamedResult(std::string name, string_table&& result)
{
  resultNames.push_back(name);
  namedResults.push_back(std::move(result));
}

template <typename idx_t>
void context<idx_t>::registerVariables(db_result<idx_t>&& result)
{
  variables.push_back(std::move(result));
}

template <typename idx_t>
std::string context<idx_t>::getNamedEntry(std::string name, std::string colname, idx_t row)
{
  int pos = -1;
  for (size_t i = 0; i < resultNames.size(); i++) {
    if (resultNames[i] == name) pos = i;
  }
  CUGRAPH_EXPECTS(pos > 0, "Named result not found");
  return namedResults[pos][colname][row];
}

template <typename idx_t>
std::string context<idx_t>::getNamedEntry(std::string name, idx_t col, idx_t row)
{
  int pos = -1;
  for (size_t i = 0; i < resultNames.size(); i++) {
    if (resultNames[i] == name) pos = i;
  }
  CUGRAPH_EXPECTS(pos > 0, "Named result not found");
  return namedResults[pos][col][row];
}

template <typename idx_t>
idx_t context<idx_t>::getNamedRows(std::string name)
{
  int pos = -1;
  for (size_t i = 0; i < resultNames.size(); i++) {
    if (resultNames[i] == name) pos = i;
  }
  CUGRAPH_EXPECTS(pos > 0, "Named result not found");
  return namedResults[pos][0].size();
}

template <typename idx_t>
std::vector<idx_t>&& context<idx_t>::getVariableColumn(std::string name)
{
  int pos = -1;
  for (size_t i = 0; i < variables.size(); i++)
    if (variables[i].hasVariable(name)) pos = i;
  CUGRAPH_EXPECTS(pos > 0, "Variable not found");
  return variables[pos].getHostColumn(name);
}

template <typename idx_t>
bool context<idx_t>::hasVariable(std::string name)
{
  for (size_t i = 0; i < variables.size(); i++) {
    if (variables[i].hasVariable(name)) return true;
  }
  return false;
}

template <typename idx_t>
bool context<idx_t>::hasNamed(std::string name)
{
  for (size_t i = 0; i < resultNames.size(); i++) {
    if (resultNames[i] == name) return true;
  }
  return false;
}

template class context<int32_t>;
template class context<int64_t>;

}  // namespace db
}  // namespace cugraph
