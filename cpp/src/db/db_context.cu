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

template class encoder<int32_t>;
template class encoder<int64_t>;

template <typename idx_t>
context<idx_t>::context()
{
  relationshipsTable          = nullptr;
  nodeLabelsTable             = nullptr;
  nodePropertiesTable         = nullptr;
  relationshipPropertiesTable = nullptr;
  nextId                      = 0;
}

template <typename idx_t>
context<idx_t>::context(db_table<idx_t>* rt,
                        db_table<idx_t>* nlt,
                        db_table<idx_t>* npt,
                        db_table<idx_t>* rpt)
{
  relationshipsTable          = rt;
  nodeLabelsTable             = nlt;
  nodePropertiesTable         = npt;
  relationshipPropertiesTable = rpt;
  nextId                      = 0;
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
std::string context<idx_t>::getUniqueId()
{
  std::stringstream ss;
  ss << "UniqueIdentifier_" << nextId++;
  return ss.str();
}

template class context<int32_t>;
template class context<int64_t>;

}  // namespace db
}  // namespace cugraph
