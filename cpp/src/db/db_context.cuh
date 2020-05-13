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

#include <db/db_table.cuh>
#include <map>
#include <string>

namespace cugraph {
namespace db {

template <typename idx_t>
class encoder {
  std::map<idx_t, std::string> idToString;
  std::map<std::string, idx_t> stringToId;
  idx_t nextId;

 public:
  encoder();
  idx_t encode(std::string val);
  std::string decode(idx_t val);
  idx_t getId();
};

template <typename idx_t>
class context {
  db_table<idx_t>* relationshipsTable;
  db_table<idx_t>* nodeLabelsTable;
  db_table<idx_t>* nodePropertiesTable;
  db_table<idx_t>* relationshipPropertiesTable;
  idx_t nextId;

 public:
  context();
  context(db_table<idx_t>* rt, db_table<idx_t>* nlt, db_table<idx_t>* npt, db_table<idx_t>* rpt);
  std::string getUniqueId();
  db_table<idx_t>* getRelationshipsTable();
  db_table<idx_t>* getNodeLabelsTable();
  db_table<idx_t>* getNodePropertiesTable();
  db_table<idx_t>* getRelationshipPropertiesTable();
};

}  // namespace db
}  // namespace cugraph
