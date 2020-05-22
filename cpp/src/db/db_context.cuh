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

#include <db/db_results.cuh>
#include <db/db_table.cuh>
#include <map>
#include <string>
#include <vector>

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
  encoder<idx_t>* coder;
  db_table<idx_t>* relationshipsTable;
  db_table<idx_t>* nodeLabelsTable;
  db_table<idx_t>* nodePropertiesTable;
  db_table<idx_t>* relationshipPropertiesTable;
  idx_t nextId;
  std::vector<string_table> namedResults;
  std::vector<std::string> resultNames;
  std::vector<db_result<idx_t>> variables;

 public:
  context();
  context(encoder<idx_t>* e,
          db_table<idx_t>* rt,
          db_table<idx_t>* nlt,
          db_table<idx_t>* npt,
          db_table<idx_t>* rpt);
  context(const context& other) = delete;
  context(context&& other);
  context& operator=(const context& other) = delete;
  context& operator                        =(context&& other);
  std::string getUniqueId();
  db_table<idx_t>* getRelationshipsTable();
  db_table<idx_t>* getNodeLabelsTable();
  db_table<idx_t>* getNodePropertiesTable();
  db_table<idx_t>* getRelationshipPropertiesTable();
  encoder<idx_t>* getEncoder();
  void registerNamedResult(std::string name, string_table&& result);
  void registerVariables(db_result<idx_t>&& result);
  std::string getNamedEntry(std::string name, std::string colname, idx_t row);
  std::string getNamedEntry(std::string name, idx_t colId, idx_t row);
};

}  // namespace db
}  // namespace cugraph
