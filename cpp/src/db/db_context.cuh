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
  std::map<idx_t, std::string> id_to_string;
  std::map<std::string, idx_t> string_to_id;
  idx_t next_id;

 public:
  encoder();
  idx_t encode(std::string val);
  std::string decode(idx_t val);
  idx_t get_id();

  /**
   * For debugging purposes only
   * @return A human representation of the encoder's content
   */
  std::string to_string();
};

template <typename idx_t>
class context {
  encoder<idx_t>* coder;
  db_table<idx_t>* relationships_table;
  db_table<idx_t>* node_labels_table;
  db_table<idx_t>* node_properties_table;
  db_table<idx_t>* relationship_properties_table;
  idx_t next_id;
  std::vector<string_table> named_results;
  std::vector<std::string> result_names;
  std::vector<db_result<idx_t>> variables;

 public:
  context() = default;
  context(encoder<idx_t>* e,
          db_table<idx_t>* rt,
          db_table<idx_t>* nlt,
          db_table<idx_t>* npt,
          db_table<idx_t>* rpt);
  context(const context& other) = delete;
  context(context&& other);
  context& operator=(const context& other) = delete;
  context& operator                        =(context&& other);
  std::string get_unique_id();
  db_table<idx_t>* get_relationships_table();
  db_table<idx_t>* get_node_labels_table();
  db_table<idx_t>* get_node_properties_table();
  db_table<idx_t>* get_relationship_properties_table();
  encoder<idx_t>* get_encoder();
  void register_named_result(std::string name, string_table&& result);
  void register_variables(db_result<idx_t>&& result);
  std::string get_named_entry(std::string name, std::string colname, idx_t row);
  std::string get_named_entry(std::string name, idx_t colId, idx_t row);
  idx_t get_named_rows(std::string name);
  std::vector<idx_t>&& get_variable_column(std::string name);
  bool has_variable(std::string name);
  bool has_named(std::string name);
};

}  // namespace db
}  // namespace cugraph
