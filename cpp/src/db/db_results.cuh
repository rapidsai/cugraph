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

#include <string>
#include <vector>
#include "rmm/device_buffer.hpp"

namespace cugraph {
namespace db {

/**
 * Class which encapsulates a result set binding
 */
template <typename idx_t>
class db_result {
  std::vector<rmm::device_buffer> columns;
  std::vector<std::string> names;
  bool dataValid;
  idx_t columnSize;

 public:
  db_result();
  db_result(db_result&& other);
  db_result(db_result& other)       = delete;
  db_result(const db_result& other) = delete;
  ~db_result()                      = default;
  db_result& operator               =(db_result&& other);
  db_result& operator=(const db_result& other) = delete;
  idx_t getSize();
  idx_t* getData(std::string idx);
  void addColumn(std::string columnName);
  void allocateColumns(idx_t size);
  std::string getIdentifier();
  bool hasVariable(std::string name);
  /**
   * For debugging purposes
   * @return Human readable representation
   */
  std::string toString();
};

/**
 * Class which encapsulates a table of strings, such as that used to store the
 * result of a LOAD CSV command.
 */
class string_table {
  std::vector<std::vector<std::string>> columns;
  std::vector<std::string> names;

 public:
  string_table()                          = default;
  string_table(const string_table& other) = delete;
  string_table(string_table&& other);
  string_table(std::string csvFile, bool with_headers, std::string delim = ",");
  string_table& operator=(const string_table& other) = delete;
  string_table& operator                             =(string_table&& other);
  std::vector<std::string>& operator[](std::string colName);
  std::vector<std::string>& operator[](int colIdx);
};

}  // namespace db
}  // namespace cugraph
