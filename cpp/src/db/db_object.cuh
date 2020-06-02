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

#include "db/db_context.cuh"
#include "db/db_table.cuh"

namespace cugraph {
namespace db {

/**
 * The main database object. It stores the needed tables and provides a method hook to run
 * a query on the data.
 */
template <typename idx_t>
class db_object {
  encoder<idx_t> idEncoder;

  // The relationship table
  db_table<idx_t> relationshipsTable;

  // The node labels table
  db_table<idx_t> nodeLabelsTable;

  // The node properties table
  db_table<idx_t> nodePropertiesTable;

  // The relationship property table
  db_table<idx_t> relationshipPropertiesTable;

 public:
  db_object();
  std::string query(std::string query);

  /**
   * For debugging purposes only.
   * @return Human readable representation.
   */
  std::string toString();
};

}  // namespace db
}  // namespace cugraph
