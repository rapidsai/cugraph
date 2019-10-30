/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cugraph.h>
#include <vector>
#include <map>
#include "utilities/graph_utils.cuh"

namespace cugraph { 
namespace db {
  /**
   * Class for representing an entry in a pattern, which may either be a variable or constant value
   * See description of db_pattern for more info on how this is used.
   */
  template <typename idx_t>
  class db_pattern_entry {
    bool is_var;
    idx_t constantValue;
    std::string variableName;
  public:
    db_pattern_entry(std::string variable);
    db_pattern_entry(idx_t constant);
    db_pattern_entry(const db_pattern_entry<idx_t>& other);
    db_pattern_entry& operator=(const db_pattern_entry<idx_t>& other);
    bool isVariable() const;
    idx_t getConstant() const;
    std::string getVariable() const;
  };

  /**
   * Class for representing a pattern (usually a triple pattern, but it's extensible)
   * A pattern in this sense consists of a sequence of entries each element is either a constant
   * value (an integer, since we dictionary encode everything) or a variable. Variables stand
   * in for unknown values that are being searched for. For example: if we have a pattern like
   * {'a', :haslabel, Person} (Where :haslabel and Person are dictionary encoded constants and
   * 'a' is a variable) We are looking for all nodes that have the label Person.
   */
  template <typename idx_t>
  class db_pattern {
    std::vector<db_pattern_entry<idx_t>> entries;
  public:
    db_pattern();
    db_pattern(const db_pattern<idx_t>& other);
    db_pattern& operator=(const db_pattern<idx_t>& other);
    int getSize() const;
    const db_pattern_entry<idx_t>& getEntry(int position) const;
    void addEntry(db_pattern_entry<idx_t>& entry);
    bool isAllConstants();
  };

  /**
   * Class which encapsulates a CSR-style index on a column
   */
  template <typename idx_t>
  class db_column_index {
    gdf_column* offsets;
    gdf_column* indirection;
    void deleteData();
  public:
    db_column_index();
    db_column_index(gdf_column* offsets, gdf_column* indirection);
    db_column_index(const db_column_index& other) = delete;
    db_column_index(db_column_index&& other);
    ~db_column_index();
    db_column_index& operator=(const db_column_index& other) = delete;
    db_column_index& operator=(db_column_index&& other);
    void resetData(gdf_column* offsets, gdf_column* indirection);
    gdf_column* getOffsets();
    gdf_column* getIndirection();
  };

  /**
   * Class which encapsulates a result set binding
   */
  template <typename idx_t>
  class db_result {
    std::vector<idx_t*> columns;
    std::vector<std::string> names;
    bool dataValid;
    idx_t columnSize;
  public:
    db_result();
    db_result(db_result&& other);
    db_result(db_result& other) = delete;
    db_result(const db_result& other) = delete;
    ~db_result();
    db_result& operator=(db_result&& other);
    db_result& operator=(db_result& other) = delete;
    db_result& operator=(const db_result& other) = delete;
    void deleteData();
    idx_t getSize();
    idx_t* getData(std::string idx);
    void addColumn(std::string columnName);
    void allocateColumns(idx_t size);
    /**
     * For debugging purposes
     * @return Human readable representation
     */
    std::string toString();
  };

  /**
   * Class which glues an arbitrary number of columns together to form a table
   */
  template <typename idx_t>
  class db_table {
    std::vector<gdf_column*> columns;
    std::vector<std::string> names;
    std::vector<db_pattern<idx_t>> inputBuffer;
    std::vector<db_column_index<idx_t>> indices;
  public:
    db_table();
    ~db_table();
    void addColumn(std::string name);
    void addEntry(db_pattern<idx_t>& pattern);

    /**
     * This method will rebuild the indices for each column in the table. This is done by
     * sorting a copy of the column along with an array which is a 0..n sequence, where
     * n is the number of entries in the column. The sorted column is used to produce an
     * offsets array and the sequence array becomes a permutation which maps the offset
     * position into the original table.
     */
    void rebuildIndices();

    /**
     * This method takes all the temporary input in the input buffer and appends it onto
     * the existing table.
     */
    void flush_input();

    /**
     * This method is for debugging purposes. It returns a human readable string representation
     * of the table.
     * @return Human readable string representation
     */
    std::string toString();
    db_column_index<idx_t>& getIndex(int idx);
    gdf_column* getColumn(int idx);
  };

  /**
   * The main database object. It stores the needed tables and provides a method hook to run
   * a query on the data.
   */
  template <typename idx_t>
  class db_object {
    // The dictionary and reverse dictionary encoding strings to ids and vice versa
    std::map<std::string, idx_t> valueToId;
    std::map<idx_t, std::string> idToValue;
    idx_t next_id;

    // The relationship table
    db_table<idx_t> relationshipsTable;

    // The relationship property table
    db_table<idx_t> relationshipPropertiesTable;

  public:
    db_object();
    std::string query(std::string query);
  };
} } //namespace
