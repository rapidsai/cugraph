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

#include <raft/cudart_utils.h>
#include <db/db_results.cuh>
#include <fstream>
#include <sstream>
#include <utilities/error.hpp>

namespace cugraph {
namespace db {

template <typename idx_t>
db_result<idx_t>::db_result()
{
  data_valid  = false;
  column_size = 0;
}

template <typename idx_t>
db_result<idx_t>::db_result(db_result&& other)
{
  *this = std::move(other);
}

template <typename idx_t>
db_result<idx_t>& db_result<idx_t>::operator=(db_result<idx_t>&& other)
{
  data_valid       = other.data_valid;
  columns          = std::move(other.columns);
  names            = std::move(other.names);
  other.data_valid = false;
  return *this;
}

template <typename idx_t>
idx_t db_result<idx_t>::get_size()
{
  return column_size;
}

template <typename idx_t>
idx_t* db_result<idx_t>::get_data(std::string idx)
{
  CUGRAPH_EXPECTS(data_valid, "Data not valid");

  idx_t* returnPtr = nullptr;
  for (size_t i = 0; i < names.size(); i++)
    if (names[i] == idx) returnPtr = reinterpret_cast<idx_t*>(columns[i].data());
  return returnPtr;
}

template <typename idx_t>
void db_result<idx_t>::add_column(std::string columnName)
{
  CUGRAPH_EXPECTS(!data_valid, "Cannot add a column to an allocated result.");
  names.push_back(columnName);
}

template <typename idx_t>
void db_result<idx_t>::allocate_columns(idx_t size)
{
  CUGRAPH_EXPECTS(!data_valid, "Already allocated columns");

  for (size_t i = 0; i < names.size(); i++) {
    rmm::device_buffer col(sizeof(idx_t) * size);
    columns.push_back(std::move(col));
  }
  data_valid  = true;
  column_size = size;
}

template <typename idx_t>
std::string db_result<idx_t>::get_identifier()
{
  std::stringstream ss;
  for (size_t i = 0; i < names.size() - 1; i++) ss << names[i] << ",";
  ss << names[names.size() - 1];
  return ss.str();
}

template <typename idx_t>
bool db_result<idx_t>::has_variable(std::string name)
{
  for (size_t i = 0; i < names.size(); i++)
    if (names[i] == name) return true;
  return false;
}

template <typename idx_t>
std::vector<idx_t>&& db_result<idx_t>::get_host_column(std::string name)
{
  int pos = -1;
  for (size_t i = 0; i < names.size(); i++) {
    if (names[i] == name) pos = i;
  }
  CUGRAPH_EXPECTS(pos > 0, "Given variable name not found");
  std::vector<idx_t> result(column_size);
  CUDA_TRY(
    cudaMemcpy(result.data(), columns[pos].data(), sizeof(idx_t) * column_size, cudaMemcpyDefault));
  return std::move(result);
}

template <typename idx_t>
std::string db_result<idx_t>::to_string()
{
  std::stringstream ss;
  ss << "db_result with " << columns.size() << " columns of length " << column_size << "\n";
  for (size_t i = 0; i < columns.size(); i++) ss << names[i] << " ";
  ss << "\n";
  std::vector<std::vector<idx_t>> hostColumns;
  hostColumns.resize(columns.size());
  for (size_t i = 0; i < columns.size(); i++) {
    hostColumns[i].resize(column_size);
    CUDA_TRY(cudaMemcpy(
      hostColumns[i].data(), columns[i].data(), sizeof(idx_t) * column_size, cudaMemcpyDefault));
  }
  for (idx_t i = 0; i < column_size; i++) {
    for (size_t j = 0; j < hostColumns.size(); j++) ss << hostColumns[j][i] << " ";
    ss << "\n";
  }
  return ss.str();
}

template class db_result<int32_t>;
template class db_result<int64_t>;

string_table::string_table(string_table&& other) { *this = std::move(other); }

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
    int colId = 0;
    while (std::getline(ss, col, delim[0])) {
      columns[colId].push_back(col);
      colId++;
    }
  }
  myFileStream.close();
}

string_table& string_table::operator=(string_table&& other)
{
  this->names   = std::move(other.names);
  this->columns = std::move(other.columns);
  return *this;
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

}  // namespace db
}  // namespace cugraph
