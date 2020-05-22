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
#include <db/db_results.cuh>
#include <fstream>
#include <sstream>

namespace cugraph {
namespace db {

template <typename idx_t>
db_result<idx_t>::db_result()
{
  dataValid  = false;
  columnSize = 0;
}

template <typename idx_t>
db_result<idx_t>::db_result(db_result&& other)
{
  dataValid       = other.dataValid;
  columns         = std::move(other.columns);
  names           = std::move(other.names);
  other.dataValid = false;
}

template <typename idx_t>
db_result<idx_t>& db_result<idx_t>::operator=(db_result<idx_t>&& other)
{
  dataValid       = other.dataValid;
  columns         = std::move(other.columns);
  names           = std::move(other.names);
  other.dataValid = false;
  return *this;
}

template <typename idx_t>
idx_t db_result<idx_t>::getSize()
{
  return columnSize;
}

template <typename idx_t>
idx_t* db_result<idx_t>::getData(std::string idx)
{
  CUGRAPH_EXPECTS(dataValid, "Data not valid");

  idx_t* returnPtr = nullptr;
  for (size_t i = 0; i < names.size(); i++)
    if (names[i] == idx) returnPtr = reinterpret_cast<idx_t*>(columns[i].data());
  return returnPtr;
}

template <typename idx_t>
void db_result<idx_t>::addColumn(std::string columnName)
{
  CUGRAPH_EXPECTS(!dataValid, "Cannot add a column to an allocated result.");
  names.push_back(columnName);
}

template <typename idx_t>
void db_result<idx_t>::allocateColumns(idx_t size)
{
  CUGRAPH_EXPECTS(!dataValid, "Already allocated columns");

  for (size_t i = 0; i < names.size(); i++) {
    rmm::device_buffer col(sizeof(idx_t) * size);
    columns.push_back(std::move(col));
  }
  dataValid  = true;
  columnSize = size;
}

template <typename idx_t>
std::string db_result<idx_t>::getIdentifier()
{
  std::stringstream ss;
  for (size_t i = 0; i < names.size() - 1; i++) ss << names[i] << ",";
  ss << names[names.size() - 1];
  return ss.str();
}

template <typename idx_t>
bool db_result<idx_t>::hasVariable(std::string name)
{
  for (size_t i = 0; i < names.size(); i++)
    if (names[i] == name) return true;
  return false;
}

template <typename idx_t>
std::string db_result<idx_t>::toString()
{
  std::stringstream ss;
  ss << "db_result with " << columns.size() << " columns of length " << columnSize << "\n";
  for (size_t i = 0; i < columns.size(); i++) ss << names[i] << " ";
  ss << "\n";
  std::vector<std::vector<idx_t>> hostColumns;
  hostColumns.resize(columns.size());
  for (size_t i = 0; i < columns.size(); i++) {
    hostColumns[i].resize(columnSize);
    CUDA_TRY(cudaMemcpy(
      hostColumns[i].data(), columns[i].data(), sizeof(idx_t) * columnSize, cudaMemcpyDefault));
  }
  for (idx_t i = 0; i < columnSize; i++) {
    for (size_t j = 0; j < hostColumns.size(); j++) ss << hostColumns[j][i] << " ";
    ss << "\n";
  }
  return ss.str();
}

template class db_result<int32_t>;
template class db_result<int64_t>;

string_table::string_table(string_table&& other)
{
  this->names   = std::move(other.names);
  this->columns = std::move(other.columns);
}

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
