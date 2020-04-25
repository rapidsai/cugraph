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

#include <cugraph.h>
#include <rmm_utils.h>
#include <db/db_object.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <sstream>
#include <thrust/binary_search.h>

namespace cugraph {
namespace db {
// Define kernel for copying run length encoded values into offset slots.
template<typename T>
__global__ void offsetsKernel(T runCounts, T* unique, T* counts, T* offsets) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < runCounts)
    offsets[unique[tid]] = counts[tid];
}

template<typename idx_t>
db_pattern_entry<idx_t>::db_pattern_entry(std::string variable) {
  is_var = true;
  variableName = variable;
}

template<typename idx_t>
db_pattern_entry<idx_t>::db_pattern_entry(idx_t constant) {
  is_var = false;
  constantValue = constant;
}

template<typename idx_t>
db_pattern_entry<idx_t>::db_pattern_entry(const db_pattern_entry<idx_t>& other) {
  is_var = other.is_var;
  constantValue = other.constantValue;
  variableName = other.variableName;
}

template<typename idx_t>
db_pattern_entry<idx_t>& db_pattern_entry<idx_t>::operator=(const db_pattern_entry<idx_t>& other) {
  is_var = other.is_var;
  constantValue = other.constantValue;
  variableName = other.variableName;
  return *this;
}

template<typename idx_t>
bool db_pattern_entry<idx_t>::isVariable() const {
  return is_var;
}

template<typename idx_t>
idx_t db_pattern_entry<idx_t>::getConstant() const {
  return constantValue;
}

template<typename idx_t>
std::string db_pattern_entry<idx_t>::getVariable() const {
  return variableName;
}

template class db_pattern_entry<int32_t>;
template class db_pattern_entry<int64_t>;

template<typename idx_t>
db_pattern<idx_t>::db_pattern() {

}

template<typename idx_t>
db_pattern<idx_t>::db_pattern(const db_pattern<idx_t>& other) {
  for (size_t i = 0; i < other.entries.size(); i++) {
    entries.push_back(other.getEntry(i));
  }
}

template<typename idx_t>
db_pattern<idx_t>& db_pattern<idx_t>::operator=(const db_pattern<idx_t>& other) {
  entries = other.entries;
  return *this;
}

template<typename idx_t>
int db_pattern<idx_t>::getSize() const {
  return entries.size();
}

template<typename idx_t>
const db_pattern_entry<idx_t>& db_pattern<idx_t>::getEntry(int position) const {
  return entries[position];
}

template<typename idx_t>
void db_pattern<idx_t>::addEntry(db_pattern_entry<idx_t>& entry) {
  entries.push_back(entry);
}

template<typename idx_t>
bool db_pattern<idx_t>::isAllConstants() {
  for (size_t i = 0; i < entries.size(); i++)
    if (entries[i].isVariable())
      return false;
  return true;
}

template class db_pattern<int32_t>;
template class db_pattern<int64_t>;

template<typename idx_t>
db_column_index<idx_t>::db_column_index() {
  offsets_size = 0;
  indirection_size = 0;
}

template<typename idx_t>
db_column_index<idx_t>::db_column_index(rmm::device_buffer&& _offsets,
                                        idx_t _offsets_size,
                                        rmm::device_buffer&& _indirection,
                                        idx_t _indirection_size) {
  offsets = std::move(_offsets);
  offsets_size = _offsets_size;
  indirection = std::move(_indirection);
  indirection_size = _indirection_size;
}

template<typename idx_t>
db_column_index<idx_t>::db_column_index(db_column_index<idx_t>&& other) {
  offsets = std::move(other.offsets);
  offsets_size = other.offsets_size;
  indirection = std::move(other.indirection);
  indirection_size = other.indirection_size;
  other.offsets_size = 0;
  other.indirection_size = 0;
}

template<typename idx_t>
db_column_index<idx_t>::~db_column_index() {
}

template<typename idx_t>
db_column_index<idx_t>& db_column_index<idx_t>::operator=(db_column_index<idx_t>&& other) {
  offsets = std::move(other.offsets);
  offsets_size = other.offsets_size;
  indirection = std::move(other.indirection);
  indirection_size = other.indirection_size;
  other.offsets_size = 0;
  other.indirection_size = 0;
  return *this;
}

template<typename idx_t>
void db_column_index<idx_t>::resetData(rmm::device_buffer&& _offsets,
                                       idx_t _offsets_size,
                                       rmm::device_buffer&& _indirection,
                                       idx_t _indirection_size) {
  offsets = std::move(_offsets);
  offsets_size = _offsets_size;
  indirection = std::move(_indirection);
  indirection_size = _indirection_size;
}

template<typename idx_t>
idx_t* db_column_index<idx_t>::getOffsets() {
  return (idx_t*) offsets.data();
}

template<typename idx_t>
idx_t db_column_index<idx_t>::getOffsetsSize() {
  return offsets_size;
}

template<typename idx_t>
idx_t* db_column_index<idx_t>::getIndirection() {
  return (idx_t*) indirection.data();
}

template<typename idx_t>
idx_t db_column_index<idx_t>::getIndirectionSize() {
  return indirection_size;
}

template<typename idx_t>
std::string db_column_index<idx_t>::toString() {
  std::stringstream ss;
  ss << "db_column_index:\n";
  ss << "Offsets: ";
  idx_t* hostOffsets = (idx_t*) malloc(sizeof(idx_t) * offsets_size);
  cudaMemcpy(hostOffsets, offsets.data(), sizeof(idx_t) * offsets_size, cudaMemcpyDefault);
  for (idx_t i = 0; i < offsets_size; i++) {
    ss << hostOffsets[i] << " ";
  }
  free(hostOffsets);
  ss << "\nIndirection: ";
  idx_t* hostIndirection = (idx_t*) malloc(sizeof(idx_t) * indirection_size);
  cudaMemcpy(hostIndirection,
             indirection.data(),
             sizeof(idx_t) * indirection_size,
             cudaMemcpyDefault);
  for (idx_t i = 0; i < indirection_size; i++) {
    ss << hostIndirection[i] << " ";
  }
  free(hostIndirection);
  ss << "\n";
  return ss.str();
}

template class db_column_index<int32_t>;
template class db_column_index<int64_t>;

template<typename idx_t>
db_result<idx_t>::db_result() {
  dataValid = false;
  columnSize = 0;
}

template<typename idx_t>
db_result<idx_t>::db_result(db_result&& other) {
  dataValid = other.dataValid;
  columns = std::move(other.columns);
  names = std::move(other.names);
  other.dataValid = false;
}

template<typename idx_t>
db_result<idx_t>& db_result<idx_t>::operator =(db_result<idx_t>&& other) {
  dataValid = other.dataValid;
  columns = std::move(other.columns);
  names = std::move(other.names);
  other.dataValid = false;
  return *this;
}

template<typename idx_t>
db_result<idx_t>::~db_result() {
}

template<typename idx_t>
idx_t db_result<idx_t>::getSize() {
  return columnSize;
}

template<typename idx_t>
idx_t* db_result<idx_t>::getData(std::string idx) {
  if (!dataValid)
    throw new std::invalid_argument("Data not valid");

  idx_t* returnPtr = nullptr;
  for (size_t i = 0; i < names.size(); i++)
    if (names[i] == idx)
      returnPtr = (idx_t*) columns[i].data();
  return returnPtr;
}

template<typename idx_t>
void db_result<idx_t>::addColumn(std::string columnName) {
  if (dataValid)
    throw new std::invalid_argument("Cannot add a column to an allocated result");
  names.push_back(columnName);
}

template<typename idx_t>
void db_result<idx_t>::allocateColumns(idx_t size) {
  if (dataValid)
    throw new std::invalid_argument("Already allocated columns");
  for (size_t i = 0; i < names.size(); i++) {
    rmm::device_buffer col(sizeof(idx_t) * size);
    columns.push_back(std::move(col));
  }
  dataValid = true;
  columnSize = size;
}

template<typename idx_t>
std::string db_result<idx_t>::toString() {
  std::stringstream ss;
  ss << "db_result with " << columns.size() << " columns of length " << columnSize << "\n";
  for (size_t i = 0; i < columns.size(); i++)
    ss << names[i] << " ";
  ss << "\n";
  std::vector<idx_t*> hostColumns;
  for (size_t i = 0; i < columns.size(); i++) {
    idx_t* hostColumn = (idx_t*) malloc(sizeof(idx_t) * columnSize);
    cudaMemcpy(hostColumn, columns[i].data(), sizeof(idx_t) * columnSize, cudaMemcpyDefault);
    hostColumns.push_back(hostColumn);
  }
  for (idx_t i = 0; i < columnSize; i++) {
    for (size_t j = 0; j < hostColumns.size(); j++)
      ss << hostColumns[j][i] << " ";
    ss << "\n";
  }
  for (size_t i = 0; i < hostColumns.size(); i++)
    free(hostColumns[i]);
  return ss.str();
}

template class db_result<int32_t>;
template class db_result<int64_t>;

template<typename idx_t>
db_table<idx_t>::db_table() {
  column_size = 0;
}

template<typename idx_t>
db_table<idx_t>::~db_table() {
}

template<typename idx_t>
void db_table<idx_t>::addColumn(std::string name) {
  if (columns.size() > size_t { 0 } && column_size > 0)
    throw new std::invalid_argument("Can't add a column to a non-empty table");

  rmm::device_buffer _col;
  columns.push_back(std::move(_col));
  names.push_back(name);
  indices.resize(indices.size() + 1);
}

template<typename idx_t>
void db_table<idx_t>::addEntry(db_pattern<idx_t>& pattern) {
  if (!pattern.isAllConstants())
    throw new std::invalid_argument("Can't add an entry that isn't all constants");
  if (static_cast<size_t>(pattern.getSize()) != columns.size())
    throw new std::invalid_argument("Can't add an entry that isn't the right size");
  inputBuffer.push_back(pattern);
}

template<typename idx_t>
void db_table<idx_t>::rebuildIndices() {
  for (size_t i = 0; i < columns.size(); i++) {
    // Copy the column's data to a new array
    idx_t size = column_size;
    rmm::device_buffer tempColumn(sizeof(idx_t) * size);
    cudaMemcpy(tempColumn.data(), columns[i].data(), sizeof(idx_t) * size, cudaMemcpyDefault);

    // Construct an array of ascending integers
    rmm::device_buffer indirection(sizeof(idx_t) * size);
    thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr),
                     (idx_t*) indirection.data(),
                     (idx_t*) indirection.data() + size);

    // Sort the arrays together
    thrust::sort_by_key(rmm::exec_policy(nullptr)->on(nullptr),
                        (idx_t*) tempColumn.data(),
                        (idx_t*) tempColumn.data() + size,
                        (idx_t*) indirection.data());

    // Compute offsets array based on sorted column
    idx_t maxId;
    cudaMemcpy(&maxId, (idx_t*) tempColumn.data() + size - 1, sizeof(idx_t), cudaMemcpyDefault);
    rmm::device_buffer offsets(sizeof(idx_t) * (maxId + 2));
    thrust::lower_bound(rmm::exec_policy(nullptr)->on(nullptr),
                        (idx_t*) tempColumn.data(),
                        (idx_t*) tempColumn.data() + size,
                        thrust::counting_iterator<idx_t>(0),
                        thrust::counting_iterator<idx_t>(maxId + 2),
                        (idx_t*) offsets.data());

    // Assign new offsets array and indirection vector to index
    indices[i].resetData(std::move(offsets), maxId + 2, std::move(indirection), size);
  }
}

template<typename idx_t>
void db_table<idx_t>::flush_input() {
  if (inputBuffer.size() == size_t { 0 })
    return;
  idx_t tempSize = inputBuffer.size();
  std::vector<idx_t*> tempColumns;
  for (size_t i = 0; i < columns.size(); i++) {
    tempColumns.push_back((idx_t*) malloc(sizeof(idx_t) * tempSize));
    for (idx_t j = 0; j < tempSize; j++) {
      tempColumns.back()[j] = inputBuffer[j].getEntry(i).getConstant();
    }
  }
  inputBuffer.clear();
  idx_t currentSize = column_size;
  idx_t newSize = currentSize + tempSize;
  std::vector<rmm::device_buffer> newColumns;
  for (size_t i = 0; i < columns.size(); i++) {
    rmm::device_buffer newCol(sizeof(idx_t) * newSize);
    newColumns.push_back(std::move(newCol));
  }
  for (size_t i = 0; i < columns.size(); i++) {
    if (currentSize > 0)
      cudaMemcpy(newColumns[i].data(),
                 columns[i].data(),
                 sizeof(idx_t) * currentSize,
                 cudaMemcpyDefault);
    cudaMemcpy((idx_t*)newColumns[i].data() + currentSize,
               tempColumns[i],
               sizeof(idx_t) * tempSize,
               cudaMemcpyDefault);
    free(tempColumns[i]);
    columns[i] = std::move(newColumns[i]);
    column_size = newSize;
  }

  rebuildIndices();
}

template<typename idx_t>
std::string db_table<idx_t>::toString() {
  idx_t columnSize = 0;
  if (columns.size() > 0)
    columnSize = column_size;
  std::stringstream ss;
  ss << "Table with " << columns.size() << " columns of length " << columnSize << "\n";
  for (size_t i = 0; i < names.size(); i++)
    ss << names[i] << " ";
  ss << "\n";
  std::vector<idx_t*> hostColumns;
  for (size_t i = 0; i < columns.size(); i++) {
    idx_t* hostColumn = (idx_t*) malloc(sizeof(idx_t) * columnSize);
    cudaMemcpy(hostColumn, columns[i].data(), sizeof(idx_t) * columnSize, cudaMemcpyDefault);
    hostColumns.push_back(hostColumn);
  }
  for (idx_t i = 0; i < columnSize; i++) {
    for (size_t j = 0; j < hostColumns.size(); j++)
      ss << hostColumns[j][i] << " ";
    ss << "\n";
  }
  for (size_t i = 0; i < hostColumns.size(); i++)
    free(hostColumns[i]);
  return ss.str();
}

template<typename idx_t>
db_column_index<idx_t>& db_table<idx_t>::getIndex(int idx) {
  return indices[idx];
}

template<typename idx_t>
idx_t* db_table<idx_t>::getColumn(int idx) {
  return (idx_t*)columns[idx].data();
}

template class db_table<int32_t>;
template class db_table<int64_t>;

template<typename idx_t>
db_object<idx_t>::db_object() {
  next_id = 0;
  relationshipsTable.addColumn("begin");
  relationshipsTable.addColumn("end");
  relationshipsTable.addColumn("type");
  relationshipPropertiesTable.addColumn("id");
  relationshipPropertiesTable.addColumn("name");
  relationshipPropertiesTable.addColumn("value");
}

template<typename idx_t>
std::string db_object<idx_t>::query(std::string query) {
  return "";
}

template class db_object<int32_t>;
template class db_object<int64_t>;
}
} //namespace
