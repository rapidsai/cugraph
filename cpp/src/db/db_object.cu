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
#include <thrust/binary_search.h>
#include <cub/device/device_run_length_encode.cuh>
#include <db/db_object.cuh>
#include <sstream>

namespace cugraph {
namespace db {
// Define kernel for copying run length encoded values into offset slots.
template <typename T>
__global__ void offsetsKernel(T runCounts, T* unique, T* counts, T* offsets)
{
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < runCounts) offsets[unique[tid]] = counts[tid];
}

template <typename idx_t>
db_pattern_entry<idx_t>::db_pattern_entry(std::string variable)
{
  is_var       = true;
  variableName = variable;
}

template <typename idx_t>
db_pattern_entry<idx_t>::db_pattern_entry(idx_t constant)
{
  is_var        = false;
  constantValue = constant;
}

template <typename idx_t>
db_pattern_entry<idx_t>::db_pattern_entry(const db_pattern_entry<idx_t>& other)
{
  is_var        = other.is_var;
  constantValue = other.constantValue;
  variableName  = other.variableName;
}

template <typename idx_t>
db_pattern_entry<idx_t>& db_pattern_entry<idx_t>::operator=(const db_pattern_entry<idx_t>& other)
{
  is_var        = other.is_var;
  constantValue = other.constantValue;
  variableName  = other.variableName;
  return *this;
}

template <typename idx_t>
bool db_pattern_entry<idx_t>::isVariable() const
{
  return is_var;
}

template <typename idx_t>
idx_t db_pattern_entry<idx_t>::getConstant() const
{
  return constantValue;
}

template <typename idx_t>
std::string db_pattern_entry<idx_t>::getVariable() const
{
  return variableName;
}

template class db_pattern_entry<int32_t>;
template class db_pattern_entry<int64_t>;

template <typename idx_t>
db_pattern<idx_t>::db_pattern()
{
}

template <typename idx_t>
db_pattern<idx_t>::db_pattern(const db_pattern<idx_t>& other)
{
  for (size_t i = 0; i < other.entries.size(); i++) { entries.push_back(other.getEntry(i)); }
}

template <typename idx_t>
db_pattern<idx_t>& db_pattern<idx_t>::operator=(const db_pattern<idx_t>& other)
{
  entries = other.entries;
  return *this;
}

template <typename idx_t>
int db_pattern<idx_t>::getSize() const
{
  return entries.size();
}

template <typename idx_t>
const db_pattern_entry<idx_t>& db_pattern<idx_t>::getEntry(int position) const
{
  return entries[position];
}

template <typename idx_t>
void db_pattern<idx_t>::addEntry(db_pattern_entry<idx_t>& entry)
{
  entries.push_back(entry);
}

template <typename idx_t>
bool db_pattern<idx_t>::isAllConstants()
{
  for (size_t i = 0; i < entries.size(); i++)
    if (entries[i].isVariable()) return false;
  return true;
}

template class db_pattern<int32_t>;
template class db_pattern<int64_t>;

template <typename idx_t>
db_column_index<idx_t>::db_column_index(rmm::device_buffer&& off, rmm::device_buffer&& ind)
{
  offsets     = std::move(off);
  indirection = std::move(ind);
}

template <typename idx_t>
void db_column_index<idx_t>::resetData(rmm::device_buffer&& off, rmm::device_buffer&& ind)
{
  offsets     = std::move(off);
  indirection = std::move(ind);
}

template <typename idx_t>
idx_t* db_column_index<idx_t>::getOffsets()
{
  return reinterpret_cast<idx_t*>(offsets.data());
}

template <typename idx_t>
idx_t db_column_index<idx_t>::getOffsetsSize()
{
  return offsets.size() / sizeof(idx_t);
}

template <typename idx_t>
idx_t* db_column_index<idx_t>::getIndirection()
{
  return reinterpret_cast<idx_t*>(indirection.data());
}

template <typename idx_t>
idx_t db_column_index<idx_t>::getIndirectionSize()
{
  return indirection.size() / sizeof(idx_t);
}

template <typename idx_t>
std::string db_column_index<idx_t>::toString()
{
  std::stringstream ss;
  ss << "db_column_index:\n";
  ss << "Offsets: ";
  std::vector<idx_t> hostOff(getOffsetsSize());
  idx_t* hostOffsets = reinterpret_cast<idx_t*>(hostOff.data());
  CUDA_TRY(
    cudaMemcpy(hostOffsets, offsets.data(), sizeof(idx_t) * getOffsetsSize(), cudaMemcpyDefault));
  for (idx_t i = 0; i < getOffsetsSize(); i++) { ss << hostOff[i] << " "; }
  ss << "\nIndirection: ";
  std::vector<idx_t> hostInd(getIndirectionSize());
  idx_t* hostIndirection = reinterpret_cast<idx_t*>(hostInd.data());
  CUDA_TRY(cudaMemcpy(
    hostIndirection, indirection.data(), sizeof(idx_t) * getIndirectionSize(), cudaMemcpyDefault));
  for (idx_t i = 0; i < getIndirectionSize(); i++) { ss << hostInd[i] << " "; }
  ss << "\n";
  return ss.str();
}

template class db_column_index<int32_t>;
template class db_column_index<int64_t>;

template <typename idx_t>
db_table<idx_t>::db_table()
{
  column_size = 0;
}

template <typename idx_t>
void db_table<idx_t>::addColumn(std::string name)
{
  CUGRAPH_EXPECTS(column_size == 0, "Can't add a column to a non-empty table");

  rmm::device_buffer _col;
  columns.push_back(std::move(_col));
  names.push_back(name);
  indices.resize(indices.size() + 1);
}

template <typename idx_t>
void db_table<idx_t>::addEntry(db_pattern<idx_t>& pattern)
{
  CUGRAPH_EXPECTS(pattern.isAllConstants(), "Can't add an entry that isn't all constants");
  CUGRAPH_EXPECTS(static_cast<size_t>(pattern.getSize()) == columns.size(),
                  "Can't add an entry that isn't the right size");
  inputBuffer.push_back(pattern);
}

template <typename idx_t>
void db_table<idx_t>::rebuildIndices()
{
  for (size_t i = 0; i < columns.size(); i++) {
    // Copy the column's data to a new array
    idx_t size = column_size;
    rmm::device_buffer tempColumn(sizeof(idx_t) * size);
    cudaMemcpy(tempColumn.data(), columns[i].data(), sizeof(idx_t) * size, cudaMemcpyDefault);

    // Construct an array of ascending integers
    rmm::device_buffer indirection(sizeof(idx_t) * size);
    thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr),
                     reinterpret_cast<idx_t*>(indirection.data()),
                     reinterpret_cast<idx_t*>(indirection.data()) + size);

    // Sort the arrays together
    thrust::sort_by_key(rmm::exec_policy(nullptr)->on(nullptr),
                        reinterpret_cast<idx_t*>(tempColumn.data()),
                        reinterpret_cast<idx_t*>(tempColumn.data()) + size,
                        reinterpret_cast<idx_t*>(indirection.data()));

    // Compute offsets array based on sorted column
    idx_t maxId;
    CUDA_TRY(cudaMemcpy(&maxId,
                        reinterpret_cast<idx_t*>(tempColumn.data()) + size - 1,
                        sizeof(idx_t),
                        cudaMemcpyDefault));
    rmm::device_buffer offsets(sizeof(idx_t) * (maxId + 2));
    thrust::lower_bound(rmm::exec_policy(nullptr)->on(nullptr),
                        reinterpret_cast<idx_t*>(tempColumn.data()),
                        reinterpret_cast<idx_t*>(tempColumn.data()) + size,
                        thrust::counting_iterator<idx_t>(0),
                        thrust::counting_iterator<idx_t>(maxId + 2),
                        reinterpret_cast<idx_t*>(offsets.data()));

    // Assign new offsets array and indirection vector to index
    indices[i].resetData(std::move(offsets), std::move(indirection));
  }
}

template <typename idx_t>
void db_table<idx_t>::flush_input()
{
  if (inputBuffer.size() == size_t{0}) return;
  idx_t tempSize = inputBuffer.size();
  std::vector<std::vector<idx_t>> tempColumns(columns.size());
  for (size_t i = 0; i < columns.size(); i++) {
    tempColumns[i].resize(tempSize);
    for (idx_t j = 0; j < tempSize; j++) {
      tempColumns[i][j] = inputBuffer[j].getEntry(i).getConstant();
    }
  }
  inputBuffer.clear();
  idx_t currentSize = column_size;
  idx_t newSize     = currentSize + tempSize;
  std::vector<rmm::device_buffer> newColumns;
  for (size_t i = 0; i < columns.size(); i++) { newColumns.emplace_back(sizeof(idx_t) * newSize); }
  for (size_t i = 0; i < columns.size(); i++) {
    if (currentSize > 0)
      CUDA_TRY(cudaMemcpy(
        newColumns[i].data(), columns[i].data(), sizeof(idx_t) * currentSize, cudaMemcpyDefault));
    CUDA_TRY(cudaMemcpy(reinterpret_cast<idx_t*>(newColumns[i].data()) + currentSize,
                        tempColumns[i].data(),
                        sizeof(idx_t) * tempSize,
                        cudaMemcpyDefault));
    columns[i]  = std::move(newColumns[i]);
    column_size = newSize;
  }

  rebuildIndices();
}

template <typename idx_t>
std::string db_table<idx_t>::toString()
{
  idx_t columnSize = 0;
  if (columns.size() > 0) columnSize = column_size;
  std::stringstream ss;
  ss << "Table with " << columns.size() << " columns of length " << columnSize << "\n";
  for (size_t i = 0; i < names.size(); i++) ss << names[i] << " ";
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

template <typename idx_t>
db_column_index<idx_t>& db_table<idx_t>::getIndex(int idx)
{
  return indices[idx];
}

template <typename idx_t>
idx_t* db_table<idx_t>::getColumn(int idx)
{
  return reinterpret_cast<idx_t*>(columns[idx].data());
}

template class db_table<int32_t>;
template class db_table<int64_t>;

template <typename idx_t>
db_object<idx_t>::db_object()
{
  next_id = 0;
  relationshipsTable.addColumn("begin");
  relationshipsTable.addColumn("end");
  relationshipsTable.addColumn("type");
  nodeLabelsTable.addColumn("nodeId");
  nodeLabelsTable.addColumn("Label Id");
  nodePropertiesTable.addColumn("nodeId");
  nodePropertiesTable.addColumn("propertyLabel");
  nodePropertiesTable.addColumn("value");
  relationshipPropertiesTable.addColumn("id");
  relationshipPropertiesTable.addColumn("name");
  relationshipPropertiesTable.addColumn("value");
}

template <typename idx_t>
std::string db_object<idx_t>::query(std::string query)
{
  return "";
}

template class db_object<int32_t>;
template class db_object<int64_t>;
}  // namespace db
}  // namespace cugraph
