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

#include <db/db_context.cuh>
#include <sstream>
#include <utilities/error.hpp>

namespace cugraph {
namespace db {

template <typename idx_t>
encoder<idx_t>::encoder()
{
  next_id = 0;
}

template <typename idx_t>
idx_t encoder<idx_t>::encode(std::string val)
{
  if (string_to_id.count(val) == 1) {
    return string_to_id[val];
  } else {
    idx_t myId = next_id;
    ++next_id;
    id_to_string[myId] = val;
    string_to_id[val]  = myId;
    return myId;
  }
}

template <typename idx_t>
std::string encoder<idx_t>::decode(idx_t val)
{
  if (id_to_string.count(val) == 1) {
    return id_to_string[val];
  } else {
    return std::string("");
  }
}

template <typename idx_t>
idx_t encoder<idx_t>::get_id()
{
  idx_t myId = next_id;
  ++next_id;
  return myId;
}

template <typename idx_t>
std::string encoder<idx_t>::to_string()
{
  std::stringstream ss;
  idx_t encoded = id_to_string.size();
  idx_t ids     = next_id - encoded;
  ss << "Encoder object with " << encoded << " encoded values and " << ids << " allocated ids:\n";
  for (auto it = id_to_string.begin(); it != id_to_string.end(); it++) {
    ss << it->first << " == " << it->second << "\n";
  }
  return ss.str();
}

template class encoder<int32_t>;
template class encoder<int64_t>;

template <typename idx_t>
context<idx_t>::context(encoder<idx_t>* e,
                        db_table<idx_t>* rt,
                        db_table<idx_t>* nlt,
                        db_table<idx_t>* npt,
                        db_table<idx_t>* rpt)
{
  coder                         = e;
  relationships_table           = rt;
  node_labels_table             = nlt;
  node_properties_table         = npt;
  relationship_properties_table = rpt;
  next_id                       = 0;
}

template <typename idx_t>
context<idx_t>::context(context<idx_t>&& other)
{
  *this = std::move(other);
}

template <typename idx_t>
context<idx_t>& context<idx_t>::operator=(context<idx_t>&& other)
{
  if (this != &other) {
    coder                               = other.coder;
    other.coder                         = nullptr;
    relationships_table                 = other.relationships_table;
    other.relationships_table           = nullptr;
    node_labels_table                   = other.node_labels_table;
    other.node_labels_table             = nullptr;
    node_properties_table               = other.node_properties_table;
    other.node_properties_table         = nullptr;
    relationship_properties_table       = other.relationship_properties_table;
    other.relationship_properties_table = nullptr;
    next_id                             = other.next_id;
    named_results                       = std::move(other.named_results);
    result_names                        = std::move(other.result_names);
    variables                           = std::move(other.variables);
  }
  return *this;
}

template <typename idx_t>
std::string context<idx_t>::get_unique_id()
{
  std::stringstream ss;
  ss << "UniqueIdentifier_" << next_id++;
  return ss.str();
}

template <typename idx_t>
db_table<idx_t>* context<idx_t>::get_relationships_table()
{
  return relationships_table;
}

template <typename idx_t>
db_table<idx_t>* context<idx_t>::get_node_labels_table()
{
  return node_labels_table;
}

template <typename idx_t>
db_table<idx_t>* context<idx_t>::get_node_properties_table()
{
  return node_properties_table;
}

template <typename idx_t>
db_table<idx_t>* context<idx_t>::get_relationship_properties_table()
{
  return relationship_properties_table;
}

template <typename idx_t>
encoder<idx_t>* context<idx_t>::get_encoder()
{
  return coder;
}

template <typename idx_t>
void context<idx_t>::register_named_result(std::string name, string_table&& result)
{
  result_names.push_back(name);
  named_results.push_back(std::move(result));
}

template <typename idx_t>
void context<idx_t>::register_variables(db_result<idx_t>&& result)
{
  variables.push_back(std::move(result));
}

template <typename idx_t>
std::string context<idx_t>::get_named_entry(std::string name, std::string colname, idx_t row)
{
  int pos = -1;
  for (size_t i = 0; i < result_names.size(); i++) {
    if (result_names[i] == name) pos = i;
  }
  CUGRAPH_EXPECTS(pos > 0, "Named result not found");
  return named_results[pos][colname][row];
}

template <typename idx_t>
std::string context<idx_t>::get_named_entry(std::string name, idx_t col, idx_t row)
{
  int pos = -1;
  for (size_t i = 0; i < result_names.size(); i++) {
    if (result_names[i] == name) pos = i;
  }
  CUGRAPH_EXPECTS(pos > 0, "Named result not found");
  return named_results[pos][col][row];
}

template <typename idx_t>
idx_t context<idx_t>::get_named_rows(std::string name)
{
  int pos = -1;
  for (size_t i = 0; i < result_names.size(); i++) {
    if (result_names[i] == name) pos = i;
  }
  CUGRAPH_EXPECTS(pos > 0, "Named result not found");
  return named_results[pos][0].size();
}

template <typename idx_t>
std::vector<idx_t>&& context<idx_t>::get_variable_column(std::string name)
{
  int pos = -1;
  for (size_t i = 0; i < variables.size(); i++)
    if (variables[i].hasVariable(name)) pos = i;
  CUGRAPH_EXPECTS(pos > 0, "Variable not found");
  return variables[pos].getHostColumn(name);
}

template <typename idx_t>
bool context<idx_t>::has_variable(std::string name)
{
  for (size_t i = 0; i < variables.size(); i++) {
    if (variables[i].hasVariable(name)) return true;
  }
  return false;
}

template <typename idx_t>
bool context<idx_t>::has_named(std::string name)
{
  for (size_t i = 0; i < result_names.size(); i++) {
    if (result_names[i] == name) return true;
  }
  return false;
}

template class context<int32_t>;
template class context<int64_t>;

}  // namespace db
}  // namespace cugraph
