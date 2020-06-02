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
#include <cypher-parser.h>
#include <rmm_utils.h>
#include <db/db_execution.cuh>
#include <db/db_object.cuh>
#include <sstream>

namespace cugraph {
namespace db {

template <typename idx_t>
db_object<idx_t>::db_object()
{
  relationshipsTable.addColumn("begin");
  relationshipsTable.addColumn("end");
  relationshipsTable.addColumn("type");
  nodeLabelsTable.addColumn("nodeId");
  nodeLabelsTable.addColumn("LabelId");
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
  // Create a context to be used for this query
  context<idx_t> ctx(&idEncoder,
                     &relationshipsTable,
                     &nodeLabelsTable,
                     &nodePropertiesTable,
                     &relationshipPropertiesTable);

  // Parse the query
  cypher_parse_result_t* result = cypher_parse(query.c_str(), NULL, NULL, 0);

  // Construct a query plan
  query_plan<idx_t> plan(result, std::move(ctx));

  // Free up the result from the parser
  cypher_parse_result_free(result);

  // Execute the query plan
  std::string output = plan.execute();

  // Complete updates on tables
  relationshipsTable.flush_input();
  nodeLabelsTable.flush_input();
  nodePropertiesTable.flush_input();
  relationshipPropertiesTable.flush_input();

  // Return the result.
  return output;
}

template <typename idx_t>
std::string db_object<idx_t>::toString()
{
  std::stringstream ss;
  ss << "Database:\n";
  ss << "Encoder:\n";
  ss << idEncoder.toString();
  ss << "Relationships Table:\n";
  ss << relationshipsTable.toString();
  ss << "Node Labels Table:\n";
  ss << nodeLabelsTable.toString();
  ss << "Node Properties Table:\n";
  ss << nodePropertiesTable.toString();
  ss << "Relationship Properties Table:\n";
  ss << relationshipPropertiesTable.toString();
  return ss.str();
}

template class db_object<int32_t>;
template class db_object<int64_t>;
}  // namespace db
}  // namespace cugraph
