/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <raft/core/handle.hpp>

namespace cugraph {
namespace etl {

/**
 * @brief     Renumber a pair cudf tables
 *
 * Given a src table and a dst table, each corresponding entry represents an
 * edge in a graph.  Renumber the vertices so that they are represented
 * by an integer of the range [0, number_of_unique_vertices).
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @param src_table   each row of the table identifies the source vertex
 *                    for the graph
 * @param dst_table   each row of the table identifies the destination vertex
 *                    for the graph
 * @param dtype       the data type of the returned columns, should be INT32
 *                    or INT64.
 *
 * @return tuple with the following three values:
 *    1) column (of type dtype) with the source vertices represented as integers
 *    2) column (of type dtype) with the destination vertices represented as integers
 *    3) table with the vertex id as a column and the vertex columns from src_table
 *       and/or dst_table that correspond to the vertex id
 *
 */
std::
  tuple<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>>
  renumber_cudf_tables(raft::handle_t const& handle,
                       cudf::table_view const& src_table,
                       cudf::table_view const& dst_table,
                       cudf::type_id dtype);

}  // namespace etl
}  // namespace cugraph
