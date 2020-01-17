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
#include <db/db_object.cuh>
#include <rmm_utils.h>

#define MAXBLOCKS 65535
#define FIND_MATCHES_BLOCK_SIZE 512

namespace cugraph { 
namespace db {
    /**
     * Method to find matches to a pattern against an indexed table.
     * @param pattern The pattern to match against. It is assumed that the order of the entries
     *  matches the order of the columns in the table being searched.
     * @param table The table to find matching entries within.
     * @param frontier The frontier of already bound values. The search is restricted to entries in the table
     *  which match at least the frontier entry. If the frontier is null, then the entire table will be
     *  scanned.
     * @param indexColumn The name of the variable in the pattern which is bound to the frontier
     *  and which indicates which index should be used on the table.
     * @return A result table with columns for each variable in the given pattern containing the bound
     *  values to those variables.
     */
    template<typename idx_t>
    db_result<idx_t> findMatches(db_pattern<idx_t>& pattern,
                                 db_table<idx_t>& table,
                                 gdf_column* frontier,
                                 int indexPosition);
} } //namespace
