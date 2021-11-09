/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cugraph_etl/functions.hpp>

#include <cugraph/utilities/error.hpp>

namespace cugraph {
namespace etl {

std::
  tuple<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>>
  renumber_cudf_tables(cudf::table_view const& src_table,
                       cudf::table_view const& dst_table,
                       cudf::type_id dtype)
{
  CUGRAPH_FAIL("not implemented yet");
}

}  // namespace etl
}  // namespace cugraph
