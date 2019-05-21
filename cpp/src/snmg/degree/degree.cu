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

#include "degree.cuh"

template<typename idx_t>
gdf_error gdf_snmg_degree_impl(int x,
                               size_t* part_offsets,
                               gdf_column* off,
                               gdf_column* ind,
                               gdf_column** x_cols) {
  std::cout << "gdf_snmg_degree_impl()\n";
  GDF_REQUIRE(off->size > 0, GDF_INVALID_API_CALL);
  GDF_REQUIRE(ind->size > 0, GDF_INVALID_API_CALL);
  GDF_REQUIRE(off->dtype == ind->dtype, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(off->null_count + ind->null_count == 0, GDF_VALIDITY_UNSUPPORTED);

  gdf_error status;
  auto p = omp_get_num_threads();

  idx_t* degree[p];
  for (auto i = 0; i < p; ++i) {
    GDF_REQUIRE(x_cols[i] != nullptr, GDF_INVALID_API_CALL);
    GDF_REQUIRE(x_cols[i]->size > 0, GDF_INVALID_API_CALL);
    degree[i] = static_cast<idx_t*>(x_cols[i]->data);
  }

  status = cugraph::snmg_degree(x,
                                part_offsets,
                                static_cast<idx_t*>(off->data),
                                static_cast<idx_t*>(ind->data),
                                degree);
  return status;
}

gdf_error gdf_snmg_degree(int x,
                          size_t* part_offsets,
                          gdf_column* off,
                          gdf_column* ind,
                          gdf_column** x_cols) {
  std::cout << "gdf_snmg_degree()\n";
  GDF_REQUIRE(part_offsets != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(off != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(ind != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(x_cols != nullptr, GDF_INVALID_API_CALL);
  switch (off->dtype) {
    case GDF_INT32:
      return gdf_snmg_degree_impl<int32_t>(x, part_offsets, off, ind, x_cols);
    case GDF_INT64:
      return gdf_snmg_degree_impl<int64_t>(x, part_offsets, off, ind, x_cols);
    default:
      return GDF_INVALID_API_CALL;
  }
}
