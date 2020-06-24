/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <functions.hpp>
#include "COOtoCSR.cuh"

namespace cugraph {

//unsigned + float
template std::unique_ptr<experimental::GraphCSR<unsigned, unsigned, float>>
coo_to_csr<unsigned, unsigned, float>(
  experimental::GraphCOOView<unsigned, unsigned, float> const &graph,
  rmm::mr::device_memory_resource *);

//unsigned + double
template std::unique_ptr<experimental::GraphCSR<unsigned, unsigned, double>>
coo_to_csr<unsigned, unsigned, double>(
  experimental::GraphCOOView<unsigned, unsigned, double> const &graph,
  rmm::mr::device_memory_resource *);

//int + float
template std::unique_ptr<experimental::GraphCSR<int32_t, int32_t, float>>
coo_to_csr<int32_t, int32_t, float>(
  experimental::GraphCOOView<int32_t, int32_t, float> const &graph,
  rmm::mr::device_memory_resource *);

//int + double
template std::unique_ptr<experimental::GraphCSR<int32_t, int32_t, double>>
coo_to_csr<int32_t, int32_t, double>(
  experimental::GraphCOOView<int32_t, int32_t, double> const &graph,
  rmm::mr::device_memory_resource *);

//long + float
template std::unique_ptr<experimental::GraphCSR<long, long, float>>
coo_to_csr<long, long, float>(
  experimental::GraphCOOView<long, long, float> const &graph,
  rmm::mr::device_memory_resource *);

//long + double 
template std::unique_ptr<experimental::GraphCSR<long, long, double>>
coo_to_csr<long, long, double>(
  experimental::GraphCOOView<long, long, double> const &graph,
  rmm::mr::device_memory_resource *);

}  // namespace cugraph
