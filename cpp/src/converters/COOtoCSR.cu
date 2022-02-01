/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "COOtoCSR.cuh"
#include <cugraph/functions.hpp>

namespace cugraph {

// Explicit instantiation for uint32_t + float
template std::unique_ptr<legacy::GraphCSR<uint32_t, uint32_t, float>>
coo_to_csr<uint32_t, uint32_t, float>(legacy::GraphCOOView<uint32_t, uint32_t, float> const& graph,
                                      rmm::mr::device_memory_resource*);

// Explicit instantiation for uint32_t + double
template std::unique_ptr<legacy::GraphCSR<uint32_t, uint32_t, double>>
coo_to_csr<uint32_t, uint32_t, double>(
  legacy::GraphCOOView<uint32_t, uint32_t, double> const& graph, rmm::mr::device_memory_resource*);

// Explicit instantiation for int + float
template std::unique_ptr<legacy::GraphCSR<int32_t, int32_t, float>>
coo_to_csr<int32_t, int32_t, float>(legacy::GraphCOOView<int32_t, int32_t, float> const& graph,
                                    rmm::mr::device_memory_resource*);

// Explicit instantiation for int + double
template std::unique_ptr<legacy::GraphCSR<int32_t, int32_t, double>>
coo_to_csr<int32_t, int32_t, double>(legacy::GraphCOOView<int32_t, int32_t, double> const& graph,
                                     rmm::mr::device_memory_resource*);

// Explicit instantiation for int64_t + float
template std::unique_ptr<legacy::GraphCSR<int64_t, int64_t, float>>
coo_to_csr<int64_t, int64_t, float>(legacy::GraphCOOView<int64_t, int64_t, float> const& graph,
                                    rmm::mr::device_memory_resource*);

// Explicit instantiation for int64_t + double
template std::unique_ptr<legacy::GraphCSR<int64_t, int64_t, double>>
coo_to_csr<int64_t, int64_t, double>(legacy::GraphCOOView<int64_t, int64_t, double> const& graph,
                                     rmm::mr::device_memory_resource*);

// in-place versions:
//
// Explicit instantiation for uint32_t + float
template void coo_to_csr_inplace<uint32_t, uint32_t, float>(
  legacy::GraphCOOView<uint32_t, uint32_t, float>& graph,
  legacy::GraphCSRView<uint32_t, uint32_t, float>& result);

// Explicit instantiation for uint32_t + double
template void coo_to_csr_inplace<uint32_t, uint32_t, double>(
  legacy::GraphCOOView<uint32_t, uint32_t, double>& graph,
  legacy::GraphCSRView<uint32_t, uint32_t, double>& result);

// Explicit instantiation for int + float
template void coo_to_csr_inplace<int32_t, int32_t, float>(
  legacy::GraphCOOView<int32_t, int32_t, float>& graph,
  legacy::GraphCSRView<int32_t, int32_t, float>& result);

// Explicit instantiation for int + double
template void coo_to_csr_inplace<int32_t, int32_t, double>(
  legacy::GraphCOOView<int32_t, int32_t, double>& graph,
  legacy::GraphCSRView<int32_t, int32_t, double>& result);

// Explicit instantiation for int64_t + float
template void coo_to_csr_inplace<int64_t, int64_t, float>(
  legacy::GraphCOOView<int64_t, int64_t, float>& graph,
  legacy::GraphCSRView<int64_t, int64_t, float>& result);

// Explicit instantiation for int64_t + double
template void coo_to_csr_inplace<int64_t, int64_t, double>(
  legacy::GraphCOOView<int64_t, int64_t, double>& graph,
  legacy::GraphCSRView<int64_t, int64_t, double>& result);

}  // namespace cugraph
