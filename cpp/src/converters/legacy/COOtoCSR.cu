/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "COOtoCSR.cuh"

#include <rmm/resource_ref.hpp>

namespace cugraph {

// Explicit instantiation for uint32_t + float
template std::unique_ptr<legacy::GraphCSR<uint32_t, uint32_t, float>>
coo_to_csr<uint32_t, uint32_t, float>(legacy::GraphCOOView<uint32_t, uint32_t, float> const& graph,
                                      rmm::device_async_resource_ref);

// Explicit instantiation for uint32_t + double
template std::unique_ptr<legacy::GraphCSR<uint32_t, uint32_t, double>>
coo_to_csr<uint32_t, uint32_t, double>(
  legacy::GraphCOOView<uint32_t, uint32_t, double> const& graph, rmm::device_async_resource_ref);

// Explicit instantiation for int + float
template std::unique_ptr<legacy::GraphCSR<int32_t, int32_t, float>>
coo_to_csr<int32_t, int32_t, float>(legacy::GraphCOOView<int32_t, int32_t, float> const& graph,
                                    rmm::device_async_resource_ref);

// Explicit instantiation for int + double
template std::unique_ptr<legacy::GraphCSR<int32_t, int32_t, double>>
coo_to_csr<int32_t, int32_t, double>(legacy::GraphCOOView<int32_t, int32_t, double> const& graph,
                                     rmm::device_async_resource_ref);

// Explicit instantiation for int64_t + float
template std::unique_ptr<legacy::GraphCSR<int64_t, int64_t, float>>
coo_to_csr<int64_t, int64_t, float>(legacy::GraphCOOView<int64_t, int64_t, float> const& graph,
                                    rmm::device_async_resource_ref);

// Explicit instantiation for int64_t + double
template std::unique_ptr<legacy::GraphCSR<int64_t, int64_t, double>>
coo_to_csr<int64_t, int64_t, double>(legacy::GraphCOOView<int64_t, int64_t, double> const& graph,
                                     rmm::device_async_resource_ref);

}  // namespace cugraph
