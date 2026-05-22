/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations of cugraph::detail::device_sort_impl for thrust::zip_iterator ranges used
 * across libcugraph (lexicographic multi-column sorts). Iterator types match
 * utility_wrappers_device_sort_zip_types.hpp (see @ref device_sort_supported_v).
 */
#include "detail/utility_wrappers_impl.cuh"

#include <cugraph/detail/utility_wrappers_device_sort_zip_types.hpp>

namespace cugraph {
namespace detail {

#define CUGRAPH_DEVICE_SORT_ZIP_INST(ZipType)                     \
  template void device_sort_impl<ZipType>(                        \
    rmm::exec_policy const& policy, ZipType first, ZipType last); \
  template void device_sort_impl<ZipType>(                        \
    rmm::exec_policy_nosync const& policy, ZipType first, ZipType last)

CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i32_i32);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i64_i64);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i64_i32);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i32_i64);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_sz_i32);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_sz_i64);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_f_sz);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_d_sz);

CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i32_i32_f);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i32_i32_d);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i64_i64_f);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i64_i64_d);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_sz_i32_i32);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_sz_i64_i64);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i32_i32_sz);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i64_i64_sz);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i32_i32_i32);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i32_i64_i32);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i64_i32_i32);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i64_i64_i32);

CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i32_i32_sz_i);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i64_i64_sz_i);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i32_i32_i32_sz);
CUGRAPH_DEVICE_SORT_ZIP_INST(zip_i64_i64_i64_sz);

#undef CUGRAPH_DEVICE_SORT_ZIP_INST

}  // namespace detail
}  // namespace cugraph
