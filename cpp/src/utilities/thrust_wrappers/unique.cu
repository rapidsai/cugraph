/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/utilities/thrust_wrappers/unique.hpp.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers/unique.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/unique.h>

#include <cstdint>

namespace cugraph {
namespace detail {

template <typename RandomAccessIterator>
RandomAccessIterator unique_impl(rmm::exec_policy const& policy,
                                 RandomAccessIterator first,
                                 RandomAccessIterator last)
{
  return thrust::unique(policy, first, last);
}

template <typename RandomAccessIterator>
RandomAccessIterator unique_impl(rmm::exec_policy_nosync const& policy,
                                 RandomAccessIterator first,
                                 RandomAccessIterator last)
{
  return thrust::unique(policy, first, last);
}

template CUGRAPH_EXPORT int32_t* unique_impl<int32_t*>(rmm::exec_policy const& policy,
                                                       int32_t* first,
                                                       int32_t* last);
template CUGRAPH_EXPORT int32_t* unique_impl<int32_t*>(rmm::exec_policy_nosync const& policy,
                                                       int32_t* first,
                                                       int32_t* last);

template CUGRAPH_EXPORT uint32_t* unique_impl<uint32_t*>(rmm::exec_policy const& policy,
                                                         uint32_t* first,
                                                         uint32_t* last);
template CUGRAPH_EXPORT uint32_t* unique_impl<uint32_t*>(rmm::exec_policy_nosync const& policy,
                                                         uint32_t* first,
                                                         uint32_t* last);

template CUGRAPH_EXPORT int64_t* unique_impl<int64_t*>(rmm::exec_policy const& policy,
                                                       int64_t* first,
                                                       int64_t* last);
template CUGRAPH_EXPORT int64_t* unique_impl<int64_t*>(rmm::exec_policy_nosync const& policy,
                                                       int64_t* first,
                                                       int64_t* last);

#define CUGRAPH_UNIQUE_ZIP_INST(ZipType)                          \
  template CUGRAPH_EXPORT ZipType unique_impl<ZipType>(           \
    rmm::exec_policy const& policy, ZipType first, ZipType last); \
  template CUGRAPH_EXPORT ZipType unique_impl<ZipType>(           \
    rmm::exec_policy_nosync const& policy, ZipType first, ZipType last)

CUGRAPH_UNIQUE_ZIP_INST(zip_i32_i32);
CUGRAPH_UNIQUE_ZIP_INST(zip_i64_i64);
CUGRAPH_UNIQUE_ZIP_INST(zip_i64_i32);
CUGRAPH_UNIQUE_ZIP_INST(zip_i32_i64);
CUGRAPH_UNIQUE_ZIP_INST(zip_sz_i32);
CUGRAPH_UNIQUE_ZIP_INST(zip_sz_i64);
CUGRAPH_UNIQUE_ZIP_INST(zip_f_sz);
CUGRAPH_UNIQUE_ZIP_INST(zip_d_sz);

CUGRAPH_UNIQUE_ZIP_INST(zip_i32_i32_f);
CUGRAPH_UNIQUE_ZIP_INST(zip_i32_i32_d);
CUGRAPH_UNIQUE_ZIP_INST(zip_i64_i64_f);
CUGRAPH_UNIQUE_ZIP_INST(zip_i64_i64_d);
CUGRAPH_UNIQUE_ZIP_INST(zip_sz_i32_i32);
CUGRAPH_UNIQUE_ZIP_INST(zip_sz_i64_i64);
CUGRAPH_UNIQUE_ZIP_INST(zip_i32_i32_sz);
CUGRAPH_UNIQUE_ZIP_INST(zip_i64_i64_sz);
CUGRAPH_UNIQUE_ZIP_INST(zip_i32_i32_i32);
CUGRAPH_UNIQUE_ZIP_INST(zip_i32_i64_i32);
CUGRAPH_UNIQUE_ZIP_INST(zip_i64_i32_i32);
CUGRAPH_UNIQUE_ZIP_INST(zip_i64_i64_i32);

CUGRAPH_UNIQUE_ZIP_INST(zip_i32_i32_sz_i);
CUGRAPH_UNIQUE_ZIP_INST(zip_i64_i64_sz_i);
CUGRAPH_UNIQUE_ZIP_INST(zip_i32_i32_i32_sz);
CUGRAPH_UNIQUE_ZIP_INST(zip_i64_i64_i64_sz);

#undef CUGRAPH_UNIQUE_ZIP_INST

}  // namespace detail
}  // namespace cugraph
