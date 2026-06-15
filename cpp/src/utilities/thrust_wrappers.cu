/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/utilities/thrust_wrappers.hpp.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

template <typename RandomAccessIterator>
void sort_impl(rmm::exec_policy const& policy,
               RandomAccessIterator first,
               RandomAccessIterator last)
{
  thrust::sort(policy, first, last);
}

template <typename RandomAccessIterator>
void sort_impl(rmm::exec_policy_nosync const& policy,
               RandomAccessIterator first,
               RandomAccessIterator last)
{
  thrust::sort(policy, first, last);
}

template CUGRAPH_EXPORT void sort_impl<int32_t*>(rmm::exec_policy const& policy,
                                                 int32_t* first,
                                                 int32_t* last);
template CUGRAPH_EXPORT void sort_impl<int32_t*>(rmm::exec_policy_nosync const& policy,
                                                 int32_t* first,
                                                 int32_t* last);

template CUGRAPH_EXPORT void sort_impl<uint32_t*>(rmm::exec_policy const& policy,
                                                  uint32_t* first,
                                                  uint32_t* last);
template CUGRAPH_EXPORT void sort_impl<uint32_t*>(rmm::exec_policy_nosync const& policy,
                                                  uint32_t* first,
                                                  uint32_t* last);

template CUGRAPH_EXPORT void sort_impl<int64_t*>(rmm::exec_policy const& policy,
                                                 int64_t* first,
                                                 int64_t* last);
template CUGRAPH_EXPORT void sort_impl<int64_t*>(rmm::exec_policy_nosync const& policy,
                                                 int64_t* first,
                                                 int64_t* last);

#define CUGRAPH_SORT_ZIP_INST(ZipType)                            \
  template CUGRAPH_EXPORT void sort_impl<ZipType>(                \
    rmm::exec_policy const& policy, ZipType first, ZipType last); \
  template CUGRAPH_EXPORT void sort_impl<ZipType>(                \
    rmm::exec_policy_nosync const& policy, ZipType first, ZipType last)

CUGRAPH_SORT_ZIP_INST(zip_i32_i32);
CUGRAPH_SORT_ZIP_INST(zip_i64_i64);
CUGRAPH_SORT_ZIP_INST(zip_i64_i32);
CUGRAPH_SORT_ZIP_INST(zip_i32_i64);
CUGRAPH_SORT_ZIP_INST(zip_sz_i32);
CUGRAPH_SORT_ZIP_INST(zip_sz_i64);
CUGRAPH_SORT_ZIP_INST(zip_f_sz);
CUGRAPH_SORT_ZIP_INST(zip_d_sz);

CUGRAPH_SORT_ZIP_INST(zip_i32_i32_f);
CUGRAPH_SORT_ZIP_INST(zip_i32_i32_d);
CUGRAPH_SORT_ZIP_INST(zip_i64_i64_f);
CUGRAPH_SORT_ZIP_INST(zip_i64_i64_d);
CUGRAPH_SORT_ZIP_INST(zip_sz_i32_i32);
CUGRAPH_SORT_ZIP_INST(zip_sz_i64_i64);
CUGRAPH_SORT_ZIP_INST(zip_i32_i32_sz);
CUGRAPH_SORT_ZIP_INST(zip_i64_i64_sz);
CUGRAPH_SORT_ZIP_INST(zip_i32_i32_i32);
CUGRAPH_SORT_ZIP_INST(zip_i32_i64_i32);
CUGRAPH_SORT_ZIP_INST(zip_i64_i32_i32);
CUGRAPH_SORT_ZIP_INST(zip_i64_i64_i32);

CUGRAPH_SORT_ZIP_INST(zip_i32_i32_sz_i);
CUGRAPH_SORT_ZIP_INST(zip_i64_i64_sz_i);
CUGRAPH_SORT_ZIP_INST(zip_i32_i32_i32_sz);
CUGRAPH_SORT_ZIP_INST(zip_i64_i64_i64_sz);

#undef CUGRAPH_SORT_ZIP_INST

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

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan_impl(rmm::exec_policy const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result)
{
  return thrust::inclusive_scan(policy, first, last, result);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result)
{
  return thrust::inclusive_scan(policy, first, last, result);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan_impl(rmm::exec_policy const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result)
{
  return thrust::exclusive_scan(policy, first, last, result);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result)
{
  return thrust::exclusive_scan(policy, first, last, result);
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan_impl(rmm::exec_policy const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result,
                                   T init)
{
  return thrust::exclusive_scan(policy, first, last, result, init);
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result,
                                   T init)
{
  return thrust::exclusive_scan(policy, first, last, result, init);
}

template CUGRAPH_EXPORT std::size_t* inclusive_scan_impl(rmm::exec_policy const& policy,
                                                         std::size_t* first,
                                                         std::size_t* last,
                                                         std::size_t* result);
template CUGRAPH_EXPORT std::size_t* inclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                                         std::size_t* first,
                                                         std::size_t* last,
                                                         std::size_t* result);
template CUGRAPH_EXPORT std::int32_t* inclusive_scan_impl(rmm::exec_policy const& policy,
                                                          std::int32_t* first,
                                                          std::int32_t* last,
                                                          std::int32_t* result);
template CUGRAPH_EXPORT std::int32_t* inclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                                          std::int32_t* first,
                                                          std::int32_t* last,
                                                          std::int32_t* result);
template CUGRAPH_EXPORT std::int64_t* inclusive_scan_impl(rmm::exec_policy const& policy,
                                                          std::int64_t* first,
                                                          std::int64_t* last,
                                                          std::int64_t* result);
template CUGRAPH_EXPORT std::int64_t* inclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                                          std::int64_t* first,
                                                          std::int64_t* last,
                                                          std::int64_t* result);

template CUGRAPH_EXPORT std::size_t* exclusive_scan_impl(rmm::exec_policy const& policy,
                                                         std::size_t* first,
                                                         std::size_t* last,
                                                         std::size_t* result);
template CUGRAPH_EXPORT std::size_t* exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                                         std::size_t* first,
                                                         std::size_t* last,
                                                         std::size_t* result);
template CUGRAPH_EXPORT std::size_t* exclusive_scan_impl(rmm::exec_policy const& policy,
                                                         std::size_t* first,
                                                         std::size_t* last,
                                                         std::size_t* result,
                                                         std::size_t init);
template CUGRAPH_EXPORT std::size_t* exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                                         std::size_t* first,
                                                         std::size_t* last,
                                                         std::size_t* result,
                                                         std::size_t init);

template CUGRAPH_EXPORT std::int32_t* exclusive_scan_impl(rmm::exec_policy const& policy,
                                                          std::int32_t* first,
                                                          std::int32_t* last,
                                                          std::int32_t* result);
template CUGRAPH_EXPORT std::int32_t* exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                                          std::int32_t* first,
                                                          std::int32_t* last,
                                                          std::int32_t* result);
template CUGRAPH_EXPORT std::int32_t* exclusive_scan_impl(rmm::exec_policy const& policy,
                                                          std::int32_t* first,
                                                          std::int32_t* last,
                                                          std::int32_t* result,
                                                          std::int32_t init);
template CUGRAPH_EXPORT std::int32_t* exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                                          std::int32_t* first,
                                                          std::int32_t* last,
                                                          std::int32_t* result,
                                                          std::int32_t init);

template CUGRAPH_EXPORT std::int64_t* exclusive_scan_impl(rmm::exec_policy const& policy,
                                                          std::int64_t* first,
                                                          std::int64_t* last,
                                                          std::int64_t* result);
template CUGRAPH_EXPORT std::int64_t* exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                                          std::int64_t* first,
                                                          std::int64_t* last,
                                                          std::int64_t* result);
template CUGRAPH_EXPORT std::int64_t* exclusive_scan_impl(rmm::exec_policy const& policy,
                                                          std::int64_t* first,
                                                          std::int64_t* last,
                                                          std::int64_t* result,
                                                          std::int64_t init);
template CUGRAPH_EXPORT std::int64_t* exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                                          std::int64_t* first,
                                                          std::int64_t* last,
                                                          std::int64_t* result,
                                                          std::int64_t init);

template <typename ForwardIterator, typename T>
void fill_impl(rmm::exec_policy const& policy,
               ForwardIterator first,
               ForwardIterator last,
               T const& value)
{
  thrust::fill(policy, first, last, value);
}

template <typename ForwardIterator, typename T>
void fill_impl(rmm::exec_policy_nosync const& policy,
               ForwardIterator first,
               ForwardIterator last,
               T const& value)
{
  thrust::fill(policy, first, last, value);
}

template <typename ForwardIterator, typename T>
void sequence_impl(
  rmm::exec_policy const& policy, ForwardIterator first, ForwardIterator last, T init, T step)
{
  thrust::sequence(policy, first, last, init, step);
}

template <typename ForwardIterator, typename T>
void sequence_impl(rmm::exec_policy_nosync const& policy,
                   ForwardIterator first,
                   ForwardIterator last,
                   T init,
                   T step)
{
  thrust::sequence(policy, first, last, init, step);
}

#define CUGRAPH_FILL_SCALAR_INST(ScalarType)                                                       \
  template CUGRAPH_EXPORT void fill_impl<ScalarType*>(                                             \
    rmm::exec_policy const& policy, ScalarType* first, ScalarType* last, ScalarType const& value); \
  template CUGRAPH_EXPORT void fill_impl<ScalarType*>(rmm::exec_policy_nosync const& policy,       \
                                                      ScalarType* first,                           \
                                                      ScalarType* last,                            \
                                                      ScalarType const& value)

#define CUGRAPH_SEQUENCE_SCALAR_INST(ScalarType)                       \
  template CUGRAPH_EXPORT void sequence_impl<ScalarType*, ScalarType>( \
    rmm::exec_policy const& policy,                                    \
    ScalarType* first,                                                 \
    ScalarType* last,                                                  \
    ScalarType init,                                                   \
    ScalarType step);                                                  \
  template CUGRAPH_EXPORT void sequence_impl<ScalarType*, ScalarType>( \
    rmm::exec_policy_nosync const& policy,                             \
    ScalarType* first,                                                 \
    ScalarType* last,                                                  \
    ScalarType init,                                                   \
    ScalarType step)

CUGRAPH_FILL_SCALAR_INST(std::size_t);
CUGRAPH_FILL_SCALAR_INST(std::uint32_t);
CUGRAPH_FILL_SCALAR_INST(std::int32_t);
CUGRAPH_FILL_SCALAR_INST(std::int64_t);
CUGRAPH_FILL_SCALAR_INST(float);
CUGRAPH_FILL_SCALAR_INST(double);

CUGRAPH_SEQUENCE_SCALAR_INST(std::size_t);
CUGRAPH_SEQUENCE_SCALAR_INST(std::int32_t);
CUGRAPH_SEQUENCE_SCALAR_INST(std::int64_t);

#undef CUGRAPH_FILL_SCALAR_INST
#undef CUGRAPH_SEQUENCE_SCALAR_INST

}  // namespace detail

}  // namespace cugraph
