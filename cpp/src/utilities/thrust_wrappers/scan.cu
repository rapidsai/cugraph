/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/utilities/thrust_wrappers/scan.hpp.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

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

}  // namespace detail
}  // namespace cugraph
