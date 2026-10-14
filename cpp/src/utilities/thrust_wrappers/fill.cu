/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/utilities/thrust_wrappers/fill.hpp.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

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

#define CUGRAPH_FILL_SCALAR_INST(ScalarType)                                                       \
  template CUGRAPH_EXPORT void fill_impl<ScalarType*>(                                             \
    rmm::exec_policy const& policy, ScalarType* first, ScalarType* last, ScalarType const& value); \
  template CUGRAPH_EXPORT void fill_impl<ScalarType*>(rmm::exec_policy_nosync const& policy,       \
                                                      ScalarType* first,                           \
                                                      ScalarType* last,                            \
                                                      ScalarType const& value)

CUGRAPH_FILL_SCALAR_INST(std::size_t);
CUGRAPH_FILL_SCALAR_INST(std::uint32_t);
CUGRAPH_FILL_SCALAR_INST(std::int32_t);
CUGRAPH_FILL_SCALAR_INST(std::int64_t);
CUGRAPH_FILL_SCALAR_INST(float);
CUGRAPH_FILL_SCALAR_INST(double);

#undef CUGRAPH_FILL_SCALAR_INST

}  // namespace detail
}  // namespace cugraph
