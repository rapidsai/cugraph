/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/utilities/thrust_wrappers/sequence.hpp.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

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

CUGRAPH_SEQUENCE_SCALAR_INST(std::size_t);
CUGRAPH_SEQUENCE_SCALAR_INST(std::int32_t);
CUGRAPH_SEQUENCE_SCALAR_INST(std::int64_t);

#undef CUGRAPH_SEQUENCE_SCALAR_INST

}  // namespace detail
}  // namespace cugraph
