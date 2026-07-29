/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/utilities/permute_wrappers.cuh.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/permute_wrappers.cuh>
#include <cugraph/utilities/thrust_wrappers/scatter.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

template <typename T>
void permute_in_place_impl(T* first,
                           std::size_t const* map_first,
                           std::size_t num_elements,
                           rmm::cuda_stream_view stream_view)
{
  auto const policy = rmm::exec_policy(stream_view);
  rmm::device_uvector<T> tmp(num_elements, stream_view);
  scatter_impl(policy, first, first + num_elements, map_first, tmp.data());
  thrust::copy(policy, tmp.begin(), tmp.end(), first);
}

#define CUGRAPH_PERMUTE_IN_PLACE_SCALAR_INST(ScalarType)          \
  template CUGRAPH_EXPORT void permute_in_place_impl<ScalarType>( \
    ScalarType * first,                                           \
    std::size_t const* map_first,                                 \
    std::size_t num_elements,                                     \
    rmm::cuda_stream_view stream_view)

CUGRAPH_PERMUTE_IN_PLACE_SCALAR_INST(std::int32_t);
CUGRAPH_PERMUTE_IN_PLACE_SCALAR_INST(std::int64_t);
CUGRAPH_PERMUTE_IN_PLACE_SCALAR_INST(float);
CUGRAPH_PERMUTE_IN_PLACE_SCALAR_INST(double);
CUGRAPH_PERMUTE_IN_PLACE_SCALAR_INST(std::size_t);

#undef CUGRAPH_PERMUTE_IN_PLACE_SCALAR_INST

}  // namespace detail
}  // namespace cugraph
