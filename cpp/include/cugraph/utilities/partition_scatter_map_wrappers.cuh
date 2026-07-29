/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cugraph {
namespace detail {

template <typename gid_offset_t, typename offset_t>
inline constexpr bool compute_partition_scatter_map_supported_v =
  (std::is_same_v<gid_offset_t, std::uint8_t> && std::is_same_v<offset_t, std::uint32_t>) ||
  (std::is_same_v<gid_offset_t, int> && std::is_same_v<offset_t, std::size_t>);

template <typename gid_offset_t, typename offset_t>
CUGRAPH_EXPORT rmm::device_uvector<size_t> compute_partition_scatter_map_impl(
  gid_offset_t const* group_id_offsets,
  offset_t const* intra_partition_displs,
  size_t const* group_displacements,
  size_t num_elements,
  rmm::cuda_stream_view stream_view);

template <typename gid_offset_t, typename offset_t>
rmm::device_uvector<size_t> compute_partition_scatter_map(
  rmm::device_uvector<gid_offset_t> const& group_id_offsets,
  rmm::device_uvector<offset_t> const& intra_partition_displs,
  rmm::device_uvector<size_t> const& group_displacements,
  rmm::cuda_stream_view stream_view)
{
  static_assert(compute_partition_scatter_map_supported_v<gid_offset_t, offset_t>,
                "compute_partition_scatter_map is not explicitly instantiated for this "
                "(gid_offset_t, offset_t) pair.");

  return compute_partition_scatter_map_impl(group_id_offsets.data(),
                                            intra_partition_displs.data(),
                                            group_displacements.data(),
                                            group_id_offsets.size(),
                                            stream_view);
}

}  // namespace detail
}  // namespace cugraph
