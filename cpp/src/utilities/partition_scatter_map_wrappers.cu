/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/utilities/partition_scatter_map_wrappers.cuh.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/partition_scatter_map_wrappers.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/tuple>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <cstddef>
#include <cstdint>

namespace cugraph {
namespace detail {

template <typename gid_offset_t, typename offset_t>
rmm::device_uvector<size_t> compute_partition_scatter_map_impl(
  gid_offset_t const* group_id_offsets,
  offset_t const* intra_partition_displs,
  size_t const* group_displacements,
  size_t num_elements,
  rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<size_t> scatter_map(num_elements, stream_view);
  thrust::transform(
    rmm::exec_policy(stream_view),
    thrust::make_zip_iterator(group_id_offsets, intra_partition_displs),
    thrust::make_zip_iterator(group_id_offsets + num_elements,
                              intra_partition_displs + num_elements),
    scatter_map.begin(),
    cuda::proclaim_return_type<size_t>(
      [group_displacements] __device__(cuda::std::tuple<gid_offset_t, offset_t> pair) {
        return group_displacements[cuda::std::get<0>(pair)] +
               static_cast<size_t>(cuda::std::get<1>(pair));
      }));
  return scatter_map;
}

#define CUGRAPH_PARTITION_SCATTER_MAP_INST(gid_t, offset_t)                                   \
  template CUGRAPH_EXPORT rmm::device_uvector<size_t>                                         \
  compute_partition_scatter_map_impl<gid_t, offset_t>(gid_t const* group_id_offsets,          \
                                                      offset_t const* intra_partition_displs, \
                                                      size_t const* group_displacements,      \
                                                      size_t num_elements,                    \
                                                      rmm::cuda_stream_view stream_view)

CUGRAPH_PARTITION_SCATTER_MAP_INST(std::uint8_t, std::uint32_t);
CUGRAPH_PARTITION_SCATTER_MAP_INST(int, std::size_t);

#undef CUGRAPH_PARTITION_SCATTER_MAP_INST

}  // namespace detail
}  // namespace cugraph
