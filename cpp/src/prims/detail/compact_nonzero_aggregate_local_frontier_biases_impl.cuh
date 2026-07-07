/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/prims/detail/compact_nonzero_aggregate_local_frontier_biases.cuh>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>

#include <raft/core/device_span.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include <limits>

namespace cugraph {
namespace detail {

template <typename edge_t, typename bias_t>
std::tuple<rmm::device_uvector<bias_t>, rmm::device_uvector<edge_t>, rmm::device_uvector<size_t>>
compact_nonzero_aggregate_local_frontier_biases(
  raft::handle_t const& handle,
  rmm::device_uvector<bias_t>&& aggregate_local_frontier_biases,
  rmm::device_uvector<size_t>&& aggregate_local_frontier_local_degree_offsets,
  size_t local_frontier_size,
  bool do_expensive_check,
  bool multi_gpu)
{
  if (do_expensive_check) {
    auto num_invalid_biases = thrust::count_if(
      handle.get_thrust_policy(),
      aggregate_local_frontier_biases.begin(),
      aggregate_local_frontier_biases.end(),
      check_out_of_range_t<bias_t>{bias_t{0.0}, std::numeric_limits<bias_t>::max()});
    if (multi_gpu) {
      num_invalid_biases = host_scalar_allreduce(
        handle.get_comms(), num_invalid_biases, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_biases == 0,
                    "invalid_input_argument: bias_e_op return values should be non-negative and "
                    "should not exceed std::numeirc_limits<bias_t>::max().");
  }

  rmm::device_uvector<size_t> aggregate_local_frontier_local_degrees(local_frontier_size,
                                                                     handle.get_stream());
  {
    thrust::adjacent_difference(handle.get_thrust_policy(),
                                aggregate_local_frontier_local_degree_offsets.begin() + 1,
                                aggregate_local_frontier_local_degree_offsets.end(),
                                aggregate_local_frontier_local_degrees.begin());

    auto pair_first = thrust::make_zip_iterator(aggregate_local_frontier_biases.begin(),
                                                thrust::make_counting_iterator(size_t{0}));
    thrust::for_each(handle.get_thrust_policy(),
                     pair_first,
                     pair_first + aggregate_local_frontier_biases.size(),
                     [offsets = raft::device_span<size_t const>(
                        aggregate_local_frontier_local_degree_offsets.data(),
                        aggregate_local_frontier_local_degree_offsets.size()),
                      degrees = raft::device_span<size_t>(
                        aggregate_local_frontier_local_degrees.data(),
                        aggregate_local_frontier_local_degrees.size())] __device__(auto pair) {
                       auto bias = cuda::std::get<0>(pair);
                       if (bias == 0.0) {
                         auto i   = cuda::std::get<1>(pair);
                         auto idx = cuda::std::distance(
                           offsets.begin() + 1,
                           thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i));
                         cuda::atomic_ref<size_t, cuda::thread_scope_device> degree(degrees[idx]);
                         degree.fetch_sub(size_t{1}, cuda::std::memory_order_relaxed);
                       }
                     });
  }

  auto num_nz_bias_nbrs = thrust::reduce(handle.get_thrust_policy(),
                                         aggregate_local_frontier_local_degrees.begin(),
                                         aggregate_local_frontier_local_degrees.end());

  rmm::device_uvector<edge_t> aggregate_local_frontier_nz_bias_indices(num_nz_bias_nbrs,
                                                                       handle.get_stream());
  {
    auto nz_biases  = rmm::device_uvector<bias_t>(num_nz_bias_nbrs, handle.get_stream());
    auto pair_first = thrust::make_zip_iterator(
      aggregate_local_frontier_biases.begin(),
      cuda::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<edge_t>(
          [offsets = raft::device_span<size_t const>(
             aggregate_local_frontier_local_degree_offsets.data(),
             aggregate_local_frontier_local_degree_offsets.size())] __device__(size_t i) {
            auto idx = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i));
            return static_cast<edge_t>(i - offsets[idx]);
          })));
    thrust::copy_if(handle.get_thrust_policy(),
                    pair_first,
                    pair_first + aggregate_local_frontier_biases.size(),
                    aggregate_local_frontier_biases.begin(),
                    thrust::make_zip_iterator(nz_biases.begin(),
                                              aggregate_local_frontier_nz_bias_indices.begin()),
                    cuda::proclaim_return_type<bool>([] __device__(bias_t b) { return b != 0.0; }));
    aggregate_local_frontier_biases = std::move(nz_biases);
  }

  cugraph::inclusive_scan(handle.get_thrust_policy(),
                          aggregate_local_frontier_local_degrees.begin(),
                          aggregate_local_frontier_local_degrees.end(),
                          aggregate_local_frontier_local_degree_offsets.begin() + 1);

  return std::make_tuple(std::move(aggregate_local_frontier_biases),
                         std::move(aggregate_local_frontier_nz_bias_indices),
                         std::move(aggregate_local_frontier_local_degree_offsets));
}

}  // namespace detail
}  // namespace cugraph
