/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cugraph/detail/collect_comm_wrapper.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

namespace cugraph {

namespace detail {

template <typename vertex_t>
rmm::device_uvector<vertex_t> permute_range(raft::handle_t const& handle,
                                            raft::random::RngState& rng_state,
                                            vertex_t local_range_start,
                                            vertex_t local_range_size,
                                            bool multi_gpu,
                                            bool do_expensive_check)
{
  if (do_expensive_check && multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto global_start =
      cugraph::host_scalar_bcast(handle.get_comms(), local_range_start, 0, handle.get_stream());
    auto sub_range_sizes =
      cugraph::host_scalar_allgather(handle.get_comms(), local_range_size, handle.get_stream());
    std::exclusive_scan(
      sub_range_sizes.begin(), sub_range_sizes.end(), sub_range_sizes.begin(), global_start);
    CUGRAPH_EXPECTS(
      sub_range_sizes[comm_rank] == local_range_start,
      "Invalid input arguments: a range must have contiguous and non-overlapping values");
  }
  rmm::device_uvector<vertex_t> permuted_integers(local_range_size, handle.get_stream());

  // generate as many integers as #local_range_size on each GPU
  detail::sequence_fill(
    handle.get_stream(), permuted_integers.begin(), permuted_integers.size(), local_range_start);

  if (multi_gpu) {
    // randomly distribute integers to all GPUs
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    std::vector<size_t> tx_value_counts(comm_size, 0);

    {
      rmm::device_uvector<vertex_t> d_target_ranks(permuted_integers.size(), handle.get_stream());

      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           d_target_ranks.data(),
                                           d_target_ranks.size(),
                                           vertex_t{0},
                                           vertex_t{comm_size},
                                           rng_state);

      thrust::sort_by_key(handle.get_thrust_policy(),
                          d_target_ranks.begin(),
                          d_target_ranks.end(),
                          permuted_integers.begin());

      rmm::device_uvector<vertex_t> d_reduced_ranks(comm_size, handle.get_stream());
      rmm::device_uvector<vertex_t> d_reduced_counts(comm_size, handle.get_stream());

      auto output_end = thrust::reduce_by_key(handle.get_thrust_policy(),
                                              d_target_ranks.begin(),
                                              d_target_ranks.end(),
                                              thrust::make_constant_iterator(1),
                                              d_reduced_ranks.begin(),
                                              d_reduced_counts.begin(),
                                              thrust::equal_to<int>());

      auto nr_output_pairs =
        static_cast<vertex_t>(cuda::std::distance(d_reduced_ranks.begin(), output_end.first));

      std::vector<vertex_t> h_reduced_ranks(comm_size);
      std::vector<vertex_t> h_reduced_counts(comm_size);

      raft::update_host(
        h_reduced_ranks.data(), d_reduced_ranks.data(), nr_output_pairs, handle.get_stream());

      raft::update_host(
        h_reduced_counts.data(), d_reduced_counts.data(), nr_output_pairs, handle.get_stream());

      for (int i = 0; i < static_cast<int>(nr_output_pairs); i++) {
        tx_value_counts[h_reduced_ranks[i]] = static_cast<size_t>(h_reduced_counts[i]);
      }
    }

    std::tie(permuted_integers, std::ignore) = cugraph::shuffle_values(
      handle.get_comms(),
      permuted_integers.begin(),
      raft::host_span<size_t const>(tx_value_counts.data(), tx_value_counts.size()),
      handle.get_stream());
  }

  // permute locally
  rmm::device_uvector<float> fractional_random_numbers(permuted_integers.size(),
                                                       handle.get_stream());

  cugraph::detail::uniform_random_fill(handle.get_stream(),
                                       fractional_random_numbers.data(),
                                       fractional_random_numbers.size(),
                                       float{0.0},
                                       float{1.0},
                                       rng_state);
  thrust::sort_by_key(handle.get_thrust_policy(),
                      fractional_random_numbers.begin(),
                      fractional_random_numbers.end(),
                      permuted_integers.begin());

  if (multi_gpu) {
    // take care of deficits and extras numbers
    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();

    size_t nr_extras{0};
    size_t nr_deficits{0};
    if (permuted_integers.size() > static_cast<size_t>(local_range_size)) {
      nr_extras = permuted_integers.size() - static_cast<size_t>(local_range_size);
    } else {
      nr_deficits = static_cast<size_t>(local_range_size) - permuted_integers.size();
    }

    auto extra_cluster_ids = cugraph::detail::device_allgatherv(
      handle,
      comm,
      raft::device_span<vertex_t const>(permuted_integers.data() + local_range_size,
                                        nr_extras > 0 ? nr_extras : 0));

    permuted_integers.resize(local_range_size, handle.get_stream());
    auto deficits =
      cugraph::host_scalar_allgather(handle.get_comms(), nr_deficits, handle.get_stream());

    std::exclusive_scan(deficits.begin(), deficits.end(), deficits.begin(), vertex_t{0});

    raft::copy(permuted_integers.data() + local_range_size - nr_deficits,
               extra_cluster_ids.begin() + deficits[comm_rank],
               nr_deficits,
               handle.get_stream());
  }

  assert(permuted_integers.size() == local_range_size);
  return permuted_integers;
}

}  // namespace detail
}  // namespace cugraph
