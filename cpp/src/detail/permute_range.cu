/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

namespace cugraph {

namespace detail {

template <typename vertex_t>
rmm::device_uvector<vertex_t> permute_range(raft::handle_t const& handle,
                                            raft::random::RngState& rng_state,
                                            vertex_t local_range_size,
                                            vertex_t local_range_start,
                                            bool multi_gpu)
{
  if (multi_gpu) {
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
      "Invalid input arguments: a rage must have contiguous and non-overlapping values");
  }
  rmm::device_uvector<vertex_t> permuted_intergers(local_range_size, handle.get_stream());

  // generate as many number as #local_vertices on each GPU
  detail::sequence_fill(
    handle.get_stream(), permuted_intergers.begin(), permuted_intergers.size(), local_range_start);

  // shuffle/permute locally
  rmm::device_uvector<float> fractional_random_numbers(permuted_intergers.size(),
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
                      permuted_intergers.begin());

  if (multi_gpu) {
    // distribute shuffled/permuted numbers to other GPUs
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    std::vector<size_t> tx_value_counts(comm_size);
    std::fill(tx_value_counts.begin(), tx_value_counts.end(), 0);

    {
      rmm::device_uvector<vertex_t> d_target_ranks(permuted_intergers.size(), handle.get_stream());

      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           d_target_ranks.data(),
                                           d_target_ranks.size(),
                                           vertex_t{0},
                                           vertex_t{comm_size},
                                           rng_state);

      thrust::sort_by_key(handle.get_thrust_policy(),
                          d_target_ranks.begin(),
                          d_target_ranks.end(),
                          permuted_intergers.begin());

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
        static_cast<vertex_t>(thrust::distance(d_reduced_ranks.begin(), output_end.first));

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

    std::tie(permuted_intergers, std::ignore) = cugraph::shuffle_values(
      handle.get_comms(), permuted_intergers.begin(), tx_value_counts, handle.get_stream());

    // shuffle/permute locally again
    fractional_random_numbers.resize(permuted_intergers.size(), handle.get_stream());

    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         fractional_random_numbers.data(),
                                         fractional_random_numbers.size(),
                                         float{0.0},
                                         float{1.0},
                                         rng_state);
    thrust::sort_by_key(handle.get_thrust_policy(),
                        fractional_random_numbers.begin(),
                        fractional_random_numbers.end(),
                        permuted_intergers.begin());

    // take care of deficits and extras numbers

    int nr_extras =
      static_cast<int>(permuted_intergers.size()) - static_cast<int>(local_range_size);
    int nr_deficits = nr_extras >= 0 ? 0 : -nr_extras;

    auto extra_cluster_ids = cugraph::detail::device_allgatherv(
      handle,
      comm,
      raft::device_span<vertex_t const>(permuted_intergers.data() + local_range_size,
                                        nr_extras > 0 ? nr_extras : 0));

    permuted_intergers.resize(local_range_size, handle.get_stream());
    auto deficits =
      cugraph::host_scalar_allgather(handle.get_comms(), nr_deficits, handle.get_stream());

    std::exclusive_scan(deficits.begin(), deficits.end(), deficits.begin(), vertex_t{0});

    raft::copy(permuted_intergers.data() + local_range_size - nr_deficits,
               extra_cluster_ids.begin() + deficits[comm_rank],
               nr_deficits,
               handle.get_stream());
  }

  assert(permuted_intergers.size() == local_range_size);
  return permuted_intergers;
}

template rmm::device_uvector<int32_t> permute_range(raft::handle_t const& handle,
                                                    raft::random::RngState& rng_state,
                                                    int32_t local_range_size,
                                                    int32_t local_range_start,
                                                    bool multi_gpu);

template rmm::device_uvector<int64_t> permute_range(raft::handle_t const& handle,
                                                    raft::random::RngState& rng_state,
                                                    int64_t local_range_size,
                                                    int64_t local_range_start,
                                                    bool multi_gpu);

}  // namespace detail
}  // namespace cugraph
