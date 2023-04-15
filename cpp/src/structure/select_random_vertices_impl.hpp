/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <detail/graph_partition_utils.cuh>

#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#ifndef NO_CUGRAPH_OPS
#include <cugraph-ops/graph/sampling.hpp>
#endif

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cstdlib>
#include <iostream>
namespace cugraph {

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<vertex_t> select_random_vertices(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<rmm::device_uvector<vertex_t>>&& given_vertices,
  raft::random::RngState& rng_state,
  size_t select_count,
  bool with_replacement,
  bool sort_vertices)
{
  if (given_vertices) {
    bool is_any_v_out_of_range =
      thrust::any_of(handle.get_thrust_policy(),
                     (*given_vertices).begin(),
                     (*given_vertices).end(),
                     [v_first = graph_view.local_vertex_partition_range_first(),
                      v_last  = graph_view.local_vertex_partition_range_last()] __device__(auto v) {
                       return !((v >= v_first) && (v < v_last));
                     });

    CUGRAPH_EXPECTS(!is_any_v_out_of_range, "All given vertices must be within range [0, V)");

    size_t nr_given_vertices = static_cast<size_t>((*given_vertices).size());
    if (multi_gpu) {
      nr_given_vertices = host_scalar_allreduce(
        handle.get_comms(), nr_given_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }

    CUGRAPH_EXPECTS(
      with_replacement || select_count <= nr_given_vertices,
      "Invalid input arguments: select_count should not exceed the number of given vertices if "
      "with_replacement == false.");

  } else {
    CUGRAPH_EXPECTS(
      with_replacement || select_count <= static_cast<size_t>(graph_view.number_of_vertices()),
      "Invalid input arguments: select_count should not exceed the number of vertices if "
      "with_replacement == false.");
  }

  rmm::device_uvector<vertex_t> mg_sample_buffer(0, handle.get_stream());

  size_t this_gpu_select_count{0};
  if constexpr (multi_gpu) {
    auto const comm_rank = handle.get_comms().get_rank();
    auto const comm_size = handle.get_comms().get_size();

    this_gpu_select_count =
      select_count / static_cast<size_t>(comm_size) +
      (static_cast<size_t>(comm_rank) < static_cast<size_t>(select_count % comm_size) ? size_t{1}
                                                                                      : size_t{0});
  } else {
    this_gpu_select_count = select_count;
  }

  if (with_replacement) {
    // FIXME: need to double check uniform_random_fill generates random numbers in [0, V) (not [0,
    // V])
    mg_sample_buffer.resize(this_gpu_select_count, handle.get_stream());

    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         mg_sample_buffer.data(),
                                         mg_sample_buffer.size(),
                                         vertex_t{0},
                                         given_vertices
                                           ? static_cast<vertex_t>((*given_vertices).size())
                                           : graph_view.number_of_vertices(),
                                         rng_state);

    if (given_vertices) {
      thrust::gather(handle.get_thrust_policy(),
                     mg_sample_buffer.begin(),
                     mg_sample_buffer.end(),
                     (*given_vertices).begin(),
                     mg_sample_buffer.begin());
    }

  } else {
    size_t mg_buffer_size = 0;
    vertex_t start_idx;

    if (given_vertices) {
      start_idx = vertex_t{0};
      if constexpr (multi_gpu) {
        auto const comm_rank = handle.get_comms().get_rank();
        auto const comm_size = handle.get_comms().get_size();

        auto recvcounts =
          host_scalar_allgather(handle.get_comms(), (*given_vertices).size(), handle.get_stream());
        std::vector<size_t> displacements(recvcounts.size(), size_t{0});
        std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);

        start_idx = displacements[comm_rank];
      }
      mg_buffer_size = (*given_vertices).size();
    } else {
      auto local_vertex_partition_range_first = graph_view.local_vertex_partition_range_first();
      auto local_vertex_partition_range_last  = graph_view.local_vertex_partition_range_last();

      mg_buffer_size = local_vertex_partition_range_last - local_vertex_partition_range_first;
      start_idx      = local_vertex_partition_range_first;
    }

    mg_sample_buffer = rmm::device_uvector<vertex_t>(mg_buffer_size, handle.get_stream());
    thrust::sequence(
      handle.get_thrust_policy(), mg_sample_buffer.begin(), mg_sample_buffer.end(), start_idx);

    {  // random shuffle (use this instead of thrust::shuffle to use raft::random::RngState)
      rmm::device_uvector<float> random_numbers(mg_sample_buffer.size(), handle.get_stream());
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           random_numbers.data(),
                                           random_numbers.size(),
                                           float{0.0},
                                           float{1.0},
                                           rng_state);
      thrust::sort_by_key(handle.get_thrust_policy(),
                          random_numbers.begin(),
                          random_numbers.end(),
                          mg_sample_buffer.begin());
    }

    if constexpr (multi_gpu) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      std::vector<size_t> tx_value_counts(comm_size);
      for (int i = 0; i < comm_size; ++i) {
        tx_value_counts[i] = mg_sample_buffer.size() / comm_size;
      }

      srand((unsigned)time(NULL));
      for (int i = 0; i < static_cast<int>(mg_sample_buffer.size() % comm_size); i++) {
        tx_value_counts[rand() % comm_size]++;
      }

      std::tie(mg_sample_buffer, std::ignore) = cugraph::shuffle_values(
        handle.get_comms(), mg_sample_buffer.begin(), tx_value_counts, handle.get_stream());

      {  // random shuffle (use this instead of thrust::shuffle to use raft::random::RngState)
        rmm::device_uvector<float> random_numbers(mg_sample_buffer.size(), handle.get_stream());
        cugraph::detail::uniform_random_fill(handle.get_stream(),
                                             random_numbers.data(),
                                             random_numbers.size(),
                                             float{0.0},
                                             float{1.0},
                                             rng_state);
        thrust::sort_by_key(handle.get_thrust_policy(),
                            random_numbers.begin(),
                            random_numbers.end(),
                            mg_sample_buffer.begin());
      }

      auto buffer_sizes = cugraph::host_scalar_allgather(
        handle.get_comms(), mg_sample_buffer.size(), handle.get_stream());
      auto min_buffer_size = *std::min_element(buffer_sizes.begin(), buffer_sizes.end());
      if (min_buffer_size <= select_count / comm_size) {
        auto new_sizes    = std::vector<size_t>(comm_size, min_buffer_size);
        auto num_deficits = select_count - min_buffer_size * comm_size;
        for (int i = 0; i < comm_size; ++i) {
          auto delta = std::min(num_deficits, buffer_sizes[i] - min_buffer_size);
          new_sizes[i] += delta;
          num_deficits -= delta;
        }
        this_gpu_select_count = new_sizes[comm_rank];
      }
    }

    mg_sample_buffer.resize(this_gpu_select_count, handle.get_stream());
    mg_sample_buffer.shrink_to_fit(handle.get_stream());
  }

  if constexpr (multi_gpu) {
    auto vertex_partition_range_lasts =
      given_vertices ? cugraph::partition_manager::compute_vertex_partition_range_lasts(
                         handle, static_cast<vertex_t>((*given_vertices).size()))
                     : graph_view.vertex_partition_range_lasts();

    mg_sample_buffer = cugraph::detail::shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
      handle, std::move(mg_sample_buffer), vertex_partition_range_lasts);

    if (given_vertices) {
      auto const comm_rank = handle.get_comms().get_rank();
      auto const comm_size = handle.get_comms().get_size();

      auto start_idx = comm_rank ? vertex_partition_range_lasts[comm_rank - 1] : 0;

      thrust::transform(handle.get_thrust_policy(),
                        mg_sample_buffer.begin(),
                        mg_sample_buffer.end(),
                        mg_sample_buffer.begin(),
                        [start_idx = start_idx] __device__(auto idx) { return idx - start_idx; });
    }
  }

  if (given_vertices) {
    thrust::gather(handle.get_thrust_policy(),
                   mg_sample_buffer.begin(),
                   mg_sample_buffer.end(),
                   (*given_vertices).begin(),
                   mg_sample_buffer.begin());
  }

  if (sort_vertices) {
    thrust::sort(handle.get_thrust_policy(), mg_sample_buffer.begin(), mg_sample_buffer.end());
  }

  return mg_sample_buffer;
}

}  // namespace cugraph
