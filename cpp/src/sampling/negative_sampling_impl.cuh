/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cugraph/sampling_functions.hpp>

#include <thrust/remove.h>
#include <thrust/unique.h>

namespace cugraph {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> negative_sampling(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  size_t num_samples,
  std::optional<raft::device_span<weight_t const>> src_bias,
  std::optional<raft::device_span<weight_t const>> dst_bias,
  bool remove_duplicates,
  bool remove_false_negatives,
  bool exact_number_of_samples,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst(0, handle.get_stream());

  // Optimistically assume we can do this in one pass
  size_t samples_in_this_batch = num_samples;

  while (samples_in_this_batch > 0) {
    if constexpr (multi_gpu) {
      size_t num_gpus = handle.get_comms().get_size();
      size_t rank     = handle.get_comms().get_rank();

      samples_in_this_batch =
        (samples_in_this_batch / num_gpus) + (rank < (samples_in_this_batch % num_gpus) ? 1 : 0);
    }

    rmm::device_uvector<vertex_t> batch_src(samples_in_this_batch, handle.get_stream());
    rmm::device_uvector<vertex_t> batch_dst(samples_in_this_batch, handle.get_stream());

    if (src_bias) {
      detail::biased_random_fill(handle,
                                 rng_state,
                                 raft::device_span<vertex_t>{batch_src.data(), batch_src.size()},
                                 *src_bias);
    } else {
      detail::uniform_random_fill(handle.get_stream(),
                                  batch_src.data(),
                                  batch_src.size(),
                                  vertex_t{0},
                                  graph_view.number_of_vertices(),
                                  rng_state);
    }

    if (dst_bias) {
      detail::biased_random_fill(handle,
                                 rng_state,
                                 raft::device_span<vertex_t>{batch_dst.data(), batch_dst.size()},
                                 *dst_bias);
    } else {
      detail::uniform_random_fill(handle.get_stream(),
                                  batch_dst.data(),
                                  batch_dst.size(),
                                  vertex_t{0},
                                  graph_view.number_of_vertices(),
                                  rng_state);
    }

    if constexpr (multi_gpu) {
      auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();

      std::tie(batch_src, batch_dst, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       float,
                                                                                       int>(
          handle,
          std::move(batch_src),
          std::move(batch_dst),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          vertex_partition_range_lasts);
    }

    if (remove_false_negatives) {
      auto has_edge_flags =
        graph_view.has_edge(handle,
                            raft::device_span<vertex_t const>{batch_src.data(), batch_src.size()},
                            raft::device_span<vertex_t const>{batch_dst.data(), batch_dst.size()},
                            do_expensive_check);

      auto begin_iter =
        thrust::make_zip_iterator(batch_src.begin(), batch_dst.begin(), has_edge_flags.begin());
      auto new_end = thrust::remove_if(handle.get_thrust_policy(),
                                       begin_iter,
                                       begin_iter + batch_src.size(),
                                       [] __device__(auto tuple) { return thrust::get<2>(tuple); });
      batch_src.resize(thrust::distance(begin_iter, new_end), handle.get_stream());
      batch_dst.resize(thrust::distance(begin_iter, new_end), handle.get_stream());
    }

    if (remove_duplicates) {
      auto begin_iter = thrust::make_zip_iterator(batch_src.begin(), batch_dst.begin());
      thrust::sort(handle.get_thrust_policy(), begin_iter, begin_iter + batch_src.size());

      auto new_end =
        thrust::unique(handle.get_thrust_policy(), begin_iter, begin_iter + batch_src.size());

      size_t unique_size = thrust::distance(begin_iter, new_end);

      if (src.size() > 0) {
        new_end =
          thrust::remove_if(handle.get_thrust_policy(),
                            begin_iter,
                            begin_iter + unique_size,
                            [local_src = raft::device_span<vertex_t const>{src.data(), src.size()},
                             local_dst = raft::device_span<vertex_t const>{
                               dst.data(), dst.size()}] __device__(auto tuple) {
                              return thrust::binary_search(
                                thrust::seq,
                                thrust::make_zip_iterator(local_src.begin(), local_dst.begin()),
                                thrust::make_zip_iterator(local_src.end(), local_dst.end()),
                                tuple);
                            });

        unique_size = thrust::distance(begin_iter, new_end);
      }

      batch_src.resize(unique_size, handle.get_stream());
      batch_dst.resize(unique_size, handle.get_stream());
    }

    if (src.size() > 0) {
      size_t current_end = src.size();

      src.resize(src.size() + batch_src.size(), handle.get_stream());
      dst.resize(dst.size() + batch_dst.size(), handle.get_stream());

      thrust::copy(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(batch_src.begin(), batch_dst.begin()),
                   thrust::make_zip_iterator(batch_src.end(), batch_dst.end()),
                   thrust::make_zip_iterator(src.begin(), dst.begin()) + current_end);

      auto begin_iter = thrust::make_zip_iterator(src.begin(), dst.begin());
      thrust::sort(handle.get_thrust_policy(), begin_iter, begin_iter + src.size());
    } else {
      src = std::move(batch_src);
      dst = std::move(batch_dst);

      if (!remove_duplicates) {
        auto begin_iter = thrust::make_zip_iterator(src.begin(), dst.begin());
        thrust::sort(handle.get_thrust_policy(), begin_iter, begin_iter + src.size());
      }
    }

    if (exact_number_of_samples) {
      size_t num_batch_samples = src.size();
      if constexpr (multi_gpu) {
        num_batch_samples = cugraph::host_scalar_allreduce(
          handle.get_comms(), num_batch_samples, raft::comms::op_t::SUM, handle.get_stream());
      }

      // FIXME: We could oversample and discard the unnecessary samples
      // to reduce the number of iterations in the outer loop, but it seems like
      // exact_number_of_samples is an edge case not worth optimizing for at this time.
      samples_in_this_batch = num_samples - num_batch_samples;
    } else {
      samples_in_this_batch = 0;
    }
  }

  return std::make_tuple(std::move(src), std::move(dst));
}

}  // namespace cugraph
