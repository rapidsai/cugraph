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

#include "prims/reduce_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <rmm/device_scalar.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace cugraph {

namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<weight_t>>>
normalize_biases(raft::handle_t const& handle,
                 graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
                 std::optional<raft::device_span<weight_t const>> biases)
{
  std::optional<rmm::device_uvector<weight_t>> normalized_biases{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> gpu_biases{std::nullopt};

  if (biases) {
    // Need to normalize the biases
    normalized_biases =
      std::make_optional<rmm::device_uvector<weight_t>>(biases->size(), handle.get_stream());

    weight_t sum =
      thrust::reduce(handle.get_thrust_policy(), biases->begin(), biases->end(), weight_t{0});

    weight_t aggregate_sum{sum};

    if constexpr (multi_gpu) {
      aggregate_sum =
        host_scalar_allreduce(handle.get_comms(), sum, raft::comms::op_t::SUM, handle.get_stream());
    }

    thrust::transform(handle.get_thrust_policy(),
                      biases->begin(),
                      biases->end(),
                      normalized_biases->begin(),
                      divider_t<weight_t>{sum});

    thrust::inclusive_scan(handle.get_thrust_policy(),
                           normalized_biases->begin(),
                           normalized_biases->end(),
                           normalized_biases->begin());

    if constexpr (multi_gpu) {
      rmm::device_scalar<weight_t> d_sum((sum / aggregate_sum), handle.get_stream());
      gpu_biases = cugraph::device_allgatherv(
        handle, handle.get_comms(), raft::device_span<weight_t const>{d_sum.data(), d_sum.size()});

      thrust::inclusive_scan(
        handle.get_thrust_policy(), gpu_biases->begin(), gpu_biases->end(), gpu_biases->begin());

      weight_t force_to_one{1.1};
      raft::update_device(
        gpu_biases->data() + gpu_biases->size() - 1, &force_to_one, 1, handle.get_stream());
    }
  }

  return std::make_tuple(std::move(normalized_biases), std::move(gpu_biases));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
rmm::device_uvector<vertex_t> create_local_samples(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<rmm::device_uvector<weight_t>> const& normalized_biases,
  std::optional<rmm::device_uvector<weight_t>> const& gpu_biases,
  size_t samples_in_this_batch)
{
  rmm::device_uvector<vertex_t> samples(0, handle.get_stream());

  if (normalized_biases) {
    size_t samples_to_generate{samples_in_this_batch};
    std::vector<size_t> sample_count_from_each_gpu;

    rmm::device_uvector<size_t> position(0, handle.get_stream());

    if constexpr (multi_gpu) {
      // Determine how many vertices are generated on each GPU
      auto const comm_size = handle.get_comms().get_size();
      auto const rank      = handle.get_comms().get_rank();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      sample_count_from_each_gpu.resize(comm_size);

      rmm::device_uvector<size_t> gpu_counts(comm_size, handle.get_stream());
      position.resize(samples_in_this_batch, handle.get_stream());

      thrust::fill(handle.get_thrust_policy(), gpu_counts.begin(), gpu_counts.end(), size_t{0});
      thrust::sequence(handle.get_thrust_policy(), position.begin(), position.end());

      rmm::device_uvector<weight_t> random_values(samples_in_this_batch, handle.get_stream());
      detail::uniform_random_fill(handle.get_stream(),
                                  random_values.data(),
                                  random_values.size(),
                                  weight_t{0},
                                  weight_t{1},
                                  rng_state);

      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(random_values.begin(), position.begin()),
                   thrust::make_zip_iterator(random_values.end(), position.end()));

      thrust::upper_bound(handle.get_thrust_policy(),
                          random_values.begin(),
                          random_values.end(),
                          gpu_biases->begin(),
                          gpu_biases->end(),
                          gpu_counts.begin());

      thrust::adjacent_difference(
        handle.get_thrust_policy(), gpu_counts.begin(), gpu_counts.end(), gpu_counts.begin());

      // all_gpu_counts[i][j] will be how many vertices need to be generated on GPU j to be sent to
      // GPU i
      auto all_gpu_counts = cugraph::device_allgatherv(
        handle,
        handle.get_comms(),
        raft::device_span<size_t const>{gpu_counts.data(), gpu_counts.size()});

      auto begin_iter = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_t>(0),
        cuda::proclaim_return_type<size_t>(
          [rank, stride = comm_size, counts = all_gpu_counts.data()] __device__(size_t idx) {
            return counts[idx * stride + rank];
          }));

      samples_to_generate =
        thrust::reduce(handle.get_thrust_policy(), begin_iter, begin_iter + comm_size, size_t{0});

      rmm::device_uvector<size_t> d_sample_count_from_each_gpu(comm_size, handle.get_stream());

      thrust::copy(handle.get_thrust_policy(),
                   begin_iter,
                   begin_iter + comm_size,
                   d_sample_count_from_each_gpu.begin());

      raft::update_host(sample_count_from_each_gpu.data(),
                        d_sample_count_from_each_gpu.data(),
                        d_sample_count_from_each_gpu.size(),
                        handle.get_stream());
    }

    // Generate samples
    //  FIXME: We could save this memory if we had an iterator that
    //         generated random values.
    rmm::device_uvector<weight_t> random_values(samples_to_generate, handle.get_stream());
    samples.resize(samples_to_generate, handle.get_stream());
    detail::uniform_random_fill(handle.get_stream(),
                                random_values.data(),
                                random_values.size(),
                                weight_t{0},
                                weight_t{1},
                                rng_state);

    thrust::transform(
      handle.get_thrust_policy(),
      random_values.begin(),
      random_values.end(),
      samples.begin(),
      [biases =
         raft::device_span<weight_t const>{normalized_biases->data(), normalized_biases->size()},
       offset = graph_view.local_vertex_partition_range_first()] __device__(weight_t r) {
        size_t result =
          offset +
          static_cast<vertex_t>(thrust::distance(
            biases.begin(), thrust::lower_bound(thrust::seq, biases.begin(), biases.end(), r)));

        // FIXME: https://github.com/rapidsai/raft/issues/2400
        // results in the possibility that 1 can appear as a
        // random floating point value, which results in the sampling
        // algorithm below generating a value that's OOB.
        if (result == (offset + biases.size())) --result;

        return result;
      });

    // Shuffle them back
    if constexpr (multi_gpu) {
      std::tie(samples, std::ignore) = shuffle_values(
        handle.get_comms(), samples.begin(), sample_count_from_each_gpu, handle.get_stream());

      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(position.begin(), samples.begin()),
                   thrust::make_zip_iterator(position.end(), samples.begin()));
    }
  } else {
    samples.resize(samples_in_this_batch, handle.get_stream());

    // Uniformly select a vertex from any GPU
    detail::uniform_random_fill(handle.get_stream(),
                                samples.data(),
                                samples.size(),
                                vertex_t{0},
                                graph_view.number_of_vertices(),
                                rng_state);
  }

  return samples;
}

}  // namespace detail

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
  std::optional<raft::device_span<weight_t const>> src_biases,
  std::optional<raft::device_span<weight_t const>> dst_biases,
  bool remove_duplicates,
  bool remove_existing_edges,
  bool exact_number_of_samples,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dst(0, handle.get_stream());

  // Optimistically assume we can do this in one pass
  size_t samples_in_this_batch = num_samples;

  // Normalize the biases and (for MG) determine how the biases are
  // distributed across the GPUs.
  auto [normalized_src_biases, gpu_src_biases] =
    detail::normalize_biases(handle, graph_view, src_biases);
  auto [normalized_dst_biases, gpu_dst_biases] =
    detail::normalize_biases(handle, graph_view, dst_biases);

  while (samples_in_this_batch > 0) {
    if constexpr (multi_gpu) {
      size_t comm_size = handle.get_comms().get_size();
      size_t comm_rank = handle.get_comms().get_rank();

      samples_in_this_batch = (samples_in_this_batch / comm_size) +
                              (comm_rank < (samples_in_this_batch % comm_size) ? 1 : 0);
    }

    auto batch_src = create_local_samples(
      handle, rng_state, graph_view, normalized_src_biases, gpu_src_biases, samples_in_this_batch);
    auto batch_dst = create_local_samples(
      handle, rng_state, graph_view, normalized_dst_biases, gpu_dst_biases, samples_in_this_batch);

    if constexpr (multi_gpu) {
      auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();

      std::tie(batch_src, batch_dst, std::ignore, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle,
          std::move(batch_src),
          std::move(batch_dst),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          vertex_partition_range_lasts);
    }

    if (remove_existing_edges) {
      auto has_edge_flags =
        graph_view.has_edge(handle,
                            raft::device_span<vertex_t const>{batch_src.data(), batch_src.size()},
                            raft::device_span<vertex_t const>{batch_dst.data(), batch_dst.size()},
                            do_expensive_check);

      auto begin_iter = thrust::make_zip_iterator(batch_src.begin(), batch_dst.begin());
      auto new_end    = thrust::remove_if(handle.get_thrust_policy(),
                                       begin_iter,
                                       begin_iter + batch_src.size(),
                                       has_edge_flags.begin(),
                                       thrust::identity<bool>());

      batch_src.resize(thrust::distance(begin_iter, new_end), handle.get_stream());
      batch_dst.resize(thrust::distance(begin_iter, new_end), handle.get_stream());
    }

    if (remove_duplicates) {
      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(batch_src.begin(), batch_dst.begin()),
                   thrust::make_zip_iterator(batch_src.end(), batch_dst.end()));

      auto new_end = thrust::unique(handle.get_thrust_policy(),
                                    thrust::make_zip_iterator(batch_src.begin(), batch_dst.begin()),
                                    thrust::make_zip_iterator(batch_src.end(), batch_dst.end()));

      size_t new_size =
        thrust::distance(thrust::make_zip_iterator(batch_src.begin(), batch_dst.begin()), new_end);

      if (src.size() > 0) {
        rmm::device_uvector<vertex_t> new_src(src.size() + new_size, handle.get_stream());
        rmm::device_uvector<vertex_t> new_dst(dst.size() + new_size, handle.get_stream());

        thrust::merge(handle.get_thrust_policy(),
                      thrust::make_zip_iterator(batch_src.begin(), batch_dst.begin()),
                      new_end,
                      thrust::make_zip_iterator(src.begin(), dst.begin()),
                      thrust::make_zip_iterator(src.end(), dst.end()),
                      thrust::make_zip_iterator(new_src.begin(), new_dst.begin()));

        new_end = thrust::unique(handle.get_thrust_policy(),
                                 thrust::make_zip_iterator(new_src.begin(), new_dst.begin()),
                                 thrust::make_zip_iterator(new_src.end(), new_dst.end()));

        new_size =
          thrust::distance(thrust::make_zip_iterator(new_src.begin(), new_dst.begin()), new_end);

        src = std::move(new_src);
        dst = std::move(new_dst);
      } else {
        src = std::move(batch_src);
        dst = std::move(batch_dst);
      }

      src.resize(new_size, handle.get_stream());
      dst.resize(new_size, handle.get_stream());
    } else if (src.size() > 0) {
      size_t current_end = src.size();

      src.resize(src.size() + batch_src.size(), handle.get_stream());
      dst.resize(dst.size() + batch_dst.size(), handle.get_stream());

      thrust::copy(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(batch_src.begin(), batch_dst.begin()),
                   thrust::make_zip_iterator(batch_src.end(), batch_dst.end()),
                   thrust::make_zip_iterator(src.begin(), dst.begin()) + current_end);
    } else {
      src = std::move(batch_src);
      dst = std::move(batch_dst);
    }

    if (exact_number_of_samples) {
      size_t current_sample_size = src.size();
      if constexpr (multi_gpu) {
        current_sample_size = cugraph::host_scalar_allreduce(
          handle.get_comms(), current_sample_size, raft::comms::op_t::SUM, handle.get_stream());
      }

      // FIXME: We could oversample and discard the unnecessary samples
      // to reduce the number of iterations in the outer loop, but it seems like
      // exact_number_of_samples is an edge case not worth optimizing for at this time.
      samples_in_this_batch = num_samples - current_sample_size;
    } else {
      samples_in_this_batch = 0;
    }
  }

  src.shrink_to_fit(handle.get_stream());
  dst.shrink_to_fit(handle.get_stream());

  return std::make_tuple(std::move(src), std::move(dst));
}

}  // namespace cugraph
