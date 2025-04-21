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

#include "cugraph/detail/collect_comm_wrapper.hpp"
#include "cugraph/utilities/device_comm.hpp"
#include "prims/reduce_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "thrust/iterator/zip_iterator.h"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <tuple>

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
                 raft::device_span<weight_t const> biases)
{
  std::optional<rmm::device_uvector<weight_t>> normalized_biases{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> gpu_biases{std::nullopt};

  // Need to normalize the biases
  normalized_biases =
    std::make_optional<rmm::device_uvector<weight_t>>(biases.size(), handle.get_stream());

  weight_t sum =
    thrust::reduce(handle.get_thrust_policy(), biases.begin(), biases.end(), weight_t{0});

  thrust::transform(handle.get_thrust_policy(),
                    biases.begin(),
                    biases.end(),
                    normalized_biases->begin(),
                    divider_t<weight_t>{sum});

  thrust::inclusive_scan(handle.get_thrust_policy(),
                         normalized_biases->begin(),
                         normalized_biases->end(),
                         normalized_biases->begin());

  if constexpr (multi_gpu) {
    rmm::device_scalar<weight_t> d_sum(sum, handle.get_stream());

    gpu_biases = cugraph::device_allgatherv(
      handle, handle.get_comms(), raft::device_span<weight_t const>{d_sum.data(), d_sum.size()});

    weight_t aggregate_sum = thrust::reduce(
      handle.get_thrust_policy(), gpu_biases->begin(), gpu_biases->end(), weight_t{0});

    // FIXME: https://github.com/rapidsai/raft/issues/2400 results in the possibility
    // that 1 can appear as a random floating point value.  We're going to use
    // thrust::upper_bound to assign random values to GPUs, we need the value 1.0 to
    // be part of the upper-most range.  We'll compute the last non-zero value in the
    // gpu_biases array here and below we will fill it with a value larger than 1.0
    size_t trailing_zeros = cuda::std::distance(
      thrust::make_reverse_iterator(gpu_biases->end()),
      thrust::find_if(handle.get_thrust_policy(),
                      thrust::make_reverse_iterator(gpu_biases->end()),
                      thrust::make_reverse_iterator(gpu_biases->begin()),
                      [] __device__(weight_t bias) { return bias > weight_t{0}; }));

    thrust::transform(handle.get_thrust_policy(),
                      gpu_biases->begin(),
                      gpu_biases->end(),
                      gpu_biases->begin(),
                      divider_t<weight_t>{aggregate_sum});

    thrust::inclusive_scan(
      handle.get_thrust_policy(), gpu_biases->begin(), gpu_biases->end(), gpu_biases->begin());

    // FIXME: conclusion of above.  Using 1.1 since it is > 1.0 and easy to type
    thrust::copy_n(handle.get_thrust_policy(),
                   thrust::make_constant_iterator<weight_t>(1.1),
                   trailing_zeros + 1,
                   gpu_biases->begin() + gpu_biases->size() - trailing_zeros - 1);
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
      auto const comm_rank = handle.get_comms().get_rank();

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

      std::vector<size_t> tx_counts(gpu_counts.size());
      std::fill(tx_counts.begin(), tx_counts.end(), size_t{1});

      rmm::device_uvector<size_t> d_sample_count_from_each_gpu(0, handle.get_stream());

      std::tie(d_sample_count_from_each_gpu, std::ignore) =
        shuffle_values(handle.get_comms(),
                       gpu_counts.begin(),
                       raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                       handle.get_stream());

      samples_to_generate = thrust::reduce(handle.get_thrust_policy(),
                                           d_sample_count_from_each_gpu.begin(),
                                           d_sample_count_from_each_gpu.end(),
                                           size_t{0});

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
          static_cast<vertex_t>(cuda::std::distance(
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
      std::tie(samples, std::ignore) =
        shuffle_values(handle.get_comms(),
                       samples.begin(),
                       raft::host_span<size_t const>(sample_count_from_each_gpu.data(),
                                                     sample_count_from_each_gpu.size()),
                       handle.get_stream());

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
  std::optional<raft::device_span<weight_t const>> src_biases,
  std::optional<raft::device_span<weight_t const>> dst_biases,
  size_t num_samples,
  bool remove_duplicates,
  bool remove_existing_edges,
  bool exact_number_of_samples,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());

  // Optimistically assume we can do this in one pass
  size_t total_samples{num_samples};
  std::vector<size_t> samples_per_gpu;

  if constexpr (multi_gpu) {
    samples_per_gpu = host_scalar_allgather(handle.get_comms(), num_samples, handle.get_stream());
    total_samples   = std::reduce(samples_per_gpu.begin(), samples_per_gpu.end());
  }

  size_t samples_in_this_batch = total_samples;

  // Normalize the biases and (for MG) determine how the biases are
  // distributed across the GPUs.
  std::optional<rmm::device_uvector<weight_t>> normalized_src_biases{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> gpu_src_biases{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> normalized_dst_biases{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> gpu_dst_biases{std::nullopt};

  if (src_biases)
    std::tie(normalized_src_biases, gpu_src_biases) =
      detail::normalize_biases(handle, graph_view, *src_biases);

  if (dst_biases)
    std::tie(normalized_dst_biases, gpu_dst_biases) =
      detail::normalize_biases(handle, graph_view, *dst_biases);

  while (samples_in_this_batch > 0) {
    if constexpr (multi_gpu) {
      auto const comm_size = handle.get_comms().get_size();
      auto const comm_rank = handle.get_comms().get_rank();

      samples_in_this_batch =
        (samples_in_this_batch / static_cast<size_t>(comm_size)) +
        (static_cast<size_t>(comm_rank) < (samples_in_this_batch % static_cast<size_t>(comm_size))
           ? 1
           : 0);
    }

    auto batch_srcs = create_local_samples(
      handle, rng_state, graph_view, normalized_src_biases, gpu_src_biases, samples_in_this_batch);
    auto batch_dsts = create_local_samples(
      handle, rng_state, graph_view, normalized_dst_biases, gpu_dst_biases, samples_in_this_batch);

    if constexpr (multi_gpu) {
      auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();

      std::tie(batch_srcs,
               batch_dsts,
               std::ignore,
               std::ignore,
               std::ignore,
               std::ignore,
               std::ignore,
               std::ignore) =
        detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t,
                                                                                       int32_t>(
          handle,
          std::move(batch_srcs),
          std::move(batch_dsts),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          vertex_partition_range_lasts);
    }

    if (remove_existing_edges) {
      auto has_edge_flags =
        graph_view.has_edge(handle,
                            raft::device_span<vertex_t const>{batch_srcs.data(), batch_srcs.size()},
                            raft::device_span<vertex_t const>{batch_dsts.data(), batch_dsts.size()},
                            do_expensive_check);

      auto begin_iter = thrust::make_zip_iterator(batch_srcs.begin(), batch_dsts.begin());
      auto new_end    = thrust::remove_if(handle.get_thrust_policy(),
                                       begin_iter,
                                       begin_iter + batch_srcs.size(),
                                       has_edge_flags.begin(),
                                       cuda::std::identity());

      batch_srcs.resize(cuda::std::distance(begin_iter, new_end), handle.get_stream());
      batch_dsts.resize(cuda::std::distance(begin_iter, new_end), handle.get_stream());
    }

    if (remove_duplicates) {
      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(batch_srcs.begin(), batch_dsts.begin()),
                   thrust::make_zip_iterator(batch_srcs.end(), batch_dsts.end()));

      auto new_end =
        thrust::unique(handle.get_thrust_policy(),
                       thrust::make_zip_iterator(batch_srcs.begin(), batch_dsts.begin()),
                       thrust::make_zip_iterator(batch_srcs.end(), batch_dsts.end()));

      size_t new_size = cuda::std::distance(
        thrust::make_zip_iterator(batch_srcs.begin(), batch_dsts.begin()), new_end);

      if (srcs.size() > 0) {
        rmm::device_uvector<vertex_t> new_src(srcs.size() + new_size, handle.get_stream());
        rmm::device_uvector<vertex_t> new_dst(dsts.size() + new_size, handle.get_stream());

        thrust::merge(handle.get_thrust_policy(),
                      thrust::make_zip_iterator(batch_srcs.begin(), batch_dsts.begin()),
                      new_end,
                      thrust::make_zip_iterator(srcs.begin(), dsts.begin()),
                      thrust::make_zip_iterator(srcs.end(), dsts.end()),
                      thrust::make_zip_iterator(new_src.begin(), new_dst.begin()));

        new_end = thrust::unique(handle.get_thrust_policy(),
                                 thrust::make_zip_iterator(new_src.begin(), new_dst.begin()),
                                 thrust::make_zip_iterator(new_src.end(), new_dst.end()));

        new_size =
          cuda::std::distance(thrust::make_zip_iterator(new_src.begin(), new_dst.begin()), new_end);

        srcs = std::move(new_src);
        dsts = std::move(new_dst);
      } else {
        srcs = std::move(batch_srcs);
        dsts = std::move(batch_dsts);
      }

      srcs.resize(new_size, handle.get_stream());
      dsts.resize(new_size, handle.get_stream());
    } else if (srcs.size() > 0) {
      size_t current_end = srcs.size();

      srcs.resize(srcs.size() + batch_srcs.size(), handle.get_stream());
      dsts.resize(dsts.size() + batch_dsts.size(), handle.get_stream());

      thrust::copy(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(batch_srcs.begin(), batch_dsts.begin()),
                   thrust::make_zip_iterator(batch_srcs.end(), batch_dsts.end()),
                   thrust::make_zip_iterator(srcs.begin(), dsts.begin()) + current_end);
    } else {
      srcs = std::move(batch_srcs);
      dsts = std::move(batch_dsts);
    }

    if (exact_number_of_samples) {
      size_t current_sample_size = srcs.size();
      if constexpr (multi_gpu) {
        current_sample_size = cugraph::host_scalar_allreduce(
          handle.get_comms(), current_sample_size, raft::comms::op_t::SUM, handle.get_stream());
      }

      // FIXME: We could oversample and discard the unnecessary samples
      // to reduce the number of iterations in the outer loop, but it seems like
      // exact_number_of_samples is an edge case not worth optimizing for at this time.
      samples_in_this_batch = total_samples - current_sample_size;
    } else {
      samples_in_this_batch = 0;
    }
  }

  srcs.shrink_to_fit(handle.get_stream());
  dsts.shrink_to_fit(handle.get_stream());

  if constexpr (multi_gpu) {
    auto const& comm     = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    // Randomly shuffle the samples so that each gpu gets their
    // desired number of samples

    if (!exact_number_of_samples) {
      // If we didn't force generating the exact number of samples,
      // we might have fewer samples than requested.  We need to
      // accommodate this situation.  For now we'll just
      // uniformly(-ish) reduce the requested size.
      size_t total_extracted = host_scalar_allreduce(
        handle.get_comms(), srcs.size(), raft::comms::op_t::SUM, handle.get_stream());
      size_t reduction = total_samples - total_extracted;

      while (reduction > 0) {
        size_t est_reduction_per_gpu = (reduction + comm_size - 1) / comm_size;
        for (size_t i = 0; i < samples_per_gpu.size(); ++i) {
          if (samples_per_gpu[i] > est_reduction_per_gpu) {
            samples_per_gpu[i] -= est_reduction_per_gpu;
            reduction -= est_reduction_per_gpu;
          } else {
            reduction -= samples_per_gpu[i];
            samples_per_gpu[i] = 0;
          }

          if (reduction < est_reduction_per_gpu) est_reduction_per_gpu = reduction;
        }
      }
      num_samples = samples_per_gpu[comm_rank];
    }

    // Mimic the logic of permute_range...
    //
    //  1) Randomly assign each entry to a GPU
    //  2) Count how many are assigned to each GPU
    //  3) Allgatherv (allgather?) to give each GPU a count for how many entries are destined for
    //  that GPU 4) Identify extras/deficits for each GPU, arbitrarily adjust counts to make correct
    //  5) Shuffle accordingly
    //
    rmm::device_uvector<int> gpu_assignment(srcs.size(), handle.get_stream());

    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         gpu_assignment.data(),
                                         gpu_assignment.size(),
                                         int{0},
                                         int{comm_size},
                                         rng_state);

    thrust::sort_by_key(handle.get_thrust_policy(),
                        gpu_assignment.begin(),
                        gpu_assignment.end(),
                        thrust::make_zip_iterator(srcs.begin(), dsts.begin()));

    rmm::device_uvector<size_t> d_send_counts(comm_size, handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      d_send_counts.begin(),
      d_send_counts.end(),
      [gpu_assignment_span = raft::device_span<const int>{
         gpu_assignment.data(), gpu_assignment.size()}] __device__(size_t i) {
        auto begin = thrust::lower_bound(
          thrust::seq, gpu_assignment_span.begin(), gpu_assignment_span.end(), static_cast<int>(i));
        auto end =
          thrust::upper_bound(thrust::seq, begin, gpu_assignment_span.end(), static_cast<int>(i));
        return cuda::std::distance(begin, end);
      });

    std::vector<size_t> tx_value_counts(comm_size, 0);
    raft::update_host(
      tx_value_counts.data(), d_send_counts.data(), d_send_counts.size(), handle.get_stream());

    std::forward_as_tuple(std::tie(srcs, dsts), std::ignore) = cugraph::shuffle_values(
      handle.get_comms(),
      thrust::make_zip_iterator(srcs.begin(), dsts.begin()),
      raft::host_span<size_t const>(tx_value_counts.data(), tx_value_counts.size()),
      handle.get_stream());

    rmm::device_uvector<float> fractional_random_numbers(srcs.size(), handle.get_stream());

    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         fractional_random_numbers.data(),
                                         fractional_random_numbers.size(),
                                         float{0.0},
                                         float{1.0},
                                         rng_state);
    thrust::sort_by_key(handle.get_thrust_policy(),
                        fractional_random_numbers.begin(),
                        fractional_random_numbers.end(),
                        thrust::make_zip_iterator(srcs.begin(), dsts.begin()));

    size_t nr_extras{0};
    size_t nr_deficits{0};
    if (srcs.size() > num_samples) {
      nr_extras = srcs.size() - static_cast<size_t>(num_samples);
    } else {
      nr_deficits = static_cast<size_t>(num_samples) - srcs.size();
    }

    auto extra_srcs = cugraph::detail::device_allgatherv(
      handle, comm, raft::device_span<vertex_t const>(srcs.data() + num_samples, nr_extras));
    // nr_extras > 0 ? nr_extras : 0));
    auto extra_dsts = cugraph::detail::device_allgatherv(
      handle, comm, raft::device_span<vertex_t const>(dsts.data() + num_samples, nr_extras));
    // nr_extras > 0 ? nr_extras : 0));

    srcs.resize(num_samples, handle.get_stream());
    dsts.resize(num_samples, handle.get_stream());
    auto deficits =
      cugraph::host_scalar_allgather(handle.get_comms(), nr_deficits, handle.get_stream());

    std::exclusive_scan(deficits.begin(), deficits.end(), deficits.begin(), vertex_t{0});

    raft::copy(srcs.data() + num_samples - nr_deficits,
               extra_srcs.begin() + deficits[comm_rank],
               nr_deficits,
               handle.get_stream());

    raft::copy(dsts.data() + num_samples - nr_deficits,
               extra_dsts.begin() + deficits[comm_rank],
               nr_deficits,
               handle.get_stream());
  }

  return std::make_tuple(std::move(srcs), std::move(dsts));
}

}  // namespace cugraph
