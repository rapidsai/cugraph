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

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
class negative_sampling_impl_t {
 private:
  static const bool store_transposed = false;

 public:
  negative_sampling_impl_t(
    raft::handle_t const& handle,
    graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
    std::optional<raft::device_span<weight_t const>> src_bias,
    std::optional<raft::device_span<weight_t const>> dst_bias)
    : gpu_bias_v_(0, handle.get_stream()),
      src_bias_v_(0, handle.get_stream()),
      dst_bias_v_(0, handle.get_stream()),
      src_bias_cache_(std::nullopt),
      dst_bias_cache_(std::nullopt)
  {
    // Need to normalize the src_bias
    if (src_bias) {
      // Normalize the src bias.
      rmm::device_uvector<weight_t> normalized_bias(graph_view.local_vertex_partition_range_size(),
                                                    handle.get_stream());

      weight_t sum = reduce_v(handle, graph_view, src_bias->begin());

      if constexpr (multi_gpu) {
        sum = host_scalar_allreduce(
          handle.get_comms(), sum, raft::comms::op_t::SUM, handle.get_stream());
      }

      thrust::transform(handle.get_thrust_policy(),
                        src_bias->begin(),
                        src_bias->end(),
                        normalized_bias.begin(),
                        divider_t<weight_t>{sum});

      // Distribute the src bias around the edge partitions
      src_bias_cache_ = std::make_optional<
        edge_src_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>(
        handle, graph_view);
      update_edge_src_property(
        handle, graph_view, normalized_bias.begin(), src_bias_cache_->mutable_view());
    }

    if (dst_bias) {
      // Normalize the dst bias.
      rmm::device_uvector<weight_t> normalized_bias(graph_view.local_vertex_partition_range_size(),
                                                    handle.get_stream());

      weight_t sum = reduce_v(handle, graph_view, dst_bias->begin());

      if constexpr (multi_gpu) {
        sum = host_scalar_allreduce(
          handle.get_comms(), sum, raft::comms::op_t::SUM, handle.get_stream());
      }

      thrust::transform(handle.get_thrust_policy(),
                        dst_bias->begin(),
                        dst_bias->end(),
                        normalized_bias.begin(),
                        divider_t<weight_t>{sum});

      dst_bias_cache_ = std::make_optional<
        edge_dst_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>(
        handle, graph_view);
      update_edge_dst_property(
        handle, graph_view, normalized_bias.begin(), dst_bias_cache_->mutable_view());
    }

    if constexpr (multi_gpu) {
      weight_t dst_bias_sum{0};

      if (dst_bias) {
        // Compute the dst_bias sum for this partition and normalize cached values
        dst_bias_sum = thrust::reduce(
          handle.get_thrust_policy(),
          dst_bias_cache_->view().value_first(),
          dst_bias_cache_->view().value_first() + graph_view.local_edge_partition_dst_range_size(),
          weight_t{0});

        thrust::transform(handle.get_thrust_policy(),
                          dst_bias_cache_->mutable_view().value_first(),
                          dst_bias_cache_->mutable_view().value_first() +
                            graph_view.local_edge_partition_dst_range_size(),
                          dst_bias_cache_->mutable_view().value_first(),
                          divider_t<weight_t>{dst_bias_sum});

        thrust::inclusive_scan(handle.get_thrust_policy(),
                               dst_bias_cache_->mutable_view().value_first(),
                               dst_bias_cache_->mutable_view().value_first() +
                                 graph_view.local_edge_partition_dst_range_size(),
                               dst_bias_cache_->mutable_view().value_first());
      } else {
        dst_bias_sum = static_cast<weight_t>(graph_view.local_edge_partition_dst_range_size()) /
                       static_cast<weight_t>(graph_view.number_of_vertices());
      }

      std::vector<weight_t> h_gpu_bias;
      h_gpu_bias.reserve(graph_view.number_of_local_edge_partitions());

      for (size_t partition_idx = 0; partition_idx < graph_view.number_of_local_edge_partitions();
           ++partition_idx) {
        weight_t src_bias_sum{
          static_cast<weight_t>(graph_view.local_edge_partition_src_range_size(partition_idx)) /
          static_cast<weight_t>(graph_view.number_of_vertices())};

        if (src_bias) {
          // Normalize each batch of biases and compute the inclusive prefix sum
          src_bias_sum =
            thrust::reduce(handle.get_thrust_policy(),
                           src_bias_cache_->view().value_firsts()[partition_idx],
                           src_bias_cache_->view().value_firsts()[partition_idx] +
                             graph_view.local_edge_partition_src_range_size(partition_idx),
                           weight_t{0});

          thrust::transform(handle.get_thrust_policy(),
                            src_bias_cache_->mutable_view().value_firsts()[partition_idx],
                            src_bias_cache_->mutable_view().value_firsts()[partition_idx] +
                              graph_view.local_edge_partition_src_range_size(partition_idx),
                            src_bias_cache_->mutable_view().value_firsts()[partition_idx],
                            divider_t<weight_t>{src_bias_sum});

          thrust::inclusive_scan(handle.get_thrust_policy(),
                                 src_bias_cache_->mutable_view().value_firsts()[partition_idx],
                                 src_bias_cache_->mutable_view().value_firsts()[partition_idx] +
                                   graph_view.local_edge_partition_src_range_size(partition_idx),
                                 src_bias_cache_->mutable_view().value_firsts()[partition_idx]);
        }

        // Because src_bias and dst_bias are normalized, the probability of a random edge appearing
        // on this partition is (src_bias_sum * dst_bias_sum)
        h_gpu_bias.push_back(src_bias_sum * dst_bias_sum);
      }

      rmm::device_uvector<weight_t> d_gpu_bias(h_gpu_bias.size(), handle.get_stream());
      raft::update_device(
        d_gpu_bias.data(), h_gpu_bias.data(), h_gpu_bias.size(), handle.get_stream());

      gpu_bias_v_ = cugraph::device_allgatherv(
        handle,
        handle.get_comms(),
        raft::device_span<weight_t const>{d_gpu_bias.data(), d_gpu_bias.size()});

      thrust::inclusive_scan(
        handle.get_thrust_policy(), gpu_bias_v_.begin(), gpu_bias_v_.end(), gpu_bias_v_.begin());
    } else {
      if (dst_bias_cache_)
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               dst_bias_cache_->mutable_view().value_first(),
                               dst_bias_cache_->mutable_view().value_first() +
                                 graph_view.local_edge_partition_dst_range_size(),
                               dst_bias_cache_->mutable_view().value_first());

      if (src_bias_cache_)
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               src_bias_cache_->mutable_view().value_firsts()[0],
                               src_bias_cache_->mutable_view().value_firsts()[0] +
                                 graph_view.local_edge_partition_src_range_size(0),
                               src_bias_cache_->mutable_view().value_firsts()[0]);
    }
  }

  std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> create_local_samples(
    raft::handle_t const& handle,
    raft::random::RngState& rng_state,
    graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
    size_t num_samples)
  {
    rmm::device_uvector<vertex_t> src(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dst(0, handle.get_stream());

    std::vector<size_t> sample_counts;

    // Determine sample counts per GPU edge partition
    if constexpr (multi_gpu) {
      auto const comm_size = handle.get_comms().get_size();
      auto const rank      = handle.get_comms().get_rank();
      auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      // First step is to count how many go on each edge_partition
      rmm::device_uvector<size_t> gpu_counts(gpu_bias_v_.size(), handle.get_stream());
      thrust::fill(handle.get_thrust_policy(), gpu_counts.begin(), gpu_counts.end(), int{0});

      rmm::device_uvector<weight_t> random_values(num_samples, handle.get_stream());
      detail::uniform_random_fill(handle.get_stream(),
                                  random_values.data(),
                                  random_values.size(),
                                  weight_t{0},
                                  weight_t{1},
                                  rng_state);

      thrust::sort(handle.get_thrust_policy(), random_values.begin(), random_values.end());

      thrust::upper_bound(handle.get_thrust_policy(),
                          random_values.begin(),
                          random_values.end(),
                          gpu_bias_v_.begin(),
                          gpu_bias_v_.end(),
                          gpu_counts.begin());

      thrust::adjacent_difference(
        handle.get_thrust_policy(), gpu_counts.begin(), gpu_counts.end(), gpu_counts.begin());

      device_allreduce(handle.get_comms(),
                       gpu_counts.begin(),
                       gpu_counts.begin(),
                       gpu_counts.size(),
                       raft::comms::op_t::SUM,
                       handle.get_stream());

      num_samples = thrust::reduce(handle.get_thrust_policy(),
                                   gpu_counts.begin() + rank * minor_comm_size,
                                   gpu_counts.begin() + rank * minor_comm_size + minor_comm_size,
                                   size_t{0});

      sample_counts.resize(minor_comm_size);
      raft::update_host(sample_counts.data(),
                        gpu_counts.data() + rank * minor_comm_size,
                        minor_comm_size,
                        handle.get_stream());

    } else {
      // SG is only one partition
      sample_counts.push_back(num_samples);
    }

    src.resize(num_samples, handle.get_stream());
    dst.resize(num_samples, handle.get_stream());

    size_t current_pos{0};

    for (size_t partition_idx = 0; partition_idx < graph_view.number_of_local_edge_partitions();
         ++partition_idx) {
      if (sample_counts[partition_idx] > 0) {
        if (src_bias_cache_) {
          rmm::device_uvector<weight_t> random_values(sample_counts[partition_idx],
                                                      handle.get_stream());
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
            src.begin() + current_pos,
            [biases =
               raft::device_span<weight_t const>{
                 src_bias_cache_->view().value_firsts()[partition_idx],
                 static_cast<size_t>(
                   graph_view.local_edge_partition_src_range_size(partition_idx))},
             offset = graph_view.local_edge_partition_src_range_first(
               partition_idx)] __device__(weight_t r) {
              size_t result =
                offset + static_cast<vertex_t>(thrust::distance(
                           biases.begin(),
                           thrust::lower_bound(thrust::seq, biases.begin(), biases.end(), r)));

              // FIXME: https://github.com/rapidsai/raft/issues/2400
              // results in the possibility that 1 can appear as a
              // random floating point value, which results in the sampling
              // algorithm below generating a value that's OOB.
              if (result == (offset + biases.size())) --result;

              return result;
            });
        } else {
          detail::uniform_random_fill(
            handle.get_stream(),
            src.data() + current_pos,
            sample_counts[partition_idx],
            graph_view.local_edge_partition_src_range_first(partition_idx),
            graph_view.local_edge_partition_src_range_last(partition_idx),
            rng_state);
        }

        if (dst_bias_cache_) {
          rmm::device_uvector<weight_t> random_values(sample_counts[partition_idx],
                                                      handle.get_stream());
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
            dst.begin() + current_pos,
            [biases =
               raft::device_span<weight_t const>{
                 dst_bias_cache_->view().value_first(),
                 static_cast<size_t>(graph_view.local_edge_partition_dst_range_size())},
             offset = graph_view.local_edge_partition_dst_range_first()] __device__(weight_t r) {
              size_t result =
                offset + static_cast<vertex_t>(thrust::distance(
                           biases.begin(),
                           thrust::lower_bound(thrust::seq, biases.begin(), biases.end(), r)));

              // FIXME: https://github.com/rapidsai/raft/issues/2400
              // results in the possibility that 1 can appear as a
              // random floating point value, which results in the sampling
              // algorithm below generating a value that's OOB.
              if (result == (offset + biases.size())) --result;

              return result;
            });
        } else {
          detail::uniform_random_fill(handle.get_stream(),
                                      dst.data() + current_pos,
                                      sample_counts[partition_idx],
                                      graph_view.local_edge_partition_dst_range_first(),
                                      graph_view.local_edge_partition_dst_range_last(),
                                      rng_state);
        }

        current_pos += sample_counts[partition_idx];
      }
    }

    return std::make_tuple(std::move(src), std::move(dst));
  }

 private:
  rmm::device_uvector<weight_t> gpu_bias_v_;
  rmm::device_uvector<weight_t> src_bias_v_;
  rmm::device_uvector<weight_t> dst_bias_v_;
  std::optional<
    edge_src_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    src_bias_cache_;
  std::optional<
    edge_dst_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    dst_bias_cache_;
};

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
  std::optional<raft::device_span<weight_t const>> src_bias,
  std::optional<raft::device_span<weight_t const>> dst_bias,
  bool remove_duplicates,
  bool remove_false_negatives,
  bool exact_number_of_samples,
  bool do_expensive_check)
{
  detail::negative_sampling_impl_t<vertex_t, edge_t, weight_t, multi_gpu> impl(
    handle, graph_view, src_bias, dst_bias);

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

    auto [batch_src, batch_dst] =
      impl.create_local_samples(handle, rng_state, graph_view, samples_in_this_batch);

    if (remove_false_negatives) {
      auto has_edge_flags =
        graph_view.has_edge(handle,
                            raft::device_span<vertex_t const>{batch_src.data(), batch_src.size()},
                            raft::device_span<vertex_t const>{batch_dst.data(), batch_dst.size()},
                            // do_expensive_check);
                            true);

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

  return std::make_tuple(std::move(src), std::move(dst));
}

}  // namespace cugraph
