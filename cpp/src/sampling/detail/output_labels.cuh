/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <optional>

namespace cugraph {
namespace detail {

/**
 * @brief Determine which labels this GPU is responsible for reporting in the sampling output.
 *
 * The offsets array returned by @ref shuffle_and_organize_output delineates one range per label, so
 * it needs an entry for every label this GPU reports plus a trailing total.  The label set has to be
 * derived from the seeds rather than from the labels that survive sampling: a label can legitimately
 * sample no edges (for temporal sampling, when nothing incident on its seeds satisfies the time
 * window), and dropping it would shift every subsequent label's range.
 *
 * When @p label_to_output_comm_rank is specified the sampled edges are shuffled so that each label's
 * edges end up on the rank owning that label, so this GPU reports exactly the labels assigned to it.
 * Without that mapping no shuffling happens, so every GPU reports the entire label range while
 * holding only the edges it sampled locally.
 *
 * Labels are expected to be dense in [0, n).
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param starting_vertex_labels Optional label associated with each seed vertex
 * @param label_to_output_comm_rank Optional map associating each label to a comm rank
 *
 * @returns the sorted labels this GPU reports, or std::nullopt if the seeds carry no labels
 */
template <typename label_t, bool multi_gpu>
std::optional<rmm::device_uvector<label_t>> compute_output_labels(
  raft::handle_t const& handle,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank)
{
  if (!starting_vertex_labels) { return std::nullopt; }

  if constexpr (multi_gpu) {
    if (label_to_output_comm_rank) {
      auto my_rank = handle.get_comms().get_rank();

      rmm::device_uvector<label_t> output_labels(label_to_output_comm_rank->size(),
                                                 handle.get_stream());
      auto last = thrust::copy_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(label_t{0}),
        thrust::make_counting_iterator(static_cast<label_t>(label_to_output_comm_rank->size())),
        output_labels.begin(),
        [output_comm_rank = *label_to_output_comm_rank, my_rank] __device__(label_t label) {
          return output_comm_rank[label] == my_rank;
        });
      output_labels.resize(
        static_cast<size_t>(cuda::std::distance(output_labels.begin(), last)), handle.get_stream());

      return output_labels;
    }
  }

  // FIXME: The C API takes starting_vertex_label_offsets and expands them into a flat
  // starting_vertex_labels array before calling the C++ sampling API, which only accepts
  // labels (and returns offsets).  Interior empty ranges are still recoverable here via
  // max(labels) + 1, but a trailing empty range disappears because expand_sparse_offsets
  // produces no seed carrying that label.  The MG path with label_to_output_comm_rank is
  // immune because the C API sizes that mapping from offsets.size() - 1.  Closing the SG
  // (and MG-without-mapping) gap means plumbing an optional label count or the seed offsets
  // through the public C++ sampling API so trailing empties are not lost.
  auto max_label = thrust::reduce(handle.get_thrust_policy(),
                                  starting_vertex_labels->begin(),
                                  starting_vertex_labels->end(),
                                  label_t{-1},
                                  cuda::maximum<label_t>{});

  if constexpr (multi_gpu) {
    max_label = host_scalar_allreduce(
      handle.get_comms(), max_label, raft::comms::op_t::MAX, handle.get_stream());
  }

  rmm::device_uvector<label_t> output_labels(static_cast<size_t>(max_label + 1),
                                             handle.get_stream());
  thrust::sequence(
    handle.get_thrust_policy(), output_labels.begin(), output_labels.end(), label_t{0});

  return output_labels;
}

}  // namespace detail
}  // namespace cugraph
