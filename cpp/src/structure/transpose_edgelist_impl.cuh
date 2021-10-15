/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cugraph/utilities/error.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <optional>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
transpose_edgelist(raft::handle_t const& handle,
                   rmm::device_uvector<vertex_t>&& edgelist_majors,
                   rmm::device_uvector<vertex_t>&& edgelist_minors,
                   std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights)
{
  std::swap(edgelist_majors, edgelist_minors);

  if constexpr (multi_gpu) {
    std::tie(edgelist_majors, edgelist_minors, edgelist_weights) =
      detail::shuffle_edgelist_by_gpu_id(handle,
                                         std::move(edgelist_majors),
                                         std::move(edgelist_minors),
                                         std::move(edgelist_weights));
  }

  return std::make_tuple(
    std::move(edgelist_majors), std::move(edgelist_minors), std::move(edgelist_weights));
}

}  // namespace detail

}  // namespace cugraph
