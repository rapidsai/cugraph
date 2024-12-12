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

#include <limits>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename priority_t>
__host__ __device__ priority_t
rank_to_priority(int rank,
                 int root,
                 int subgroup_size /* faster interconnect within a subgroup */,
                 int comm_size,
                 vertex_t offset /* to evenly distribute traffic */)
{
  static_assert(sizeof(priority_t) == 1 || sizeof(priority_t) == 2 || sizeof(priority_t) == 4);
  using cast_t = std::conditional_t<
    sizeof(priority_t) == 1,
    int16_t,
    std::conditional_t<sizeof(priority_t) == 2, int32_t, int64_t>>;  // to prevent overflow

  if (rank == root) {
    return priority_t{0};
  } else if (rank / subgroup_size ==
             root / subgroup_size) {  // intra-subgroup communication is sufficient (priorities in
                                      // [1, subgroup_size)
    auto rank_dist =
      static_cast<int>(((static_cast<cast_t>(rank) + subgroup_size) - root) % subgroup_size);
    int modulo = subgroup_size - 1;
    return static_cast<priority_t>(1 + (static_cast<cast_t>(rank_dist - 1) + (offset % modulo)) %
                                         modulo);
  } else {  // inter-subgroup communication is necessary (priorities in [subgroup_size, comm_size)
    auto subgroup_dist =
      static_cast<int>(((static_cast<cast_t>(rank / subgroup_size) + (comm_size / subgroup_size)) -
                        (root / subgroup_size)) %
                       (comm_size / subgroup_size));
    auto intra_subgroup_rank_dist = static_cast<int>(
      ((static_cast<cast_t>(rank % subgroup_size) + subgroup_size) - (root % subgroup_size)) %
      subgroup_size);
    auto rank_dist = subgroup_dist * subgroup_size + intra_subgroup_rank_dist;
    int modulo     = comm_size - subgroup_size;
    return static_cast<priority_t>(
      subgroup_size +
      (static_cast<cast_t>(rank_dist - subgroup_size) + (offset % modulo)) % modulo);
  }
}

template <typename vertex_t, typename priority_t>
__host__ __device__ int priority_to_rank(
  priority_t priority,
  int root,
  int subgroup_size /* faster interconnect within a subgroup */,
  int comm_size,
  vertex_t offset /* to evenly distribute traffict */)
{
  static_assert(sizeof(priority_t) == 1 || sizeof(priority_t) == 2 || sizeof(priority_t) == 4);
  using cast_t = std::conditional_t<
    sizeof(priority_t) == 1,
    int16_t,
    std::conditional_t<sizeof(priority_t) == 2, int32_t, int64_t>>;  // to prevent overflow

  if (priority == priority_t{0}) {
    return root;
  } else if (priority < static_cast<priority_t>(subgroup_size)) {
    int modulo     = subgroup_size - 1;
    auto rank_dist = static_cast<int>(
      1 + ((static_cast<cast_t>(priority - 1) + modulo) - (offset % modulo)) % modulo);
    return static_cast<int>((root - (root % subgroup_size)) +
                            ((static_cast<cast_t>(root) + rank_dist) % subgroup_size));
  } else {
    int modulo     = comm_size - subgroup_size;
    auto rank_dist = static_cast<int>(
      subgroup_size +
      ((static_cast<cast_t>(priority) - subgroup_size) + (modulo - (offset % modulo))) % modulo);
    auto subgroup_dist            = rank_dist / subgroup_size;
    auto intra_subgroup_rank_dist = rank_dist % subgroup_size;
    return static_cast<int>(
      ((static_cast<cast_t>((root / subgroup_size) * subgroup_size) +
        subgroup_dist * subgroup_size) +
       (static_cast<cast_t>(root) + intra_subgroup_rank_dist) % subgroup_size) %
      comm_size);
  }
}

}  // namespace detail

}  // namespace cugraph
