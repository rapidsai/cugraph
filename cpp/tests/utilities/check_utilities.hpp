/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "utilities/conversion_utilities.hpp"

#include <raft/core/handle.hpp>
#include <raft/core/span.hpp>

#include <numeric>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace test {

// check the mapping between v0 & v1 is invertible (one to one and onto)
template <typename vertex_t>
bool check_invertible(raft::host_span<vertex_t const> v0, raft::host_span<vertex_t const> v1)
{
  if (v0.size() != v1.size()) return false;

  std::map<vertex_t, vertex_t> map{};

  for (size_t i = 0; i < v0.size(); ++i) {
    auto find_it = map.find(v0[i]);
    if (find_it == map.end()) {
      map[v0[i]] = v1[i];
    } else if (find_it->second != v1[i])
      return false;  // one value in v0 is mapped to multiple distinct values in v1, so v0 to v1 is
                     // not a function
  }

  std::map<vertex_t, vertex_t> inv_map{};
  for (auto it = map.begin(); it != map.end(); ++it) {
    auto find_it = inv_map.find(it->second);
    if (find_it == inv_map.end()) {
      inv_map[it->second] = it->first;
    } else
      return false;  // multiple distinct values in v0 are mapped to one value in v1
  }

  std::vector<vertex_t> inv_v1(v1.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    auto find_it = inv_map.find(v1[i]);
    if (find_it == inv_map.end()) return false;  // elements in v0 are mapped to only a subset of v1
    inv_v1[i] = find_it->second;
  }

  return std::equal(v0.begin(), v0.end(), inv_v1.begin());
}

template <typename vertex_t>
bool check_invertible(raft::handle_t const& handle,
                      raft::device_span<vertex_t const> v0,
                      raft::device_span<vertex_t const> v1)
{
  auto v0_copy = to_host(handle, v0);
  auto v1_copy = to_host(handle, v1);

  return check_invertible(raft::host_span<vertex_t const>(v0_copy.data(), v0_copy.size()),
                          raft::host_span<vertex_t const>(v1_copy.data(), v1_copy.size()));
}

template <typename type_t>
struct nearly_equal {
  const type_t threshold_ratio;
  const type_t threshold_magnitude;

  bool operator()(type_t lhs, type_t rhs) const
  {
    return std::abs(lhs - rhs) <
           std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
  }
};

template <typename type_t>
struct device_nearly_equal {
  const type_t threshold_ratio;
  const type_t threshold_magnitude;

  bool __device__ operator()(type_t lhs, type_t rhs) const
  {
    return std::abs(lhs - rhs) <
           thrust::max(thrust::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
  }
};

}  // namespace test
}  // namespace cugraph
