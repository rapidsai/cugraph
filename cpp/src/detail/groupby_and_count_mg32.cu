/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include "detail/graph_partition_utils.cuh"
#include "detail/groupby_and_count.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {
namespace detail {

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_edgelist_majors,
  rmm::device_uvector<int32_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_edgelist_weights,
  std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_ids,
  std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_types,
  bool groupby_and_counts_local_partition);

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_edgelist_majors,
  rmm::device_uvector<int32_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<double>>& d_edgelist_weights,
  std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_ids,
  std::optional<rmm::device_uvector<int32_t>>& d_edgelist_edge_types,
  bool groupby_and_counts_local_partition);

}  // namespace detail
}  // namespace cugraph
