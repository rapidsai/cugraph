/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "mg_louvain_helper.hpp"

#include <cugraph/graph.hpp>

#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace cugraph {
namespace test {

template <typename T>
void single_gpu_renumber_edgelist_given_number_map(raft::handle_t const& handle,
                                                   rmm::device_uvector<T>& edgelist_srcs_v,
                                                   rmm::device_uvector<T>& edgelist_dsts_v,
                                                   rmm::device_uvector<T>& renumber_map_gathered_v)
{
  rmm::device_uvector<T> index_v(renumber_map_gathered_v.size(), handle.get_stream());

  auto execution_policy = handle.get_thrust_policy();
  thrust::for_each(
    execution_policy,
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(renumber_map_gathered_v.size()),
    [d_renumber_map_gathered = renumber_map_gathered_v.data(), d_index = index_v.data()] __device__(
      auto idx) { d_index[d_renumber_map_gathered[idx]] = idx; });

  thrust::transform(execution_policy,
                    edgelist_srcs_v.begin(),
                    edgelist_srcs_v.end(),
                    edgelist_srcs_v.begin(),
                    [d_index = index_v.data()] __device__(auto v) { return d_index[v]; });

  thrust::transform(execution_policy,
                    edgelist_dsts_v.begin(),
                    edgelist_dsts_v.end(),
                    edgelist_dsts_v.begin(),
                    [d_index = index_v.data()] __device__(auto v) { return d_index[v]; });
}

// explicit instantiation

template void single_gpu_renumber_edgelist_given_number_map(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_edgelist_srcs,
  rmm::device_uvector<int32_t>& d_edgelist_dsts,
  rmm::device_uvector<int32_t>& d_renumber_map_gathered_v);

template void single_gpu_renumber_edgelist_given_number_map(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_edgelist_srcs,
  rmm::device_uvector<int64_t>& d_edgelist_dsts,
  rmm::device_uvector<int64_t>& d_renumber_map_gathered_v);

}  // namespace test
}  // namespace cugraph
