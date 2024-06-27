/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "c_api/capi_helper.hpp"
#include "structure/detail/structure_utils.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

namespace cugraph {
namespace detail {

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         raft::comms::comms_t const& comm,
                                         raft::device_span<T const> d_input)
{
  auto gathered_v = cugraph::device_allgatherv(handle, comm, d_input);

  return gathered_v;
}

}  // namespace detail
}  // namespace cugraph
