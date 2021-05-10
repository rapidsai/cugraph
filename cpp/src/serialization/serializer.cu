/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

// Andrei Schaffer, aschaffer@nvidia.com
//

#include <serialization/serializer.hpp>

#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

namespace cugraph {
namespace serializer {
template <typename value_t>
serializer::device_byte_it serializer::serialize(raft::handle_t const& handle,
                                                 rmm::device_uvector<value_t> const& src,
                                                 serializer::device_byte_it it_dev_dest) const
{
  auto byte_buff_sz     = src.size() * sizeof(value_t);
  auto it_end           = it_dev_dest + byte_buff_sz;
  auto const* byte_buff = static_cast<value_t const*>(src.data());

  thrust::copy_n(rmm::exec_policy(handle.get_stream_view()), byte_buff, byte_buff_sz, it_dev_dest);

  return it_end;
}

// Manual template instantiations (EIDir's):
//
template serializer::device_byte_it serializer::serialize(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& src,
  serializer::device_byte_it it_dev_dest) const;

template serializer::device_byte_it serializer::serialize(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& src,
  serializer::device_byte_it it_dev_dest) const;

template serializer::device_byte_it serializer::serialize(
  raft::handle_t const& handle,
  rmm::device_uvector<float> const& src,
  serializer::device_byte_it it_dev_dest) const;

template serializer::device_byte_it serializer::serialize(
  raft::handle_t const& handle,
  rmm::device_uvector<double> const& src,
  serializer::device_byte_it it_dev_dest) const;

}  // namespace serializer
}  // namespace cugraph
