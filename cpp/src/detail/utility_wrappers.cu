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
#include <cugraph/detail/utility_wrappers.hpp>

#include <raft/random/rng.cuh>

#include <thrust/sequence.h>
#include <rmm/exec_policy.hpp>

namespace cugraph {
namespace detail {

template <typename value_t>
void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                         value_t* d_value,
                         size_t size,
                         value_t min_value,
                         value_t max_value,
                         uint64_t seed)
{
  raft::random::Rng rng(seed);
  rng.uniform<value_t, size_t>(d_value, size, min_value, max_value, stream_view.value());
}

template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                  float* d_value,
                                  size_t size,
                                  float min_value,
                                  float max_value,
                                  uint64_t seed);

template void uniform_random_fill(rmm::cuda_stream_view const& stream_view,
                                  double* d_value,
                                  size_t size,
                                  double min_value,
                                  double max_value,
                                  uint64_t seed);

template <typename value_t>
void sequence_fill(rmm::cuda_stream_view const& stream_view,
                   value_t* d_value,
                   size_t size,
                   value_t start_value)
{
  thrust::sequence(rmm::exec_policy(stream_view), d_value, d_value + size, start_value);
}

template void sequence_fill(rmm::cuda_stream_view const& stream_view,
                            int32_t* d_value,
                            size_t size,
                            int32_t start_value);

template void sequence_fill(rmm::cuda_stream_view const& stream_view,
                            int64_t* d_value,
                            size_t size,
                            int64_t start_value);

template <typename vertex_t>
vertex_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                   rmm::device_uvector<vertex_t> const& d_edgelist_rows,
                                   rmm::device_uvector<vertex_t> const& d_edgelist_cols)
{
  auto edge_first =
    thrust::make_zip_iterator(thrust::make_tuple(d_edgelist_rows.begin(), d_edgelist_cols.begin()));

  return thrust::transform_reduce(
    rmm::exec_policy(stream_view),
    edge_first,
    edge_first + d_edgelist_rows.size(),
    [] __device__(auto e) { return std::max(thrust::get<0>(e), thrust::get<1>(e)); },
    vertex_t{0},
    thrust::maximum<vertex_t>());
}

template int32_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                           rmm::device_uvector<int32_t> const& d_edgelist_rows,
                                           rmm::device_uvector<int32_t> const& d_edgelist_cols);

template int64_t compute_maximum_vertex_id(rmm::cuda_stream_view const& stream_view,
                                           rmm::device_uvector<int64_t> const& d_edgelist_rows,
                                           rmm::device_uvector<int64_t> const& d_edgelist_cols);

}  // namespace detail
}  // namespace cugraph
