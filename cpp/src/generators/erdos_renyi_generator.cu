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

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_erdos_renyi_graph_edgelist_gnp(raft::handle_t const& handle,
                                        vertex_t num_vertices,
                                        float p,
                                        vertex_t base_vertex_id,
                                        uint64_t seed)
{
  CUGRAPH_EXPECTS(num_vertices < std::numeric_limits<int32_t>::max(),
                  "Implementation cannot support specified value");

  auto random_iterator = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_t>(0), [seed] __device__(size_t index) {
      thrust::default_random_engine rng(seed);
      thrust::uniform_real_distribution<float> dist(0.0, 1.0);
      rng.discard(index);
      return dist(rng);
    });

  size_t count = thrust::count_if(handle.get_thrust_policy(),
                                  random_iterator,
                                  random_iterator + num_vertices * num_vertices,
                                  [p] __device__(float prob) { return prob < p; });

  rmm::device_uvector<size_t> indices_v(count, handle.get_stream());

  thrust::copy_if(handle.get_thrust_policy(),
                  random_iterator,
                  random_iterator + num_vertices * num_vertices,
                  indices_v.begin(),
                  [p] __device__(float prob) { return prob < p; });

  rmm::device_uvector<vertex_t> src_v(count, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(count, handle.get_stream());

  thrust::transform(handle.get_thrust_policy(),
                    indices_v.begin(),
                    indices_v.end(),
                    thrust::make_zip_iterator(thrust::make_tuple(src_v.begin(), src_v.end())),
                    [num_vertices] __device__(size_t index) {
                      size_t src = index / num_vertices;
                      size_t dst = index % num_vertices;

                      return thrust::make_tuple(static_cast<vertex_t>(src),
                                                static_cast<vertex_t>(dst));
                    });

  handle.sync_stream();

  return std::make_tuple(std::move(src_v), std::move(dst_v));
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_erdos_renyi_graph_edgelist_gnm(raft::handle_t const& handle,
                                        vertex_t num_vertices,
                                        size_t m,
                                        vertex_t base_vertex_id,
                                        uint64_t seed)
{
  CUGRAPH_FAIL("Not implemented");
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_erdos_renyi_graph_edgelist_gnp(raft::handle_t const& handle,
                                        int32_t num_vertices,
                                        float p,
                                        int32_t base_vertex_id,
                                        uint64_t seed);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_erdos_renyi_graph_edgelist_gnp(raft::handle_t const& handle,
                                        int64_t num_vertices,
                                        float p,
                                        int64_t base_vertex_id,
                                        uint64_t seed);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_erdos_renyi_graph_edgelist_gnm(raft::handle_t const& handle,
                                        int32_t num_vertices,
                                        size_t m,
                                        int32_t base_vertex_id,
                                        uint64_t seed);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_erdos_renyi_graph_edgelist_gnm(raft::handle_t const& handle,
                                        int64_t num_vertices,
                                        size_t m,
                                        int64_t base_vertex_id,
                                        uint64_t seed);

}  // namespace cugraph
