/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <rmm/detail/error.hpp>
#include <tuple>

namespace cugraph {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_bipartite_rmat_edgelist(raft::handle_t const& handle,
                                 raft::random::RngState& rng_state,
                                 size_t src_scale,
                                 size_t dst_scale,
                                 size_t num_edges,
                                 double a,
                                 double b,
                                 double c)
{
  CUGRAPH_EXPECTS(
    (size_t{1} << src_scale) <= static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
    "Invalid input argument: src_scale too large for vertex_t.");
  CUGRAPH_EXPECTS(
    (size_t{1} << dst_scale) <= static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
    "Invalid input argument: dst_scale too large for vertex_t.");
  CUGRAPH_EXPECTS((a >= 0.0) && (b >= 0.0) && (c >= 0.0) && (a + b + c <= 1.0),
                  "Invalid input argument: a, b, c should be non-negative and a + b + c should not "
                  "be larger than 1.0.");

  // to limit memory footprint (1024 is a tuning parameter)
  auto max_edges_to_generate_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * 1024;
  rmm::device_uvector<float> rands(
    std::min(num_edges, max_edges_to_generate_per_iteration) * (src_scale + dst_scale),
    handle.get_stream());

  rmm::device_uvector<vertex_t> srcs(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(num_edges, handle.get_stream());

  size_t num_edges_generated{0};
  while (num_edges_generated < num_edges) {
    auto num_edges_to_generate =
      std::min(num_edges - num_edges_generated, max_edges_to_generate_per_iteration);
    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(srcs.begin(), dsts.begin())) +
                      num_edges_generated;

    detail::uniform_random_fill(handle.get_stream(),
                                rands.data(),
                                num_edges_to_generate * (src_scale + dst_scale),
                                0.0f,
                                1.0f,
                                rng_state);

    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_edges_to_generate),
      pair_first,
      // if a + b == 0.0, a_norm is irrelevant, if (1.0 - (a+b)) == 0.0, c_norm is irrelevant
      [src_scale,
       dst_scale,
       rands    = rands.data(),
       a_plus_b = a + b,
       a_plus_c = a + c,
       a_norm   = (a + b) > 0.0 ? a / (a + b) : 0.0,
       c_norm   = (1.0 - (a + b)) > 0.0 ? c / (1.0 - (a + b)) : 0.0] __device__(auto i) {
        vertex_t src{0};
        vertex_t dst{0};
        size_t rand_offset = i * (src_scale + dst_scale);
        for (int level = 0; level < static_cast<int>(std::max(src_scale, dst_scale)); ++level) {
          auto dst_threshold = a_plus_c;
          if (level < src_scale) {
            auto r           = rands[rand_offset++];
            auto src_bit_set = r > a_plus_b;
            src +=
              src_bit_set ? static_cast<vertex_t>(vertex_t{1} << (src_scale - (level + 1))) : 0;
            dst_threshold = src_bit_set ? c_norm : a_norm;
          }
          if (level < dst_scale) {
            auto r           = rands[rand_offset++];
            auto dst_bit_set = r > dst_threshold;
            dst +=
              dst_bit_set ? static_cast<vertex_t>(vertex_t{1} << (dst_scale - (level + 1))) : 0;
          }
        }
        return thrust::make_tuple(src, dst);
      });
    num_edges_generated += num_edges_to_generate;
  }

  return std::make_tuple(std::move(srcs), std::move(dsts));
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_bipartite_rmat_edgelist<int32_t>(raft::handle_t const& handle,
                                          raft::random::RngState& rng_state,
                                          size_t src_scale,
                                          size_t dst_scale,
                                          size_t num_edges,
                                          double a,
                                          double b,
                                          double c);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_bipartite_rmat_edgelist<int64_t>(raft::handle_t const& handle,
                                          raft::random::RngState& rng_state,
                                          size_t src_scale,
                                          size_t dst_scale,
                                          size_t num_edges,
                                          double a,
                                          double b,
                                          double c);

}  // namespace cugraph
