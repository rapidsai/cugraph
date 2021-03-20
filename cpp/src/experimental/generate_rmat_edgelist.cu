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

#include <experimental/scramble.cuh>

#include <experimental/graph_generator.hpp>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <raft/random/rng.cuh>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {
namespace experimental {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> generate_rmat_edgelist(
  raft::handle_t const& handle,
  size_t scale,
  size_t num_edges,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool clip_and_flip,
  bool scramble_vertex_ids)
{
  CUGRAPH_EXPECTS((size_t{1} << scale) <= static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
                  "Invalid input argument: scale too large for vertex_t.");
  CUGRAPH_EXPECTS((a >= 0.0) && (b >= 0.0) && (c >= 0.0) && (a + b + c <= 1.0),
                  "Invalid input argument: a, b, c should be non-negative and a + b + c should not "
                  "be larger than 1.0.");

  raft::random::Rng rng(seed);
  // to limit memory footprint (1024 is a tuning parameter)
  auto max_edges_to_generate_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * 1024;
  rmm::device_uvector<float> rands(
    std::min(num_edges, max_edges_to_generate_per_iteration) * 2 * scale, handle.get_stream());

  rmm::device_uvector<vertex_t> srcs(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(num_edges, handle.get_stream());

  size_t num_edges_generated{0};
  while (num_edges_generated < num_edges) {
    auto num_edges_to_generate =
      std::min(num_edges - num_edges_generated, max_edges_to_generate_per_iteration);
    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(srcs.begin(), dsts.begin())) +
                      num_edges_generated;
    rng.uniform<float, size_t>(
      rands.data(), num_edges_to_generate * 2 * scale, 0.0f, 1.0f, handle.get_stream());
    thrust::transform(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_edges_to_generate),
      pair_first,
      // if a + b == 0.0, a_norm is irrelevant, if (1.0 - (a+b)) == 0.0, c_norm is irrelevant
      [scale,
       clip_and_flip,
       rands    = rands.data(),
       a_plus_b = a + b,
       a_norm   = (a + b) > 0.0 ? a / (a + b) : 0.0,
       c_norm   = (1.0 - (a + b)) > 0.0 ? c / (1.0 - (a + b)) : 0.0] __device__(auto i) {
        vertex_t src{0};
        vertex_t dst{0};
        for (size_t bit = scale - 1; bit != 0; --bit) {
          auto r0          = rands[i * 2 * scale + 2 * bit];
          auto r1          = rands[i * 2 * scale + 2 * bit + 1];
          auto src_bit_set = r0 > a_plus_b;
          auto dst_bit_set = r1 > (src_bit_set ? c_norm : a_norm);
          if (clip_and_flip) {
            if (src == dst) {
              if (!src_bit_set && dst_bit_set) {
                src_bit_set = !src_bit_set;
                dst_bit_set = !dst_bit_set;
              }
            }
          }
          src += src_bit_set ? static_cast<vertex_t>(1 << bit) : 0;
          dst += dst_bit_set ? static_cast<vertex_t>(1 << bit) : 0;
        }
        return thrust::make_tuple(src, dst);
      });
    num_edges_generated += num_edges_to_generate;
  }

  if (scramble_vertex_ids) {
    rands.resize(0, handle.get_stream());
    rands.shrink_to_fit(handle.get_stream());

    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(srcs.begin(), dsts.begin()));
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      pair_first,
                      pair_first + srcs.size(),
                      pair_first,
                      [scale] __device__(auto pair) {
                        return thrust::make_tuple(detail::scramble(thrust::get<0>(pair), scale),
                                                  detail::scramble(thrust::get<1>(pair), scale));
                      });
  }

  return std::make_tuple(std::move(srcs), std::move(dsts));
}

// explicit instantiation

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_rmat_edgelist<int32_t>(raft::handle_t const& handle,
                                size_t scale,
                                size_t num_edges,
                                double a,
                                double b,
                                double c,
                                uint64_t seed,
                                bool clip_and_flip,
                                bool scramble_vertex_ids);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_rmat_edgelist<int64_t>(raft::handle_t const& handle,
                                size_t scale,
                                size_t num_edges,
                                double a,
                                double b,
                                double c,
                                uint64_t seed,
                                bool clip_and_flip,
                                bool scramble_vertex_ids);

}  // namespace experimental
}  // namespace cugraph
