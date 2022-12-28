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

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <random>
#include <rmm/detail/error.hpp>
#include <tuple>

namespace cugraph {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> generate_rmat_edgelist(
  raft::handle_t const& handle,
  size_t scale,
  size_t num_edges,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool clip_and_flip)
{
  CUGRAPH_EXPECTS((size_t{1} << scale) <= static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
                  "Invalid input argument: scale too large for vertex_t.");
  CUGRAPH_EXPECTS((a >= 0.0) && (b >= 0.0) && (c >= 0.0) && (a + b + c <= 1.0),
                  "Invalid input argument: a, b, c should be non-negative and a + b + c should not "
                  "be larger than 1.0.");

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

    detail::uniform_random_fill(
      handle.get_stream(), rands.data(), num_edges_to_generate * 2 * scale, 0.0f, 1.0f, seed);
    seed += num_edges_to_generate * 2 * scale;

    thrust::transform(
      handle.get_thrust_policy(),
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
        for (int bit = static_cast<int>(scale) - 1; bit >= 0; --bit) {
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
          src += src_bit_set ? static_cast<vertex_t>(vertex_t{1} << bit) : 0;
          dst += dst_bit_set ? static_cast<vertex_t>(vertex_t{1} << bit) : 0;
        }
        return thrust::make_tuple(src, dst);
      });
    num_edges_generated += num_edges_to_generate;
  }

  return std::make_tuple(std::move(srcs), std::move(dsts));
}

template <typename vertex_t>
std::vector<std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>>
generate_rmat_edgelists(raft::handle_t const& handle,
                        size_t n_edgelists,
                        size_t min_scale,
                        size_t max_scale,
                        size_t edge_factor,
                        generator_distribution_t component_distribution,
                        generator_distribution_t edge_distribution,
                        uint64_t seed,
                        bool clip_and_flip)
{
  CUGRAPH_EXPECTS(min_scale > 0, "minimum graph scale is 1.");
  CUGRAPH_EXPECTS(
    size_t{1} << max_scale <= static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
    "Invalid input argument: scale too large for vertex_t.");

  std::vector<std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>> output{};
  output.reserve(n_edgelists);
  std::vector<vertex_t> scale(n_edgelists);

  std::default_random_engine eng;
  eng.seed(seed);
  if (component_distribution == generator_distribution_t::UNIFORM) {
    std::uniform_int_distribution<vertex_t> dist(min_scale, max_scale);
    std::generate(scale.begin(), scale.end(), [&dist, &eng]() { return dist(eng); });
  } else {
    // May expose this as a parameter in the future
    std::exponential_distribution<float> dist(4);
    // The modulo is here to protect the range because exponential distribution is defined on
    // [0,infinity). With exponent 4 most values are between 0 and 1
    auto range = max_scale - min_scale;
    std::generate(scale.begin(), scale.end(), [&dist, &eng, &min_scale, &range]() {
      return min_scale + static_cast<vertex_t>(static_cast<float>(range) * dist(eng)) % range;
    });
  }

  // intialized to standard powerlaw values
  double a = 0.57, b = 0.19, c = 0.19;
  if (edge_distribution == generator_distribution_t::UNIFORM) {
    a = 0.25;
    b = a;
    c = a;
  }

  for (size_t i = 0; i < n_edgelists; i++) {
    output.push_back(generate_rmat_edgelist<vertex_t>(
      handle, scale[i], scale[i] * edge_factor, a, b, c, i, clip_and_flip));
  }
  return output;
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_rmat_edgelist<int32_t>(raft::handle_t const& handle,
                                size_t scale,
                                size_t num_edges,
                                double a,
                                double b,
                                double c,
                                uint64_t seed,
                                bool clip_and_flip);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_rmat_edgelist<int64_t>(raft::handle_t const& handle,
                                size_t scale,
                                size_t num_edges,
                                double a,
                                double b,
                                double c,
                                uint64_t seed,
                                bool clip_and_flip);

template std::vector<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
generate_rmat_edgelists<int32_t>(raft::handle_t const& handle,
                                 size_t n_edgelists,
                                 size_t min_scale,
                                 size_t max_scale,
                                 size_t edge_factor,
                                 generator_distribution_t component_distribution,
                                 generator_distribution_t edge_distribution,
                                 uint64_t seed,
                                 bool clip_and_flip);

template std::vector<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>>
generate_rmat_edgelists<int64_t>(raft::handle_t const& handle,
                                 size_t n_edgelists,
                                 size_t min_scale,
                                 size_t max_scale,
                                 size_t edge_factor,
                                 generator_distribution_t component_distribution,
                                 generator_distribution_t edge_distribution,
                                 uint64_t seed,
                                 bool clip_and_flip);

}  // namespace cugraph
