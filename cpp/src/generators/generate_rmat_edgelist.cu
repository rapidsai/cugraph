/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> generate_rmat_edgelist(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  size_t scale,
  size_t num_edges,
  double a,
  double b,
  double c,
  bool clip_and_flip,
  bool scramble_vertex_ids)
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
      handle.get_stream(), rands.data(), num_edges_to_generate * 2 * scale, 0.0f, 1.0f, rng_state);

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

  if (scramble_vertex_ids) {
    return cugraph::scramble_vertex_ids<vertex_t>(handle, std::move(srcs), std::move(dsts), scale);
  } else {
    return std::make_tuple(std::move(srcs), std::move(dsts));
  }
}

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
  raft::random::RngState rng_state(seed);

  return generate_rmat_edgelist<vertex_t>(
    handle, rng_state, scale, num_edges, a, b, c, clip_and_flip, scramble_vertex_ids);
}

template <typename vertex_t>
std::vector<std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>>
generate_rmat_edgelists(raft::handle_t const& handle,
                        raft::random::RngState& rng_state,
                        size_t n_edgelists,
                        size_t min_scale,
                        size_t max_scale,
                        size_t edge_factor,
                        generator_distribution_t size_distribution,
                        generator_distribution_t edge_distribution,
                        bool clip_and_flip,
                        bool scramble_vertex_ids)
{
  CUGRAPH_EXPECTS(min_scale > 0, "minimum graph scale is 1.");
  CUGRAPH_EXPECTS(
    size_t{1} << max_scale <= static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
    "Invalid input argument: scale too large for vertex_t.");

  std::vector<std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>> output{};
  output.reserve(n_edgelists);
  std::vector<vertex_t> scale(n_edgelists);
  rmm::device_uvector<vertex_t> d_scale(n_edgelists, handle.get_stream());

  if (size_distribution == generator_distribution_t::UNIFORM) {
    detail::uniform_random_fill(handle.get_stream(),
                                d_scale.data(),
                                d_scale.size(),
                                static_cast<vertex_t>(min_scale),
                                static_cast<vertex_t>(max_scale),
                                rng_state);
  } else {
    // May expose lambda as a parameter in the future
    rmm::device_uvector<float> rand(n_edgelists, handle.get_stream());
    raft::random::exponential(handle, rng_state, rand.data(), rand.size(), float{4});
    // The modulo is here to protect the range because exponential distribution is defined on
    // [0,infinity). With exponent 4 most values are between 0 and 1
    auto range = max_scale - min_scale;
    thrust::transform(handle.get_thrust_policy(),
                      rand.begin(),
                      rand.end(),
                      d_scale.begin(),
                      [min_scale, range] __device__(auto rnd) {
                        return min_scale +
                               static_cast<vertex_t>(static_cast<float>(range) * rnd) % range;
                      });
  }

  raft::update_host(scale.data(), d_scale.data(), d_scale.size(), handle.get_stream());

  // intialized to standard powerlaw values
  double a = 0.57, b = 0.19, c = 0.19;
  if (edge_distribution == generator_distribution_t::UNIFORM) {
    a = 0.25;
    b = a;
    c = a;
  }

  for (size_t i = 0; i < n_edgelists; i++) {
    output.push_back(generate_rmat_edgelist<vertex_t>(handle,
                                                      rng_state,
                                                      scale[i],
                                                      scale[i] * edge_factor,
                                                      a,
                                                      b,
                                                      c,
                                                      clip_and_flip,
                                                      scramble_vertex_ids));
  }
  return output;
}

template <typename vertex_t>
std::vector<std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>>
generate_rmat_edgelists(raft::handle_t const& handle,
                        size_t n_edgelists,
                        size_t min_scale,
                        size_t max_scale,
                        size_t edge_factor,
                        generator_distribution_t size_distribution,
                        generator_distribution_t edge_distribution,
                        uint64_t seed,
                        bool clip_and_flip,
                        bool scramble_vertex_ids)
{
  raft::random::RngState rng_state(seed);

  return generate_rmat_edgelists<vertex_t>(handle,
                                           rng_state,
                                           n_edgelists,
                                           min_scale,
                                           max_scale,
                                           edge_factor,
                                           size_distribution,
                                           edge_distribution,
                                           clip_and_flip,
                                           scramble_vertex_ids);
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_rmat_edgelist<int32_t>(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                size_t scale,
                                size_t num_edges,
                                double a,
                                double b,
                                double c,
                                bool clip_and_flip,
                                bool scramble_vertex_ids);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_rmat_edgelist<int64_t>(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                size_t scale,
                                size_t num_edges,
                                double a,
                                double b,
                                double c,
                                bool clip_and_flip,
                                bool scramble_vertex_ids);

template std::vector<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
generate_rmat_edgelists<int32_t>(raft::handle_t const& handle,
                                 raft::random::RngState& rng_state,
                                 size_t n_edgelists,
                                 size_t min_scale,
                                 size_t max_scale,
                                 size_t edge_factor,
                                 generator_distribution_t size_distribution,
                                 generator_distribution_t edge_distribution,
                                 bool clip_and_flip,
                                 bool scramble_vertex_ids);

template std::vector<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>>
generate_rmat_edgelists<int64_t>(raft::handle_t const& handle,
                                 raft::random::RngState& rng_state,
                                 size_t n_edgelists,
                                 size_t min_scale,
                                 size_t max_scale,
                                 size_t edge_factor,
                                 generator_distribution_t size_distribution,
                                 generator_distribution_t edge_distribution,
                                 bool clip_and_flip,
                                 bool scramble_vertex_ids);

/* DEPRECATED */
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

template std::vector<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
generate_rmat_edgelists<int32_t>(raft::handle_t const& handle,
                                 size_t n_edgelists,
                                 size_t min_scale,
                                 size_t max_scale,
                                 size_t edge_factor,
                                 generator_distribution_t size_distribution,
                                 generator_distribution_t edge_distribution,
                                 uint64_t seed,
                                 bool clip_and_flip,
                                 bool scramble_vertex_ids);

template std::vector<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>>
generate_rmat_edgelists<int64_t>(raft::handle_t const& handle,
                                 size_t n_edgelists,
                                 size_t min_scale,
                                 size_t max_scale,
                                 size_t edge_factor,
                                 generator_distribution_t size_distribution,
                                 generator_distribution_t edge_distribution,
                                 uint64_t seed,
                                 bool clip_and_flip,
                                 bool scramble_vertex_ids);

}  // namespace cugraph
