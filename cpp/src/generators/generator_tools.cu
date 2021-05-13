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

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>
#include <experimental/scramble.cuh>

#include <raft/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

#include <numeric>

namespace cugraph {

namespace detail {

template <typename T>
void append_all(raft::handle_t const &handle,
                std::vector<rmm::device_uvector<T>> &&input,
                rmm::device_uvector<T> &output)
{
  auto output_iter = output.begin();

  for (size_t i = 0; i < input.size(); ++i) {
    raft::copy(output_iter, input[i].begin(), input[i].size(), handle.get_stream());
    output_iter += input[i].size();
  }
}

}  // namespace detail

template <typename vertex_t>
void scramble_vertex_ids(raft::handle_t const &handle,
                         rmm::device_uvector<vertex_t> &d_src_v,
                         rmm::device_uvector<vertex_t> &d_dst_v,
                         vertex_t vertex_id_offset,
                         uint64_t seed)
{
  vertex_t scale = raft::log2(d_src_v.size());

  auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));
  thrust::transform(rmm::exec_policy(handle.get_stream()),
                    pair_first,
                    pair_first + d_src_v.size(),
                    pair_first,
                    [scale] __device__(auto pair) {
                      return thrust::make_tuple(
                        experimental::detail::scramble(thrust::get<0>(pair), scale),
                        experimental::detail::scramble(thrust::get<1>(pair), scale));
                    });
}

template <typename vertex_t, typename weight_t>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  combine_edgelists(raft::handle_t const &handle,
                    std::vector<rmm::device_uvector<vertex_t>> &&sources,
                    std::vector<rmm::device_uvector<vertex_t>> &&dests,
                    std::optional<std::vector<rmm::device_uvector<weight_t>>> &&optional_d_weights)
{
  CUGRAPH_EXPECTS(sources.size() == dests.size(),
                  "sources and dests vertex lists must be the same size");

  if (optional_d_weights) {
    CUGRAPH_EXPECTS(sources.size() == optional_d_weights.value().size(),
                    "has_weights is specified, sources and weights must be the same size");

    thrust::for_each_n(
      thrust::host,
      thrust::make_zip_iterator(
        thrust::make_tuple(sources.begin(), dests.begin(), optional_d_weights.value().begin())),
      sources.size(),
      [](auto tuple) {
        CUGRAPH_EXPECTS(thrust::get<0>(tuple).size() != thrust::get<1>(tuple).size(),
                        "source vertex and dest vertex uvectors must be same size");
        CUGRAPH_EXPECTS(thrust::get<0>(tuple).size() != thrust::get<2>(tuple).size(),
                        "source vertex and weights uvectors must be same size");
      });
  } else {
    thrust::for_each_n(
      thrust::host,
      thrust::make_zip_iterator(thrust::make_tuple(sources.begin(), dests.begin())),
      sources.size(),
      [](auto tuple) {
        CUGRAPH_EXPECTS(thrust::get<0>(tuple).size() == thrust::get<1>(tuple).size(),
                        "source vertex and dest vertex uvectors must be same size");
      });
  }

  std::vector<rmm::device_uvector<weight_t>> d_weights;

  size_t size{0};
  for (size_t i = 0; i < sources.size(); ++i) size += sources[i].size();

  rmm::device_uvector<vertex_t> srcs_v(size, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts_v(size, handle.get_stream());
  rmm::device_uvector<weight_t> weights_v(0, handle.get_stream());

  detail::append_all<vertex_t>(handle, std::move(sources), srcs_v);
  detail::append_all<vertex_t>(handle, std::move(dests), dsts_v);

  if (optional_d_weights) {
    weights_v.resize(size, handle.get_stream());
    detail::append_all(handle, std::move(optional_d_weights.value()), weights_v);
  }

  size_t number_of_edges{0};

  if (optional_d_weights) {
    thrust::sort(
      rmm::exec_policy(handle.get_stream()),
      thrust::make_zip_iterator(
        thrust::make_tuple(srcs_v.begin(), dsts_v.begin(), weights_v.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(srcs_v.end(), dsts_v.end(), weights_v.end())));

    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(srcs_v.begin(), dsts_v.begin()));
    auto end_iter   = thrust::unique_by_key(rmm::exec_policy(handle.get_stream()),
                                          pair_first,
                                          pair_first + srcs_v.size(),
                                          weights_v.begin());

    number_of_edges = thrust::distance(pair_first, thrust::get<0>(end_iter));
  } else {
    thrust::sort(rmm::exec_policy(handle.get_stream()),
                 thrust::make_zip_iterator(thrust::make_tuple(srcs_v.begin(), dsts_v.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(srcs_v.end(), dsts_v.end())));

    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(srcs_v.begin(), dsts_v.begin()));

    auto end_iter =
      thrust::unique(rmm::exec_policy(handle.get_stream()),
                     thrust::make_zip_iterator(thrust::make_tuple(srcs_v.begin(), dsts_v.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(srcs_v.end(), dsts_v.end())));

    number_of_edges = thrust::distance(pair_first, end_iter);
  }

  srcs_v.resize(number_of_edges, handle.get_stream());
  srcs_v.shrink_to_fit(handle.get_stream());
  dsts_v.resize(number_of_edges, handle.get_stream());
  dsts_v.shrink_to_fit(handle.get_stream());

  if (optional_d_weights) {
    weights_v.resize(number_of_edges, handle.get_stream());
    weights_v.shrink_to_fit(handle.get_stream());
  }

  return std::make_tuple(std::move(srcs_v), std::move(dsts_v), std::move(weights_v));
}

template <typename vertex_t, typename weight_t>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  symmetrize_edgelist(raft::handle_t const &handle,
                      rmm::device_uvector<vertex_t> &&d_src_v,
                      rmm::device_uvector<vertex_t> &&d_dst_v,
                      std::optional<rmm::device_uvector<weight_t>> &&optional_d_weights_v)
{
  rmm::device_uvector<vertex_t> copy_src_v(d_src_v.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> copy_dst_v(d_dst_v.size(), handle.get_stream());

  raft::copy(copy_src_v.begin(), d_src_v.begin(), d_src_v.size(), handle.get_stream());
  raft::copy(copy_dst_v.begin(), d_dst_v.begin(), d_src_v.size(), handle.get_stream());

  std::vector<rmm::device_uvector<vertex_t>> srcs_v{};
  std::vector<rmm::device_uvector<vertex_t>> dsts_v{};

  srcs_v.push_back(std::move(d_src_v));
  srcs_v.push_back(std::move(copy_dst_v));

  dsts_v.push_back(std::move(d_dst_v));
  dsts_v.push_back(std::move(copy_src_v));

  if (optional_d_weights_v) {
    std::vector<rmm::device_uvector<weight_t>> weights_v{};

    rmm::device_uvector<weight_t> d_weights_v = std::move(optional_d_weights_v.value());
    rmm::device_uvector<weight_t> copy_weights_v(d_weights_v.size(), handle.get_stream());

    raft::copy(
      copy_weights_v.begin(), d_weights_v.begin(), d_weights_v.size(), handle.get_stream());

    weights_v.push_back(std::move(d_weights_v));
    weights_v.push_back(std::move(copy_weights_v));

    return combine_edgelists<vertex_t, weight_t>(
      handle, std::move(srcs_v), std::move(dsts_v), std::move(weights_v));
  } else {
    return combine_edgelists<vertex_t, weight_t>(
      handle, std::move(srcs_v), std::move(dsts_v), std::nullopt);
  }
}

template void scramble_vertex_ids(raft::handle_t const &handle,
                                  rmm::device_uvector<int32_t> &d_src_v,
                                  rmm::device_uvector<int32_t> &d_dst_v,
                                  int32_t vertex_id_offset,
                                  uint64_t seed);

template void scramble_vertex_ids(raft::handle_t const &handle,
                                  rmm::device_uvector<int64_t> &d_src_v,
                                  rmm::device_uvector<int64_t> &d_dst_v,
                                  int64_t vertex_id_offset,
                                  uint64_t seed);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
  combine_edgelists(raft::handle_t const &handle,
                    std::vector<rmm::device_uvector<int32_t>> &&sources,
                    std::vector<rmm::device_uvector<int32_t>> &&dests,
                    std::optional<std::vector<rmm::device_uvector<float>>> &&optional_d_weights);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
  combine_edgelists(raft::handle_t const &handle,
                    std::vector<rmm::device_uvector<int64_t>> &&sources,
                    std::vector<rmm::device_uvector<int64_t>> &&dests,
                    std::optional<std::vector<rmm::device_uvector<float>>> &&optional_d_weights);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
  combine_edgelists(raft::handle_t const &handle,
                    std::vector<rmm::device_uvector<int32_t>> &&sources,
                    std::vector<rmm::device_uvector<int32_t>> &&dests,
                    std::optional<std::vector<rmm::device_uvector<double>>> &&optional_d_weights);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
  combine_edgelists(raft::handle_t const &handle,
                    std::vector<rmm::device_uvector<int64_t>> &&sources,
                    std::vector<rmm::device_uvector<int64_t>> &&dests,
                    std::optional<std::vector<rmm::device_uvector<double>>> &&optional_d_weights);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
  symmetrize_edgelist(raft::handle_t const &handle,
                      rmm::device_uvector<int32_t> &&d_src_v,
                      rmm::device_uvector<int32_t> &&d_dst_v,
                      std::optional<rmm::device_uvector<float>> &&optional_d_weights_v);
template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
  symmetrize_edgelist(raft::handle_t const &handle,
                      rmm::device_uvector<int64_t> &&d_src_v,
                      rmm::device_uvector<int64_t> &&d_dst_v,
                      std::optional<rmm::device_uvector<float>> &&optional_d_weights_v);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
  symmetrize_edgelist(raft::handle_t const &handle,
                      rmm::device_uvector<int32_t> &&d_src_v,
                      rmm::device_uvector<int32_t> &&d_dst_v,
                      std::optional<rmm::device_uvector<double>> &&optional_d_weights_v);
template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
  symmetrize_edgelist(raft::handle_t const &handle,
                      rmm::device_uvector<int64_t> &&d_src_v,
                      rmm::device_uvector<int64_t> &&d_dst_v,
                      std::optional<rmm::device_uvector<double>> &&optional_d_weights_v);

}  // namespace cugraph
