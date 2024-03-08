/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "generators/scramble.cuh"

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <numeric>

namespace cugraph {

namespace detail {

template <typename T>
rmm::device_uvector<T> append_all(raft::handle_t const& handle,
                                  std::vector<rmm::device_uvector<T>>&& input)
{
  auto size = std::transform_reduce(
    input.begin(), input.end(), size_t{0}, std::plus<size_t>{}, [](auto const& element) {
      return element.size();
    });

  rmm::device_uvector<T> output(size, handle.get_stream());
  auto output_iter = output.begin();

  for (auto& element : input) {
    raft::copy(output_iter, element.begin(), element.size(), handle.get_stream());
    output_iter += element.size();
  }

  return output;
}

}  // namespace detail

template <typename vertex_t>
rmm::device_uvector<vertex_t> scramble_vertex_ids(raft::handle_t const& handle,
                                                  rmm::device_uvector<vertex_t>&& vertices,
                                                  size_t lgN)
{
  thrust::transform(handle.get_thrust_policy(),
                    vertices.begin(),
                    vertices.end(),
                    vertices.begin(),
                    [lgN] __device__(auto v) { return detail::scramble(v, lgN); });

  return std::move(vertices);
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> scramble_vertex_ids(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& srcs,
  rmm::device_uvector<vertex_t>&& dsts,
  size_t lgN)
{
  auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(srcs.begin(), dsts.begin()));
  thrust::transform(handle.get_thrust_policy(),
                    pair_first,
                    pair_first + srcs.size(),
                    pair_first,
                    [lgN] __device__(auto pair) {
                      return thrust::make_tuple(detail::scramble(thrust::get<0>(pair), lgN),
                                                detail::scramble(thrust::get<1>(pair), lgN));
                    });

  return std::make_tuple(std::move(srcs), std::move(dsts));
}

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
combine_edgelists(raft::handle_t const& handle,
                  std::vector<rmm::device_uvector<vertex_t>>&& sources,
                  std::vector<rmm::device_uvector<vertex_t>>&& dests,
                  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& optional_d_weights,
                  bool remove_multi_edges)
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

  rmm::device_uvector<vertex_t> srcs_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts_v(0, handle.get_stream());
  rmm::device_uvector<weight_t> weights_v(0, handle.get_stream());

  srcs_v = detail::append_all<vertex_t>(handle, std::move(sources));
  dsts_v = detail::append_all<vertex_t>(handle, std::move(dests));

  if (optional_d_weights) {
    weights_v = detail::append_all(handle, std::move(optional_d_weights.value()));
  }

  if (remove_multi_edges) {
    size_t number_of_edges{srcs_v.size()};

    if (optional_d_weights) {
      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(
          thrust::make_tuple(srcs_v.begin(), dsts_v.begin(), weights_v.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(srcs_v.end(), dsts_v.end(), weights_v.end())));

      auto pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(srcs_v.begin(), dsts_v.begin()));
      auto end_iter = thrust::unique_by_key(
        handle.get_thrust_policy(), pair_first, pair_first + srcs_v.size(), weights_v.begin());

      number_of_edges = thrust::distance(pair_first, thrust::get<0>(end_iter));
    } else {
      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(thrust::make_tuple(srcs_v.begin(), dsts_v.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(srcs_v.end(), dsts_v.end())));

      auto pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(srcs_v.begin(), dsts_v.begin()));

      auto end_iter = thrust::unique(
        handle.get_thrust_policy(),
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
  }

  return std::make_tuple(
    std::move(srcs_v),
    std::move(dsts_v),
    optional_d_weights
      ? std::move(std::optional<rmm::device_uvector<weight_t>>(std::move(weights_v)))
      : std::nullopt);
}

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
symmetrize_edgelist_from_triangular(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& d_src_v,
  rmm::device_uvector<vertex_t>&& d_dst_v,
  std::optional<rmm::device_uvector<weight_t>>&& optional_d_weights_v,
  bool check_diagonal)
{
  auto num_strictly_triangular_edges = d_src_v.size();
  if (check_diagonal) {
    if (optional_d_weights_v) {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(d_src_v.begin(), d_dst_v.begin(), (*optional_d_weights_v).begin()));
      auto strictly_triangular_last = thrust::partition(
        handle.get_thrust_policy(), edge_first, edge_first + d_src_v.size(), [] __device__(auto e) {
          return thrust::get<0>(e) != thrust::get<1>(e);
        });
      num_strictly_triangular_edges =
        static_cast<size_t>(thrust::distance(edge_first, strictly_triangular_last));
    } else {
      auto edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));
      auto strictly_triangular_last = thrust::partition(
        handle.get_thrust_policy(), edge_first, edge_first + d_src_v.size(), [] __device__(auto e) {
          return thrust::get<0>(e) != thrust::get<1>(e);
        });
      num_strictly_triangular_edges =
        static_cast<size_t>(thrust::distance(edge_first, strictly_triangular_last));
    }
  }

  auto offset = d_src_v.size();
  d_src_v.resize(offset + num_strictly_triangular_edges, handle.get_stream());
  d_dst_v.resize(offset + num_strictly_triangular_edges, handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               d_dst_v.begin(),
               d_dst_v.begin() + num_strictly_triangular_edges,
               d_src_v.begin() + offset);
  thrust::copy(handle.get_thrust_policy(),
               d_src_v.begin(),
               d_src_v.begin() + num_strictly_triangular_edges,
               d_dst_v.begin() + offset);
  if (optional_d_weights_v) {
    optional_d_weights_v->resize(d_src_v.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 optional_d_weights_v->begin(),
                 optional_d_weights_v->begin() + num_strictly_triangular_edges,
                 optional_d_weights_v->begin() + offset);
  }

  return std::make_tuple(std::move(d_src_v),
                         std::move(d_dst_v),
                         optional_d_weights_v ? std::move(optional_d_weights_v) : std::nullopt);
}

template rmm::device_uvector<int32_t> scramble_vertex_ids(raft::handle_t const& handle,
                                                          rmm::device_uvector<int32_t>&& vertices,
                                                          size_t lgN);

template rmm::device_uvector<int64_t> scramble_vertex_ids(raft::handle_t const& handle,
                                                          rmm::device_uvector<int64_t>&& vertices,
                                                          size_t lgN);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> scramble_vertex_ids(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& srcs,
  rmm::device_uvector<int32_t>&& dsts,
  size_t lgN);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> scramble_vertex_ids(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& srcs,
  rmm::device_uvector<int64_t>&& dsts,
  size_t lgN);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
combine_edgelists(raft::handle_t const& handle,
                  std::vector<rmm::device_uvector<int32_t>>&& sources,
                  std::vector<rmm::device_uvector<int32_t>>&& dests,
                  std::optional<std::vector<rmm::device_uvector<float>>>&& optional_d_weights,
                  bool remove_multi_edges);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
combine_edgelists(raft::handle_t const& handle,
                  std::vector<rmm::device_uvector<int64_t>>&& sources,
                  std::vector<rmm::device_uvector<int64_t>>&& dests,
                  std::optional<std::vector<rmm::device_uvector<float>>>&& optional_d_weights,
                  bool remove_multi_edges);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
combine_edgelists(raft::handle_t const& handle,
                  std::vector<rmm::device_uvector<int32_t>>&& sources,
                  std::vector<rmm::device_uvector<int32_t>>&& dests,
                  std::optional<std::vector<rmm::device_uvector<double>>>&& optional_d_weights,
                  bool remove_multi_edges);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
combine_edgelists(raft::handle_t const& handle,
                  std::vector<rmm::device_uvector<int64_t>>&& sources,
                  std::vector<rmm::device_uvector<int64_t>>&& dests,
                  std::optional<std::vector<rmm::device_uvector<double>>>&& optional_d_weights,
                  bool remove_multi_edges);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
symmetrize_edgelist_from_triangular(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_src_v,
  rmm::device_uvector<int32_t>&& d_dst_v,
  std::optional<rmm::device_uvector<float>>&& optional_d_weights_v,
  bool check_diagonal);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
symmetrize_edgelist_from_triangular(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_src_v,
  rmm::device_uvector<int64_t>&& d_dst_v,
  std::optional<rmm::device_uvector<float>>&& optional_d_weights_v,
  bool check_diagonal);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
symmetrize_edgelist_from_triangular(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_src_v,
  rmm::device_uvector<int32_t>&& d_dst_v,
  std::optional<rmm::device_uvector<double>>&& optional_d_weights_v,
  bool check_diagonal);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
symmetrize_edgelist_from_triangular(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_src_v,
  rmm::device_uvector<int64_t>&& d_dst_v,
  std::optional<rmm::device_uvector<double>>&& optional_d_weights_v,
  bool check_diagonal);

}  // namespace cugraph
