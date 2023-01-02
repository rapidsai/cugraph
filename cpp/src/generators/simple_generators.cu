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

#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <numeric>

namespace cugraph {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_path_graph_edgelist(raft::handle_t const& handle,
                             std::vector<std::tuple<vertex_t, vertex_t>> const& component_parms_v)
{
  size_t num_edges =
    std::transform_reduce(component_parms_v.begin(),
                          component_parms_v.end(),
                          size_t{0},
                          std::plus<size_t>(),
                          [](auto tuple) { return static_cast<size_t>(std::get<0>(tuple) - 1); });

  bool edge_off_end{false};

  if (handle.comms_initialized()) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    if (comm_size > 1) {
      if (comm_rank < comm_size) {
        num_edges += component_parms_v.size();
        edge_off_end = true;
      }
    }
  }

  rmm::device_uvector<vertex_t> d_src_v(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(num_edges, handle.get_stream());

  auto src_iterator = d_src_v.begin();
  auto dst_iterator = d_dst_v.begin();

  for (auto tuple : component_parms_v) {
    vertex_t num_vertices, base_vertex_id;
    std::tie(num_vertices, base_vertex_id) = tuple;

    vertex_t num_edges{num_vertices - 1};

    if (edge_off_end) ++num_edges;

    thrust::sequence(
      handle.get_thrust_policy(), src_iterator, src_iterator + num_edges, base_vertex_id);

    thrust::sequence(
      handle.get_thrust_policy(), dst_iterator, dst_iterator + num_edges, base_vertex_id + 1);

    src_iterator += num_edges;
    dst_iterator += num_edges;
  }

  handle.sync_stream();

  return std::make_tuple(std::move(d_src_v), std::move(d_dst_v));
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<vertex_t, vertex_t, vertex_t>> const& component_parms_v)
{
  size_t num_edges = std::transform_reduce(component_parms_v.begin(),
                                           component_parms_v.end(),
                                           size_t{0},
                                           std::plus<size_t>(),
                                           [](auto tuple) {
                                             vertex_t x, y;
                                             std::tie(x, y, std::ignore) = tuple;

                                             return ((x - 1) * y) + (x * (y - 1));
                                           });

  rmm::device_uvector<vertex_t> d_src_v(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(num_edges, handle.get_stream());

  auto output_iterator =
    thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));

  for (auto tuple : component_parms_v) {
    vertex_t x, y, base_vertex_id;
    std::tie(x, y, base_vertex_id) = tuple;

    vertex_t num_vertices = x * y;

    auto x_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(base_vertex_id),
                         thrust::make_counting_iterator<vertex_t>(base_vertex_id + 1)));

    output_iterator = thrust::copy_if(handle.get_thrust_policy(),
                                      x_iterator,
                                      x_iterator + num_vertices - 1,
                                      output_iterator,
                                      [base_vertex_id, x] __device__(auto pair) {
                                        vertex_t dst = thrust::get<1>(pair);
                                        // Want to skip if dst is in the last column of a graph
                                        return ((dst - base_vertex_id) % x) != 0;
                                      });

    auto y_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(base_vertex_id),
                         thrust::make_counting_iterator<vertex_t>(base_vertex_id + x)));

    output_iterator = thrust::copy_if(handle.get_thrust_policy(),
                                      y_iterator,
                                      y_iterator + num_vertices - x,
                                      output_iterator,
                                      [base_vertex_id, x, y] __device__(auto pair) {
                                        vertex_t dst = thrust::get<1>(pair);

                                        // Want to skip if dst is in the first row of a new graph
                                        return ((dst - base_vertex_id) % (x * y)) >= x;
                                      });
  }

  handle.sync_stream();

  return std::make_tuple(std::move(d_src_v), std::move(d_dst_v));
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<vertex_t, vertex_t, vertex_t, vertex_t>> const& component_parms_v)
{
  size_t num_edges =
    std::transform_reduce(component_parms_v.begin(),
                          component_parms_v.end(),
                          size_t{0},
                          std::plus<size_t>(),
                          [](auto tuple) {
                            vertex_t x, y, z;
                            std::tie(x, y, z, std::ignore) = tuple;

                            return ((x - 1) * y * z) + (x * (y - 1) * z) + (x * y * (z - 1));
                          });

  rmm::device_uvector<vertex_t> d_src_v(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(num_edges, handle.get_stream());

  auto output_iterator =
    thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));

  for (auto tuple : component_parms_v) {
    vertex_t x, y, z, base_vertex_id;
    std::tie(x, y, z, base_vertex_id) = tuple;

    vertex_t num_vertices = x * y * z;

    auto x_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(base_vertex_id),
                         thrust::make_counting_iterator<vertex_t>(base_vertex_id + 1)));

    output_iterator = thrust::copy_if(handle.get_thrust_policy(),
                                      x_iterator,
                                      x_iterator + num_vertices - 1,
                                      output_iterator,
                                      [base_vertex_id, x] __device__(auto pair) {
                                        vertex_t dst = thrust::get<1>(pair);
                                        // Want to skip if dst is in the last column of a graph
                                        return ((dst - base_vertex_id) % x) != 0;
                                      });

    auto y_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(base_vertex_id),
                         thrust::make_counting_iterator<vertex_t>(base_vertex_id + x)));

    output_iterator = thrust::copy_if(handle.get_thrust_policy(),
                                      y_iterator,
                                      y_iterator + num_vertices - x,
                                      output_iterator,
                                      [base_vertex_id, x, y] __device__(auto pair) {
                                        vertex_t dst = thrust::get<1>(pair);
                                        // Want to skip if dst is in the first row of a new graph
                                        return ((dst - base_vertex_id) % (x * y)) >= x;
                                      });

    auto z_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(base_vertex_id),
                         thrust::make_counting_iterator<vertex_t>(base_vertex_id + x * y)));

    output_iterator = thrust::copy_if(handle.get_thrust_policy(),
                                      z_iterator,
                                      z_iterator + num_vertices - x * y,
                                      output_iterator,
                                      [base_vertex_id, x, y, z] __device__(auto pair) {
                                        vertex_t dst = thrust::get<1>(pair);
                                        // Want to skip if dst is in the first row of a new graph
                                        return ((dst - base_vertex_id) % (x * y * z)) >= (x * y);
                                      });
  }

  handle.sync_stream();

  return std::make_tuple(std::move(d_src_v), std::move(d_dst_v));
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_complete_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<vertex_t, vertex_t>> const& component_parms_v)
{
  std::for_each(component_parms_v.begin(), component_parms_v.end(), [](auto tuple) {
    vertex_t num_vertices = std::get<0>(tuple);
    CUGRAPH_EXPECTS(num_vertices < std::numeric_limits<int32_t>::max(),
                    "Implementation cannot support specified value");
  });

  size_t num_edges = std::transform_reduce(component_parms_v.begin(),
                                           component_parms_v.end(),
                                           size_t{0},
                                           std::plus<size_t>(),
                                           [](auto tuple) {
                                             vertex_t num_vertices = std::get<0>(tuple);
                                             return num_vertices * (num_vertices - 1) / 2;
                                           });

  vertex_t invalid_vertex{std::numeric_limits<vertex_t>::max()};

  rmm::device_uvector<vertex_t> d_src_v(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(num_edges, handle.get_stream());

  auto output_iterator =
    thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));

  for (auto tuple : component_parms_v) {
    vertex_t num_vertices, base_vertex_id;
    std::tie(num_vertices, base_vertex_id) = tuple;

    auto transform_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0),
      [base_vertex_id, num_vertices, invalid_vertex] __device__(size_t index) {
        size_t graph_index = index / (num_vertices * num_vertices);
        size_t local_index = index % (num_vertices * num_vertices);

        vertex_t src = base_vertex_id + static_cast<vertex_t>(local_index / num_vertices);
        vertex_t dst = base_vertex_id + static_cast<vertex_t>(local_index % num_vertices);

        if (src == dst) {
          src = invalid_vertex;
          dst = invalid_vertex;
        } else {
          src += (graph_index * num_vertices);
          dst += (graph_index * num_vertices);
        }

        return thrust::make_tuple(src, dst);
      });

    output_iterator = thrust::copy_if(handle.get_thrust_policy(),
                                      transform_iter,
                                      transform_iter + num_vertices * num_vertices,
                                      output_iterator,
                                      [invalid_vertex] __device__(auto tuple) {
                                        auto src = thrust::get<0>(tuple);
                                        auto dst = thrust::get<1>(tuple);

                                        return (src != invalid_vertex) && (src < dst);
                                      });
  }

  handle.sync_stream();

  return std::make_tuple(std::move(d_src_v), std::move(d_dst_v));
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_path_graph_edgelist(raft::handle_t const& handle,
                             std::vector<std::tuple<int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_path_graph_edgelist(raft::handle_t const& handle,
                             std::vector<std::tuple<int64_t, int64_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int32_t, int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int64_t, int64_t, int64_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_complete_graph_edgelist(
  raft::handle_t const& handle, std::vector<std::tuple<int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_complete_graph_edgelist(
  raft::handle_t const& handle, std::vector<std::tuple<int64_t, int64_t>> const& component_parms_v);

}  // namespace cugraph
