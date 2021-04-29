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

#include <graph_generators.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>

#include <raft/cudart_utils.h>

namespace cugraph {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_path_graph_edgelist(raft::handle_t const& handle, size_t num_vertices, bool symmetrize)
{
  size_t num_edges{num_vertices - 1};
  vertex_t start_vertex{0};

  if (handle.comms_initialized()) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    if (comm_size > 1) {
      if (comm_rank < comm_size) ++num_edges;

      start_vertex = static_cast<vertex_t>(comm_rank * num_vertices);
    }
  }

  if (symmetrize) num_edges *= 2;

  rmm::device_uvector<vertex_t> d_src_v(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(num_edges, handle.get_stream());

  thrust::sequence(rmm::exec_policy(handle.get_stream()),
                   d_src_v.begin(),
                   d_src_v.begin() + (num_vertices - 1),
                   start_vertex);

  thrust::sequence(rmm::exec_policy(handle.get_stream()),
                   d_dst_v.begin(),
                   d_dst_v.begin() + (num_vertices - 1),
                   start_vertex + 1);

  if (symmetrize) {
    thrust::sequence(rmm::exec_policy(handle.get_stream()),
                     d_src_v.begin() + (num_vertices - 1),
                     d_src_v.end(),
                     start_vertex + 1);

    thrust::sequence(rmm::exec_policy(handle.get_stream()),
                     d_dst_v.begin() + (num_vertices - 1),
                     d_dst_v.end(),
                     start_vertex);
  }

  handle.get_stream_view().synchronize();

  return std::make_tuple(std::move(d_src_v), std::move(d_dst_v));
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle, size_t x, size_t y, size_t num_graphs, bool symmetrize)
{
  vertex_t num_vertices = static_cast<vertex_t>(x * y * num_graphs);
  size_t num_edges      = (((x - 1) * y) + (x * (y - 1))) * num_graphs;

  vertex_t start_vertex{0};

  if (symmetrize) num_edges *= 2;

  rmm::device_uvector<vertex_t> d_src_v(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(num_edges, handle.get_stream());

  auto current_iter =
    thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));

  auto x_iterator = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex),
                       thrust::make_counting_iterator<vertex_t>(start_vertex + 1)));

  current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                 x_iterator,
                                 x_iterator + num_vertices - 1,
                                 current_iter,
                                 [x] __device__(auto pair) {
                                   vertex_t dst = thrust::get<1>(pair);
                                   // Want to skip if dst is in the last column of a graph
                                   return (dst % x) != 0;
                                 });

  if (symmetrize) {
    auto x_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex + 1),
                         thrust::make_counting_iterator<vertex_t>(start_vertex)));

    current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                   x_iterator,
                                   x_iterator + num_vertices - 1,
                                   current_iter,
                                   [x] __device__(auto pair) {
                                     vertex_t src = thrust::get<0>(pair);
                                     // Want to skip if src is in the last column of a graph
                                     return (src % x) != 0;
                                   });
  }

  auto y_iterator = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex),
                       thrust::make_counting_iterator<vertex_t>(start_vertex + x)));

  current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                 y_iterator,
                                 y_iterator + num_vertices - x,
                                 current_iter,
                                 [x, y] __device__(auto pair) {
                                   vertex_t dst = thrust::get<1>(pair);

                                   // Want to skip if dst is in the first row of a new graph
                                   return (dst % (x * y)) >= x;
                                 });

  if (symmetrize) {
    auto y_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex + x),
                         thrust::make_counting_iterator<vertex_t>(start_vertex)));

    current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                   y_iterator,
                                   y_iterator + num_vertices - x,
                                   current_iter,
                                   [x, y] __device__(auto pair) {
                                     vertex_t src = thrust::get<0>(pair);
                                     // Want to skip if src is in the first row of a new graph
                                     return (src % (x * y)) >= x;
                                   });
  }

  handle.get_stream_view().synchronize();

  return std::make_tuple(std::move(d_src_v), std::move(d_dst_v));
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle, size_t x, size_t y, size_t z, size_t num_graphs, bool symmetrize)
{
  // TODO:  Implement 3d mesh
  vertex_t num_vertices = static_cast<vertex_t>(x * y * z * num_graphs);
  size_t num_edges      = (((x - 1) * y * z) + (x * (y - 1) * z) + (x * y * (z - 1))) * num_graphs;

  vertex_t start_vertex{0};

  if (symmetrize) num_edges *= 2;

  rmm::device_uvector<vertex_t> d_src_v(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(num_edges, handle.get_stream());

  auto current_iter =
    thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));

  auto x_iterator = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex),
                       thrust::make_counting_iterator<vertex_t>(start_vertex + 1)));

  current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                 x_iterator,
                                 x_iterator + num_vertices - 1,
                                 current_iter,
                                 [x] __device__(auto pair) {
                                   vertex_t dst = thrust::get<1>(pair);
                                   // Want to skip if dst is in the last column of a graph
                                   return (dst % x) != 0;
                                 });

  if (symmetrize) {
    auto x_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex + 1),
                         thrust::make_counting_iterator<vertex_t>(start_vertex)));

    current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                   x_iterator,
                                   x_iterator + num_vertices - 1,
                                   current_iter,
                                   [x] __device__(auto pair) {
                                     vertex_t src = thrust::get<0>(pair);
                                     // Want to skip if src is in the last column of a graph
                                     return (src % x) != 0;
                                   });
  }

  auto y_iterator = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex),
                       thrust::make_counting_iterator<vertex_t>(start_vertex + x)));

  current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                 y_iterator,
                                 y_iterator + num_vertices - x,
                                 current_iter,
                                 [x, y] __device__(auto pair) {
                                   vertex_t dst = thrust::get<1>(pair);
                                   // Want to skip if dst is in the first row of a new graph
                                   return (dst % (x * y)) >= x;
                                 });

  if (symmetrize) {
    auto y_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex + x),
                         thrust::make_counting_iterator<vertex_t>(start_vertex)));

    current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                   y_iterator,
                                   y_iterator + num_vertices - x,
                                   current_iter,
                                   [x, y] __device__(auto pair) {
                                     vertex_t src = thrust::get<0>(pair);
                                     // Want to skip if src is in the first row of a new graph
                                     return (src % (x * y)) >= x;
                                   });
  }

  auto z_iterator = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex),
                       thrust::make_counting_iterator<vertex_t>(start_vertex + x * y)));

  current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                 z_iterator,
                                 z_iterator + num_vertices - x * y,
                                 current_iter,
                                 [x, y, z] __device__(auto pair) {
                                   vertex_t dst = thrust::get<1>(pair);
                                   // Want to skip if dst is in the first row of a new graph
                                   return (dst % (x * y * z)) >= (x * y);
                                 });

  if (symmetrize) {
    auto z_iterator = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator<vertex_t>(start_vertex + x * y),
                         thrust::make_counting_iterator<vertex_t>(start_vertex)));

    current_iter = thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                                   z_iterator,
                                   z_iterator + num_vertices - x * y,
                                   current_iter,
                                   [x, y, z] __device__(auto pair) {
                                     vertex_t src = thrust::get<0>(pair);
                                     // Want to skip if src is in the first row of a new graph
                                     return (src % (x * y * z)) >= (x * y);
                                   });
  }

  handle.get_stream_view().synchronize();

  return std::make_tuple(std::move(d_src_v), std::move(d_dst_v));
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_complete_graph_edgelist(raft::handle_t const& handle,
                                 size_t num_vertices,
                                 size_t num_graphs,
                                 bool symmetrize)
{
  size_t num_edges = num_vertices * (num_vertices - 1) * num_graphs;
  vertex_t start_vertex{0};
  vertex_t invalid_vertex{std::numeric_limits<vertex_t>::max()};

  if (!symmetrize) num_edges /= 2;

  if (handle.comms_initialized()) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    if (comm_size > 1) { start_vertex = static_cast<vertex_t>(comm_rank * num_vertices); }
  }

  rmm::device_uvector<vertex_t> d_src_v(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(num_edges, handle.get_stream());

  auto current_iter =
    thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));

  auto transform_iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_t>(0),
    [start_vertex, num_vertices, invalid_vertex] __device__(size_t index) {
      size_t graph_index = index / (num_vertices * num_vertices);
      size_t local_index = index % (num_vertices * num_vertices);

      vertex_t src = static_cast<vertex_t>(local_index / num_vertices);
      vertex_t dst = static_cast<vertex_t>(local_index % num_vertices);

      if (src == dst) {
        src = invalid_vertex;
        dst = invalid_vertex;
      } else {
        src += (graph_index * num_vertices);
        dst += (graph_index * num_vertices);
      }

      return thrust::make_tuple(src + start_vertex, dst + start_vertex);
    });

  thrust::copy_if(rmm::exec_policy(handle.get_stream()),
                  transform_iter,
                  transform_iter + num_vertices * num_vertices * num_graphs,
                  current_iter,
                  [symmetrize, invalid_vertex] __device__(auto tuple) {
                    auto src = thrust::get<0>(tuple);
                    auto dst = thrust::get<1>(tuple);

                    return (src != invalid_vertex) && (symmetrize || (src < dst));
                  });

 handle.get_stream_view().synchronize();

  return std::make_tuple(std::move(d_src_v), std::move(d_dst_v));
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_path_graph_edgelist(raft::handle_t const& handle, size_t num_vertices, bool symmetrize);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_path_graph_edgelist(raft::handle_t const& handle, size_t num_vertices, bool symmetrize);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle, size_t x, size_t y, size_t num_graphs, bool symmetrize);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle, size_t x, size_t y, size_t num_graphs, bool symmetrize);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle, size_t x, size_t y, size_t z, size_t num_graphs, bool symmetrize);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle, size_t x, size_t y, size_t z, size_t num_graphs, bool symmetrize);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_complete_graph_edgelist(raft::handle_t const& handle,
                                 size_t num_vertices,
                                 size_t num_graphs,
                                 bool symmetrize);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_complete_graph_edgelist(raft::handle_t const& handle,
                                 size_t num_vertices,
                                 size_t num_graphs,
                                 bool symmetrize);
}  // namespace cugraph
