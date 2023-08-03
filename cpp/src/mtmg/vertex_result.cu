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
#include <cugraph/mtmg/vertex_result.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <detail/graph_partition_utils.cuh>
#include <mtmg/xxx.cuh>

#include <thrust/gather.h>

namespace cugraph {
namespace mtmg {

template <typename result_t>
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<result_t> vertex_result_t<result_t>::gather(
  handle_t const& handle,
  raft::device_span<vertex_t const> vertices,
  cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view)
{
  std::cout << "inside vertex_result_t::gather" << std::endl;

  auto my_graph_view = graph_view.get_pointer(handle);

  rmm::device_uvector<vertex_t> local_vertices(vertices.size(), handle.get_stream());
  rmm::device_uvector<int> vertex_gpu(vertices.size(), handle.get_stream());
  rmm::device_uvector<size_t> vertex_pos(vertices.size(), handle.get_stream());
  rmm::device_uvector<result_t> result(vertices.size(), handle.get_stream());

  raft::copy(local_vertices.data(), vertices.data(), vertices.size(), handle.get_stream());
  cugraph::detail::scalar_fill(
    handle.get_stream(), vertex_gpu.data(), vertex_gpu.size(), handle.get_rank());
  cugraph::detail::sequence_fill(
    handle.get_stream(), vertex_pos.data(), vertex_pos.size(), size_t{0});

  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
    my_graph_view->vertex_partition_range_lasts().size(), handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      my_graph_view->vertex_partition_range_lasts().data(),
                      my_graph_view->vertex_partition_range_lasts().size(),
                      handle.get_stream());

  auto const major_comm_size =
    handle.raft_handle().get_subcomm(cugraph::partition_manager::major_comm_name()).get_size();
  auto const minor_comm_size =
    handle.raft_handle().get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size();

  sleep(handle.raft_handle().get_comms().get_rank());
  std::cout << "first shuffle, rank = " << handle.raft_handle().get_comms().get_rank() << std::endl;
  raft::print_device_vector("from vertex_partition_range_lasts",
                            d_vertex_partition_range_lasts.data(),
                            d_vertex_partition_range_lasts.size(),
                            std::cout);

  std::forward_as_tuple(local_vertices, std::tie(vertex_gpu, vertex_pos), std::ignore) =
    groupby_gpu_id_and_shuffle_kv_pairs2(
      handle.raft_handle().get_comms(),
      local_vertices.begin(),
      local_vertices.end(),
      thrust::make_zip_iterator(vertex_gpu.begin(), vertex_pos.begin()),
      cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
        raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()),
        major_comm_size,
        minor_comm_size},
      handle.get_stream());

  //
  //  Now gather
  //
  std::cout << "after first shuffle, rank = " << handle.raft_handle().get_comms().get_rank()
            << ", local size = " << local_vertices.size() << std::endl;
  RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

  rmm::device_uvector<result_t> tmp_result(local_vertices.size(), handle.get_stream());

  auto pointer = this->get_pointer(handle);

  auto vertex_partition = vertex_partition_device_view_t<vertex_t, multi_gpu>(
    my_graph_view->local_vertex_partition_view());

  auto iter =
    thrust::make_transform_iterator(local_vertices.begin(), [vertex_partition] __device__(auto v) {
      return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
    });

  thrust::gather(handle.raft_handle().get_thrust_policy(),
                 iter,
                 iter + local_vertices.size(),
                 pointer->begin(),
                 tmp_result.begin());

  RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
  sleep(handle.raft_handle().get_comms().get_rank());
  std::cout << "second shuffle, rank = " << handle.raft_handle().get_comms().get_rank()
            << std::endl;
  raft::print_device_vector("  vertex_gpu", vertex_gpu.data(), vertex_gpu.size(), std::cout);
  raft::print_device_vector(
    "  local_vertices", local_vertices.data(), local_vertices.size(), std::cout);
  raft::print_device_vector("  vertex_pos", vertex_pos.data(), vertex_pos.size(), std::cout);
  raft::print_device_vector("  tmp_result", tmp_result.data(), tmp_result.size(), std::cout);

  //
  // Shuffle back
  //
  std::forward_as_tuple(
    std::ignore, std::tie(local_vertices, vertex_pos, tmp_result), std::ignore) =
    groupby_gpu_id_and_shuffle_kv_pairs2(
      handle.raft_handle().get_comms(),
      vertex_gpu.begin(),
      vertex_gpu.end(),
      thrust::make_zip_iterator(local_vertices.begin(), vertex_pos.begin(), tmp_result.begin()),
      [] __device__(int gpu) { return gpu; },
      handle.get_stream());

  RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
  //
  // Finally, reorder result
  //
  std::cout << "scatter, rank = " << handle.raft_handle().get_comms().get_rank() << std::endl;

  thrust::scatter(handle.raft_handle().get_thrust_policy(),
                  tmp_result.begin(),
                  tmp_result.end(),
                  vertex_pos.begin(),
                  result.begin());

  sleep(handle.raft_handle().get_comms().get_rank());
  std::cout << "after scatter, rank = " << handle.raft_handle().get_comms().get_rank() << std::endl;
  raft::print_device_vector("  vertices", vertices.data(), vertices.size(), std::cout);
  raft::print_device_vector("  result", result.data(), result.size(), std::cout);

  return result;
}

template rmm::device_uvector<float> vertex_result_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, true, false> const& graph_view);

template rmm::device_uvector<float> vertex_result_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, true, false> const& graph_view);

template rmm::device_uvector<float> vertex_result_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, true, false> const& graph_view);

template rmm::device_uvector<float> vertex_result_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, true, true> const& graph_view);

template rmm::device_uvector<float> vertex_result_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, true, true> const& graph_view);

template rmm::device_uvector<float> vertex_result_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, true, true> const& graph_view);

}  // namespace mtmg
}  // namespace cugraph
