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

#include "mg_louvain_helper.hpp"

#include <experimental/graph.hpp>

#include <utilities/device_comm.cuh>
#include <utilities/error.hpp>
#include <utilities/host_scalar_comm.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace cugraph {
namespace test {

template <typename T>
rmm::device_uvector<T> gather_distributed_vector(raft::handle_t const &handle,
                                                 T const *d_input,
                                                 size_t size)
{
  auto rx_sizes =
    cugraph::experimental::host_scalar_gather(handle.get_comms(), size, 0, handle.get_stream());
  std::vector<size_t> rx_displs(static_cast<size_t>(handle.get_comms().get_rank()) == 0
                                  ? handle.get_comms().get_size()
                                  : int{0},
                                size_t{0});
  if (static_cast<size_t>(handle.get_comms().get_rank()) == 0) {
    std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);
  }

  auto total_size = thrust::reduce(thrust::host, rx_sizes.begin(), rx_sizes.end());
  rmm::device_uvector<T> gathered_v(total_size, handle.get_stream());

  cugraph::experimental::device_gatherv(handle.get_comms(),
                                        d_input,
                                        gathered_v.data(),
                                        size,
                                        rx_sizes,
                                        rx_displs,
                                        0,
                                        handle.get_stream());

  return gathered_v;
}

template <typename vertex_t>
bool compare_renumbered_vectors(raft::handle_t const &handle,
                                rmm::device_uvector<vertex_t> const &v1,
                                rmm::device_uvector<vertex_t> const &v2)
{
  vertex_t max = 1 + thrust::reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    v1.begin(),
                                    v1.end(),
                                    vertex_t{0});

  rmm::device_uvector<size_t> map(max, cudaStream_t{nullptr});

  auto iter = thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin()));

  thrust::for_each(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   iter,
                   iter + v1.size(),
                   [d_map = map.data()] __device__(auto pair) {
                     vertex_t e1 = thrust::get<0>(pair);
                     vertex_t e2 = thrust::get<1>(pair);

                     d_map[e1] = e2;
                   });

  auto error_count =
    thrust::count_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     iter,
                     iter + v1.size(),
                     [d_map = map.data()] __device__(auto pair) {
                       vertex_t e1 = thrust::get<0>(pair);
                       vertex_t e2 = thrust::get<1>(pair);

                       return (d_map[e1] != e2);
                     });

  return (error_count == 0);
}

template <typename T>
void single_gpu_renumber_edgelist_given_number_map(raft::handle_t const &handle,
                                                   rmm::device_uvector<T> &edgelist_rows_v,
                                                   rmm::device_uvector<T> &edgelist_cols_v,
                                                   rmm::device_uvector<T> &renumber_map_gathered_v)
{
  rmm::device_uvector<T> index_v(renumber_map_gathered_v.size(), handle.get_stream());

  thrust::for_each(
    rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(renumber_map_gathered_v.size()),
    [d_renumber_map_gathered = renumber_map_gathered_v.data(), d_index = index_v.data()] __device__(
      auto idx) { d_index[d_renumber_map_gathered[idx]] = idx; });

  thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    edgelist_rows_v.begin(),
                    edgelist_rows_v.end(),
                    edgelist_rows_v.begin(),
                    [d_index = index_v.data()] __device__(auto v) { return d_index[v]; });

  thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    edgelist_cols_v.begin(),
                    edgelist_cols_v.end(),
                    edgelist_cols_v.begin(),
                    [d_index = index_v.data()] __device__(auto v) { return d_index[v]; });
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  compressed_sparse_to_edgelist(edge_t const *compressed_sparse_offsets,
                                vertex_t const *compressed_sparse_indices,
                                weight_t const *compressed_sparse_weights,
                                vertex_t major_first,
                                vertex_t major_last,
                                cudaStream_t stream)
{
  edge_t number_of_edges{0};
  raft::update_host(
    &number_of_edges, compressed_sparse_offsets + (major_last - major_first), 1, stream);
  CUDA_TRY(cudaStreamSynchronize(stream));
  rmm::device_uvector<vertex_t> edgelist_major_vertices(number_of_edges, stream);
  rmm::device_uvector<vertex_t> edgelist_minor_vertices(number_of_edges, stream);
  rmm::device_uvector<weight_t> edgelist_weights(
    compressed_sparse_weights != nullptr ? number_of_edges : 0, stream);

  // FIXME: this is highly inefficient for very high-degree vertices, for better performance, we can
  // fill high-degree vertices using one CUDA block per vertex, mid-degree vertices using one CUDA
  // warp per vertex, and low-degree vertices using one CUDA thread per block
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator(major_first),
                   thrust::make_counting_iterator(major_last),
                   [compressed_sparse_offsets,
                    major_first,
                    p_majors = edgelist_major_vertices.begin()] __device__(auto v) {
                     auto first = compressed_sparse_offsets[v - major_first];
                     auto last  = compressed_sparse_offsets[v - major_first + 1];
                     thrust::fill(thrust::seq, p_majors + first, p_majors + last, v);
                   });
  thrust::copy(rmm::exec_policy(stream)->on(stream),
               compressed_sparse_indices,
               compressed_sparse_indices + number_of_edges,
               edgelist_minor_vertices.begin());
  if (compressed_sparse_weights != nullptr) {
    thrust::copy(rmm::exec_policy(stream)->on(stream),
                 compressed_sparse_weights,
                 compressed_sparse_weights + number_of_edges,
                 edgelist_weights.data());
  }

  return std::make_tuple(std::move(edgelist_major_vertices),
                         std::move(edgelist_minor_vertices),
                         std::move(edgelist_weights));
}

template <typename vertex_t, typename weight_t>
void sort_and_coarsen_edgelist(rmm::device_uvector<vertex_t> &edgelist_major_vertices /* [INOUT] */,
                               rmm::device_uvector<vertex_t> &edgelist_minor_vertices /* [INOUT] */,
                               rmm::device_uvector<weight_t> &edgelist_weights /* [INOUT] */,
                               cudaStream_t stream)
{
  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin()));

  size_t number_of_edges{0};
  if (edgelist_weights.size() > 0) {
    thrust::sort_by_key(rmm::exec_policy(stream)->on(stream),
                        pair_first,
                        pair_first + edgelist_major_vertices.size(),
                        edgelist_weights.begin());

    rmm::device_uvector<vertex_t> tmp_edgelist_major_vertices(edgelist_major_vertices.size(),
                                                              stream);
    rmm::device_uvector<vertex_t> tmp_edgelist_minor_vertices(tmp_edgelist_major_vertices.size(),
                                                              stream);
    rmm::device_uvector<weight_t> tmp_edgelist_weights(tmp_edgelist_major_vertices.size(), stream);
    auto it = thrust::reduce_by_key(
      rmm::exec_policy(stream)->on(stream),
      pair_first,
      pair_first + edgelist_major_vertices.size(),
      edgelist_weights.begin(),
      thrust::make_zip_iterator(thrust::make_tuple(tmp_edgelist_major_vertices.begin(),
                                                   tmp_edgelist_minor_vertices.begin())),
      tmp_edgelist_weights.begin());
    number_of_edges = thrust::distance(tmp_edgelist_weights.begin(), thrust::get<1>(it));

    edgelist_major_vertices = std::move(tmp_edgelist_major_vertices);
    edgelist_minor_vertices = std::move(tmp_edgelist_minor_vertices);
    edgelist_weights        = std::move(tmp_edgelist_weights);
  } else {
    thrust::sort(rmm::exec_policy(stream)->on(stream),
                 pair_first,
                 pair_first + edgelist_major_vertices.size());
    auto it         = thrust::unique(rmm::exec_policy(stream)->on(stream),
                             pair_first,
                             pair_first + edgelist_major_vertices.size());
    number_of_edges = thrust::distance(pair_first, it);
  }

  edgelist_major_vertices.resize(number_of_edges, stream);
  edgelist_minor_vertices.resize(number_of_edges, stream);
  edgelist_weights.resize(number_of_edges, stream);
  edgelist_major_vertices.shrink_to_fit(stream);
  edgelist_minor_vertices.shrink_to_fit(stream);
  edgelist_weights.shrink_to_fit(stream);
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  compressed_sparse_to_relabeled_and_sorted_and_coarsened_edgelist(
    edge_t const *compressed_sparse_offsets,
    vertex_t const *compressed_sparse_indices,
    weight_t const *compressed_sparse_weights,
    vertex_t const *p_major_labels,
    vertex_t const *p_minor_labels,
    vertex_t major_first,
    vertex_t major_last,
    vertex_t minor_first,
    vertex_t minor_last,
    cudaStream_t stream)
{
  // FIXME: it might be possible to directly create relabled & coarsened edgelist from the
  // compressed sparse format to save memory

  rmm::device_uvector<vertex_t> edgelist_major_vertices(0, stream);
  rmm::device_uvector<vertex_t> edgelist_minor_vertices(0, stream);
  rmm::device_uvector<weight_t> edgelist_weights(0, stream);
  std::tie(edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights) =
    compressed_sparse_to_edgelist(compressed_sparse_offsets,
                                  compressed_sparse_indices,
                                  compressed_sparse_weights,
                                  major_first,
                                  major_last,
                                  stream);

  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin()));
  thrust::transform(
    rmm::exec_policy(stream)->on(stream),
    pair_first,
    pair_first + edgelist_major_vertices.size(),
    pair_first,
    [p_major_labels, p_minor_labels, major_first, minor_first] __device__(auto val) {
      return thrust::make_tuple(p_major_labels[thrust::get<0>(val) - major_first],
                                p_minor_labels[thrust::get<1>(val) - minor_first]);
    });

  sort_and_coarsen_edgelist(
    edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights, stream);

  return std::make_tuple(std::move(edgelist_major_vertices),
                         std::move(edgelist_minor_vertices),
                         std::move(edgelist_weights));
}

// single-GPU version
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
std::unique_ptr<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, false>>
coarsen_graph(
  raft::handle_t const &handle,
  cugraph::experimental::graph_view_t<vertex_t, edge_t, weight_t, store_transposed, false> const
    &graph_view,
  vertex_t const *labels)
{
  rmm::device_uvector<vertex_t> coarsened_edgelist_major_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> coarsened_edgelist_minor_vertices(0, handle.get_stream());
  rmm::device_uvector<weight_t> coarsened_edgelist_weights(0, handle.get_stream());
  std::tie(coarsened_edgelist_major_vertices,
           coarsened_edgelist_minor_vertices,
           coarsened_edgelist_weights) =
    compressed_sparse_to_relabeled_and_sorted_and_coarsened_edgelist(
      graph_view.offsets(),
      graph_view.indices(),
      graph_view.weights(),
      labels,
      labels,
      vertex_t{0},
      graph_view.get_number_of_vertices(),
      vertex_t{0},
      graph_view.get_number_of_vertices(),
      handle.get_stream());

  cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{};
  edgelist.p_src_vertices = store_transposed ? coarsened_edgelist_minor_vertices.data()
                                             : coarsened_edgelist_major_vertices.data();
  edgelist.p_dst_vertices = store_transposed ? coarsened_edgelist_major_vertices.data()
                                             : coarsened_edgelist_minor_vertices.data();
  edgelist.p_edge_weights  = coarsened_edgelist_weights.data();
  edgelist.number_of_edges = static_cast<edge_t>(coarsened_edgelist_major_vertices.size());

  vertex_t new_number_of_vertices =
    1 + thrust::reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       labels,
                       labels + graph_view.get_number_of_vertices(),
                       vertex_t{0},
                       thrust::maximum<vertex_t>());

  return std::make_unique<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, false>>(
    handle,
    edgelist,
    new_number_of_vertices,
    cugraph::experimental::graph_properties_t{
      graph_view.is_symmetric(), false, graph_view.is_weighted()},
    true);
}

// explicit instantiation

template void single_gpu_renumber_edgelist_given_number_map(
  raft::handle_t const &handle,
  rmm::device_uvector<int> &d_edgelist_rows,
  rmm::device_uvector<int> &d_edgelist_cols,
  rmm::device_uvector<int> &d_renumber_map_gathered_v);

template rmm::device_uvector<int> gather_distributed_vector(raft::handle_t const &handle,
                                                            int const *d_input,
                                                            size_t size);

template bool compare_renumbered_vectors(raft::handle_t const &handle,
                                         rmm::device_uvector<int> const &v1,
                                         rmm::device_uvector<int> const &v2);

template std::unique_ptr<cugraph::experimental::graph_t<int32_t, int32_t, float, false, false>>
coarsen_graph(
  raft::handle_t const &handle,
  cugraph::experimental::graph_view_t<int32_t, int32_t, float, false, false> const &graph_view,
  int32_t const *labels);

}  // namespace test
}  // namespace cugraph
