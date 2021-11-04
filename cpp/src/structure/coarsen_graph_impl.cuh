/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#pragma once

#include <cugraph/detail/decompress_matrix_partition.cuh>
#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_to_adj_matrix_row_col.cuh>
#include <cugraph/prims/row_col_properties.cuh>
#include <cugraph/utilities/error.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {

namespace {

template <typename vertex_t, typename edge_t, typename weight_t>
edge_t groupby_e_and_coarsen_edgelist(vertex_t* edgelist_major_vertices /* [INOUT] */,
                                      vertex_t* edgelist_minor_vertices /* [INOUT] */,
                                      std::optional<weight_t*> edgelist_weights /* [INOUT] */,
                                      edge_t number_of_edges,
                                      cudaStream_t stream)
{
  auto pair_first =
    thrust::make_zip_iterator(thrust::make_tuple(edgelist_major_vertices, edgelist_minor_vertices));

  if (edgelist_weights) {
    thrust::sort_by_key(
      rmm::exec_policy(stream), pair_first, pair_first + number_of_edges, *edgelist_weights);

    rmm::device_uvector<vertex_t> tmp_edgelist_major_vertices(number_of_edges, stream);
    rmm::device_uvector<vertex_t> tmp_edgelist_minor_vertices(tmp_edgelist_major_vertices.size(),
                                                              stream);
    rmm::device_uvector<weight_t> tmp_edgelist_weights(tmp_edgelist_major_vertices.size(), stream);
    auto it = thrust::reduce_by_key(
      rmm::exec_policy(stream),
      pair_first,
      pair_first + number_of_edges,
      (*edgelist_weights),
      thrust::make_zip_iterator(thrust::make_tuple(tmp_edgelist_major_vertices.begin(),
                                                   tmp_edgelist_minor_vertices.begin())),
      tmp_edgelist_weights.begin());
    auto ret =
      static_cast<edge_t>(thrust::distance(tmp_edgelist_weights.begin(), thrust::get<1>(it)));

    auto edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(tmp_edgelist_major_vertices.begin(),
                                                   tmp_edgelist_minor_vertices.begin(),
                                                   tmp_edgelist_weights.begin()));
    thrust::copy(rmm::exec_policy(stream),
                 edge_first,
                 edge_first + ret,
                 thrust::make_zip_iterator(thrust::make_tuple(
                   edgelist_major_vertices, edgelist_minor_vertices, *edgelist_weights)));

    return ret;
  } else {
    thrust::sort(rmm::exec_policy(stream), pair_first, pair_first + number_of_edges);
    return static_cast<edge_t>(thrust::distance(
      pair_first,
      thrust::unique(rmm::exec_policy(stream), pair_first, pair_first + number_of_edges)));
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename AdjMatrixMinorLabelInputWrapper>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
decompress_matrix_partition_to_relabeled_and_grouped_and_coarsened_edgelist(
  raft::handle_t const& handle,
  matrix_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> const matrix_partition,
  vertex_t const* major_label_first,
  AdjMatrixMinorLabelInputWrapper const minor_label_input,
  std::optional<std::vector<vertex_t>> const& segment_offsets)
{
  static_assert(std::is_same_v<typename AdjMatrixMinorLabelInputWrapper::value_type, vertex_t>);

  // FIXME: it might be possible to directly create relabled & coarsened edgelist from the
  // compressed sparse format to save memory

  rmm::device_uvector<vertex_t> edgelist_major_vertices(matrix_partition.get_number_of_edges(),
                                                        handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minor_vertices(edgelist_major_vertices.size(),
                                                        handle.get_stream());
  auto edgelist_weights = matrix_partition.get_weights()
                            ? std::make_optional<rmm::device_uvector<weight_t>>(
                                edgelist_major_vertices.size(), handle.get_stream())
                            : std::nullopt;
  detail::decompress_matrix_partition_to_edgelist(
    handle,
    matrix_partition,
    edgelist_major_vertices.data(),
    edgelist_minor_vertices.data(),
    edgelist_weights ? std::optional<weight_t*>{(*edgelist_weights).data()} : std::nullopt,
    segment_offsets);

  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin()));
  thrust::transform(handle.get_thrust_policy(),
                    pair_first,
                    pair_first + edgelist_major_vertices.size(),
                    pair_first,
                    [major_label_first,
                     minor_label_input,
                     major_first = matrix_partition.get_major_first(),
                     minor_first = matrix_partition.get_minor_first()] __device__(auto val) {
                      return thrust::make_tuple(
                        *(major_label_first + (thrust::get<0>(val) - major_first)),
                        minor_label_input.get(thrust::get<1>(val) - minor_first));
                    });

  auto number_of_edges = groupby_e_and_coarsen_edgelist(
    edgelist_major_vertices.data(),
    edgelist_minor_vertices.data(),
    edgelist_weights ? std::optional<weight_t*>{(*edgelist_weights).data()} : std::nullopt,
    static_cast<edge_t>(edgelist_major_vertices.size()),
    handle.get_stream());
  edgelist_major_vertices.resize(number_of_edges, handle.get_stream());
  edgelist_major_vertices.shrink_to_fit(handle.get_stream());
  edgelist_minor_vertices.resize(number_of_edges, handle.get_stream());
  edgelist_minor_vertices.shrink_to_fit(handle.get_stream());
  if (edgelist_weights) {
    (*edgelist_weights).resize(number_of_edges, handle.get_stream());
    (*edgelist_weights).shrink_to_fit(handle.get_stream());
  }

  return std::make_tuple(std::move(edgelist_major_vertices),
                         std::move(edgelist_minor_vertices),
                         std::move(edgelist_weights));
}

}  // namespace

namespace detail {

// multi-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
             rmm::device_uvector<vertex_t>>>
coarsen_graph(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* labels,
  bool do_expensive_check)
{
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto const comm_rank     = comm.get_rank();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto const row_comm_rank = row_comm.get_rank();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();
  auto const col_comm_rank = col_comm.get_rank();

  if (do_expensive_check) {
    // currently, nothing to do
  }

  // 1. construct coarsened edge list

  std::conditional_t<
    store_transposed,
    row_properties_t<graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
                     vertex_t>,
    col_properties_t<graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
                     vertex_t>>
    adj_matrix_minor_labels(handle, graph_view);
  if constexpr (store_transposed) {
    copy_to_adj_matrix_row(handle, graph_view, labels, adj_matrix_minor_labels);
  } else {
    copy_to_adj_matrix_col(handle, graph_view, labels, adj_matrix_minor_labels);
  }

  std::vector<rmm::device_uvector<vertex_t>> coarsened_edgelist_major_vertices{};
  std::vector<rmm::device_uvector<vertex_t>> coarsened_edgelist_minor_vertices{};
  auto coarsened_edgelist_weights =
    graph_view.is_weighted() ? std::make_optional<std::vector<rmm::device_uvector<weight_t>>>({})
                             : std::nullopt;
  coarsened_edgelist_major_vertices.reserve(graph_view.get_number_of_local_adj_matrix_partitions());
  coarsened_edgelist_minor_vertices.reserve(coarsened_edgelist_major_vertices.size());
  if (coarsened_edgelist_weights) {
    (*coarsened_edgelist_weights).reserve(coarsened_edgelist_major_vertices.size());
  }
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    coarsened_edgelist_major_vertices.emplace_back(0, handle.get_stream());
    coarsened_edgelist_minor_vertices.emplace_back(0, handle.get_stream());
    if (coarsened_edgelist_weights) {
      (*coarsened_edgelist_weights).emplace_back(0, handle.get_stream());
    }
  }
  // FIXME: we may compare performance/memory footprint with the hash_based approach especially when
  // cuco::dynamic_map becomes available (so we don't need to preallocate memory assuming the worst
  // case). We may be able to limit the memory requirement close to the final coarsened edgelist
  // with the hash based approach.
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    // 1-1. locally construct coarsened edge list

    rmm::device_uvector<vertex_t> major_labels(
      store_transposed ? graph_view.get_number_of_local_adj_matrix_partition_cols(i)
                       : graph_view.get_number_of_local_adj_matrix_partition_rows(i),
      handle.get_stream());
    device_bcast(col_comm,
                 labels,
                 major_labels.data(),
                 major_labels.size(),
                 static_cast<int>(i),
                 handle.get_stream());

    auto [edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights] =
      decompress_matrix_partition_to_relabeled_and_grouped_and_coarsened_edgelist(
        handle,
        matrix_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu>(
          graph_view.get_matrix_partition_view(i)),
        major_labels.data(),
        adj_matrix_minor_labels.device_view(),
        graph_view.get_local_adj_matrix_partition_segment_offsets(i));

    // 1-2. globally shuffle

    std::tie(edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights) =
      cugraph::detail::shuffle_edgelist_by_gpu_id(handle,
                                                  std::move(edgelist_major_vertices),
                                                  std::move(edgelist_minor_vertices),
                                                  std::move(edgelist_weights));

    // 1-3. append data to local adjacency matrix partitions

    // FIXME: we can skip this if groupby_gpuid_and_shuffle_values is updated to return sorted edge
    // list based on the final matrix partition (maybe add
    // groupby_adj_matrix_partition_and_shuffle_values).

    auto counts = cugraph::detail::groupby_and_count_edgelist_by_local_partition_id(
      handle, edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights);

    std::vector<size_t> h_counts(counts.size());
    raft::update_host(h_counts.data(), counts.data(), counts.size(), handle.get_stream());
    handle.get_stream_view().synchronize();

    std::vector<size_t> h_displacements(h_counts.size(), size_t{0});
    std::partial_sum(h_counts.begin(), h_counts.end() - 1, h_displacements.begin() + 1);

    for (int j = 0; j < col_comm_size; ++j) {
      auto number_of_partition_edges = groupby_e_and_coarsen_edgelist(
        edgelist_major_vertices.begin() + h_displacements[j],
        edgelist_minor_vertices.begin() + h_displacements[j],
        edgelist_weights ? std::optional<weight_t*>{(*edgelist_weights).data() + h_displacements[j]}
                         : std::nullopt,
        h_counts[j],
        handle.get_stream());

      auto cur_size = coarsened_edgelist_major_vertices[j].size();
      // FIXME: this can lead to frequent costly reallocation; we may be able to avoid this if we
      // can reserve address space to avoid expensive reallocation.
      // https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management
      coarsened_edgelist_major_vertices[j].resize(cur_size + number_of_partition_edges,
                                                  handle.get_stream());
      coarsened_edgelist_minor_vertices[j].resize(coarsened_edgelist_major_vertices[j].size(),
                                                  handle.get_stream());

      if (coarsened_edgelist_weights) {
        (*coarsened_edgelist_weights)[j].resize(coarsened_edgelist_major_vertices[j].size(),
                                                handle.get_stream());

        auto src_edge_first =
          thrust::make_zip_iterator(thrust::make_tuple(edgelist_major_vertices.begin(),
                                                       edgelist_minor_vertices.begin(),
                                                       (*edgelist_weights).begin())) +
          h_displacements[j];
        auto dst_edge_first =
          thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_major_vertices[j].begin(),
                                                       coarsened_edgelist_minor_vertices[j].begin(),
                                                       (*coarsened_edgelist_weights)[j].begin())) +
          cur_size;
        thrust::copy(handle.get_thrust_policy(),
                     src_edge_first,
                     src_edge_first + number_of_partition_edges,
                     dst_edge_first);
      } else {
        auto src_edge_first = thrust::make_zip_iterator(thrust::make_tuple(
                                edgelist_major_vertices.begin(), edgelist_minor_vertices.begin())) +
                              h_displacements[j];
        auto dst_edge_first = thrust::make_zip_iterator(
                                thrust::make_tuple(coarsened_edgelist_major_vertices[j].begin(),
                                                   coarsened_edgelist_minor_vertices[j].begin())) +
                              cur_size;
        thrust::copy(handle.get_thrust_policy(),
                     src_edge_first,
                     src_edge_first + number_of_partition_edges,
                     dst_edge_first);
      }
    }
  }

  for (size_t i = 0; i < coarsened_edgelist_major_vertices.size(); ++i) {
    auto number_of_partition_edges = groupby_e_and_coarsen_edgelist(
      coarsened_edgelist_major_vertices[i].data(),
      coarsened_edgelist_minor_vertices[i].data(),
      coarsened_edgelist_weights ? std::optional<weight_t*>{(*coarsened_edgelist_weights)[i].data()}
                                 : std::nullopt,
      static_cast<edge_t>(coarsened_edgelist_major_vertices[i].size()),
      handle.get_stream());
    coarsened_edgelist_major_vertices[i].resize(number_of_partition_edges, handle.get_stream());
    coarsened_edgelist_major_vertices[i].shrink_to_fit(handle.get_stream());
    coarsened_edgelist_minor_vertices[i].resize(number_of_partition_edges, handle.get_stream());
    coarsened_edgelist_minor_vertices[i].shrink_to_fit(handle.get_stream());
    if (coarsened_edgelist_weights) {
      (*coarsened_edgelist_weights)[i].resize(number_of_partition_edges, handle.get_stream());
      (*coarsened_edgelist_weights)[i].shrink_to_fit(handle.get_stream());
    }
  }

  // 3. find unique labels for this GPU

  rmm::device_uvector<vertex_t> unique_labels(graph_view.get_number_of_local_vertices(),
                                              handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), labels, labels + unique_labels.size(), unique_labels.begin());
  thrust::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
  unique_labels.resize(
    thrust::distance(
      unique_labels.begin(),
      thrust::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end())),
    handle.get_stream());

  unique_labels = cugraph::detail::shuffle_vertices_by_gpu_id(handle, std::move(unique_labels));

  thrust::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
  unique_labels.resize(
    thrust::distance(
      unique_labels.begin(),
      thrust::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end())),
    handle.get_stream());

  // 4. renumber

  rmm::device_uvector<vertex_t> renumber_map_labels(0, handle.get_stream());
  renumber_meta_t<vertex_t, edge_t, multi_gpu> meta{};
  {
    std::vector<vertex_t*> major_ptrs(coarsened_edgelist_major_vertices.size());
    std::vector<vertex_t*> minor_ptrs(major_ptrs.size());
    std::vector<edge_t> counts(major_ptrs.size());
    for (size_t i = 0; i < coarsened_edgelist_major_vertices.size(); ++i) {
      major_ptrs[i] = coarsened_edgelist_major_vertices[i].data();
      minor_ptrs[i] = coarsened_edgelist_minor_vertices[i].data();
      counts[i]     = static_cast<edge_t>(coarsened_edgelist_major_vertices[i].size());
    }
    std::tie(renumber_map_labels, meta) = renumber_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      std::optional<rmm::device_uvector<vertex_t>>{std::move(unique_labels)},
      major_ptrs,
      minor_ptrs,
      counts,
      std::nullopt,
      do_expensive_check);
  }

  // 5. build a graph

  std::vector<edgelist_t<vertex_t, edge_t, weight_t>> edgelists{};
  edgelists.resize(graph_view.get_number_of_local_adj_matrix_partitions());
  for (size_t i = 0; i < edgelists.size(); ++i) {
    edgelists[i].p_src_vertices = store_transposed ? coarsened_edgelist_minor_vertices[i].data()
                                                   : coarsened_edgelist_major_vertices[i].data();
    edgelists[i].p_dst_vertices = store_transposed ? coarsened_edgelist_major_vertices[i].data()
                                                   : coarsened_edgelist_minor_vertices[i].data();
    edgelists[i].p_edge_weights =
      coarsened_edgelist_weights
        ? std::optional<weight_t const*>{(*coarsened_edgelist_weights)[i].data()}
        : std::nullopt,
    edgelists[i].number_of_edges = static_cast<edge_t>(coarsened_edgelist_major_vertices[i].size());
  }

  return std::make_tuple(
    std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
      handle,
      edgelists,
      graph_meta_t<vertex_t, edge_t, multi_gpu>{
        meta.number_of_vertices,
        meta.number_of_edges,
        graph_properties_t{graph_view.is_symmetric(), false},
        meta.partition,
        meta.segment_offsets,
        store_transposed ? meta.num_local_unique_edge_minors : meta.num_local_unique_edge_majors,
        store_transposed ? meta.num_local_unique_edge_majors : meta.num_local_unique_edge_minors}),
    std::move(renumber_map_labels));
}

// single-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
             rmm::device_uvector<vertex_t>>>
coarsen_graph(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* labels,
  bool do_expensive_check)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto [coarsened_edgelist_major_vertices,
        coarsened_edgelist_minor_vertices,
        coarsened_edgelist_weights] =
    decompress_matrix_partition_to_relabeled_and_grouped_and_coarsened_edgelist(
      handle,
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu>(
        graph_view.get_matrix_partition_view()),
      labels,
      detail::minor_properties_device_view_t<vertex_t, vertex_t const*>(labels),
      graph_view.get_local_adj_matrix_partition_segment_offsets(0));

  rmm::device_uvector<vertex_t> unique_labels(graph_view.get_number_of_vertices(),
                                              handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), labels, labels + unique_labels.size(), unique_labels.begin());
  thrust::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
  unique_labels.resize(
    thrust::distance(
      unique_labels.begin(),
      thrust::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end())),
    handle.get_stream());

  auto [renumber_map_labels, meta] = renumber_edgelist<vertex_t, edge_t, multi_gpu>(
    handle,
    std::optional<rmm::device_uvector<vertex_t>>{std::move(unique_labels)},
    coarsened_edgelist_major_vertices.data(),
    coarsened_edgelist_minor_vertices.data(),
    static_cast<edge_t>(coarsened_edgelist_major_vertices.size()),
    do_expensive_check);

  edgelist_t<vertex_t, edge_t, weight_t> edgelist{};
  edgelist.p_src_vertices  = store_transposed ? coarsened_edgelist_minor_vertices.data()
                                              : coarsened_edgelist_major_vertices.data();
  edgelist.p_dst_vertices  = store_transposed ? coarsened_edgelist_major_vertices.data()
                                              : coarsened_edgelist_minor_vertices.data();
  edgelist.p_edge_weights  = coarsened_edgelist_weights
                               ? std::optional<weight_t const*>{(*coarsened_edgelist_weights).data()}
                               : std::nullopt;
  edgelist.number_of_edges = static_cast<edge_t>(coarsened_edgelist_major_vertices.size());

  return std::make_tuple(
    std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
      handle,
      edgelist,
      graph_meta_t<vertex_t, edge_t, multi_gpu>{
        static_cast<vertex_t>(renumber_map_labels.size()),
        graph_properties_t{graph_view.is_symmetric(), false},
        meta.segment_offsets}),
    std::move(renumber_map_labels));
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
           rmm::device_uvector<vertex_t>>
coarsen_graph(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* labels,
  bool do_expensive_check)
{
  return detail::coarsen_graph(handle, graph_view, labels, do_expensive_check);
}

}  // namespace cugraph
