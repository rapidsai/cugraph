/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <detail/graph_utils.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/cython.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/graph_traits.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/path_retrieval.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/handle.hpp>
#include <raft/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

#include <numeric>
#include <vector>

namespace cugraph {
namespace cython {

namespace detail {

// workaround for CUDA extended lambda restrictions
template <typename vertex_t>
struct compute_local_partition_id_t {
  vertex_t const* lasts{nullptr};
  size_t num_local_partitions{0};

  __device__ size_t operator()(vertex_t v)
  {
    for (size_t i = 0; i < num_local_partitions; ++i) {
      if (v < lasts[i]) { return i; }
    }
    return num_local_partitions;
  }
};

// FIXME: this is unnecessary if edge_counts_ in the major_minor_weights_t object returned by
// call_shuffle() is passed back, better be fixed. this code assumes that the entire set of edges
// for each partition are consecutively stored.
template <typename vertex_t, typename edge_t, bool transposed>
std::vector<edge_t> compute_edge_counts(raft::handle_t const& handle,
                                        graph_container_t const& graph_container)
{
  auto num_local_partitions = static_cast<size_t>(graph_container.col_comm_size);

  std::vector<vertex_t> partition_offsets_vector(
    reinterpret_cast<vertex_t*>(graph_container.vertex_partition_offsets),
    reinterpret_cast<vertex_t*>(graph_container.vertex_partition_offsets) +
      (graph_container.row_comm_size * graph_container.col_comm_size) + 1);

  std::vector<vertex_t> h_lasts(num_local_partitions);
  for (size_t i = 0; i < h_lasts.size(); ++i) {
    h_lasts[i] = partition_offsets_vector[graph_container.row_comm_size * (i + 1)];
  }
  rmm::device_uvector<vertex_t> d_lasts(h_lasts.size(), handle.get_stream());
  raft::update_device(d_lasts.data(), h_lasts.data(), h_lasts.size(), handle.get_stream());
  auto major_vertices = transposed
                          ? reinterpret_cast<vertex_t const*>(graph_container.dst_vertices)
                          : reinterpret_cast<vertex_t const*>(graph_container.src_vertices);
  auto key_first      = thrust::make_transform_iterator(
    major_vertices, compute_local_partition_id_t<vertex_t>{d_lasts.data(), num_local_partitions});
  rmm::device_uvector<size_t> d_local_partition_ids(num_local_partitions, handle.get_stream());
  rmm::device_uvector<edge_t> d_edge_counts(d_local_partition_ids.size(), handle.get_stream());
  auto it = thrust::reduce_by_key(handle.get_thrust_policy(),
                                  key_first,
                                  key_first + graph_container.num_local_edges,
                                  thrust::make_constant_iterator(edge_t{1}),
                                  d_local_partition_ids.begin(),
                                  d_edge_counts.begin());
  if (static_cast<size_t>(thrust::distance(d_local_partition_ids.begin(), thrust::get<0>(it))) <
      num_local_partitions) {
    rmm::device_uvector<edge_t> d_counts(num_local_partitions, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), d_counts.begin(), d_counts.end(), edge_t{0});
    thrust::scatter(handle.get_thrust_policy(),
                    d_edge_counts.begin(),
                    thrust::get<1>(it),
                    d_local_partition_ids.begin(),
                    d_counts.begin());
    d_edge_counts = std::move(d_counts);
  }
  std::vector<edge_t> h_edge_counts(num_local_partitions, 0);
  raft::update_host(
    h_edge_counts.data(), d_edge_counts.data(), d_edge_counts.size(), handle.get_stream());
  handle.sync_stream();

  return h_edge_counts;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool transposed,
          bool multi_gpu,
          std::enable_if_t<multi_gpu>* = nullptr>
std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>> create_graph(
  raft::handle_t const& handle, graph_container_t const& graph_container)
{
  CUGRAPH_EXPECTS(graph_container.segment_offsets != nullptr,
                  "graph_container.segment_offsets shouldn't be nullptr in multi-GPU.");

  auto num_local_partitions = static_cast<size_t>(graph_container.col_comm_size);

  std::vector<vertex_t> partition_offsets_vector(
    reinterpret_cast<vertex_t*>(graph_container.vertex_partition_offsets),
    reinterpret_cast<vertex_t*>(graph_container.vertex_partition_offsets) +
      (graph_container.row_comm_size * graph_container.col_comm_size) + 1);

  auto edge_counts = compute_edge_counts<vertex_t, edge_t, transposed>(handle, graph_container);

  std::vector<edge_t> displacements(edge_counts.size(), 0);
  std::partial_sum(edge_counts.begin(), edge_counts.end() - 1, displacements.begin() + 1);

  std::vector<cugraph::edgelist_t<vertex_t, edge_t, weight_t>> edgelists(num_local_partitions);
  for (size_t i = 0; i < edgelists.size(); ++i) {
    edgelists[i] = cugraph::edgelist_t<vertex_t, edge_t, weight_t>{
      raft::device_span<vertex_t const>(
        reinterpret_cast<vertex_t const*>(graph_container.src_vertices) + displacements[i],
        edge_counts[i]),
      raft::device_span<vertex_t const>(
        reinterpret_cast<vertex_t const*>(graph_container.dst_vertices) + displacements[i],
        edge_counts[i]),
      graph_container.is_weighted
        ? std::make_optional<raft::device_span<weight_t const>>(
            reinterpret_cast<weight_t const*>(graph_container.weights) + displacements[i],
            edge_counts[i])
        : std::nullopt};
  }

  partition_t<vertex_t> partition(partition_offsets_vector,
                                  graph_container.row_comm_size,
                                  graph_container.col_comm_size,
                                  graph_container.row_comm_rank,
                                  graph_container.col_comm_rank);

  return std::make_unique<graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>(
    handle,
    edgelists,
    graph_meta_t<vertex_t, edge_t, multi_gpu>{
      static_cast<vertex_t>(graph_container.num_global_vertices),
      static_cast<edge_t>(graph_container.num_global_edges),
      graph_container.graph_props,
      partition,
      std::vector<vertex_t>(static_cast<vertex_t const*>(graph_container.segment_offsets),
                            static_cast<vertex_t const*>(graph_container.segment_offsets) +
                              graph_container.num_segments + 1),
      // FIXME: disable (key, value) pairs at this moment (should be enabled once fully tuned).
      std::numeric_limits<vertex_t>::max(),
      std::numeric_limits<vertex_t>::max()},
    graph_container.do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool transposed,
          bool multi_gpu,
          std::enable_if_t<!multi_gpu>* = nullptr>
std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>> create_graph(
  raft::handle_t const& handle, graph_container_t const& graph_container)
{
  edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    raft::device_span<vertex_t const>(
      reinterpret_cast<vertex_t const*>(graph_container.src_vertices),
      graph_container.num_local_edges),
    raft::device_span<vertex_t const>(
      reinterpret_cast<vertex_t const*>(graph_container.dst_vertices),
      graph_container.num_local_edges),
    graph_container.is_weighted ? std::make_optional<raft::device_span<weight_t const>>(
                                    reinterpret_cast<weight_t const*>(graph_container.weights),
                                    graph_container.num_local_edges)
                                : std::nullopt};
  return std::make_unique<graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>(
    handle,
    edgelist,
    graph_meta_t<vertex_t, edge_t, multi_gpu>{
      static_cast<vertex_t>(graph_container.num_global_vertices),
      graph_container.graph_props,
      graph_container.segment_offsets != nullptr
        ? std::make_optional<std::vector<vertex_t>>(
            static_cast<vertex_t const*>(graph_container.segment_offsets),
            static_cast<vertex_t const*>(graph_container.segment_offsets) +
              graph_container.num_segments + 1)
        : std::nullopt},
    graph_container.do_expensive_check);
}

}  // namespace detail

// Populates a graph_container_t with a pointer to a new graph object and sets
// the meta-data accordingly.  The graph container owns the pointer and it is
// assumed it will delete it on destruction.
void populate_graph_container(graph_container_t& graph_container,
                              raft::handle_t& handle,
                              void* src_vertices,
                              void* dst_vertices,
                              void* weights,
                              void* vertex_partition_offsets,
                              void* segment_offsets,
                              size_t num_segments,
                              numberTypeEnum vertexType,
                              numberTypeEnum edgeType,
                              numberTypeEnum weightType,
                              size_t num_local_edges,
                              size_t num_global_vertices,
                              size_t num_global_edges,
                              bool is_weighted,
                              bool is_symmetric,
                              bool transposed,
                              bool multi_gpu)
{
  CUGRAPH_EXPECTS(graph_container.graph_type == graphTypeEnum::null,
                  "populate_graph_container() can only be called on an empty container.");

  bool do_expensive_check{false};

  if (multi_gpu) {
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();  // pcols
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();  // prows
    graph_container.row_comm_size = row_comm_size;
    graph_container.col_comm_size = col_comm_size;
    graph_container.row_comm_rank = row_comm_rank;
    graph_container.col_comm_rank = col_comm_rank;
  }

  graph_container.src_vertices             = src_vertices;
  graph_container.dst_vertices             = dst_vertices;
  graph_container.weights                  = weights;
  graph_container.is_weighted              = is_weighted;
  graph_container.vertex_partition_offsets = vertex_partition_offsets;
  graph_container.segment_offsets          = segment_offsets;
  graph_container.num_segments             = num_segments;
  graph_container.num_local_edges          = num_local_edges;
  graph_container.num_global_vertices      = num_global_vertices;
  graph_container.num_global_edges         = num_global_edges;
  graph_container.vertexType               = vertexType;
  graph_container.edgeType                 = edgeType;
  graph_container.weightType               = weightType;
  graph_container.transposed               = transposed;
  graph_container.is_multi_gpu             = multi_gpu;
  graph_container.do_expensive_check       = do_expensive_check;

  graph_properties_t graph_props{.is_symmetric = is_symmetric, .is_multigraph = false};
  graph_container.graph_props = graph_props;

  graph_container.graph_type = graphTypeEnum::graph_t;
}

// Wrapper for calling extract_egonet through a graph container
// FIXME : this should not be a legacy COO and it is not clear how to handle C++ api return type as
// is.graph_container Need to figure out how to return edge lists
template <typename vertex_t, typename weight_t>
std::unique_ptr<cy_multi_edgelists_t> call_egonet(raft::handle_t const& handle,
                                                  graph_container_t const& graph_container,
                                                  vertex_t* source_vertex,
                                                  vertex_t n_subgraphs,
                                                  vertex_t radius)
{
  if (graph_container.edgeType == numberTypeEnum::int32Type) {
    auto graph =
      detail::create_graph<int32_t, int32_t, weight_t, false, false>(handle, graph_container);
    auto g = cugraph::extract_ego(handle,
                                  graph->view(),
                                  reinterpret_cast<int32_t*>(source_vertex),
                                  static_cast<int32_t>(n_subgraphs),
                                  static_cast<int32_t>(radius));
    cy_multi_edgelists_t coo_contents{
      0,  // not used
      std::get<0>(g).size(),
      static_cast<size_t>(n_subgraphs),
      std::make_unique<rmm::device_buffer>(std::get<0>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<1>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<2>(g)
                                             ? (*std::get<2>(g)).release()
                                             : rmm::device_buffer(size_t{0}, handle.get_stream())),
      std::make_unique<rmm::device_buffer>(std::get<3>(g).release())};
    return std::make_unique<cy_multi_edgelists_t>(std::move(coo_contents));
  } else if (graph_container.edgeType == numberTypeEnum::int64Type) {
    auto graph =
      detail::create_graph<vertex_t, int64_t, weight_t, false, false>(handle, graph_container);
    auto g = cugraph::extract_ego(handle,
                                  graph->view(),
                                  reinterpret_cast<vertex_t*>(source_vertex),
                                  static_cast<vertex_t>(n_subgraphs),
                                  static_cast<vertex_t>(radius));
    cy_multi_edgelists_t coo_contents{
      0,  // not used
      std::get<0>(g).size(),
      static_cast<size_t>(n_subgraphs),
      std::make_unique<rmm::device_buffer>(std::get<0>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<1>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<2>(g)
                                             ? (*std::get<2>(g)).release()
                                             : rmm::device_buffer(size_t{0}, handle.get_stream())),
      std::make_unique<rmm::device_buffer>(std::get<3>(g).release())};
    return std::make_unique<cy_multi_edgelists_t>(std::move(coo_contents));
  } else {
    CUGRAPH_FAIL("vertexType/edgeType combination unsupported");
  }
}
// Wrapper for graph generate_rmat_edgelist()
// to expose the API to cython
// enum class generator_distribution_t { POWER_LAW = 0, UNIFORM };
template <typename vertex_t>
std::unique_ptr<graph_generator_t> call_generate_rmat_edgelist(raft::handle_t const& handle,
                                                               size_t scale,
                                                               size_t num_edges,
                                                               double a,
                                                               double b,
                                                               double c,
                                                               uint64_t seed,
                                                               bool clip_and_flip,
                                                               bool scramble_vertex_ids)
{
  auto src_dst_tuple = cugraph::generate_rmat_edgelist<vertex_t>(
    handle, scale, num_edges, a, b, c, seed, clip_and_flip);

  if (scramble_vertex_ids) {
    cugraph::scramble_vertex_ids<vertex_t>(
      handle, std::get<0>(src_dst_tuple), std::get<1>(src_dst_tuple), vertex_t{0}, seed);
  }

  graph_generator_t gg_vals{
    std::make_unique<rmm::device_buffer>(std::get<0>(src_dst_tuple).release()),
    std::make_unique<rmm::device_buffer>(std::get<1>(src_dst_tuple).release())};

  return std::make_unique<graph_generator_t>(std::move(gg_vals));
}

template <typename vertex_t>
std::vector<std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
call_generate_rmat_edgelists(raft::handle_t const& handle,
                             size_t n_edgelists,
                             size_t min_scale,
                             size_t max_scale,
                             size_t edge_factor,
                             cugraph::generator_distribution_t size_distribution,
                             cugraph::generator_distribution_t edge_distribution,
                             uint64_t seed,
                             bool clip_and_flip,
                             bool scramble_vertex_ids)
{
  auto src_dst_vec_tuple = cugraph::generate_rmat_edgelists<vertex_t>(handle,
                                                                      n_edgelists,
                                                                      min_scale,
                                                                      max_scale,
                                                                      edge_factor,
                                                                      size_distribution,
                                                                      edge_distribution,
                                                                      seed,
                                                                      clip_and_flip);

  if (scramble_vertex_ids) {
    std::for_each(
      src_dst_vec_tuple.begin(), src_dst_vec_tuple.end(), [&handle, seed](auto& src_dst_tuple) {
        cugraph::scramble_vertex_ids<vertex_t>(
          handle, std::get<0>(src_dst_tuple), std::get<1>(src_dst_tuple), vertex_t{0}, seed);
      });
  }

  std::vector<std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
    gg_vec;

  std::transform(
    src_dst_vec_tuple.begin(),
    src_dst_vec_tuple.end(),
    std::back_inserter(gg_vec),
    [](auto& tpl_dev_uvec) {
      return std::make_pair(
        std::move(std::make_unique<rmm::device_buffer>(std::get<0>(tpl_dev_uvec).release())),
        std::move(std::make_unique<rmm::device_buffer>(std::get<1>(tpl_dev_uvec).release())));
    });

  return gg_vec;
}

// Wrapper for random_walks() through a graph container
// to expose the API to cython.
//
template <typename vertex_t, typename edge_t>
std::enable_if_t<cugraph::is_vertex_edge_combo<vertex_t, edge_t>::value,
                 std::unique_ptr<random_walk_ret_t>>
call_random_walks(raft::handle_t const& handle,
                  graph_container_t const& graph_container,
                  vertex_t const* ptr_start_set,
                  edge_t num_paths,
                  edge_t max_depth,
                  bool use_padding)
{
  if (graph_container.weightType == numberTypeEnum::floatType) {
    using weight_t = float;

    auto graph =
      detail::create_graph<vertex_t, edge_t, weight_t, false, false>(handle, graph_container);

    auto triplet = cugraph::random_walks(
      handle, graph->view(), ptr_start_set, num_paths, max_depth, use_padding);

    random_walk_ret_t rw_tri{std::get<0>(triplet).size(),
                             std::get<1>(triplet).size(),
                             static_cast<size_t>(num_paths),
                             static_cast<size_t>(max_depth),
                             std::make_unique<rmm::device_buffer>(std::get<0>(triplet).release()),
                             std::make_unique<rmm::device_buffer>(std::get<1>(triplet).release()),
                             std::make_unique<rmm::device_buffer>(std::get<2>(triplet).release())};

    return std::make_unique<random_walk_ret_t>(std::move(rw_tri));

  } else if (graph_container.weightType == numberTypeEnum::doubleType) {
    using weight_t = double;

    auto graph =
      detail::create_graph<vertex_t, edge_t, weight_t, false, false>(handle, graph_container);

    auto triplet = cugraph::random_walks(
      handle, graph->view(), ptr_start_set, num_paths, max_depth, use_padding);

    random_walk_ret_t rw_tri{std::get<0>(triplet).size(),
                             std::get<1>(triplet).size(),
                             static_cast<size_t>(num_paths),
                             static_cast<size_t>(max_depth),
                             std::make_unique<rmm::device_buffer>(std::get<0>(triplet).release()),
                             std::make_unique<rmm::device_buffer>(std::get<1>(triplet).release()),
                             std::make_unique<rmm::device_buffer>(std::get<2>(triplet).release())};

    return std::make_unique<random_walk_ret_t>(std::move(rw_tri));

  } else {
    CUGRAPH_FAIL("Unsupported weight type.");
  }
}

template <typename index_t>
std::unique_ptr<random_walk_path_t> call_rw_paths(raft::handle_t const& handle,
                                                  index_t num_paths,
                                                  index_t const* vertex_path_sizes)
{
  auto triplet = cugraph::query_rw_sizes_offsets<index_t>(handle, num_paths, vertex_path_sizes);
  random_walk_path_t rw_path_tri{
    std::make_unique<rmm::device_buffer>(std::get<0>(triplet).release()),
    std::make_unique<rmm::device_buffer>(std::get<1>(triplet).release()),
    std::make_unique<rmm::device_buffer>(std::get<2>(triplet).release())};
  return std::make_unique<random_walk_path_t>(std::move(rw_path_tri));
}

// wrapper for weakly connected components:
//
template <typename vertex_t, typename weight_t>
void call_wcc(raft::handle_t const& handle,
              graph_container_t const& graph_container,
              vertex_t* components)
{
  if (graph_container.is_multi_gpu) {
    if (graph_container.edgeType == numberTypeEnum::int32Type) {
      auto graph =
        detail::create_graph<int32_t, int32_t, weight_t, false, true>(handle, graph_container);
      cugraph::weakly_connected_components(
        handle, graph->view(), reinterpret_cast<int32_t*>(components), false);

    } else if (graph_container.edgeType == numberTypeEnum::int64Type) {
      auto graph =
        detail::create_graph<vertex_t, int64_t, weight_t, false, true>(handle, graph_container);
      cugraph::weakly_connected_components(
        handle, graph->view(), reinterpret_cast<vertex_t*>(components), false);
    }
  } else {
    if (graph_container.edgeType == numberTypeEnum::int32Type) {
      auto graph =
        detail::create_graph<int32_t, int32_t, weight_t, false, false>(handle, graph_container);
      cugraph::weakly_connected_components(
        handle, graph->view(), reinterpret_cast<int32_t*>(components), false);
    } else if (graph_container.edgeType == numberTypeEnum::int64Type) {
      auto graph =
        detail::create_graph<vertex_t, int64_t, weight_t, false, false>(handle, graph_container);
      cugraph::weakly_connected_components(
        handle, graph->view(), reinterpret_cast<vertex_t*>(components), false);
    }
  }
}

// wrapper for shuffling:
//
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<major_minor_weights_t<vertex_t, edge_t, weight_t>> call_shuffle(
  raft::handle_t const& handle,
  vertex_t*
    edgelist_major_vertices,  // [IN / OUT]: groupby_gpu_id_and_shuffle_values() sorts in-place
  vertex_t* edgelist_minor_vertices,  // [IN / OUT]
  weight_t* edgelist_weights,         // [IN / OUT]
  edge_t num_edgelist_edges,
  bool is_weighted)
{
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  std::unique_ptr<major_minor_weights_t<vertex_t, edge_t, weight_t>> ptr_ret =
    std::make_unique<major_minor_weights_t<vertex_t, edge_t, weight_t>>(handle);

  if (is_weighted) {
    auto zip_edge = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights));

    std::forward_as_tuple(
      std::tie(ptr_ret->get_major(), ptr_ret->get_minor(), ptr_ret->get_weights()),
      std::ignore) =
      cugraph::groupby_gpu_id_and_shuffle_values(
        comm,  // handle.get_comms(),
        zip_edge,
        zip_edge + num_edgelist_edges,
        [key_func =
           cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
             comm.get_size(), row_comm.get_size(), col_comm.get_size()}] __device__(auto val) {
          return key_func(thrust::get<0>(val), thrust::get<1>(val));
        },
        handle.get_stream());
  } else {
    auto zip_edge = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_major_vertices, edgelist_minor_vertices));

    std::forward_as_tuple(std::tie(ptr_ret->get_major(), ptr_ret->get_minor()),
                          std::ignore) =
      cugraph::groupby_gpu_id_and_shuffle_values(
        comm,  // handle.get_comms(),
        zip_edge,
        zip_edge + num_edgelist_edges,
        [key_func =
           cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
             comm.get_size(), row_comm.get_size(), col_comm.get_size()}] __device__(auto val) {
          return key_func(thrust::get<0>(val), thrust::get<1>(val));
        },
        handle.get_stream());
  }

  auto local_partition_id_op =
    [comm_size,
     key_func = cugraph::detail::compute_partition_id_from_ext_edge_endpoints_t<vertex_t>{
       comm_size, row_comm_size, col_comm_size}] __device__(auto pair) {
      return key_func(thrust::get<0>(pair), thrust::get<1>(pair)) /
             comm_size;  // global partition id to local partition id
    };
  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(ptr_ret->get_major().data(), ptr_ret->get_minor().data()));

  auto edge_counts = (is_weighted)
                       ? cugraph::groupby_and_count(pair_first,
                                                    pair_first + ptr_ret->get_major().size(),
                                                    ptr_ret->get_weights().data(),
                                                    local_partition_id_op,
                                                    col_comm_size,
                                                    false,
                                                    handle.get_stream())
                       : cugraph::groupby_and_count(pair_first,
                                                    pair_first + ptr_ret->get_major().size(),
                                                    local_partition_id_op,
                                                    col_comm_size,
                                                    false,
                                                    handle.get_stream());

  std::vector<size_t> h_edge_counts(edge_counts.size());
  raft::update_host(
    h_edge_counts.data(), edge_counts.data(), edge_counts.size(), handle.get_stream());
  handle.sync_stream();

  ptr_ret->get_edge_counts().resize(h_edge_counts.size());
  for (size_t i = 0; i < h_edge_counts.size(); ++i) {
    ptr_ret->get_edge_counts()[i] = static_cast<edge_t>(h_edge_counts[i]);
  }

  return ptr_ret;  // RVO-ed
}

// Wrapper for calling renumber_edeglist() inplace:
// TODO: check if return type needs further handling...
//
template <typename vertex_t, typename edge_t>
std::unique_ptr<renum_tuple_t<vertex_t, edge_t>> call_renumber(
  raft::handle_t const& handle,
  vertex_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  vertex_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<edge_t> const& edge_counts,
  bool store_transposed,
  bool do_expensive_check,
  bool multi_gpu)  // bc. cython cannot take non-type template params
{
  // caveat: return values have different types on the 2 branches below:
  //
  std::unique_ptr<renum_tuple_t<vertex_t, edge_t>> p_ret =
    std::make_unique<renum_tuple_t<vertex_t, edge_t>>(handle);

  if (multi_gpu) {
    std::vector<edge_t> displacements(edge_counts.size(), edge_t{0});
    std::partial_sum(edge_counts.begin(), edge_counts.end() - 1, displacements.begin() + 1);
    std::vector<vertex_t*> src_ptrs(edge_counts.size());
    std::vector<vertex_t*> dst_ptrs(src_ptrs.size());
    for (size_t i = 0; i < edge_counts.size(); ++i) {
      src_ptrs[i] = shuffled_edgelist_src_vertices + displacements[i];
      dst_ptrs[i] = shuffled_edgelist_dst_vertices + displacements[i];
    }

    cugraph::renumber_meta_t<vertex_t, edge_t, true> meta{};
    std::tie(p_ret->get_dv(), meta) =
      cugraph::renumber_edgelist<vertex_t, edge_t, true>(handle,
                                                         std::nullopt,
                                                         src_ptrs,
                                                         dst_ptrs,
                                                         edge_counts,
                                                         std::nullopt,
                                                         store_transposed,
                                                         do_expensive_check);
    p_ret->get_num_vertices()    = meta.number_of_vertices;
    p_ret->get_num_edges()       = meta.number_of_edges;
    p_ret->get_partition()       = meta.partition;
    p_ret->get_segment_offsets() = meta.edge_partition_segment_offsets;
  } else {
    cugraph::renumber_meta_t<vertex_t, edge_t, false> meta{};
    std::tie(p_ret->get_dv(), meta) =
      cugraph::renumber_edgelist<vertex_t, edge_t, false>(handle,
                                                          std::nullopt,
                                                          shuffled_edgelist_src_vertices,
                                                          shuffled_edgelist_dst_vertices,
                                                          edge_counts[0],
                                                          store_transposed,
                                                          do_expensive_check);

    p_ret->get_num_vertices()    = static_cast<vertex_t>(p_ret->get_dv().size());
    p_ret->get_num_edges()       = edge_counts[0];
    p_ret->get_partition()       = cugraph::partition_t<vertex_t>{};  // dummy
    p_ret->get_segment_offsets() = meta.segment_offsets;
  }

  return p_ret;  // RVO-ed (copy ellision)
}

// Helper for setting up subcommunicators
void init_subcomms(raft::handle_t& handle, size_t row_comm_size)
{
  partition_2d::subcomm_factory_t<partition_2d::key_naming_t> subcomm_factory(handle,
                                                                              row_comm_size);
}

// Explicit instantiations

template std::unique_ptr<cy_multi_edgelists_t> call_egonet<int32_t, float>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int32_t* source_vertex,
  int32_t n_subgraphs,
  int32_t radius);

template std::unique_ptr<cy_multi_edgelists_t> call_egonet<int32_t, double>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int32_t* source_vertex,
  int32_t n_subgraphs,
  int32_t radius);

template std::unique_ptr<cy_multi_edgelists_t> call_egonet<int64_t, float>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int64_t* source_vertex,
  int64_t n_subgraphs,
  int64_t radius);

template std::unique_ptr<cy_multi_edgelists_t> call_egonet<int64_t, double>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int64_t* source_vertex,
  int64_t n_subgraphs,
  int64_t radius);

template std::unique_ptr<random_walk_ret_t> call_random_walks<int32_t, int32_t>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int32_t const* ptr_start_set,
  int32_t num_paths,
  int32_t max_depth,
  bool use_padding);

template std::unique_ptr<random_walk_ret_t> call_random_walks<int32_t, int64_t>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int32_t const* ptr_start_set,
  int64_t num_paths,
  int64_t max_depth,
  bool use_padding);

template std::unique_ptr<random_walk_ret_t> call_random_walks<int64_t, int64_t>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int64_t const* ptr_start_set,
  int64_t num_paths,
  int64_t max_depth,
  bool use_padding);

template std::unique_ptr<random_walk_path_t> call_rw_paths<int32_t>(
  raft::handle_t const& handle, int32_t num_paths, int32_t const* vertex_path_sizes);

template std::unique_ptr<random_walk_path_t> call_rw_paths<int64_t>(
  raft::handle_t const& handle, int64_t num_paths, int64_t const* vertex_path_sizes);

template void call_wcc<int32_t, float>(raft::handle_t const& handle,
                                       graph_container_t const& graph_container,
                                       int32_t* components);

template void call_wcc<int32_t, double>(raft::handle_t const& handle,
                                        graph_container_t const& graph_container,
                                        int32_t* components);

template void call_wcc<int64_t, float>(raft::handle_t const& handle,
                                       graph_container_t const& graph_container,
                                       int64_t* components);

template void call_wcc<int64_t, double>(raft::handle_t const& handle,
                                        graph_container_t const& graph_container,
                                        int64_t* components);

template std::unique_ptr<major_minor_weights_t<int32_t, int32_t, float>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  float* edgelist_weights,
  int32_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int32_t, int64_t, float>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  float* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int32_t, int32_t, double>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  double* edgelist_weights,
  int32_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int32_t, int64_t, double>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  double* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int64_t, int64_t, float>> call_shuffle(
  raft::handle_t const& handle,
  int64_t* edgelist_major_vertices,
  int64_t* edgelist_minor_vertices,
  float* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_weighted);

template std::unique_ptr<major_minor_weights_t<int64_t, int64_t, double>> call_shuffle(
  raft::handle_t const& handle,
  int64_t* edgelist_major_vertices,
  int64_t* edgelist_minor_vertices,
  double* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_weighted);

// TODO: add the remaining relevant EIDIr's:
//
template std::unique_ptr<renum_tuple_t<int32_t, int32_t>> call_renumber(
  raft::handle_t const& handle,
  int32_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  int32_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<int32_t> const& edge_counts,
  bool store_transposed,
  bool do_expensive_check,
  bool multi_gpu);

template std::unique_ptr<renum_tuple_t<int32_t, int64_t>> call_renumber(
  raft::handle_t const& handle,
  int32_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  int32_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<int64_t> const& edge_counts,
  bool store_transposed,
  bool do_expensive_check,
  bool multi_gpu);

template std::unique_ptr<renum_tuple_t<int64_t, int64_t>> call_renumber(
  raft::handle_t const& handle,
  int64_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  int64_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<int64_t> const& edge_counts,
  bool store_transposed,
  bool do_expensive_check,
  bool multi_gpu);

template std::unique_ptr<graph_generator_t> call_generate_rmat_edgelist<int32_t>(
  raft::handle_t const& handle,
  size_t scale,
  size_t num_edges,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool clip_and_flip,
  bool scramble_vertex_ids);

template std::unique_ptr<graph_generator_t> call_generate_rmat_edgelist<int64_t>(
  raft::handle_t const& handle,
  size_t scale,
  size_t num_edges,
  double a,
  double b,
  double c,
  uint64_t seed,
  bool clip_and_flip,
  bool scramble_vertex_ids);

template std::vector<
  std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
call_generate_rmat_edgelists<int32_t>(raft::handle_t const& handle,
                                      size_t n_edgelists,
                                      size_t min_scale,
                                      size_t max_scale,
                                      size_t edge_factor,
                                      cugraph::generator_distribution_t size_distribution,
                                      cugraph::generator_distribution_t edge_distribution,
                                      uint64_t seed,
                                      bool clip_and_flip,
                                      bool scramble_vertex_ids);

template std::vector<
  std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
call_generate_rmat_edgelists<int64_t>(raft::handle_t const& handle,
                                      size_t n_edgelists,
                                      size_t min_scale,
                                      size_t max_scale,
                                      size_t edge_factor,
                                      cugraph::generator_distribution_t size_distribution,
                                      cugraph::generator_distribution_t edge_distribution,
                                      uint64_t seed,
                                      bool clip_and_flip,
                                      bool scramble_vertex_ids);

}  // namespace cython
}  // namespace cugraph
