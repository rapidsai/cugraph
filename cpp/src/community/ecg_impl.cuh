#pragma once
#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>
#include <prims/update_edge_src_dst_property.cuh>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <prims/fill_edge_property.cuh>
#include <prims/transform_e.cuh>
#include <rmm/device_uvector.hpp>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, size_t, weight_t> ecg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  weight_t min_weight,
  size_t ensemble_size,
  size_t max_level,
  weight_t threshold,
  weight_t resolution,
  weight_t theta = 1.0)
{
  using graph_view_t = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  edge_src_property_t<graph_view_t, vertex_t> src_cluster_assignments(handle, graph_view);
  edge_dst_property_t<graph_view_t, vertex_t> dst_cluster_assignments(handle, graph_view);
  edge_property_t<graph_view_t, weight_t> modified_edge_weights(handle, graph_view);

  cugraph::fill_edge_property(handle, graph_view, weight_t{0}, modified_edge_weights);

  weight_t modularity = -1.0;
  rmm::device_uvector<vertex_t> cluster_assignments(graph_view.local_vertex_partition_range_size(),
                                                    handle.get_stream());

  for (size_t i = 0; i < ensemble_size; i++) {
    std::tie(std::ignore, modularity) = cugraph::louvain(
      handle,
      std::make_optional(std::reference_wrapper<raft::random::RngState>(rng_state)),
      graph_view,
      edge_weight_view,
      cluster_assignments.data(),
      size_t{1},
      threshold,
      resolution);

    // std::tie(std::ignore, modularity) = cugraph::leiden(handle,
    //                                                     rng_state,
    //                                                     graph_view,
    //                                                     edge_weight_view,
    //                                                     cluster_assignments.data(),
    //                                                     size_t{1},
    //                                                     resolution,
    //                                                     theta);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector(
      "\tcluster_assignments:", cluster_assignments.data(), cluster_assignments.size(), std::cout);
    std::cout << "\tmodularity: " << modularity << std::endl;

    cugraph::update_edge_src_property(
      handle, graph_view, cluster_assignments.begin(), src_cluster_assignments);
    cugraph::update_edge_dst_property(
      handle, graph_view, cluster_assignments.begin(), dst_cluster_assignments);

    cugraph::transform_e(
      handle,
      graph_view,
      src_cluster_assignments.view(),
      dst_cluster_assignments.view(),
      modified_edge_weights.view(),
      [] __device__(auto src, auto dst, auto src_property, auto dst_property, auto edge_property) {
        printf("\n\t src %d dst %d sc %f  dc %f  mew %f\n",
               static_cast<int>(src),
               static_cast<int>(dst),
               static_cast<float>(src_property),
               static_cast<float>(dst_property),
               static_cast<float>(edge_property));
        return edge_property + (src_property == dst_property);
      },
      modified_edge_weights.mutable_view());

    for (size_t ep_idx = 0; ep_idx < graph_view.number_of_local_edge_partitions(); ++ep_idx) {
      // // Toplogy
      // auto edge_partition = graph_view.local_edge_partition_view(ep_idx);

      // RAFT_CUDA_TRY(cudaDeviceSynchronize());
      // std::cout << "rank: " << r << ", #edges = " << edge_partition.number_of_edges()
      //           << std::endl;

      // auto number_of_edges = edge_partition.number_of_edges();
      // auto offsets         = edge_partition.offsets();
      // auto indices         = edge_partition.indices();

      // RAFT_CUDA_TRY(cudaDeviceSynchronize());
      // raft::print_device_vector("offsets:", offsets.begin(), offsets.size(), std::cout);
      // RAFT_CUDA_TRY(cudaDeviceSynchronize());
      // raft::print_device_vector("indices:", indices.begin(), indices.size(), std::cout);

      // Edge property values
      {
        auto value_firsts = modified_edge_weights.view().value_firsts();
        auto edge_counts  = modified_edge_weights.view().edge_counts();

        // assert(number_of_edges == edge_counts[ep_idx]);

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        raft::print_device_vector(
          "\tmodified weights:", value_firsts[ep_idx], edge_counts[ep_idx], std::cout);
      }
    }
  }

  cugraph::transform_e(
    handle,
    graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    view_concat(*edge_weight_view, modified_edge_weights.view()),
    [min_weight, ensemble_size] __device__(
      auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, auto edge_properties) {
      auto e_weight    = thrust::get<0>(edge_properties);
      auto e_frequency = thrust::get<1>(edge_properties);

      printf("\n\t src %d dst %d e_weight %f e_freq %f\n",
             static_cast<int>(src),
             static_cast<int>(dst),
             static_cast<float>(e_weight),
             static_cast<float>(e_frequency));

      return min_weight + (e_weight - min_weight) * e_frequency / ensemble_size;
    },
    modified_edge_weights.mutable_view());

  std::tie(max_level, modularity) =

    cugraph::louvain(handle,
                     std::make_optional(std::reference_wrapper<raft::random::RngState>(rng_state)),
                     graph_view,
                     std::make_optional(modified_edge_weights.view()),
                     cluster_assignments.data(),
                     max_level,
                     threshold,
                     resolution);

  // cugraph::leiden(handle,
  //                 rng_state,
  //                 graph_view,
  //                 std::make_optional(modified_edge_weights.view()),
  //                 cluster_assignments.data(),
  //                 max_level,
  //                 resolution,
  //                 theta);

  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  std::cout << "\tfinal modularity: " << modularity << std::endl;
  raft::print_device_vector("\tfinal cluster_assignments:",
                            cluster_assignments.data(),
                            cluster_assignments.size(),
                            std::cout);

  return std::make_tuple(std::move(cluster_assignments), max_level, modularity);
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, size_t, weight_t> ecg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  weight_t min_weight,
  size_t ensemble_size,
  size_t max_level,
  weight_t threshold,
  weight_t resolution)
{
  return detail::ecg(handle,
                     rng_state,
                     graph_view,
                     edge_weight_view,
                     min_weight,
                     ensemble_size,
                     max_level,
                     threshold,
                     resolution);
}

}  // namespace cugraph
