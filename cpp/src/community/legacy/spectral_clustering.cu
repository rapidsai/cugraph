/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/copy.hpp>
#include <raft/random/rng_state.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/spectral/modularity_maximization.cuh>
#include <raft/spectral/partition.cuh>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <cuvs/cluster/spectral.hpp>
#include <cuvs/preprocessing/spectral_embedding.hpp>

#include <ctime>

namespace cugraph {

namespace ext_raft {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void balancedCutClustering_impl(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                vertex_t n_clusters,
                                vertex_t n_eig_vects,
                                weight_t evs_tolerance,
                                int evs_max_iter,
                                weight_t kmean_tolerance,
                                int kmean_max_iter,
                                vertex_t* clustering,
                                weight_t* eig_vals,
                                weight_t* eig_vects)
{
  RAFT_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  RAFT_EXPECTS(evs_tolerance >= weight_t{0.0},
               "API error, evs_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(evs_tolerance < weight_t{1.0},
               "API error, evs_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(kmean_tolerance >= weight_t{0.0},
               "API error, kmean_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(kmean_tolerance < weight_t{1.0},
               "API error, kmean_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(n_clusters > 1, "API error, must specify more than 1 cluster");
  RAFT_EXPECTS(n_clusters < graph.number_of_vertices,
               "API error, number of clusters must be smaller than number of vertices");
  RAFT_EXPECTS(n_eig_vects <= n_clusters,
               "API error, cannot specify more eigenvectors than clusters");
  RAFT_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");
  RAFT_EXPECTS(eig_vals != nullptr, "API error, must specify valid eigenvalues");
  RAFT_EXPECTS(eig_vects != nullptr, "API error, must specify valid eigenvectors");

  // Convert CSR to COO using raft::sparse::convert::csr_to_coo
  rmm::device_uvector<vertex_t> src_indices(graph.number_of_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_indices(graph.number_of_edges, handle.get_stream());

  // Copy destination indices (already in COO format)
  raft::copy(dst_indices.data(), graph.indices, graph.number_of_edges, handle.get_stream());

  // Convert CSR row offsets to COO source indices
  raft::sparse::convert::csr_to_coo<vertex_t>(graph.offsets,
                                              static_cast<vertex_t>(graph.number_of_vertices),
                                              src_indices.data(),
                                              static_cast<edge_t>(graph.number_of_edges),
                                              handle.get_stream());

  // Create coordinate structure view from converted COO data
  auto coord_view = raft::make_device_coordinate_structure_view<vertex_t, vertex_t, vertex_t>(
    src_indices.data(),
    dst_indices.data(),
    graph.number_of_vertices,
    graph.number_of_vertices,
    graph.number_of_edges);

  // Create COO matrix view using coordinate structure view and CSR edge data
  auto coo_matrix = raft::make_device_coo_matrix_view<weight_t>(graph.edge_data, coord_view);

  // Use seed from RNG state instead of hardcoded 0
  rmm::device_uvector<unsigned long long> d_seed(1, handle.get_stream());

  raft::random::uniformInt<unsigned long long>(
    rng_state, d_seed.data(), 1, 0, std::numeric_limits<vertex_t>::max() - 1, handle.get_stream());

  unsigned long long seed{0};
  raft::update_host(&seed, d_seed.data(), d_seed.size(), handle.get_stream());

  cuvs::cluster::spectral::params params;

  params.rng_state    = rng_state;
  params.n_clusters   = n_clusters;
  params.n_components = n_eig_vects;
  params.n_init       = 10;  // Multiple initializations for better results
  params.n_neighbors =
    std::min(static_cast<int>(graph.number_of_vertices) - 1, 15);  // Adaptive neighbor count

  cuvs::cluster::spectral::fit_predict(
    handle,
    params,
    coo_matrix,
    raft::make_device_vector_view<vertex_t, vertex_t>(clustering, graph.number_of_vertices));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void spectralModularityMaximization_impl(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
  vertex_t n_clusters,
  vertex_t n_eig_vects,
  weight_t evs_tolerance,
  int evs_max_iter,
  weight_t kmean_tolerance,
  int kmean_max_iter,
  vertex_t* clustering,
  weight_t* eig_vals,
  weight_t* eig_vects)
{
  RAFT_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  RAFT_EXPECTS(evs_tolerance >= weight_t{0.0},
               "API error, evs_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(evs_tolerance < weight_t{1.0},
               "API error, evs_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(kmean_tolerance >= weight_t{0.0},
               "API error, kmean_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(kmean_tolerance < weight_t{1.0},
               "API error, kmean_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(n_clusters > 1, "API error, must specify more than 1 cluster");
  RAFT_EXPECTS(n_clusters < graph.number_of_vertices,
               "API error, number of clusters must be smaller than number of vertices");
  RAFT_EXPECTS(n_eig_vects <= n_clusters,
               "API error, cannot specify more eigenvectors than clusters");
  RAFT_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");
  RAFT_EXPECTS(eig_vals != nullptr, "API error, must specify valid eigenvalues");
  RAFT_EXPECTS(eig_vects != nullptr, "API error, must specify valid eigenvectors");

  // Convert CSR to COO using raft::sparse::convert::csr_to_coo
  rmm::device_uvector<vertex_t> src_indices(graph.number_of_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_indices(graph.number_of_edges, handle.get_stream());

  // Copy destination indices (already in COO format)
  raft::copy(dst_indices.data(), graph.indices, graph.number_of_edges, handle.get_stream());

  // Convert CSR row offsets to COO source indices
  raft::sparse::convert::csr_to_coo<vertex_t>(graph.offsets,
                                              static_cast<vertex_t>(graph.number_of_vertices),
                                              src_indices.data(),
                                              static_cast<edge_t>(graph.number_of_edges),
                                              handle.get_stream());

  // Create coordinate structure view from converted COO data
  auto coord_view = raft::make_device_coordinate_structure_view<vertex_t, vertex_t, vertex_t>(
    src_indices.data(),
    dst_indices.data(),
    graph.number_of_vertices,
    graph.number_of_vertices,
    graph.number_of_edges);

  // Create COO matrix view using coordinate structure view and CSR edge data
  auto coo_matrix = raft::make_device_coo_matrix_view<weight_t>(graph.edge_data, coord_view);

  // Use seed from RNG state instead of hardcoded 0
  rmm::device_uvector<unsigned long long> d_seed(1, handle.get_stream());

  raft::random::uniformInt<unsigned long long>(
    rng_state, d_seed.data(), 1, 0, std::numeric_limits<vertex_t>::max() - 1, handle.get_stream());

  unsigned long long seed{0};
  raft::update_host(&seed, d_seed.data(), d_seed.size(), handle.get_stream());

  cuvs::cluster::spectral::params params;

  params.rng_state    = rng_state;
  params.n_clusters   = n_clusters;
  params.n_components = n_eig_vects;
  params.n_init       = 10;  // Multiple initializations for better results
  params.n_neighbors =
    std::min(static_cast<int>(graph.number_of_vertices) - 1, 15);  // Adaptive neighbor count

  cuvs::cluster::spectral::fit_predict(
    handle,
    params,
    coo_matrix,
    raft::make_device_vector_view<vertex_t, vertex_t>(clustering, graph.number_of_vertices));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void analyzeModularityClustering_impl(raft::handle_t const& handle,
                                      legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                      int n_clusters,
                                      vertex_t const* clustering,
                                      weight_t* modularity)
{
  using index_type = vertex_t;
  using value_type = weight_t;
  using nnz_type   = edge_t;

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type, nnz_type> const r_csr_m{handle,
                                                                                          graph};

  weight_t mod;
  raft::spectral::analyzeModularity(handle, r_csr_m, n_clusters, clustering, mod);
  *modularity = mod;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void analyzeBalancedCut_impl(raft::handle_t const& handle,
                             legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                             vertex_t n_clusters,
                             vertex_t const* clustering,
                             weight_t* edgeCut,
                             weight_t* ratioCut)
{
  RAFT_EXPECTS(n_clusters <= graph.number_of_vertices,
               "API error: number of clusters must be <= number of vertices");
  RAFT_EXPECTS(n_clusters > 0, "API error: number of clusters must be > 0)");

  weight_t edge_cut;
  weight_t cost{0};

  using index_type = vertex_t;
  using value_type = weight_t;
  using nnz_type   = edge_t;

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type, nnz_type> const r_csr_m{handle,
                                                                                          graph};

  raft::spectral::analyzePartition(handle, r_csr_m, n_clusters, clustering, edge_cut, cost);

  *edgeCut  = edge_cut;
  *ratioCut = cost;
}

}  // namespace detail

template <typename VT, typename ET, typename WT>
void balancedCutClustering(raft::handle_t const& handle,
                           raft::random::RngState& rng_state,
                           legacy::GraphCSRView<VT, ET, WT> const& graph,
                           VT num_clusters,
                           VT num_eigen_vects,
                           WT evs_tolerance,
                           int evs_max_iter,
                           WT kmean_tolerance,
                           int kmean_max_iter,
                           VT* clustering)
{
  rmm::device_uvector<WT> eig_vals(num_eigen_vects, handle.get_stream());
  rmm::device_uvector<WT> eig_vects(num_eigen_vects * graph.number_of_vertices,
                                    handle.get_stream());

  detail::balancedCutClustering_impl(handle,
                                     rng_state,
                                     graph,
                                     num_clusters,
                                     num_eigen_vects,
                                     evs_tolerance,
                                     evs_max_iter,
                                     kmean_tolerance,
                                     kmean_max_iter,
                                     clustering,
                                     eig_vals.data(),
                                     eig_vects.data());
}

template <typename VT, typename ET, typename WT>
void spectralModularityMaximization(raft::handle_t const& handle,
                                    raft::random::RngState& rng_state,
                                    legacy::GraphCSRView<VT, ET, WT> const& graph,
                                    VT n_clusters,
                                    VT n_eigen_vects,
                                    WT evs_tolerance,
                                    int evs_max_iter,
                                    WT kmean_tolerance,
                                    int kmean_max_iter,
                                    VT* clustering)
{
  rmm::device_uvector<WT> eig_vals(n_eigen_vects, handle.get_stream());
  rmm::device_uvector<WT> eig_vects(n_eigen_vects * graph.number_of_vertices, handle.get_stream());

  detail::spectralModularityMaximization_impl(handle,
                                              rng_state,
                                              graph,
                                              n_clusters,
                                              n_eigen_vects,
                                              evs_tolerance,
                                              evs_max_iter,
                                              kmean_tolerance,
                                              kmean_max_iter,
                                              clustering,
                                              eig_vals.data(),
                                              eig_vects.data());
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_modularity(raft::handle_t const& handle,
                                  legacy::GraphCSRView<VT, ET, WT> const& graph,
                                  int n_clusters,
                                  VT const* clustering,
                                  WT* score)
{
  detail::analyzeModularityClustering_impl(handle, graph, n_clusters, clustering, score);
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_edge_cut(raft::handle_t const& handle,
                                legacy::GraphCSRView<VT, ET, WT> const& graph,
                                int n_clusters,
                                VT const* clustering,
                                WT* score)
{
  WT dummy{0.0};
  detail::analyzeBalancedCut_impl(handle, graph, n_clusters, clustering, score, &dummy);
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_ratio_cut(raft::handle_t const& handle,
                                 legacy::GraphCSRView<VT, ET, WT> const& graph,
                                 int n_clusters,
                                 VT const* clustering,
                                 WT* score)
{
  WT dummy{0.0};
  detail::analyzeBalancedCut_impl(handle, graph, n_clusters, clustering, &dummy, score);
}

template void balancedCutClustering<int, int, float>(raft::handle_t const& handle,
                                                     raft::random::RngState&,
                                                     legacy::GraphCSRView<int, int, float> const&,
                                                     int,
                                                     int,
                                                     float,
                                                     int,
                                                     float,
                                                     int,
                                                     int*);
template void balancedCutClustering<int, int, double>(raft::handle_t const& handle,
                                                      raft::random::RngState&,
                                                      legacy::GraphCSRView<int, int, double> const&,
                                                      int,
                                                      int,
                                                      double,
                                                      int,
                                                      double,
                                                      int,
                                                      int*);
template void spectralModularityMaximization<int, int, float>(
  raft::handle_t const& handle,
  raft::random::RngState&,
  legacy::GraphCSRView<int, int, float> const&,
  int,
  int,
  float,
  int,
  float,
  int,
  int*);
template void spectralModularityMaximization<int, int, double>(
  raft::handle_t const& handle,
  raft::random::RngState&,
  legacy::GraphCSRView<int, int, double> const&,
  int,
  int,
  double,
  int,
  double,
  int,
  int*);
template void analyzeClustering_modularity<int, int, float>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, float> const&,
  int,
  int const*,
  float*);
template void analyzeClustering_modularity<int, int, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, double> const&,
  int,
  int const*,
  double*);
template void analyzeClustering_edge_cut<int, int, float>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, float> const&,
  int,
  int const*,
  float*);
template void analyzeClustering_edge_cut<int, int, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, double> const&,
  int,
  int const*,
  double*);
template void analyzeClustering_ratio_cut<int, int, float>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, float> const&,
  int,
  int const*,
  float*);
template void analyzeClustering_ratio_cut<int, int, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, double> const&,
  int,
  int const*,
  double*);

}  // namespace ext_raft
}  // namespace cugraph
