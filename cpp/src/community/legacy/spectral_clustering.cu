/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>

#include <ctime>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/transform.h>

#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>

#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.cuh>
#endif
#include <raft/spectral/modularity_maximization.cuh>
#include <raft/spectral/partition.cuh>

namespace cugraph {

namespace ext_raft {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void balancedCutClustering_impl(legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
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

  raft::handle_t handle;

  int evs_max_it{4000};
  int kmean_max_it{200};
  weight_t evs_tol{1.0E-3};
  weight_t kmean_tol{1.0E-2};

  if (evs_max_iter > 0) evs_max_it = evs_max_iter;

  if (evs_tolerance > weight_t{0.0}) evs_tol = evs_tolerance;

  if (kmean_max_iter > 0) kmean_max_it = kmean_max_iter;

  if (kmean_tolerance > weight_t{0.0}) kmean_tol = kmean_tolerance;

  int restartIter_lanczos = 15 + n_eig_vects;

  unsigned long long seed{1234567};
  bool reorthog{false};

  using index_type = vertex_t;
  using value_type = weight_t;

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const r_csr_m{handle, graph};

  raft::spectral::eigen_solver_config_t<index_type, value_type> eig_cfg{
    n_eig_vects, evs_max_it, restartIter_lanczos, evs_tol, reorthog, seed};
  raft::spectral::lanczos_solver_t<index_type, value_type> eig_solver{eig_cfg};

  raft::spectral::cluster_solver_config_t<index_type, value_type> clust_cfg{
    n_clusters, kmean_max_it, kmean_tol, seed};
  raft::spectral::kmeans_solver_t<index_type, value_type> cluster_solver{clust_cfg};

  raft::spectral::partition(
    handle, r_csr_m, eig_solver, cluster_solver, clustering, eig_vals, eig_vects);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void spectralModularityMaximization_impl(
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

  raft::handle_t handle;

  int evs_max_it{4000};
  int kmean_max_it{200};
  weight_t evs_tol{1.0E-3};
  weight_t kmean_tol{1.0E-2};

  if (evs_max_iter > 0) evs_max_it = evs_max_iter;

  if (evs_tolerance > weight_t{0.0}) evs_tol = evs_tolerance;

  if (kmean_max_iter > 0) kmean_max_it = kmean_max_iter;

  if (kmean_tolerance > weight_t{0.0}) kmean_tol = kmean_tolerance;

  int restartIter_lanczos = 15 + n_eig_vects;

  unsigned long long seed{123456};
  bool reorthog{false};

  using index_type = vertex_t;
  using value_type = weight_t;

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const r_csr_m{handle, graph};

  raft::spectral::eigen_solver_config_t<index_type, value_type> eig_cfg{
    n_eig_vects, evs_max_it, restartIter_lanczos, evs_tol, reorthog, seed};
  raft::spectral::lanczos_solver_t<index_type, value_type> eig_solver{eig_cfg};

  raft::spectral::cluster_solver_config_t<index_type, value_type> clust_cfg{
    n_clusters, kmean_max_it, kmean_tol, seed};
  raft::spectral::kmeans_solver_t<index_type, value_type> cluster_solver{clust_cfg};

  // not returned...
  // auto result =
  raft::spectral::modularity_maximization(
    handle, r_csr_m, eig_solver, cluster_solver, clustering, eig_vals, eig_vects);

  // not returned...
  // int iters_lanczos, iters_kmeans;
  // iters_lanczos = std::get<0>(result);
  // iters_kmeans  = std::get<2>(result);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void analyzeModularityClustering_impl(legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                      int n_clusters,
                                      vertex_t const* clustering,
                                      weight_t* modularity)
{
  raft::handle_t handle;

  using index_type = vertex_t;
  using value_type = weight_t;

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const r_csr_m{handle, graph};

  weight_t mod;
  raft::spectral::analyzeModularity(handle, r_csr_m, n_clusters, clustering, mod);
  *modularity = mod;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void analyzeBalancedCut_impl(legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                             vertex_t n_clusters,
                             vertex_t const* clustering,
                             weight_t* edgeCut,
                             weight_t* ratioCut)
{
  raft::handle_t handle;

  RAFT_EXPECTS(n_clusters <= graph.number_of_vertices,
               "API error: number of clusters must be <= number of vertices");
  RAFT_EXPECTS(n_clusters > 0, "API error: number of clusters must be > 0)");

  weight_t edge_cut;
  weight_t cost{0};

  using index_type = vertex_t;
  using value_type = weight_t;

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const r_csr_m{handle, graph};

  raft::spectral::analyzePartition(handle, r_csr_m, n_clusters, clustering, edge_cut, cost);

  *edgeCut  = edge_cut;
  *ratioCut = cost;
}

}  // namespace detail

template <typename VT, typename ET, typename WT>
void balancedCutClustering(legacy::GraphCSRView<VT, ET, WT> const& graph,
                           VT num_clusters,
                           VT num_eigen_vects,
                           WT evs_tolerance,
                           int evs_max_iter,
                           WT kmean_tolerance,
                           int kmean_max_iter,
                           VT* clustering)
{
  rmm::device_vector<WT> eig_vals(num_eigen_vects);
  rmm::device_vector<WT> eig_vects(num_eigen_vects * graph.number_of_vertices);

  detail::balancedCutClustering_impl(graph,
                                     num_clusters,
                                     num_eigen_vects,
                                     evs_tolerance,
                                     evs_max_iter,
                                     kmean_tolerance,
                                     kmean_max_iter,
                                     clustering,
                                     eig_vals.data().get(),
                                     eig_vects.data().get());
}

template <typename VT, typename ET, typename WT>
void spectralModularityMaximization(legacy::GraphCSRView<VT, ET, WT> const& graph,
                                    VT n_clusters,
                                    VT n_eigen_vects,
                                    WT evs_tolerance,
                                    int evs_max_iter,
                                    WT kmean_tolerance,
                                    int kmean_max_iter,
                                    VT* clustering)
{
  rmm::device_vector<WT> eig_vals(n_eigen_vects);
  rmm::device_vector<WT> eig_vects(n_eigen_vects * graph.number_of_vertices);

  detail::spectralModularityMaximization_impl(graph,
                                              n_clusters,
                                              n_eigen_vects,
                                              evs_tolerance,
                                              evs_max_iter,
                                              kmean_tolerance,
                                              kmean_max_iter,
                                              clustering,
                                              eig_vals.data().get(),
                                              eig_vects.data().get());
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_modularity(legacy::GraphCSRView<VT, ET, WT> const& graph,
                                  int n_clusters,
                                  VT const* clustering,
                                  WT* score)
{
  detail::analyzeModularityClustering_impl(graph, n_clusters, clustering, score);
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_edge_cut(legacy::GraphCSRView<VT, ET, WT> const& graph,
                                int n_clusters,
                                VT const* clustering,
                                WT* score)
{
  WT dummy{0.0};
  detail::analyzeBalancedCut_impl(graph, n_clusters, clustering, score, &dummy);
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_ratio_cut(legacy::GraphCSRView<VT, ET, WT> const& graph,
                                 int n_clusters,
                                 VT const* clustering,
                                 WT* score)
{
  WT dummy{0.0};
  detail::analyzeBalancedCut_impl(graph, n_clusters, clustering, &dummy, score);
}

template void balancedCutClustering<int, int, float>(
  legacy::GraphCSRView<int, int, float> const&, int, int, float, int, float, int, int*);
template void balancedCutClustering<int, int, double>(
  legacy::GraphCSRView<int, int, double> const&, int, int, double, int, double, int, int*);
template void spectralModularityMaximization<int, int, float>(
  legacy::GraphCSRView<int, int, float> const&, int, int, float, int, float, int, int*);
template void spectralModularityMaximization<int, int, double>(
  legacy::GraphCSRView<int, int, double> const&, int, int, double, int, double, int, int*);
template void analyzeClustering_modularity<int, int, float>(
  legacy::GraphCSRView<int, int, float> const&, int, int const*, float*);
template void analyzeClustering_modularity<int, int, double>(
  legacy::GraphCSRView<int, int, double> const&, int, int const*, double*);
template void analyzeClustering_edge_cut<int, int, float>(
  legacy::GraphCSRView<int, int, float> const&, int, int const*, float*);
template void analyzeClustering_edge_cut<int, int, double>(
  legacy::GraphCSRView<int, int, double> const&, int, int const*, double*);
template void analyzeClustering_ratio_cut<int, int, float>(
  legacy::GraphCSRView<int, int, float> const&, int, int const*, float*);
template void analyzeClustering_ratio_cut<int, int, double>(
  legacy::GraphCSRView<int, int, double> const&, int, int const*, double*);

}  // namespace ext_raft
}  // namespace cugraph
