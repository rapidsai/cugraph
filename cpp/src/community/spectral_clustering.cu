/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

/** ---------------------------------------------------------------------------*
 * @brief Wrapper functions for Nvgraph
 *
 * @file nvgraph_wrapper.cpp
 * ---------------------------------------------------------------------------**/

#include <algorithms.hpp>
#include <graph.hpp>

#include <nvgraph/include/sm_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <utilities/error.hpp>
#include <ctime>
#include <nvgraph/include/nvgraph_error.hxx>

#include <nvgraph/include/modularity_maximization.hxx>
#include <nvgraph/include/nvgraph_cublas.hxx>
#include <nvgraph/include/nvgraph_cusparse.hxx>
#include <nvgraph/include/partition.hxx>

#include <nvgraph/include/spectral_matrix.hxx>

namespace cugraph {
namespace nvgraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void balancedCutClustering_impl(experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                                vertex_t n_clusters,
                                vertex_t n_eig_vects,
                                weight_t evs_tolerance,
                                int evs_max_iter,
                                weight_t kmean_tolerance,
                                int kmean_max_iter,
                                vertex_t *clustering,
                                weight_t *eig_vals,
                                weight_t *eig_vects)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(evs_tolerance >= weight_t{0.0},
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(evs_tolerance < weight_t{1.0},
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance >= weight_t{0.0},
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance < weight_t{1.0},
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(n_clusters > 1, "API error, must specify more than 1 cluster");
  CUGRAPH_EXPECTS(n_clusters < graph.number_of_vertices,
                  "API error, number of clusters must be smaller than number of vertices");
  CUGRAPH_EXPECTS(n_eig_vects <= n_clusters,
                  "API error, cannot specify more eigenvectors than clusters");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");
  CUGRAPH_EXPECTS(eig_vals != nullptr, "API error, must specify valid eigenvalues");
  CUGRAPH_EXPECTS(eig_vects != nullptr, "API error, must specify valid eigenvectors");

  int evs_max_it{4000};
  int kmean_max_it{200};
  weight_t evs_tol{1.0E-3};
  weight_t kmean_tol{1.0E-2};

  if (evs_max_iter > 0) evs_max_it = evs_max_iter;

  if (evs_tolerance > weight_t{0.0}) evs_tol = evs_tolerance;

  if (kmean_max_iter > 0) kmean_max_it = kmean_max_iter;

  if (kmean_tolerance > weight_t{0.0}) kmean_tol = kmean_tolerance;

  int restartIter_lanczos = 15 + n_eig_vects;

  ::nvgraph::partition<vertex_t, edge_t, weight_t>(graph,
                                                   n_clusters,
                                                   n_eig_vects,
                                                   evs_max_it,
                                                   restartIter_lanczos,
                                                   evs_tol,
                                                   kmean_max_it,
                                                   kmean_tol,
                                                   clustering,
                                                   eig_vals,
                                                   eig_vects);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void spectralModularityMaximization_impl(
  experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  vertex_t n_clusters,
  vertex_t n_eig_vects,
  weight_t evs_tolerance,
  int evs_max_iter,
  weight_t kmean_tolerance,
  int kmean_max_iter,
  vertex_t *clustering,
  weight_t *eig_vals,
  weight_t *eig_vects)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(evs_tolerance >= weight_t{0.0},
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(evs_tolerance < weight_t{1.0},
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance >= weight_t{0.0},
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance < weight_t{1.0},
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(n_clusters > 1, "API error, must specify more than 1 cluster");
  CUGRAPH_EXPECTS(n_clusters < graph.number_of_vertices,
                  "API error, number of clusters must be smaller than number of vertices");
  CUGRAPH_EXPECTS(n_eig_vects <= n_clusters,
                  "API error, cannot specify more eigenvectors than clusters");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");
  CUGRAPH_EXPECTS(eig_vals != nullptr, "API error, must specify valid eigenvalues");
  CUGRAPH_EXPECTS(eig_vects != nullptr, "API error, must specify valid eigenvectors");

  int evs_max_it{4000};
  int kmean_max_it{200};
  weight_t evs_tol{1.0E-3};
  weight_t kmean_tol{1.0E-2};

  int iters_lanczos, iters_kmeans;

  if (evs_max_iter > 0) evs_max_it = evs_max_iter;

  if (evs_tolerance > weight_t{0.0}) evs_tol = evs_tolerance;

  if (kmean_max_iter > 0) kmean_max_it = kmean_max_iter;

  if (kmean_tolerance > weight_t{0.0}) kmean_tol = kmean_tolerance;

  int restartIter_lanczos = 15 + n_eig_vects;
  ::nvgraph::modularity_maximization<vertex_t, edge_t, weight_t>(graph,
                                                                 n_clusters,
                                                                 n_eig_vects,
                                                                 evs_max_it,
                                                                 restartIter_lanczos,
                                                                 evs_tol,
                                                                 kmean_max_it,
                                                                 kmean_tol,
                                                                 clustering,
                                                                 eig_vals,
                                                                 eig_vects,
                                                                 iters_lanczos,
                                                                 iters_kmeans);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void analyzeModularityClustering_impl(
  experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  int n_clusters,
  vertex_t const *clustering,
  weight_t *modularity)
{
  weight_t mod;
  ::nvgraph::analyzeModularity(graph, n_clusters, clustering, mod);
  *modularity = mod;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void analyzeBalancedCut_impl(experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                             vertex_t n_clusters,
                             vertex_t const *clustering,
                             weight_t *edgeCut,
                             weight_t *ratioCut)
{
  CUGRAPH_EXPECTS(n_clusters <= graph.number_of_vertices,
                  "API error: number of clusters must be <= number of vertices");
  CUGRAPH_EXPECTS(n_clusters > 0, "API error: number of clusters must be > 0)");

  weight_t edge_cut, ratio_cut;

  ::nvgraph::analyzePartition(graph, n_clusters, clustering, edge_cut, ratio_cut);

  *edgeCut  = edge_cut;
  *ratioCut = ratio_cut;
}

}  // namespace detail

template <typename VT, typename ET, typename WT>
void balancedCutClustering(experimental::GraphCSRView<VT, ET, WT> const &graph,
                           VT num_clusters,
                           VT num_eigen_vects,
                           WT evs_tolerance,
                           int evs_max_iter,
                           WT kmean_tolerance,
                           int kmean_max_iter,
                           VT *clustering)
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
void spectralModularityMaximization(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                    VT n_clusters,
                                    VT n_eigen_vects,
                                    WT evs_tolerance,
                                    int evs_max_iter,
                                    WT kmean_tolerance,
                                    int kmean_max_iter,
                                    VT *clustering)
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
void analyzeClustering_modularity(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                  int n_clusters,
                                  VT const *clustering,
                                  WT *score)
{
  detail::analyzeModularityClustering_impl(graph, n_clusters, clustering, score);
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_edge_cut(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                int n_clusters,
                                VT const *clustering,
                                WT *score)
{
  WT dummy{0.0};
  detail::analyzeBalancedCut_impl(graph, n_clusters, clustering, score, &dummy);
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_ratio_cut(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                 int n_clusters,
                                 VT const *clustering,
                                 WT *score)
{
  WT dummy{0.0};
  detail::analyzeBalancedCut_impl(graph, n_clusters, clustering, &dummy, score);
}

template void balancedCutClustering<int, int, float>(
  experimental::GraphCSRView<int, int, float> const &, int, int, float, int, float, int, int *);
template void balancedCutClustering<int, int, double>(
  experimental::GraphCSRView<int, int, double> const &, int, int, double, int, double, int, int *);
template void spectralModularityMaximization<int, int, float>(
  experimental::GraphCSRView<int, int, float> const &, int, int, float, int, float, int, int *);
template void spectralModularityMaximization<int, int, double>(
  experimental::GraphCSRView<int, int, double> const &, int, int, double, int, double, int, int *);
template void analyzeClustering_modularity<int, int, float>(
  experimental::GraphCSRView<int, int, float> const &, int, int const *, float *);
template void analyzeClustering_modularity<int, int, double>(
  experimental::GraphCSRView<int, int, double> const &, int, int const *, double *);
template void analyzeClustering_edge_cut<int, int, float>(
  experimental::GraphCSRView<int, int, float> const &, int, int const *, float *);
template void analyzeClustering_edge_cut<int, int, double>(
  experimental::GraphCSRView<int, int, double> const &, int, int const *, double *);
template void analyzeClustering_ratio_cut<int, int, float>(
  experimental::GraphCSRView<int, int, float> const &, int, int const *, float *);
template void analyzeClustering_ratio_cut<int, int, double>(
  experimental::GraphCSRView<int, int, double> const &, int, int const *, double *);

}  // namespace nvgraph
}  // namespace cugraph
