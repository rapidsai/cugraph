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
#pragma once

#include <graph.hpp>

namespace cugraph {

/**
 * @brief     Find the PageRank vertex values for a graph.
 *
 * cuGraph computes an approximation of the Pagerank eigenvector using the power method.
 * The number of iterations depends on the properties of the network itself; it increases
 * when the tolerance descreases and/or alpha increases toward the limiting value of 1.
 * The user is free to use default values or to provide inputs for the initial guess,
 * tolerance and maximum number of iterations.
 *
 * @throws                           cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported value : float or double.   
 *
 * @param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as a transposed adjacency list (CSR). Edge weights are not used for this algorithm.
 * @param[in] alpha                  The damping factor alpha represents the probability to follow an outgoing edge, standard value is 0.85.
                                     Thus, 1.0-alpha is the probability to “teleport” to a random vertex. Alpha should be greater than 0.0 and strictly lower than 1.0.
 *                                   The initial guess must not be the vector of 0s. Any value other than 1 or 0 is treated as an invalid value.
 * @param[in] pagerank               Array of size V. Should contain the initial guess if has_guess=true. In this case the initial guess cannot be the vector of 0s. Memory is provided and owned by the caller.
 * @param[in] personalization_subset_size (optional) The number of vertices for to personalize. Initialized to 0 by default.
 * @param[in] personalization_subset (optional) Array of size personalization_subset_size containing vertices for running personalized pagerank. Initialized to nullptr by default. Memory is provided and owned by the caller.
 * @param[in] personalization_values (optional) Array of size personalization_subset_size containing values associated with personalization_subset vertices. Initialized to nullptr by default. Memory is provided and owned by the caller.
 * @param[in] tolerance              Set the tolerance the approximation, this parameter should be a small magnitude value.
 *                                   The lower the tolerance the better the approximation. If this value is 0.0f, cuGRAPH will use the default value which is 1.0E-5.
 *                                   Setting too small a tolerance can lead to non-convergence due to numerical roundoff. Usually values between 0.01 and 0.00001 are acceptable.
 * @param[in] max_iter               (optional) The maximum number of iterations before an answer is returned. This can be used to limit the execution time and do an early exit before the solver reaches the convergence tolerance.
 *                                   If this value is lower or equal to 0 cuGRAPH will use the default value, which is 500.
 * @param[in] has_guess              (optional) This parameter is used to notify cuGRAPH if it should use a user-provided initial guess. False means the user does not have a guess, in this case cuGRAPH will use a uniform vector set to 1/V.
 *                                   If the value is True, cuGRAPH will read the pagerank parameter and use this as an initial guess.
 * @param[out] *pagerank             The PageRank : pagerank[i] is the PageRank of vertex i. Memory remains provided and owned by the caller.
 *
 */
template <typename VT, typename ET, typename WT>
void pagerank(experimental::GraphCSC<VT,ET,WT> const &graph,
              WT* pagerank,
              VT personalization_subset_size=0, 
              VT* personalization_subset=nullptr, 
              WT* personalization_values=nullptr,
              double alpha = 0.85,
              double tolerance = 1e-5, 
              int64_t max_iter = 500,
              bool has_guess = false);

/**
 * @brief     Compute betweenness centrality for a graph
 *
 * Betweenness centrality for a vertex is the sum of the fraction of
 * all pairs shortest paths that pass through the vertex.
 * 
 * @throws                           cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.   
 * @tparam result_t                  Type of computed result.  Supported values :  float
 *
 * @param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as a transposed adjacency list (CSR). Edge weights are not used for this algorithm.
 * @param[out] result                Device array of centrality scores
 *
 *
 * TODO:  Could add parameters for sampling and normalization, current implementation
 *        computes exact answer and normalizes.
 */
template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSR<VT,ET,WT> const &graph,
                            result_t *result);

} //namespace cugraph
