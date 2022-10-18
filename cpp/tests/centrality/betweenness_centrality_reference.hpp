/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#pragma once

#include <utilities/test_utilities.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <queue>
#include <stack>
#include <vector>

namespace {

template <typename vertex_t, typename edge_t>
void ref_bfs(std::vector<edge_t> const& offsets,
             std::vector<vertex_t> const& indices,
             std::queue<vertex_t>& Q,
             std::stack<vertex_t>& S,
             std::vector<vertex_t>& dist,
             std::vector<std::vector<vertex_t>>& pred,
             std::vector<double>& sigmas,
             vertex_t source)
{
  pred.clear();
  pred.resize(offsets.size() - 1);
  std::fill(dist.begin(), dist.end(), std::numeric_limits<vertex_t>::max());
  std::fill(sigmas.begin(), sigmas.end(), double{0});
  dist[source]   = 0;
  sigmas[source] = 1;
  Q.push(source);

  while (!Q.empty()) {
    vertex_t v = Q.front();
    Q.pop();
    S.push(v);
    for (edge_t nbr_idx = offsets[v]; nbr_idx < offsets[v + 1]; ++nbr_idx) {
      vertex_t nbr = indices[nbr_idx];
      // Path Discovery:
      // Found for the first time?
      if (dist[nbr] == std::numeric_limits<vertex_t>::max()) {
        dist[nbr] = dist[v] + 1;
        Q.push(nbr);
      }
      // Path counting
      // Edge(v, w) on  a shortest path?
      if (dist[nbr] == dist[v] + 1) {
        sigmas[nbr] += sigmas[v];
        pred[nbr].push_back(v);
      }
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void ref_accumulation(std::vector<weight_t>& result,
                      std::stack<vertex_t>& S,
                      std::vector<std::vector<vertex_t>>& pred,
                      std::vector<double>& sigmas,
                      std::vector<double>& deltas,
                      vertex_t source)
{
  std::fill(deltas.begin(), deltas.end(), double{0});

  while (!S.empty()) {
    vertex_t w = S.top();
    S.pop();
    for (vertex_t v : pred[w]) {
      deltas[v] += (sigmas[v] / sigmas[w]) * (1.0 + deltas[w]);
    }
    if (w != source) { result[w] += deltas[w]; }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void ref_endpoints_accumulation(std::vector<weight_t>& result,
                                std::stack<vertex_t>& S,
                                std::vector<std::vector<vertex_t>>& pred,
                                std::vector<double>& sigmas,
                                std::vector<double>& deltas,
                                vertex_t source)
{
  result[source] += S.size() - 1;
  std::fill(deltas.begin(), deltas.end(), double{0});

  while (!S.empty()) {
    vertex_t w = S.top();
    S.pop();
    for (vertex_t v : pred[w]) {
      deltas[v] += (sigmas[v] / sigmas[w]) * (1.0 + deltas[w]);
    }
    if (w != source) { result[w] += deltas[w] + 1; }
  }
}

template <typename vertex_t, typename weight_t>
void ref_edge_accumulation(std::vector<weight_t>& result,
                           std::stack<vertex_t>& S,
                           std::vector<std::vector<vertex_t>>& pred,
                           std::vector<double>& sigmas,
                           std::vector<double>& deltas,
                           vertex_t source)
{
  std::fill(deltas.begin(), deltas.end(), double{0});
  while (!S.empty()) {
    vertex_t w = S.top();
    S.pop();
    for (vertex_t v : pred[w]) {
      deltas[v] += (sigmas[v] / sigmas[w]) * (1.0 + deltas[w]);
    }
    if (w != source) { result[w] += deltas[w]; }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void reference_rescale(result_t* result,
                       bool directed,
                       bool normalize,
                       bool endpoints,
                       vertex_t const number_of_vertices,
                       vertex_t const number_of_sources)
{
  bool modified                      = false;
  result_t rescale_factor            = static_cast<result_t>(1);
  result_t casted_number_of_sources  = static_cast<result_t>(number_of_sources);
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices);
  if (normalize) {
    if (number_of_vertices > 2) {
      if (endpoints) {
        rescale_factor /= (casted_number_of_vertices * (casted_number_of_vertices - 1));
      } else {
        rescale_factor /= ((casted_number_of_vertices - 1) * (casted_number_of_vertices - 2));
      }
      modified = true;
    }
  } else {
    if (!directed) {
      rescale_factor /= static_cast<result_t>(2);
      modified = true;
    }
  }
  if (modified) {
    if (number_of_sources > 0) {
      rescale_factor *= (casted_number_of_vertices / casted_number_of_sources);
    }
  }
  for (auto idx = 0; idx < number_of_vertices; ++idx) {
    result[idx] *= rescale_factor;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::vector<weight_t> betweenness_centrality_reference(
  std::vector<edge_t> const& offsets,
  std::vector<vertex_t> const& indices,
  std::optional<std::vector<weight_t>> const& wgt,
  std::vector<vertex_t> const& seeds,
  bool count_endpoints)
{
  std::vector<weight_t> result;
  if (offsets.size() > 1) {
    result.resize(offsets.size() - 1);

    // Adapted from legacy C++ test implementation
    std::queue<vertex_t> Q;
    std::stack<vertex_t> S;

    std::vector<vertex_t> dist(result.size());
    std::vector<std::vector<vertex_t>> pred(result.size());
    std::vector<double> sigmas(result.size());
    std::vector<double> deltas(result.size());

    std::vector<vertex_t> neighbors;

    for (vertex_t s : seeds) {
      ref_bfs(offsets, indices, Q, S, dist, pred, sigmas, s);

      if (count_endpoints) {
        ref_endpoints_accumulation<vertex_t, edge_t, weight_t>(result, S, pred, sigmas, deltas, s);
      } else {
        ref_accumulation<vertex_t, edge_t, weight_t>(result, S, pred, sigmas, deltas, s);
      }
    }
  }

  return result;
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::vector<weight_t> edge_betweenness_centrality_reference(
  std::vector<edge_t> const& offsets,
  std::vector<vertex_t> const& indices,
  std::optional<std::vector<weight_t>> const& wgt,
  std::vector<vertex_t> const& seeds)
{
  std::vector<weight_t> result;
  if (indices.size() > 0) {
    result.resize(indices.size());

    // Adapted from legacy C++ test implementation
    std::queue<vertex_t> Q;
    std::stack<vertex_t> S;

    std::vector<vertex_t> dist(offsets.size() - 1);
    std::vector<std::vector<vertex_t>> pred(offsets.size() - 1);
    std::vector<double> sigmas(offsets.size() - 1);
    std::vector<double> deltas(offsets.size() - 1);

    for (vertex_t s : seeds) {
      ref_bfs(offsets, indices, Q, S, dist, pred, sigmas, s);

      ref_edge_accumulation(result, S, pred, sigmas, deltas, s);
    }
  }
  return result;
}
}  // namespace
