/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

/**
 * ---------------------------------------------------------------------------*
 * @brief Katz Centrality implementation
 *
 * @file katz_centrality.cu
 * --------------------------------------------------------------------------*/

#include <Hornet.hpp>
#include <Static/KatzCentrality/Katz.cuh>
#include <graph.hpp>
#include "utilities/error_utils.h"

namespace cugraph {

template <typename VT, typename ET, typename WT, typename result_t>
void katz_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                     result_t *result,
                     double alpha,
                     int max_iter,
                     double tol,
                     bool has_guess,
                     bool normalized)
{
  const bool isStatic = true;
  using HornetGraph   = hornet::gpu::HornetStatic<VT>;
  using HornetInit    = hornet::HornetInit<VT>;
  using Katz          = hornets_nest::KatzCentralityStatic;

  HornetInit init(graph.number_of_vertices, graph.number_of_edges, graph.offsets, graph.indices);
  HornetGraph hnt(init, hornet::DeviceType::DEVICE);
  Katz katz(hnt, alpha, max_iter, tol, normalized, isStatic, result);
  if (katz.getAlpha() < alpha) {
    CUGRAPH_FAIL("Error : alpha is not small enough for convergence");
  }
  katz.run();
  if (!katz.hasConverged()) { CUGRAPH_FAIL("Error : Convergence not reached"); }
}

template void katz_centrality<int, int, float, double>(
  experimental::GraphCSRView<int, int, float> const &, double *, double, int, double, bool, bool);

}  // namespace cugraph
