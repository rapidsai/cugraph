/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include "types.h"

namespace cugraph {

/**
 * @Synopsis   Creates source, destination and value columns based on the specified R-MAT model
 *
 * @Param[in] *argv                  String that accepts the following arguments
 *                                   rmat (default: rmat_scale = 10, a = 0.57, b = c = 0.19)
 *                                               Generate R-MAT graph as input
 *                                               --rmat_scale=<vertex-scale>
 *                                               --rmat_nodes=<number-nodes>
 *                                               --rmat_edgefactor=<edge-factor>
 *                                               --rmat_edges=<number-edges>
 *                                               --rmat_a=<factor> --rmat_b=<factor>
 * --rmat_c=<factor>
 *                                               --rmat_self_loops If this option is supplied, then
 * self loops will be retained
 *                                               --rmat_undirected If this option is not mentioned,
 * then the graps will be undirected Optional arguments:
 *                                       [--device=<device_index>] Set GPU(s) for testing (Default:
 * 0).
 *                                       [--quiet]                 No output (unless --json is
 * specified).
 *                                       [--random_seed]           This will enable usage of random
 * seed, else it will use same seed
 *
 * @Param[out] &vertices             Number of vertices in the generated edge list
 *
 * @Param[out] &edges                Number of edges in the generated edge list
 *
 * @Param[out] *src                  Columns containing the sources
 *
 * @Param[out] *dst                  Columns containing the destinations
 *
 * @Param[out] *val                  Columns containing the edge weights
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void grmat_gen(const char* argv,
               size_t& vertices,
               size_t& edges,
               gdf_column* src,
               gdf_column* dest,
               gdf_column* val);

}  // namespace cugraph
