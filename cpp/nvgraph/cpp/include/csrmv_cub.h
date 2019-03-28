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

#include "nvgraph.h"
#include "nvgraph_error.hxx"
#include "multi_valued_csr_graph.hxx"

namespace nvgraph
{

template <typename I, typename V>
class SemiringDispatch
{
public:
    template <typename SR>
    static NVGRAPH_ERROR Dispatch(
        const V*             d_values,
        const I*             d_row_offsets,
        const I*             d_column_indices,
        const V*             d_vector_x,
        V*             d_vector_y,
        V              alpha,
        V              beta, 
        I              num_rows,
        I              num_cols,
        I              num_nonzeros,
        cudaStream_t   stream);

    static NVGRAPH_ERROR InitAndLaunch(
            const nvgraph::MultiValuedCsrGraph<I, V> &graph,
            const size_t weight_index,
            const void *p_alpha,
            const size_t x_index,
            const void *p_beta,
            const size_t y_index,
            const nvgraphSemiring_t SR,
            cudaStream_t stream
        );
};


// API wrapper to avoid bloating main API object nvgraph.cpp
NVGRAPH_ERROR SemiringAPILauncher(nvgraphHandle_t handle,
                           const nvgraphGraphDescr_t descrG,
                           const size_t weight_index,
                           const void *alpha,
                           const size_t x,
                           const void *beta,
                           const size_t y,
                           const nvgraphSemiring_t sr);
} //namespace nvgraph
