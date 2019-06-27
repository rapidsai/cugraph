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

#include "cub_semiring/cub.cuh"

#include "nvgraph/nvgraph.h"

#include "include/nvgraphP.h"
#include "include/nvgraph_error.hxx"
#include "include/csrmv_cub.h"

namespace nvgraph
{

template <typename I, typename V>template <typename SR>
NVGRAPH_ERROR SemiringDispatch<I, V>::Dispatch(
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
        cudaStream_t   stream)
{
    // std::static_assert(std::is_same<typename std::remove_cv<T>::type, int>::value, "current CUB implementation supports int only for indices");
    size_t temp_buf_size = 0;
    cudaError_t err = cub_semiring::cub::DeviceSpmv::CsrMV<V, SR>( NULL, temp_buf_size, d_values, d_row_offsets, d_column_indices, d_vector_x, 
        d_vector_y, alpha, beta, num_rows, num_cols, num_nonzeros, stream);
    CHECK_CUDA(err);
    Vector<char> tmp_buf(std::max(temp_buf_size, size_t(1)), stream);
    err = cub_semiring::cub::DeviceSpmv::CsrMV<V, SR>( tmp_buf.raw(), temp_buf_size, d_values, d_row_offsets, d_column_indices, d_vector_x, 
        d_vector_y, alpha, beta, num_rows, num_cols, num_nonzeros, stream);
    CHECK_CUDA(err);
    return NVGRAPH_OK;
};

// deconstructs graph, checks parameters and dispatches semiring implementation
template <typename I, typename V>
NVGRAPH_ERROR SemiringDispatch<I, V>::InitAndLaunch(
            const nvgraph::MultiValuedCsrGraph<I, V> &graph,
            const size_t weight_index,
            const void *p_alpha,
            const size_t x_index,
            const void *p_beta,
            const size_t y_index,
            const nvgraphSemiring_t SR,
            cudaStream_t stream
        )
{
    if (weight_index >= graph.get_num_edge_dim() || x_index >= graph.get_num_vertex_dim() || y_index >= graph.get_num_vertex_dim()) // base index is 0
        return NVGRAPH_ERR_BAD_PARAMETERS;
    I n = static_cast<I>(graph.get_num_vertices());
    I nnz = static_cast<I>(graph.get_num_edges());
    const V* vals = graph.get_raw_edge_dim(weight_index);
    const V* x = graph.get_raw_vertex_dim( x_index);
    V* y = const_cast<V*>(graph.get_raw_vertex_dim(y_index));
    V alpha = *(static_cast<const V*>(p_alpha));
    V beta = *(static_cast<const V*>(p_beta));
    const I* row_ptr = graph.get_raw_row_offsets();
    const I* col_ind = graph.get_raw_column_indices(); 
    
    NVGRAPH_ERROR err = NVGRAPH_ERR_BAD_PARAMETERS;

    switch (SR)
    {
        case NVGRAPH_PLUS_TIMES_SR: 
            err = Dispatch< cub_semiring::cub::PlusTimesSemiring<V> >(vals, row_ptr, col_ind, x, y, alpha, beta, n, n, nnz, stream);
            break;
        case NVGRAPH_MIN_PLUS_SR: 
            err = Dispatch< cub_semiring::cub::MinPlusSemiring<V> >(vals, row_ptr, col_ind, x, y, alpha, beta, n, n, nnz, stream);
            break;
        case NVGRAPH_MAX_MIN_SR: 
            err = Dispatch< cub_semiring::cub::MaxMinSemiring<V> >(vals, row_ptr, col_ind, x, y, alpha, beta, n, n, nnz, stream);
            break;
        case NVGRAPH_OR_AND_SR:
            err = Dispatch< cub_semiring::cub::OrAndBoolSemiring<V> >(vals, row_ptr, col_ind, x, y, alpha, beta, n, n, nnz, stream);
            break;
        default:
            break;
    }
    return err;
};

// API wrapper to avoid bloating main API object nvgraph.cpp
NVGRAPH_ERROR SemiringAPILauncher(nvgraphHandle_t handle,
                           const nvgraphGraphDescr_t descrG,
                           const size_t weight_index,
                           const void *alpha,
                           const size_t x,
                           const void *beta,
                           const size_t y,
                           const nvgraphSemiring_t sr)
{
    typedef int I;

    if (descrG->graphStatus!=HAS_VALUES) // need a MultiValuedCsrGraph
        return NVGRAPH_ERR_BAD_PARAMETERS;

    if (descrG->TT != NVGRAPH_CSR_32) // supported topologies
        return NVGRAPH_ERR_BAD_PARAMETERS;

    cudaStream_t stream = handle->stream;

    NVGRAPH_ERROR err = NVGRAPH_ERR_NOT_IMPLEMENTED; 

    switch(descrG->T)
        {
            case CUDA_R_32F :
            {
                const nvgraph::MultiValuedCsrGraph<I, float> *mcsrg = static_cast<const nvgraph::MultiValuedCsrGraph<I, float>*> (descrG->graph_handle);
                err = SemiringDispatch<I, float>::InitAndLaunch( *mcsrg, weight_index, static_cast<const float*>(alpha), x, 
                    static_cast<const float*>(beta), y, sr, stream);
                break;
            }
            case CUDA_R_64F :
            {
                const nvgraph::MultiValuedCsrGraph<I, double> *mcsrg = static_cast<const nvgraph::MultiValuedCsrGraph<I, double>*> (descrG->graph_handle);
                err = SemiringDispatch<I, double>::InitAndLaunch( *mcsrg, weight_index, static_cast<const double*>(alpha), x, 
                    static_cast<const double*>(beta), y, sr, stream);
                break;
            }
            default:
                break;
        }
    return err;
};

} //namespace nvgraph
