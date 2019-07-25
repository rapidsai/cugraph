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

#define NEW_CSRMV

#include <algorithm>
#include <iomanip>
#include "include/valued_csr_graph.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_cusparse.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/nvgraph_error.hxx"
#include "include/nvgraph_csrmv.hxx"
#include "include/sssp.hxx"
#ifdef NEW_CSRMV
#include "include/csrmv_cub.h"
#include "cub_semiring/cub.cuh"
#endif
#include <cfloat>

namespace nvgraph
{
template <typename IndexType_, typename ValueType_>
void Sssp<IndexType_, ValueType_>::setup(IndexType source_index, Vector<ValueType>& source_connection,  Vector<ValueType>& sssp_result)
{
    
#ifdef DEBUG
    int n = static_cast<int>(m_network.get_num_vertices());
    if (n != static_cast<int>(source_connection.get_size()) || n != static_cast<int>(sssp_result.get_size()) || !( source_index>=0 && source_index<n) )
    {
        CERR() << "n : " << n << std::endl;
        CERR() << "source_index : " << source_index << std::endl;
        CERR() << "source_connection : " << source_connection.get_size() << std::endl;
        CERR() << "sssp_result : " << sssp_result.get_size() << std::endl;
        FatalError("Wrong input vector in SSSP solver.", NVGRAPH_ERR_BAD_PARAMETERS);
    }
#endif
    m_source = source_index;
    m_tmp = source_connection;
    m_sssp = sssp_result;
    //m_mask.allocate(n, m_stream);
    //m_mask.fill(1, m_stream);
    m_is_setup = true;
}
template <typename IndexType_, typename ValueType_>
bool Sssp<IndexType_, ValueType_>::solve_it()
{
    int n = static_cast<int>(m_network.get_num_vertices()), nnz =  static_cast<int>(m_network.get_num_edges());
    int inc = 1;
    ValueType_ tolerance =  static_cast<float>( 1.0E-6);
    ValueType *sssp = m_sssp.raw(),  *tmp = m_tmp.raw(); //initially set y equal to x
    // int *mask = m_mask.raw();
    
#ifdef NEW_CSRMV
    ValueType_ alpha = cub_semiring::cub::MinPlusSemiring<ValueType_>::times_ident();
    ValueType_ beta = cub_semiring::cub::MinPlusSemiring<ValueType_>::times_ident();
    SemiringDispatch<IndexType_, ValueType_>::template Dispatch< cub_semiring::cub::MinPlusSemiring<ValueType_> >(
        m_network.get_raw_values(),
        m_network.get_raw_row_offsets(),
        m_network.get_raw_column_indices(),
        tmp,
        sssp,
        alpha,
        beta, 
        n,
        n,
        nnz,
        m_stream);
#else
    ValueType_  alpha = 0.0, beta = 0.0; //times_ident = 0 for MinPlus semiring
#if __cplusplus > 199711L
    Semiring SR = Semiring::MinPlus;
#else
    Semiring SR = MinPlus;
#endif
    // y = Network^T op x op->plus x
    // *op* is (plus : min, time : +)
    
    /***************************
    ---> insert csrmv_mp here
    - semiring: (min, +)
    - mask: m_mask
    - parameters:
           (n, n, nnz, 
           alpha,
           m_network,
           tmp,
           beta,
           sssp);
    ****************************/
    csrmv_mp<IndexType_, ValueType_>(n, n, nnz,
                                    alpha,
                                    m_network,
                                    tmp,
                                    beta,
                                    sssp,
                                    SR, 
                                    m_stream);
#endif
    // CVG check : ||tmp - sssp||
    Cublas::axpy(n, (ValueType_)-1.0, sssp, inc, tmp, inc);
    m_residual = Cublas::nrm2(n, tmp, inc);
    if (m_residual < tolerance) 
    {
        return true;
    }
    else
    {
        // we do the convergence check by computing the norm two of tmp = sssp(n-1) - sssp(n)
        // hence if tmp[i] = 0, sssp[i] hasn't changed so we can skip the i th column at the n+1 iteration
        //m_tmp.flag_zeros(m_mask, m_stream);
        m_tmp.copy(m_sssp, m_stream);
        return false;
    }
}
template <typename IndexType_, typename ValueType_>
NVGRAPH_ERROR Sssp<IndexType_, ValueType_>::solve(IndexType source_index, Vector<ValueType>& source_connection, Vector<ValueType>&  sssp_result)
{
    setup(source_index, source_connection, sssp_result);
    bool converged = false;
    int max_it = static_cast<int>(m_network.get_num_edges()), i = 0;

    while (!converged && i < max_it)
    {
        converged = solve_it();
        i++;
    }
    m_iterations = i;
    return converged ? NVGRAPH_OK : NVGRAPH_ERR_NOT_CONVERGED;
}
template class Sssp<int, double>;
template class Sssp<int, float>;
} // end namespace nvgraph

