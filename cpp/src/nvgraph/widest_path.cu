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
#include <cfloat>
#include "include/nvgraph_error.hxx"
#include "include/valued_csr_graph.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_cublas.hxx"
#ifdef NEW_CSRMV
#include "include/csrmv_cub.h"
#include "cub_semiring/cub.cuh"
#endif
#include "include/nvgraph_csrmv.hxx"
#include "include/widest_path.hxx"

namespace nvgraph
{
template <typename IndexType_, typename ValueType_>
void WidestPath<IndexType_, ValueType_>::setup(IndexType source_index, Vector<ValueType>& source_connection,  Vector<ValueType>& widest_path_result)
{
    
#ifdef DEBUG
    int n = static_cast<int>(m_network.get_num_vertices());
    if (n != static_cast<int>(source_connection.get_size()) || n != static_cast<int>(widest_path_result.get_size()) || !( source_index>=0 && source_index<n) )
    {
        CERR() << "n : " << n << std::endl;
        CERR() << "source_index : " << source_index << std::endl;
        CERR() << "source_connection : " << source_connection.get_size() << std::endl;
        CERR() << "widest_path_result : " << widest_path_result.get_size() << std::endl;
        FatalError("Wrong input vector in WidestPath solver.", NVGRAPH_ERR_BAD_PARAMETERS);
    }
#endif
    m_source = source_index;
    m_tmp = source_connection;
    m_widest_path = widest_path_result;
    //m_mask.allocate(n);
    m_is_setup = true;
}
template <typename IndexType_, typename ValueType_>
bool WidestPath<IndexType_, ValueType_>::solve_it()
{
    int n = static_cast<int>(m_network.get_num_vertices()), nnz =  static_cast<int>(m_network.get_num_edges());
    int inc = 1;
    ValueType_ tolerance = static_cast<float>( 1.0E-6);
    ValueType *widest_path = m_widest_path.raw(),  *tmp = m_tmp.raw();
    // int *mask = m_mask.raw();
    // y = Network^T op x op->plus x
    // *op* is (plus : max, time : min)
    
    /***************************
    ---> insert csrmv_mp here
    - semiring: (max, min)
    - mask: m_mask    // not implemented in csrmv
    - parameters:
           (n, n, nnz, 
           alpha,
           m_network,
           tmp,
           beta,
           widest_path);
    ****************************/
  
    // About setting alpha & beta
    // 1. The general Csrmv_mp_sr does :
    //     y = alpha op->time A op->time x op->plus beta op->time y
    // 2. SR = MaxMin has :
    //     plus_ident = SR_type(-inf);
    //     times_ident = SR_type(inf);
    //     times_null = SR_type(-inf);
    // 3. In order to solve : 
    //     y = Network^T op x op->plus x
    //     We need alpha = times_ident
    //                     beta   = times_ident
    

#ifdef NEW_CSRMV
    ValueType_ alpha = cub_semiring::cub::MaxMinSemiring<ValueType_>::times_ident();
    ValueType_ beta = cub_semiring::cub::MaxMinSemiring<ValueType_>::times_ident();
    SemiringDispatch<IndexType_, ValueType_>::template Dispatch< cub_semiring::cub::MaxMinSemiring<ValueType_> >(
        m_network.get_raw_values(),
        m_network.get_raw_row_offsets(),
        m_network.get_raw_column_indices(),
        tmp,
        widest_path,
        alpha,
        beta, 
        n,
        n,
        nnz,
        m_stream);
#else

    ValueType_ inf; 
    if (typeid(ValueType_) == typeid(float)) 
        inf = FLT_MAX ;
    else if (typeid(ValueType_) == typeid(double)) 
         inf = DBL_MAX ;
    else
        FatalError("Graph value type is not supported by this semiring.", NVGRAPH_ERR_BAD_PARAMETERS);

    ValueType_  alpha = inf, beta = inf;
#if __cplusplus > 199711L
    Semiring SR = Semiring::MaxMin;
#else // new csrmv
    Semiring SR = MaxMin;
#endif

    csrmv_mp<IndexType_, ValueType_>(n, n, nnz,
                                    alpha,
                                    m_network,
                                    tmp,
                                    beta,
                                    widest_path,
                                    SR, 
                                    m_stream);
#endif // new csrmv
    // CVG check : ||tmp - widest_path||
    Cublas::axpy(n, (ValueType_)-1.0, widest_path, inc, tmp, inc);
    m_residual = Cublas::nrm2(n, tmp, inc);
    if (m_residual < tolerance) 
    {
        return true;
    }
    else
    {
        // we do the convergence check by computing the norm two of tmp = widest_path(n-1) - widest_path(n)
        // hence if tmp[i] = 0, widest_path[i] hasn't changed so we can skip the i th column at the n+1 iteration
        // m_tmp.flag_zeros(m_mask);
        m_tmp.copy(m_widest_path); // we want x+1 =  Ax +x and csrmv does y = Ax+y, so we copy x in y here.
        return false;
    }
}
template <typename IndexType_, typename ValueType_>
NVGRAPH_ERROR WidestPath<IndexType_, ValueType_>::solve(IndexType source_index, Vector<ValueType>& source_connection, Vector<ValueType>&  widest_path_result)
{
    setup(source_index, source_connection, widest_path_result);
    bool converged = false;
    int max_it = 100000, i = 0;
    while (!converged && i < max_it)
    {
        converged = solve_it();
        i++;
    }
    m_iterations = i;
    return converged ? NVGRAPH_OK : NVGRAPH_ERR_NOT_CONVERGED;
}
template class WidestPath<int, double>;
template class WidestPath<int, float>;
} // end namespace nvgraph

