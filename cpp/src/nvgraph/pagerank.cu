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
//#define NEW_CSRMV

#include "include/valued_csr_graph.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_cusparse.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/nvgraph_error.hxx"
#include "include/pagerank.hxx"
#include "include/pagerank_kernels.hxx"
#ifdef NEW_CSRMV
#include "include/csrmv_cub.h"
#include "include/cub_semiring/cub.cuh"
#endif
#include "include/nvgraph_csrmv.hxx"
#include <algorithm>
#include <iomanip>

namespace nvgraph
{
template <typename IndexType_, typename ValueType_>
Pagerank<IndexType_, ValueType_>::Pagerank(const ValuedCsrGraph <IndexType, ValueType>& network, Vector<ValueType>& dangling_nodes, cudaStream_t stream)
    :m_network(network), m_a(dangling_nodes), m_stream(stream)
{
    // initialize cuda libs outside of the solve (this is slow)
    Cusparse::get_handle();
    Cublas::get_handle();
    m_residual = 1000.0;
    m_damping_factor = 0.0;
}

template <typename IndexType_, typename ValueType_>
void Pagerank<IndexType_, ValueType_>::setup(ValueType damping_factor, Vector<ValueType>& initial_guess, Vector<ValueType>& pagerank_vector)
{
    int n = static_cast<int>(m_network.get_num_vertices());
//    int nnz = static_cast<int>(m_network.get_num_edges());
#ifdef DEBUG
    if (n != static_cast<int>(initial_guess.get_size()) || n != static_cast<int>(m_a.get_size()) || n != static_cast<int>(pagerank_vector.get_size()))
    {
        CERR() << "n : " << n << std::endl;
        CERR() << "m_network.get_num_edges() " << m_network.get_num_edges() << std::endl;
        CERR() << "m_a : " << m_a.get_size() << std::endl;
        CERR() << "initial_guess.get_size() : " << initial_guess.get_size() << std::endl;
        CERR() << "pagerank_vector.get_size() : " << pagerank_vector.get_size() << std::endl;
        FatalError("Wrong input vector in Pagerank solver.", NVGRAPH_ERR_BAD_PARAMETERS);
    }
#endif
    if (damping_factor > 0.999 || damping_factor < 0.0001)
        FatalError("Wrong damping factor value in Pagerank solver.", NVGRAPH_ERR_BAD_PARAMETERS);
	m_damping_factor = damping_factor;
    m_tmp = initial_guess;
    m_pagerank = pagerank_vector;
    //dump(m_a.raw(), 100, 0);
	update_dangling_nodes(n, m_a.raw(), this->m_damping_factor, m_stream);
    //dump(m_a.raw(), 100, 0);
	m_b.allocate(n, m_stream);
    //m_b.dump(0,n);
    ValueType_ val =  static_cast<ValueType_>( 1.0/n);

    //fill_raw_vec(m_b.raw(), n, val); 
    // auto b = m_b.raw();
     m_b.fill(val, m_stream);
    // WARNING force initialization of the initial guess
    //fill(m_tmp.raw(), n, 1.1); 
}

template <typename IndexType_, typename ValueType_>
bool Pagerank<IndexType_, ValueType_>::solve_it()
{
	
    int n = static_cast<int>(m_network.get_num_vertices()), nnz = static_cast<int>(m_network.get_num_edges());
    int inc = 1;
    ValueType_  dot_res;

    ValueType *a = m_a.raw(),
         *b = m_b.raw(),
         *pr = m_pagerank.raw(),
         *tmp = m_tmp.raw();
    
    // normalize the input vector (tmp)
    if(m_iterations == 0)
        Cublas::scal(n, (ValueType_)1.0/Cublas::nrm2(n, tmp, inc) , tmp, inc);
    
    //spmv : pr = network * tmp
#ifdef NEW_CSRMV
    ValueType_ alpha = cub_semiring::cub::PlusTimesSemiring<ValueType_>::times_ident(); // 1.
    ValueType_ beta = cub_semiring::cub::PlusTimesSemiring<ValueType_>::times_null(); // 0.
    SemiringDispatch<IndexType_, ValueType_>::template Dispatch< cub_semiring::cub::PlusTimesSemiring<ValueType_> >(
        m_network.get_raw_values(),
        m_network.get_raw_row_offsets(),
        m_network.get_raw_column_indices(),
        tmp,
        pr,
        alpha,
        beta, 
        n,
        n,
        nnz,
        m_stream);
#else
    ValueType_  alpha = 1.0, beta =0.0;
#if __cplusplus > 199711L
    Semiring SR = Semiring::PlusTimes;
#else
    Semiring SR = PlusTimes;
#endif
    csrmv_mp<IndexType_, ValueType_>(n, n, nnz, 
           alpha,
           m_network,
           tmp,
           beta,
           pr,
           SR, 
           m_stream);
#endif
    
    // Rank one updates
    Cublas::scal(n, m_damping_factor, pr, inc);
    Cublas::dot(n, a, inc, tmp, inc, &dot_res);
    Cublas::axpy(n, dot_res, b, inc, pr, inc);

    // CVG check
    // we need to normalize pr to compare it to tmp 
    // (tmp has been normalized and overwitted at the beginning)
    Cublas::scal(n, (ValueType_)1.0/Cublas::nrm2(n, pr, inc) , pr, inc);
    
    // v = v - x
    Cublas::axpy(n, (ValueType_)-1.0, pr, inc, tmp, inc);
    m_residual = Cublas::nrm2(n, tmp, inc);

    if (m_residual < m_tolerance) // We know lambda = 1 for Pagerank
    {
        // CONVERGED
        // WARNING Norm L1 is more standard for the output of PageRank
        //m_pagerank.dump(0,m_pagerank.get_size());
        Cublas::scal(m_pagerank.get_size(), (ValueType_)1.0/m_pagerank.nrm1(m_stream), pr, inc);
        return true;
    }
    else
    {
        // m_pagerank.dump(0,m_pagerank.get_size());
        std::swap(m_pagerank, m_tmp);
        return false;
    }
}

template <typename IndexType_, typename ValueType_>
NVGRAPH_ERROR Pagerank<IndexType_, ValueType_>::solve(ValueType damping_factor, Vector<ValueType>& initial_guess, Vector<ValueType>& pagerank_vector, float tolerance, int max_it)
{
    m_max_it = max_it;
    m_tolerance = static_cast<ValueType_>(tolerance);
    setup(damping_factor, initial_guess, pagerank_vector);
    bool converged = false;
    int i = 0;

    while (!converged && i < m_max_it)
    { 
        m_iterations = i;
        converged = solve_it();
        i++;
    }
    m_iterations = i;

    if (converged)    
    {
        pagerank_vector = m_pagerank;
    }
    else
    {
        // still return something even if we didn't converged 
        Cublas::scal(m_pagerank.get_size(), (ValueType_)1.0/m_tmp.nrm1(m_stream), m_tmp.raw(), 1);
        pagerank_vector = m_tmp;
    }
        //m_pagerank.dump(0,m_pagerank.get_size());
        //pagerank_vector.dump(0,pagerank_vector.get_size());
    return converged ? NVGRAPH_OK : NVGRAPH_ERR_NOT_CONVERGED;
}

template class Pagerank<int, double>;
template class Pagerank<int, float>;

// init :
// We actually need the transpose (=converse =reverse) of the original network, if the inuput is the original network then we have to transopose it	
// b is a constant and uniform vector, b = 1.0/num_vertices
// a is a constant vector that initialy store the dangling nodes then we set : a = alpha*a + (1-alpha)e
// pagerank is 0 
// tmp is random
// alpha is a constant scalar (0.85 usually)

//loop :
//  pagerank = csrmv (network, tmp)
//  scal(pagerank, alpha); //pagerank =  alpha*pagerank
//  gamma  = dot(a, tmp); //gamma  = a*tmp
//  pagerank = axpy(b, pagerank, gamma); // pagerank = pagerank+gamma*b

// convergence check
//  tmp = axpby(pagerank, tmp, -1, 1);	 // tmp = pagerank - tmp
//  residual_norm = norm(tmp);               
//  if converged (residual_norm)
	  // l1 = l1_norm(pagerank);
	  // pagerank = scal(pagerank, 1/l1);
      // return pagerank 
//  swap(tmp, pagerank)
//end loop

} // end namespace nvgraph

