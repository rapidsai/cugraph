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

#include <algorithm>
#include <iomanip>
#include <utility>
#include <curand.h>

#include "include/valued_csr_graph.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_vector_kernels.hxx"
#include "include/nvgraph_cusparse.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/nvgraph_lapack.hxx"
#include "include/nvgraph_error.hxx"
#include "include/pagerank_kernels.hxx"
#include "include/arnoldi.hxx"
#include "include/nvgraph_csrmv.hxx"
#include "include/matrix.hxx"

namespace nvgraph
{

template <typename IndexType_, typename ValueType_>
ImplicitArnoldi<IndexType_, ValueType_>::ImplicitArnoldi(const ValuedCsrGraph <IndexType, ValueType>& A)
    :m_A(A), m_markov(false), m_laplacian(false), m_tolerance(1.0E-12), m_iterations(0), m_dirty_bit(false), m_max_iter(500), has_init_guess(false)
{
//     initialize cuda libs outside of the solve (this is slow)
//    cusparseHandle_t t1 = Cusparse::get_handle();
//    cublasHandle_t t2 = Cublas::get_handle();

//  compiler is complainig, unused variables
    Cusparse::get_handle();
    Cublas::get_handle();
}

template <typename IndexType_, typename ValueType_>
ImplicitArnoldi<IndexType_, ValueType_>::ImplicitArnoldi(const ValuedCsrGraph <IndexType, ValueType>& A, int parts)
    :m_A(A), m_parts(parts), m_laplacian(true), m_markov(false), m_tolerance(1.0E-9), m_iterations(0), m_dirty_bit(false), m_max_iter(500), has_init_guess(false)
{
//     initialize cuda libs outside of the solve (this is slow)
//    cusparseHandle_t t1 = Cusparse::get_handle();
//    cublasHandle_t t2 = Cublas::get_handle();

//  compiler is complainig, unused variables
    Cusparse::get_handle();
    Cublas::get_handle();
}

template <typename IndexType_, typename ValueType_>
ImplicitArnoldi<IndexType_, ValueType_>::ImplicitArnoldi(const ValuedCsrGraph <IndexType, ValueType>& A, Vector<ValueType>& dangling_nodes, const float tolerance, const int max_iter, ValueType alpha)
    :m_A(A),  m_a(dangling_nodes), m_damping(alpha), m_markov(true), m_laplacian(false), m_tolerance(tolerance), m_iterations(0), m_dirty_bit(false), m_max_iter(max_iter), has_init_guess(false)
{
//     initialize cuda libs outside of the solve (this is slow)
//    cusparseHandle_t t1 = Cusparse::get_handle();
//    cublasHandle_t t2 = Cublas::get_handle();

//  compiler is complainig, unused variables
    Cusparse::get_handle();
    Cublas::get_handle();
}

template <typename IndexType_, typename ValueType_>
NVGRAPH_ERROR ImplicitArnoldi<IndexType_, ValueType_>::solve(const int restart_it, const int nEigVals,
                                                          Vector<ValueType>& initial_guess,
                                                          Vector<ValueType>& eigVals,
                                                          Vector<ValueType>& eigVecs,
                                                          const int nested_subspaces_freq)
{
    //try {
    m_nested_subspaces_freq = nested_subspaces_freq;

    setup(initial_guess, restart_it, nEigVals);
    m_eigenvectors = eigVecs;
    bool converged = false;
    int i = 0;
    // we can print stats after setup to have the initial residual
    while (!converged && i< m_max_iter)
    {
        // re-add the extra eigenvalue in case QR step changed it.
        m_n_eigenvalues = m_nr_eigenvalues+1; 
        converged = solve_it();
        i++;
    }
    m_iterations = i;
    if (!m_miramns)
    { 
        if (m_laplacian)
        {
            SR(m_krylov_size); 
        }
        else if  (m_markov)
        {
             LR(m_select); 
        }
        else
        {
            LM(m_krylov_size); 
        }
     }
    compute_eigenvectors();
    cudaMemcpyAsync(eigVals.raw(), &m_ritz_eigenvalues[0], (size_t)(m_nr_eigenvalues*sizeof(m_ritz_eigenvalues[0])), cudaMemcpyHostToDevice);
    cudaCheckError();
    // } catch (const std::exception &exc) {std::cout << exc.what();}
    // x = m_x; // sometime there is a mixup between pointers, need to investigate that.
    return NVGRAPH_OK;
}

template <typename IndexType_, typename ValueType_> 
void ImplicitArnoldi<IndexType_, ValueType_>::setup(Vector<ValueType>& initial_guess, const int restart_it, const int nEigVals)
{
    m_krylov_size = restart_it;
    m_select = m_krylov_size;
    m_nr_eigenvalues = nEigVals;

    // We always compute an extra eigenvalue to make sure we always have m_nr_eigenvalues
    // So even if the double shifted QR consume the m_n_eigenvalues^th eigenvalue we are fine
    m_n_eigenvalues = m_nr_eigenvalues+1;

    // General parameter check
    if(m_krylov_size >= static_cast<int>(m_A.get_num_vertices())) 
        FatalError("ARNOLDI: The krylov subspace size is larger than the matrix", NVGRAPH_ERR_BAD_PARAMETERS);
    if(m_n_eigenvalues >= m_krylov_size) 
        FatalError("ARNOLDI: The number of required eigenvalues +1 is larger than the maximum krylov subspace size", NVGRAPH_ERR_BAD_PARAMETERS);
    if(m_krylov_size < 3) 
        FatalError("ARNOLDI: Sould perform at least 3 iterations before restart", NVGRAPH_ERR_BAD_PARAMETERS);

    // Some checks on optional Markov parameters
    if (m_markov)
    {
        if (m_nr_eigenvalues != 1)
            FatalError("ARNOLDI: Only one eigenpair is needed for the equilibrium of a Markov chain", NVGRAPH_ERR_BAD_PARAMETERS);
        if (m_damping > 0.99999 || m_damping < 0.0001)
           FatalError("ARNOLDI: Wrong damping factor value", NVGRAPH_ERR_BAD_PARAMETERS);
    }

    //if (m_laplacian)
    //{
    //   if (m_parts > m_n_eigenvalues)
    //    FatalError("IRAM: ", NVGRAPH_ERR_BAD_PARAMETERS);
    //}

    // Some checks on optional miramns parameters
    if ( m_nested_subspaces_freq <= 0)
    {
        m_nested_subspaces = 0;
        m_miramns=false;
    }
    else
    {
        m_safety_lower_bound = 7;
        if( m_nested_subspaces_freq > (m_krylov_size-(m_safety_lower_bound+m_nr_eigenvalues+1))) // ie not enough space betwen the number of ev and the max size of the subspace
        {
    #ifdef DEBUG
            COUT()<<"MIRAMns Warning: Invalid frequence of nested subspaces, nested_subspaces_freq > m_max-4*n_eigVal" << std::endl;
    #endif
            m_miramns=false;
        }
        else
        {
            m_miramns=true;
            // This formula should give the number of subspaces
            // We allways count the smallest, the largest plus every size matching m_nested_subspaces_freq between them.
            m_nested_subspaces = 2 + (m_krylov_size-(m_safety_lower_bound+m_nr_eigenvalues+1)-1)/m_nested_subspaces_freq;
            
            //COUT()<<"Number of nested subspaces : "<<m_nested_subspaces << std::endl;
            //COUT()<<"nested_subspaces_freq "<< m_nested_subspaces_freq << std::endl;
        }

    }


    m_residual = 1.0E6;
    
    //Allocations
    size_t n = m_A.get_num_vertices();
//    nnz is not used
//    size_t nnz = m_A.get_num_edges();
    // Device
    m_V.allocate(n*(m_krylov_size + 1));
    m_V_tmp.allocate(n*(m_n_eigenvalues + 1));
    m_ritz_eigenvectors_d.allocate(m_krylov_size*m_krylov_size);
    m_Q_d.allocate(m_krylov_size*m_krylov_size);
    
    //Host
    m_Vi.resize(m_krylov_size + 1);
    m_ritz_eigenvalues.resize(m_krylov_size);
    m_ritz_eigenvalues_i.resize(m_krylov_size);
    m_ritz_eigenvectors.resize(m_krylov_size * m_krylov_size);
    m_H.resize(m_krylov_size * m_krylov_size);
    m_H_select.resize(m_select*m_select);
    m_H_tmp.resize(m_krylov_size * m_krylov_size);
    m_Q.resize(m_krylov_size * m_krylov_size);
    if(m_miramns)
    {
        m_mns_residuals.resize(m_nested_subspaces);
        m_mns_beta.resize(m_nested_subspaces);
    }

    for (int i = 0; i < static_cast<int>(m_Vi.size()); ++i)
    {
        m_Vi[i]=m_V.raw()+i*n;
    }
    if (!has_init_guess)
    {
      const ValueType_ one  = 1;
      const ValueType_ zero = 0;
      curandGenerator_t randGen;
      // Initialize random number generator
      CHECK_CURAND(curandCreateGenerator(&randGen,CURAND_RNG_PSEUDO_PHILOX4_32_10));
      CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(randGen, 123456/*time(NULL)*/));
      // Initialize initial  vector
      CHECK_CURAND(curandGenerateNormalX(randGen, m_V.raw(), n, zero, one));
      ValueType_ normQ1 = Cublas::nrm2(n, m_V.raw(), 1);
      Cublas::scal(n, (ValueType_)1.0/normQ1, m_V.raw(), 1);
    }
    else
     {   
        m_V.copy(initial_guess);
    }
    //dump_raw_vec (m_V.raw(), 10, 0);
    if(m_markov)
    {
        update_dangling_nodes(n, m_a.raw(), static_cast<ValueType_>( m_damping));
        //dump(m_a.raw(), 100, 0);
        m_b.allocate(n);
        ValueType_ val =  static_cast<float>(1.0/n); //
        m_b.fill(val);
        //m_b.dump(0,n);
    }

    if (m_laplacian)
    {
        // degree matrix
        m_D.allocate(n);
        m_b.allocate(n);
        ValueType_ val = 1.0;
        m_b.fill(val);
        size_t n = m_A.get_num_vertices();
        size_t nnz = m_A.get_num_edges();
        ValueType_  alpha = 1.0, beta =0.0, gamma= -1.0; 

#if __cplusplus > 199711L
        Semiring sring = Semiring::PlusTimes;   
#else 
        Semiring sring = PlusTimes;   
#endif
        csrmv_mp<IndexType_, ValueType_>(n, n, nnz, alpha, m_A, m_b.raw(), beta, m_D.raw(), sring);
        //Cusparse::csrmv(false, false, 
        //    n, n, nnz,
        //    &alpha,
        //    m_A.get_raw_values(),
        //    m_A.get_raw_row_offsets(),
        //    m_A.get_raw_column_indices(),
        //    m_b.raw(),
        //    &beta,
        //    m_D.raw());
        Cublas::scal(nnz, gamma, m_A.get_raw_values(), 1);
        
        // m_b can be deleted now
        //dump_raw_vec ( m_A.get_raw_values(), nnz, 0);
        //dump_raw_vec (m_D.raw(), n, 0);
    }


    // normalize
    Cublas::scal(n, (ValueType_)1.0/Cublas::nrm2(n, m_Vi[0], 1) , m_Vi[0], 1);
    m_iterations = 0;
    // arnoldi from 0 to k
    solve_arnoldi(0,m_krylov_size);
    
}
#ifdef DEBUG
template <typename ValueType_>
void dump_host_dense_mat(std::vector<ValueType_>& v, int ld)
{
    std::stringstream ss;
    ss.str(std::string());
    ss << std::setw(10);
    ss.precision(3);        
    for (int i = 0; i < ld; ++i)
    {
     for (int j = 0; j < ld; ++j)
     {
        ss << v[i*ld+j]  << std::setw(10);
     }  
     ss << std::endl;
    }
    COUT()<<ss.str();
}

template <typename ValueType_>
void dump_host_vec(std::vector<ValueType_>& v)
{
    std::stringstream ss;
    ss.str(std::string());
    ss << std::setw(10);
    ss.precision(4);        
    for (int i = 0; i < v.size(); ++i)
        ss << v[i]  << std::setw(10);
     ss << std::endl;
    COUT()<<ss.str();
}
#endif

template <typename IndexType_, typename ValueType_>
bool ImplicitArnoldi<IndexType_, ValueType_>::solve_arnoldi(int lower_bound, int upper_bound)
{
    int inc =1, mns_residuals_idx = 0;
    size_t n = m_A.get_num_vertices();
    size_t nnz = m_A.get_num_edges();

    ValueType_  alpha = 1.0, beta =0.0, Hji = 0, dot_res; 
   
#if __cplusplus > 199711L
    Semiring sring = Semiring::PlusTimes;   
#else
    Semiring sring = PlusTimes;   
#endif
    
    //m_V.dump(lower_bound*n,n);
    
    if (m_miramns) 
    {
        std::fill (m_mns_residuals.begin(),m_mns_residuals.end(),0.0);
    }

    for (int i = lower_bound; i < upper_bound; ++i)
    {
        // beta = norm(f); v = f/beta; 
        if (i>0 && i == lower_bound)
        {
            m_beta = Cublas::nrm2(n, m_Vi[i], 1);
            // Vi = Vi/||Vi||
            Cublas::scal(n, (ValueType_)1.0/m_beta, m_Vi[i], inc);
            // m_V.dump((i-1)*n,n);
        }

        //  Compute H, V and f
        csrmv_mp<IndexType_, ValueType_>(n, n, nnz, alpha, m_A, m_Vi[i], beta, m_Vi[i+1], sring);
        //if (i == 0) dump_raw_vec (m_Vi[i+1], n, 0);
        if (m_laplacian) 
        {
          //apply to the external diagonal
          dmv(n, alpha, m_D.raw(), m_Vi[i], alpha, m_Vi[i+1]);
          //dump_raw_vec ( m_D.raw(), 10, 0);
          //dump_raw_vec (m_Vi[i+1], 10, 0);
        }

        if(m_markov)
        {
            Cublas::scal(n, m_damping, m_Vi[i+1], inc);
            Cublas::dot(n, m_a.raw(), inc, m_Vi[i], inc, &dot_res); 
            Cublas::axpy(n, dot_res, m_b.raw(), inc,  m_Vi[i+1], inc); 
        }
        
        // Modified GS algorithm
        for (int j = 0; j <= i; ++j)
        {
            // H(j,i) = AVi.Vj
            Cublas::dot(n, m_Vi[i+1], inc, m_Vi[j], inc, &Hji);
            m_H[i*m_krylov_size + j] = Hji;
            //V(i + 1) -= H(j, i) * V(j) 
            Cublas::axpy(n, -Hji, m_Vi[j],inc, m_Vi[i+1],inc);
        }
        if (i > 0)
        {
            // H(i+1,i) = ||Vi|| <=> H(i,i-1) = ||Vi||
            m_H[(i-1)*m_krylov_size + i] = m_beta;
        }
        //||Vi+1||
        m_beta = Cublas::nrm2(n, m_Vi[i+1], 1);
        if (i+1 < upper_bound) 
        {
            
            Cublas::scal(n, (ValueType_)1.0/m_beta, m_Vi[i+1], inc);
        }

        if (m_miramns) 
        {
            // The smallest subspaces is always m_safety_lower_bound+m_nr_eigenvalues+1
            // The largest is allways max_krylov_size, 
            // Between that we check the quality at every stride (m_nested_subspaces_freq).
            if( i == m_safety_lower_bound+m_nr_eigenvalues || 
                i+1 == upper_bound || 
                (i > m_safety_lower_bound+m_nr_eigenvalues && ((i-(m_safety_lower_bound+m_nr_eigenvalues))%m_nested_subspaces_freq == 0)) )
            {
                //COUT()<<"i "<<i<<", idx "<<mns_residuals_idx << std::endl;
                //dump_host_dense_mat(m_H, m_krylov_size);
                compute_residual(i+1,true); // it is i+1 just because at an iteration i the subspace size is i+1
                //m_mns_residuals[m_krylov_size-m_n_eigenvalues-(m_krylov_size-i)] = m_residual;
                m_mns_beta[mns_residuals_idx] = m_beta;
                //store current residual
                m_mns_residuals[mns_residuals_idx] = m_residual; 
                mns_residuals_idx++;

                // early exit if converged
                if (m_residual<m_tolerance) 
                {    
                    // prepare for exit here 
                    //m_select = m_krylov_size-m_n_eigenvalues-(m_krylov_size-i)+1;
                    m_select = i+1;

                    if (m_laplacian)
                    {
                        SR(m_select); 
                    }
                    else if  (m_markov)
                    {
                         LR(m_select); 
                    }
                    else
                    {
                        LM(m_select); 
                    }

                    return true; 
                }
            }
        }
        
    }
   // dump_host_dense_mat(m_H, m_krylov_size);
    // this is where we compute the residual after the arnoldi reduction in IRAM
    if (!m_miramns)
        compute_residual(m_krylov_size, true);
    
    return m_converged; // maybe we can optimize that later
}

template <typename IndexType_, typename ValueType_>
bool ImplicitArnoldi<IndexType_, ValueType_>::solve_it()
{

    if (m_residual<m_tolerance) return true; // no need to do the k...p arnoldi steps

    if (m_miramns)
    {    
        int prev = m_select;
        select_subspace();
        extract_subspace(prev);
    }
    implicit_restart();
    
    return solve_arnoldi(m_n_eigenvalues, m_krylov_size); // arnoldi from k to m
}

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::select_subspace()
{
#if __cplusplus > 199711L
    typename std::vector<ValueType_>::iterator it = std::min_element(std::begin(m_mns_residuals), std::end(m_mns_residuals));
#else
    typename std::vector<ValueType_>::iterator it = std::min_element(m_mns_residuals.begin(), m_mns_residuals.end());
#endif

    m_residual = *it;
#if __cplusplus > 199711L
    int dist = static_cast<int>(std::distance(std::begin(m_mns_residuals), it));
#else
    int dist = static_cast<int>(std::distance(m_mns_residuals.begin(), it));
#endif
    m_select = std::min((m_safety_lower_bound+m_nr_eigenvalues) + (m_nested_subspaces_freq*dist) +1, m_krylov_size);
    m_select_idx = dist ; 
    //COUT()<<"m_select "<<m_select<< std::endl;
}

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::extract_subspace(int m)
{
    
    if (m != m_select || m_H_select.size() == 0)
    {
        m_H_select.resize(m_select*m_select);
        m_H_tmp.resize(m_select*m_select);
        m_Q.resize(m_select*m_select);
        m_Q_tmp.resize(m_select*m_select);
    }
    //m_ritz_eigenvalues.resize(m_select);; //host
    //m_ritz_eigenvectors.resize(m_select*m_select);
    // copy
    //int k = m_krylov_size-m_select;
    //int l = 0;
    //for(int i = k; i<m_krylov_size; i++)
    //{
    //    for(int j = 0; j<m_select; j++)
    //    {
    //       m_H_select[l*m_select+j] = m_H[i*m_krylov_size+j];
    //    }
    //    l++;
    //}

    for(int i = 0; i<m_select; i++)
    {
        for(int j = 0; j<m_select; j++)
        {
           m_H_select[i*m_select+j] = m_H[i*m_krylov_size+j];
        }
    }
    // retrieve || f || if needed
    if (m_select < m_krylov_size)
        m_beta = m_mns_beta[m_select_idx];

    m_dirty_bit = true;
}

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::compute_residual(int subspace_size, bool dirty_bit)
{
    //dump_host_dense_mat(m_H_select, m_select);
    if (m_miramns)  
    {
        
        if (dirty_bit)
        {
            if (static_cast<int>(m_H_tmp.size()) != subspace_size*subspace_size)
                m_H_tmp.resize(subspace_size*subspace_size);
            //std::fill (m_ritz_eigenvalues.begin(),m_ritz_eigenvalues.end(),0.0);
            //std::fill (m_ritz_eigenvectors.begin(),m_ritz_eigenvectors.end(),0.0);

            for(int i = 0; i<subspace_size; i++)
            {
                for(int j = 0; j<subspace_size; j++)
                {
                   m_H_tmp[i*subspace_size+j] = m_H[i*m_krylov_size+j];
                }
            }
            // dump_host_dense_mat(m_H_tmp,subspace_size);
            //Lapack<ValueType_>::geev(&m_H_tmp[0], &m_ritz_eigenvalues[0], &m_ritz_eigenvectors[0], subspace_size , subspace_size, subspace_size);
            Lapack<ValueType_>::geev(&m_H_tmp[0], &m_ritz_eigenvalues[0], &m_ritz_eigenvalues_i[0], &m_ritz_eigenvectors[0], NULL, subspace_size , subspace_size, subspace_size);
        }
    }
    else
    {
        if (dirty_bit)
        {
            // we change m_H_tmp size during miramns
            if (m_H_tmp.size() != m_H.size())
                m_H_tmp.resize(m_H.size());
            std::copy(m_H.begin(), m_H.end(), m_H_tmp.begin());
            //Lapack<ValueType_>::geev(&m_H_tmp[0], &m_ritz_eigenvalues[0], &m_ritz_eigenvectors[0], m_krylov_size , m_krylov_size, m_krylov_size);
            Lapack<ValueType_>::geev(&m_H_tmp[0], &m_ritz_eigenvalues[0],  &m_ritz_eigenvalues_i[0], &m_ritz_eigenvectors[0], NULL, m_krylov_size , m_krylov_size, m_krylov_size);
        }
    }

    //COUT() << "m_ritz_eigenvalues : "<<std::endl;
    //dump_host_vec(m_ritz_eigenvalues);
    //COUT() << "m_ritz_eigenvectors : "<<std::endl;
    //dump_host_dense_mat(m_ritz_eigenvectors, subspace_size);

    // sort 
    if (m_laplacian)
    {
        SR(subspace_size);
    }
    else if  (m_markov)
    {
          LR(m_select); 
     }
    else
    {
        LM(subspace_size); 
    }
    //COUT() << "m_ritz_eigenvalues : "<<std::endl;
   // dump_host_vec(m_ritz_eigenvalues);
    ValueType_ last_ritz_vector, residual_norm, tmp_residual;
    ValueType_ lam;
    m_residual = 0.0f;

    // Convergence check  by approximating the residual of the Ritz pairs.
    if  (m_markov)
     {
          last_ritz_vector = m_ritz_eigenvectors[subspace_size-1];
           //COUT() << "last_ritz_vector : "<<last_ritz_vector<<std::endl;
          // if (!last_ritz_vector)
          //  dump_host_dense_mat(m_ritz_eigenvectors, subspace_size);
          // COUT() << "m_beta : "<<m_beta<<std::endl;
         m_residual = std::abs(last_ritz_vector * m_beta);
         if (m_residual == 0.0)
            m_residual = 1.0E6;
     }
     else
    {
        for (int i = 0; i < m_n_eigenvalues; i++)
        {
            last_ritz_vector = m_ritz_eigenvectors[i * subspace_size + subspace_size-1];
            residual_norm = std::abs(last_ritz_vector * m_beta);
           if(m_ritz_eigenvalues_i[i])
               lam = std::sqrt(m_ritz_eigenvalues[i]*m_ritz_eigenvalues[i] + m_ritz_eigenvalues_i[i]*m_ritz_eigenvalues_i[i]);
           else
               lam = std::abs(m_ritz_eigenvalues[i]);

            tmp_residual = residual_norm / lam;
            if (m_residual<tmp_residual)
                m_residual = tmp_residual;
        }
    }

    if (m_residual < m_tolerance)
    {
        m_converged = true;
    }
    else
    {
        m_converged = false;
    }
}

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::implicit_restart()
{
    // optim:  avoid the cpy here 
    if (!m_miramns) std::copy(m_H.begin(), m_H.end(), m_H_select.begin());
    select_shifts(m_dirty_bit);

    qr_step();

    refine_basis();

    // optim:  avoid the cpy here 
    if (!m_miramns) std::copy(m_H_select.begin(), m_H_select.end(), m_H.begin());
}

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::select_shifts(bool dirty_bit)
{
    // dirty_bit is false by default
    if (dirty_bit)
    {
        std::copy(m_H_select.begin(), m_H_select.end(), m_H_tmp.begin()); 
        //Lapack<ValueType_>::geev(&m_H_tmp[0], &m_ritz_eigenvalues[0], &m_ritz_eigenvectors[0], m_select , m_select, m_select);
        Lapack<ValueType_>::geev(&m_H_tmp[0], &m_ritz_eigenvalues[0],&m_ritz_eigenvalues_i[0], &m_ritz_eigenvectors[0], NULL, m_select , m_select, m_select);
    }
    m_dirty_bit = false;
    if (m_laplacian)
    {
        SR(m_select); 
    }
    else if  (m_markov)
    {
         LR(m_select); 
    }
    else
    {
        LM(m_select); 
    }
    // in the future we can quikly add LM, SM, SR
    // complex (LI SI) are not supported.

}


#if __cplusplus <= 199711L
    template<typename ValueType_>
    bool cmp_LR(const std::pair<int,ValueType_> &left, const std::pair<int,ValueType_> &right){
        return left.second > right.second;
    };
#endif


template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::LR(int subspace_sz)
{
    // Eigen values of interest have the largest real part 
    std::vector<std::pair<int,ValueType_> > items;
    for (int i = 0; i < subspace_sz; ++i)
        items.push_back(std::make_pair( i, m_ritz_eigenvalues[i]));

    // this is a reverse  key value sort by algebraic value
    // in this case we select the largest eigenvalues
    // In the future we can add other shift selection strategies here
    // to converge to different eigen values (reverse sort by magnitude, or usual sort by magnitude etc ).
#if __cplusplus > 199711L
    std::sort(items.begin(), items.end(),[](const std::pair<int,ValueType_> &left, const std::pair<int,ValueType_> &right) 
                                             {return left.second > right.second; });
#else
    std::sort(items.begin(), items.end(), cmp_LR<ValueType_>);
#endif

    // Now we need to reorder the vectors accordingly
    std::vector<ValueType_> ritz_tmp(m_ritz_eigenvectors);

    for (int i = 0; i < subspace_sz; ++i)
    {
        //COUT() << "reordrering : " << items[i].first <<std::endl
        //                 << "start : " <<items[i].first*subspace_sz<<std::endl
        //                 << "end : " <<items[i].first*subspace_sz+subspace_sz<<std::endl
        //                 << "out : " <<i*subspace_sz<<std::endl;

        std::copy(ritz_tmp.begin() + (items[i].first*subspace_sz), 
                           ritz_tmp.begin() + (items[i].first*subspace_sz + subspace_sz), 
                           m_ritz_eigenvectors.begin()+(i*subspace_sz)); 
        m_ritz_eigenvalues[i] = items[i].second;
    }
    // dump_host_vec(m_ritz_eigenvalues);
    std::vector<ValueType_> tmp_i(m_ritz_eigenvalues_i);
    for (int i = 0; i < subspace_sz; ++i)
    {
        m_ritz_eigenvalues_i[i] = tmp_i[items[i].first];
    }
}


template<typename ValueType_>
bool cmp_LM(const std::pair<int,ValueType_> &left, const std::pair<int,ValueType_> &right){
    return left.second > right.second;
};

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::LM(int subspace_sz)
{ 
    std::vector<ValueType_> magnitude(subspace_sz);
    std::vector<std::pair<int, ValueType_ > > kv;
    
    for (int i = 0; i < subspace_sz; ++i)
       magnitude[i] = m_ritz_eigenvalues[i]*m_ritz_eigenvalues[i] + m_ritz_eigenvalues_i[i]*m_ritz_eigenvalues_i[i];
    
    for (int i = 0; i < subspace_sz; ++i)
        kv.push_back(std::make_pair( i, magnitude[i]));

    // this is a reverse  key value sort by magnitude 
    // in this case we select the largest magnitude

    std::sort(kv.begin(), kv.end(), cmp_LM<ValueType_>);

    // Now we need to reorder the vectors accordingly
    std::vector<ValueType_> ritz_tmp(m_ritz_eigenvectors);
    std::vector<ValueType_> ev(m_ritz_eigenvalues);
    std::vector<ValueType_> ev_i(m_ritz_eigenvalues_i);
    for (int i = 0; i < subspace_sz; ++i)
    {
        //COUT() << "reordrering : " << kv[i].first <<std::endl
        //                 << "start : " <<kv[i].first*subspace_sz<<std::endl
        //                 << "end : " <<kv[i].first*subspace_sz+subspace_sz<<std::endl
        //                 << "out : " <<i*subspace_sz<<std::endl;
        std::copy(ritz_tmp.begin() + (kv[i].first*subspace_sz), 
                  ritz_tmp.begin() + (kv[i].first*subspace_sz + subspace_sz), 
                  m_ritz_eigenvectors.begin()+(i*subspace_sz)); 
        m_ritz_eigenvalues[i] = ev[kv[i].first];
        m_ritz_eigenvalues_i[i] = ev_i[kv[i].first];
    }
}

#if __cplusplus <= 199711L
    template<typename ValueType_>
    bool cmp_SR(const std::pair<int,ValueType_> &left, const std::pair<int,ValueType_> &right){
        return left.second < right.second;
    };
#endif

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::SR(int subspace_sz)
{
    // Eigen values of interest have the largest real part 
    std::vector<std::pair<int,ValueType_> > items;
    for (int i = 0; i < subspace_sz; ++i)
        items.push_back(std::make_pair( i, m_ritz_eigenvalues[i]));

    // this is a reverse  key value sort by algebraic value
    // in this case we select the largest eigenvalues
    // In the future we can add other shift selection strategies here
    // to converge to different eigen values (reverse sort by magnitude, or usual sort by magnitude etc ).
#if __cplusplus > 199711L
    std::sort(items.begin(), items.end(),[](const std::pair<int,ValueType_> &left, const std::pair<int,ValueType_> &right) 
                                             {return left.second < right.second; });
#else
    std::sort(items.begin(), items.end(), cmp_SR<ValueType_>);
#endif

    // Now we need to reorder the vectors accordingly
    std::vector<ValueType_> ritz_tmp(m_ritz_eigenvectors);

    for (int i = 0; i < subspace_sz; ++i)
    {
        //COUT() << "reordrering : " << items[i].first <<std::endl
        //                 << "start : " <<items[i].first*subspace_sz<<std::endl
        //                 << "end : " <<items[i].first*subspace_sz+subspace_sz<<std::endl
        //                 << "out : " <<i*subspace_sz<<std::endl;

        std::copy(ritz_tmp.begin() + (items[i].first*subspace_sz), 
                           ritz_tmp.begin() + (items[i].first*subspace_sz + subspace_sz), 
                           m_ritz_eigenvectors.begin()+(i*subspace_sz)); 
        m_ritz_eigenvalues[i] = items[i].second;
    }
    // dump_host_vec(m_ritz_eigenvalues);
}

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::qr_step()
{   
    ValueType_ mu, mu_i, mu_i_sq;
    int n = m_select;
    int ld = m_select;
    std::vector<ValueType> tau(n);
    std::vector<ValueType> work(n);
    int lwork = -1; 
    // workspace query
    std::copy (m_H_select.begin(),m_H_select.end(), m_H_tmp.begin());
    Lapack<ValueType_>::geqrf(n, n, &m_H_tmp[0], ld, &tau[0], &work[0], &lwork);
    // work is a real array used as workspace. On exit, if LWORK = -1, work[0] contains the optimal LWORK.
    // it can be safely casted to int here to remove the conversion warning.
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    // Q0 = I
    m_Q.assign(m_Q.size(),0.0);
    shift(m_Q, m_select, m_select, -1);
    //for (int j = 0; j < m_select; j++)
    //    m_Q[j*m_select+j] = 1.0;
   
    int i = m_select-1;
    while (i >= m_n_eigenvalues)
    {
        //Get the shift
        mu_i = m_ritz_eigenvalues_i[i];
        mu = m_ritz_eigenvalues[i];
        shift(m_H_tmp, m_select, m_select, mu);

        if (mu_i )
        {
            //Complex case
            //Double shift
            //(H - re_mu*I)^2 + im_mu^2*I)

            if (i==m_n_eigenvalues)
            {
                // if we are in this case we will consume the  next eigen value which is a wanted eigenalue
                // fortunately  m_n_eigenvalues = m_nr_eigenvalues +1 (we alway compute one more eigenvalue)
                m_n_eigenvalues -=1;

                //COUT() << "IRAM: last ev absorded in double shift" <<std::endl;
            } 
            //COUT() << "Complex  shift"<<std::endl;
            //COUT() << "shift : " << mu  << " " << mu_i << "i" <<std::endl;   
            std::vector<ValueType> A(m_select*m_select);
            
            for (int ii = 0; ii < m_select; ii++)
                for (int k = 0; k < m_select; k++)
                    for (int j = 0; j < m_select; j++)
                        A[ii*m_select+j] +=  m_H_tmp[ii*m_select+k]* m_H_tmp[k*m_select+j];
            mu_i_sq = mu_i*mu_i;
            std::copy (A.begin(),A.end(), m_H_tmp.begin());
            shift(m_H_tmp, m_select, m_select, -mu_i_sq);

             //COUT() << "H"<< m_select-i<<std::endl;
             //dump_host_dense_mat(m_H_tmp, m_select);
        }

          // [Q,R] = qr(H - mu*I);
         Lapack<ValueType_>::geqrf(n, n, &m_H_tmp[0], ld, &tau[0], &work[0], &lwork);
        //H+ = (Q)'* H * Q ;
        Lapack<ValueType_>::ormqr(false, true, n, n, n, &m_H_tmp[0], ld, &tau[0], &m_H_select[0], n, &work[0], &lwork);
        Lapack<ValueType_>::ormqr(true, false, n, n, n, &m_H_tmp[0], ld, &tau[0], &m_H_select[0], n, &work[0], &lwork);
        
        //Q+ = Q+*Q; 
        Lapack<ValueType_>::ormqr(true, false, n, n, n, &m_H_tmp[0], ld, &tau[0], &m_Q[0], n, &work[0], &lwork);
         
        // clean up below subdiagonal (column major storage)

        cleanup_subspace(m_H_select, m_select,m_select);
        //for (int j = 0; j < m_select-1; j++)
        //   for (int k = j+2; k < m_select; k++)
        //       m_H_select[j*m_select + k] = 0;
         
        //COUT() << "shift : " << mu <<std::endl;   
        //COUT() << "H"<< m_select-i<<std::endl;
        //dump_host_dense_mat(m_H_select, m_select);
        //COUT() << "Q"<< m_select-i <<std::endl;
        //dump_host_dense_mat(m_Q, m_select);

        std::copy (m_H_select.begin(),m_H_select.end(), m_H_tmp.begin());
        // Example for how to explicitly form Q
        // Lapack<ValueType_>::orgqr(n, n, n, &m_H_tmp[0], ld, &tau[0], &work[0], &lwork); 
        // std::copy (m_H_tmp.begin(),m_H_tmp.end(), m_Q.begin());
        if (mu_i) 
              i-=2; //complex
        else 
              i-=1; //real
    }

}

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::refine_basis()
{
    ValueType_ alpha, beta;

    // update f (and send on dev at some point) 
    // Back to row major -> transpose Q and mind which element we pick in H (ie stored as Ht).
    // copy Q to dev 
    // Need Mat1*Mat2, where Mat1(n,m) is tall, skin, dense and  Mat2(m,l) is small dense with l<m and m<<n

    // something like f+1 = V(:,1:m)*Q(:,n_ev+1)*H(n_ev+1,n_ev) + f*Q(m,n_ev);
    // ie vec =  Lmat Svec scal +Svec scal , all dense (L=large S=small)
    // just local small name for variables
    int n = m_A.get_num_vertices(),
        nev = m_n_eigenvalues,
        nk = m_select;
    
    m_Q_d.fill(0);

    ValueType_ *fptr = m_V_tmp.raw()+n*nev; // = Vi[nev]
    cudaMemcpyAsync(m_Q_d.raw(), &m_Q[0], (size_t)(m_select*m_select*sizeof(m_Q[0])), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpyAsync(fptr, m_Vi[nk], (size_t)(n*sizeof(ValueType_)), cudaMemcpyDeviceToDevice); cudaCheckError();

    alpha = m_Q[(nev-1) * nk + nk - 1];
    beta = 1.0;

    // retrieve f from v[m_select] if needed 
    // We could also store the vector f for each nested subspace 
    if (m_select!=m_krylov_size)
        Cublas::scal(n, m_beta, fptr, 1);
    
    Cublas::scal(n, alpha, fptr, 1);

    alpha = m_H_select[(nev-1) * nk + nev ];

    Cublas::gemm(false, false, n, 1, nk, &alpha, m_V.raw(), n, m_Q_d.raw(), nk, &beta, fptr, n);
    
    //COUT() << "f+ : "<<std::endl;   
    //m_V_tmp.dump(2*n,n);
    //COUT() <<std::endl;   

    //V(:,1:m)*Q(:,n_ev+1)*H(n_ev+1,n_ev)
    // ie Lmat =  Lmat * Smat, all dense (L=large S=small)
    // <=> tmpT = H(n_ev, n_ev+1) V*Q in col maj
    
    alpha = 1.0;
    beta = 0.0;

    // debug cleaning
    //m_Q_d.fill(0);
    //cudaMemcpyAsync(m_Q_d.raw(), &m_Q[0], (size_t)(nev*m_select*sizeof(m_Q[0])), cudaMemcpyHostToDevice);
    //fill_raw_vec (m_V_tmp.raw(), n*(nev+1), beta);
    //fill_raw_vec (m_V.raw()+n*nk, n, beta);
    
    //COUT() << "QT : "<<std::endl;   
    //m_Q_d.dump(0,m_select);
    //m_Q_d.dump(1*m_select, m_select);
    //m_Q_d.dump(2*m_select, m_select);
    //m_Q_d.dump(3*m_select, m_select);
    //COUT() <<std::endl;   
    
    //COUT() << "VT : "<<std::endl;   
    //m_V.dump(0,n);
    //m_V.dump(1*n,n);
    //m_V.dump(2*n,n);
    //m_V.dump(3*n,n);
    ////m_V.dump(4*n,n);
    //COUT() <<std::endl;   

    //cudaDeviceSynchronize();

    Cublas::gemm(false, false, n, nev, nk, &alpha, m_V.raw(), n, m_Q_d.raw(), nk, 
                                           &beta, m_V_tmp.raw(), n);

    m_V.copy(m_V_tmp);

    // update H
    if (m_miramns)
    {
        for(int i = 0; i<m_select; i++)
            for(int j = 0; j<m_select; j++)
               m_H[i*m_krylov_size+j] = m_H_select[i*m_select+j];
        cleanup_subspace(m_H, m_krylov_size,m_n_eigenvalues);
    }
}   

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::compute_eigenvectors()
{
    //dump_host_vec(m_ritz_eigenvalues);
    //dump_host_dense_mat(m_ritz_eigenvectors,m_select);
    int n = m_A.get_num_vertices(),
        nev = m_nr_eigenvalues,
        nk = m_select;
    ValueType_ alpha=1.0, beta = 0.0;
    cudaMemcpyAsync(m_ritz_eigenvectors_d.raw(), &m_ritz_eigenvectors[0], (size_t)(m_select*m_select*sizeof(m_ritz_eigenvectors[0])), cudaMemcpyHostToDevice);
    cudaCheckError();
    Cublas::gemm(false, false, n, nev, nk, &alpha, m_V.raw(), n, 
                 m_ritz_eigenvectors_d.raw(), nk, 
                 &beta,  m_eigenvectors.raw(), n);
    //nrm 1 for pagerank
    if(m_markov) 
        Cublas::scal(n, (ValueType_)1.0/m_eigenvectors.nrm1(), m_eigenvectors.raw(), 1);
}

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::cleanup_subspace(std::vector<ValueType_>& v, int ld, int new_sz)
{

    // just a simple clean

    //    In               Out
    // * * 0 0 0        * * 0 0 0   
    // * * * 0 0        * * * 0 0 
    // * * * * 0        * * * * 0 
    // * * * * *        * * * * 0  <--- new_sz
    // * * * * *        0 0 0 0 0

    for (int i = 0; i < new_sz-1; i++)
      for (int j = i+2; j < new_sz; j++)
          v[i*ld + j] = 0;
    for (int i = new_sz; i < ld; i++)
      for (int j = 0; j < ld; j++)
        v[i*ld + j] = 0;
    for (int i = 0; i < new_sz; i++)
      for (int j = new_sz; j < ld; j++)
        v[i*ld + j] = 0;

    // Not used anymore
    //    In               Out
    // * * 0 0 0        0 0 0 0 0   
    // * * * 0 0        0 0 0 0 0 
    // * * * * 0        * * 0 0 0 <--- new_sz
    // * * * * *        * * * 0 0
    // * * * * *        * * * 0 0
    //int k = ld-new_sz;
    //for (int i = 0; i < ld; ++i)
    // for (int j = 0; j < ld; ++j)
    //    if ((i < k) ||  
    //        (j >= new_sz) || 
    //        (i >= k && j-1 > i-k ))        
    //            v[i*ld+j] = 0.0;  

}

template <typename IndexType_, typename ValueType_>
void ImplicitArnoldi<IndexType_, ValueType_>::shift(std::vector<ValueType_>& H, int ld, int m, ValueType mu)
{
    int start = ld-m;
    for (int i = start; i < ld; i++)
        H[i*ld+i-start] -= mu;
}

template <typename IndexType_, typename ValueType_>
std::vector<ValueType_> ImplicitArnoldi<IndexType_, ValueType_>::get_f_copy()
{
    std::vector<ValueType> tmp(m_A.get_num_vertices());
    cudaMemcpyAsync(&tmp[0],m_Vi[m_krylov_size], (size_t)(m_A.get_num_vertices()*sizeof(ValueType_)), cudaMemcpyDeviceToHost);
    cudaCheckError();
    return tmp;
}

template <typename IndexType_, typename ValueType_>
std::vector<ValueType_> ImplicitArnoldi<IndexType_, ValueType_>::get_fp_copy()
{
    std::vector<ValueType> tmp(m_A.get_num_vertices());
    cudaMemcpyAsync(&tmp[0],m_Vi[m_n_eigenvalues], (size_t)(m_A.get_num_vertices()*sizeof(ValueType_)), cudaMemcpyDeviceToHost);
    cudaCheckError();
    return tmp;
}

template <typename IndexType_, typename ValueType_>
std::vector<ValueType_> ImplicitArnoldi<IndexType_, ValueType_>::get_V_copy()
{
    std::vector<ValueType> tmp(m_A.get_num_vertices()*(m_krylov_size+1));
    cudaMemcpyAsync(&tmp[0],m_V.raw(), (size_t)(m_A.get_num_vertices()*(m_krylov_size+1)*sizeof(ValueType_)), cudaMemcpyDeviceToHost);
    cudaCheckError();
    return tmp;
}


template class ImplicitArnoldi<int, double>;
template class ImplicitArnoldi<int, float>;
} // end namespace nvgraph

