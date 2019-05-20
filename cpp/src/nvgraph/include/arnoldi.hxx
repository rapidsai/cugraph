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

#include <vector>   

namespace nvgraph
{

template <typename IndexType_, typename ValueType_>
class ImplicitArnoldi
{
public: 
    typedef IndexType_ IndexType;
    typedef ValueType_ ValueType;

private:
    //Arnoldi
    ValuedCsrGraph <IndexType, ValueType> m_A ;//device
    std::vector<ValueType*> m_Vi;        // Host vector of device adresses -> no it is a 2D vect
    Vector<ValueType> m_V;               // Each colum is a vector of size n, colum major storage  
    Vector<ValueType> m_Q_d;             // Device version of Q (Qt)
    Vector<ValueType> m_V_tmp;           // Output of V*Q <=> QtVt
    Vector<ValueType> m_ritz_eigenvectors_d;
    Vector<ValueType> m_eigenvectors;
    std::vector<ValueType> m_H;                //host
    std::vector<ValueType> m_H_select;                //host
    std::vector<ValueType> m_H_tmp;            //host (lapack likes to overwrite input)
    std::vector<ValueType> m_ritz_eigenvalues; //host
    std::vector<ValueType> m_ritz_eigenvalues_i; //host
    std::vector<ValueType> m_shifts;           //host 
    std::vector<ValueType> m_ritz_eigenvectors;//host
    std::vector<ValueType> m_Q;                //host
    std::vector<ValueType> m_Q_tmp;            //host (lapack likes to overwrite input)
    std::vector<ValueType> m_mns_residuals;      //host resuals of subspaces
    std::vector<ValueType> m_mns_beta;      //host resuals of subspaces

    Vector <ValueType> m_a; // Markov
    Vector <ValueType> m_b; // Markov
    Vector <ValueType> m_D; // Laplacian
    
    ValueType m_beta;     // from arnoldi projection algorithm 
    ValueType m_residual; // is set by compute_residual()
    ValueType m_damping; // for Markov and Pagerank

    float m_tolerance;

    int m_nr_eigenvalues; // the number of wanted eigenvals, also called k in the litterature
    int m_n_eigenvalues; // the number of  eigenvals we keep in the solver, this greater or equal to k, this can be m_nr_eigenvalues or m_nr_eigenvalues+1
    int m_krylov_size;   // the maximum size of the krylov sobspace, also called m in the litterature (m=k+p)
    int m_iterations;    // a counter of restart, each restart cost m_krylov_size-m_n_eigenvalues arnoldi iterations (~spmv)
    int m_max_iter; // maximum number of iterations
    
    int m_parts; // laplacian related

    //miramns related ints
    int m_nested_subspaces;     // the number of subspace to evaluate in MIRAMns
    int m_nested_subspaces_freq;     // the frequence at which we should evaluate subspaces in MIRAMns
    int m_select;        // best subspace size
    int m_select_idx; // best subspace number (0 indexed)
    int m_safety_lower_bound;   // The smallest subspace to check is m_safety_lower_bound+m_nr_eigenvalues+1

    bool m_converged;   
    bool m_is_setup;     
    bool m_has_guess;    
    bool m_markov; 
    bool m_miramns; 
    bool m_dirty_bit; // to know if H has changed, so if we need to call geev 
    bool m_laplacian;
    bool has_init_guess;

    // Warning : here an iteration is a restart 
    bool solve_it();

    //  Input:  A V[0]
    //  Output: V, H, f(=V[m_krylov_size])
    bool solve_arnoldi(int lower_bound, int upper_bound);

    //  Input:  H - a real square upper Hessenberg matrix
    //  Output: w - eigenvalues of H sorted according to which
    //              most wanted to least wanted order     
    //  Optionally compute the eigenvalues of H
    void select_shifts(bool dirty_bit=false);

    // reorder eigenpairs by largest real part
    void LR(int subspace_sz);

    // reorder eigenpairs by largest magnitude
    void LM(int subspace_sz);

    // reorder eigenpairs by smallest real part
    void SR(int subspace_sz);

    //   Input: Q       -- a real square orthogonal matrix
    //          H       -- a real square upper Hessenberg matrix
    //          mu      -- a real shift
    //   Output: Q+     -- a real orthogonal matrix  
    //           H+     -- a real square upper Hessenberg matrix
    // This step will "refine" the subspace by "pushing" the information 
    // into the top left corner
    void qr_step();
    
    // Update V and f using Q+ and H+
    void refine_basis();

    // Approximate residual of the largest Ritz pair of H
    // Optionally compute the eigenvalues of H
    void compute_residual(int subspace_size, bool dirty_bit=false);
    
    void compute_eigenvectors();

    void select_subspace();

    // extract H_select from H
    void extract_subspace(int m);

    // clean everything outside of the new_sz*new_sz hessenberg matrix (in colum major)
    void cleanup_subspace(std::vector<ValueType_>& v, int ld, int new_sz);

    // clean everything outside of the new_sz*new_sz hessenberg matrix (in colum major)
    void shift(std::vector<ValueType_>& H, int ld, int m, ValueType mu);

public:
    // Simple constructor 
    ImplicitArnoldi(void) {};
    // Simple destructor
    ~ImplicitArnoldi(void) {};

    // Create a ImplicitArnoldi Solver 
    ImplicitArnoldi(const ValuedCsrGraph <IndexType, ValueType>& A);

    // Create a ImplicitArnoldi Solver with support of graph laplacian generation
    ImplicitArnoldi(const ValuedCsrGraph <IndexType, ValueType>& A, int parts);

    // Create a ImplicitArnoldi Solver with support of damping factor and rank one updates (pagerank, markov ...)
    ImplicitArnoldi(const ValuedCsrGraph <IndexType, ValueType>& A, Vector<ValueType>& dangling_nodes, const float tolerance, const int max_iter, ValueType alpha=0.95);
 
    void setup( Vector<ValueType>& initial_guess, const int restart_it, const int nEigVals); // public because we want to use and test that directly and/or separately

    // Starting from  V, H, f :
    // Call the QRstep, project the update, launch the arnlodi with the new base 
    // and check the quality of the new result 
    void implicit_restart(); // public because we want to use and test that directly and/or separately

    // The total number of SPMV will be : m_krylov_size + (m_krylov_size-m_n_eigenvalues)*nb_restart
    NVGRAPH_ERROR solve(const int restart_it, const int nEigVals, 
                     Vector<ValueType>& initial_guess,
                     Vector<ValueType>& eigVals,
                     Vector<ValueType>& eigVecs,
                     const int n_sub_space=0);

    inline ValueType get_residual() const {return m_residual;}
    inline int get_iterations() const {return m_iterations;}

    // we use that for tests, unoptimized copies/transfers inside
    std::vector<ValueType> get_H_copy() {return m_H;}
    std::vector<ValueType> get_Hs_copy() {return m_H_select;}
    std::vector<ValueType> get_ritz_eval_copy(){return m_ritz_eigenvalues;} // should be called after select_shifts
    std::vector<ValueType> get_V_copy();
    std::vector<ValueType> get_f_copy();
    std::vector<ValueType> get_fp_copy();
};

} // end namespace nvgraph

