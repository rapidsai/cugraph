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

namespace nvgraph
{
template <typename IndexType_, typename ValueType_>
class Pagerank 
{
public: 
    typedef IndexType_ IndexType;
    typedef ValueType_ ValueType;

private:
	ValuedCsrGraph <IndexType, ValueType> m_network ;
	Vector <ValueType> m_a;
	Vector <ValueType> m_b;
	Vector <ValueType> m_pagerank;
	Vector <ValueType> m_tmp;
	ValueType m_damping_factor;
	ValueType m_residual;
	ValueType m_tolerance;
	cudaStream_t m_stream;
	int m_iterations;
	int m_max_it;
	bool m_is_setup;
	bool m_has_guess;

	bool solve_it();
	//void update_dangling_nodes(Vector<ValueType_>& dangling_nodes);
	void setup(ValueType damping_factor, Vector<ValueType>& initial_guess, Vector<ValueType>& pagerank_vector);

public:
	// Simple constructor 
	Pagerank(void) {};
	// Simple destructor
	~Pagerank(void) {};

	// Create a Pagerank Solver attached to a the transposed of a transition matrix
	// *** network is the transposed of a transition matrix***
	Pagerank(const ValuedCsrGraph <IndexType, ValueType>& network, Vector<ValueType>& dangling_nodes, cudaStream_t stream = 0);
	
	// dangling_nodes is a vector of size n where dangling_nodes[i] = 1.0 if vertex i is a dangling node and 0.0 otherwise
    // pagerank_vector is the output
    //void solve(ValueType damping_factor, Vector<ValueType>& dangling_nodes, Vector<ValueType>& pagerank_vector);
   // setup with an initial guess of the pagerank
    NVGRAPH_ERROR solve(ValueType damping_factor, Vector<ValueType>& initial_guess, Vector<ValueType>& pagerank_vector, float tolerance =1.0E-6, int max_it = 500);
    inline ValueType get_residual() const {return m_residual;}
    inline int get_iterations() const {return m_iterations;}


// init :
// We need the transpose (=converse =reverse) in input (this can be seen as a CSC matrix that we see as CSR)
// b is a constant and uniform vector, b = 1.0/num_vertices
// a is a constant vector that initialy store the dangling nodes then we set : a = alpha*a + (1-alpha)e
// pagerank is 0
// tmp is random ( 1/n is fine)
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
};

} // end namespace nvgraph

