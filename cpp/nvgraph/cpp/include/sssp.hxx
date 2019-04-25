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
#include <climits> 
namespace nvgraph
{
template <typename IndexType_, typename ValueType_>
class Sssp 
{
public: 
    typedef IndexType_ IndexType;
    typedef ValueType_ ValueType;

private:
    ValuedCsrGraph <IndexType, ValueType> m_network ;
    Vector <ValueType> m_sssp;
    Vector <ValueType> m_tmp;
    Vector <int> m_mask; // mask[i] = 0 if we can ignore the i th column in the csrmv

    IndexType m_source;
    ValueType m_residual;
    int m_iterations;
    bool m_is_setup;

    cudaStream_t m_stream;

    bool solve_it();
    void setup(IndexType source_index, Vector<ValueType>& source_connection,  Vector<ValueType>& sssp_result);

public:
    // Simple constructor 
    Sssp(void) {};
    // Simple destructor
    ~Sssp(void) {};

    // Create a Sssp solver attached to a the transposed of a  weighted network
    // *** network is the transposed/CSC*** 
    Sssp(const ValuedCsrGraph <IndexType, ValueType>& network, cudaStream_t stream = 0):m_network(network),m_is_setup(false), m_stream(stream)  {};
   
    /*! Find the sortest path from  the vertex source_index to every other vertices.
     *
     *  \param source_index The source. 
     *  \param source_connection The connectivity of the source
     *                                                  if there is a link from source_index to i, source_connection[i] = E(source_index, i) 
     *                                                  otherwise  source_connection[i] = inifinity 
     *                                                  source_connection[source_index] = 0
                                                         The source_connection is computed somewhere else.
     *  \param (output) m_sssp  m_sssp[i] contains the sortest path from  the source to the vertex i.
     */
     
    NVGRAPH_ERROR solve(IndexType source_index, Vector<ValueType>& source_connection, Vector<ValueType>& sssp_result);
    inline int get_iterations() const {return m_iterations;}
};

} // end namespace nvgraph

