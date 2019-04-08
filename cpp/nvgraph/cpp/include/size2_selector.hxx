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

//#include <common_selector.hxx>
#include <nvgraph_vector.hxx>
#include <valued_csr_graph.hxx>

namespace nvgraph {

typedef enum
{
   USER_PROVIDED = 0, // using edge values as is
   SCALED_BY_ROW_SUM   = 1,  // 0.5*(A_ij+A_ji)/max(d(i),d (j)), where d(i) is the sum of the row i
   SCALED_BY_DIAGONAL   = 2,  // 0.5*(A_ij+A_ji)/max(diag(i),diag(j)) 
}Matching_t;

template <typename IndexType_, typename ValueType_>
class Size2Selector
{

  public:
    typedef IndexType_ IndexType;
    typedef ValueType_ ValueType;

    Size2Selector();

    Size2Selector(Matching_t similarity_metric,  int deterministic = 1, int max_iterations = 15 , ValueType numUnassigned_tol = 0.05 ,bool two_phase = false, bool merge_singletons = true, cudaStream_t stream = 0) 
       :m_similarity_metric(similarity_metric), m_deterministic(deterministic), m_max_iterations(max_iterations), m_numUnassigned_tol(numUnassigned_tol), m_two_phase(two_phase), m_merge_singletons(merge_singletons), m_stream(stream)
    {
        m_aggregation_edge_weight_component = 0;
        m_weight_formula = 0;
    }

    NVGRAPH_ERROR setAggregates(const ValuedCsrGraph<IndexType, ValueType> &A, Vector<IndexType> &aggregates, int &num_aggregates);

  protected:
    NVGRAPH_ERROR setAggregates_common_sqblocks(const ValuedCsrGraph<IndexType, ValueType> &A, Vector<IndexType> &aggregates, int &num_aggregates);
    Matching_t m_similarity_metric;
    int m_deterministic;
    int m_max_iterations;
    ValueType m_numUnassigned_tol;
    bool m_two_phase;
    bool m_merge_singletons;
    cudaStream_t m_stream;    
    int m_aggregation_edge_weight_component;
    int m_weight_formula;
};

}//nvgraph
