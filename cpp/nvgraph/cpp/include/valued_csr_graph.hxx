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

#include "csr_graph.hxx"
#include "nvgraph_vector.hxx"

namespace nvgraph
{

/*! A ValuedCsrGraph is a graph strored in a CSR data structure.
    It represents an weighted graph and has storage for row_offsets and column_indices and values
 */
template <typename IndexType_, typename ValueType_>
class ValuedCsrGraph : public nvgraph::CsrGraph<IndexType_>
{
public:
    typedef IndexType_ IndexType;
    typedef ValueType_ ValueType;

private:
    typedef nvgraph::CsrGraph<IndexType> Parent;

protected:
    /*! Storage for the nonzero entries of the CSR data structure.
     */
    SHARED_PREFIX::shared_ptr<ValueType> values;

public:  

    /*! Construct an empty \p ValuedCsrGraph.
     */
    ValuedCsrGraph(void) {}
    /*! Destruct a \p ValuedCsrGraph.
     */
    ~ValuedCsrGraph(void) {}

    /*! Construct a \p ValuedCsrGraph with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_entries Number of nonzero graph entries.
     */
    ValuedCsrGraph(size_t num_rows, size_t num_entries, cudaStream_t stream)
        : Parent(num_rows, num_entries, stream),
          values(allocateDevice<ValueType>(num_entries, NULL)) {}

    /*! Construct a \p ValuedCsrGraph from another graph.
     *
     *  \param ValuedCsrGraph Another graph in csr
     */
    ValuedCsrGraph(const ValuedCsrGraph& gr): 
        Parent(gr),
        values(gr.values)
    {}

    /*! Construct a \p ValuedCsrGraph from another graph.  
     *
     *  \param ValuedCsrGraph Another graph in csr
     */
    ValuedCsrGraph(const Parent& gr, Vector<ValueType>& vals):
        Parent(gr),  
        values(vals.raw())      
    {

    }

    inline ValueType* get_raw_values()  const { return values.get(); }


    /*! Swap the contents of two \p ValuedCsrGraph objects.
     *
     *  \param graph Another graph in csr 
     */
    void swap(ValuedCsrGraph& graph);

    /*! Assignment from another graph.
     *
     *  \param graph Another graph in csr
     */
    ValuedCsrGraph& operator=(const ValuedCsrGraph& graph);

    //Accept method injection
    DEFINE_VISITABLE(IndexType_)

}; // class ValuedCsrGraph
}

