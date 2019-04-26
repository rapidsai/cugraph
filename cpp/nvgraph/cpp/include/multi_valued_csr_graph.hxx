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
#include "valued_csr_graph.hxx"
#include <vector>

namespace nvgraph
{

template <typename IndexType_, typename ValueType_>
class MultiValuedCsrGraph : public nvgraph::CsrGraph<IndexType_>
{
public:
    typedef IndexType_ IndexType;
    typedef ValueType_ ValueType;
private:
    typedef nvgraph::CsrGraph<IndexType> Parent;

protected:
    /*! Storage for the nonzero entries of the multi CSR data structure.
     */
    //std::vector <nvgraph::Vector<ValueType>*> values_dim;
    //std::vector <nvgraph::Vector<ValueType>*> vertex_dim;

    std::vector <SHARED_PREFIX::shared_ptr<nvgraph::Vector<ValueType> > > values_dim;
    std::vector <SHARED_PREFIX::shared_ptr<nvgraph::Vector<ValueType> > > vertex_dim;
public:

    /*! Storage for the nonzero entries of the Multi-CSR data structure.*/
    MultiValuedCsrGraph(void) {}
    ~MultiValuedCsrGraph(void) 
    {
       //for (int i = 0; i < n_vertex_dim; ++i)
       //    if (vertex_dim[i]) 
       //        delete vertex_dim[i]; 
       // for (int i = 0; i < n_edges_dim; ++i)
       //    if (values_dim[i])
       //        delete values_dim[i];
    }

    /*! Construct a \p MultiValuedCsrGraph with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_entries Number of nonzero graph entries.
     *  \param num_dimensions Number of dimensions (ie. number of values arrays).
     */
    MultiValuedCsrGraph(size_t num_rows, size_t num_entries, cudaStream_t stream)
    : Parent(num_rows, num_entries, stream) { }

    /*! Construct a \p MultiValuedCsrGraph from another graph.*/
    MultiValuedCsrGraph(const MultiValuedCsrGraph& gr)
    :   Parent(gr),
        values_dim(gr.values_dim),
        vertex_dim(gr.vertex_dim)

    {}
    MultiValuedCsrGraph(const Parent& gr)
    :   Parent(gr)
    {}

    inline void allocateVertexData(size_t v_dim, cudaStream_t stream) 
    {
        vertex_dim.resize(v_dim);
        for (size_t i = 0; i < vertex_dim.size(); ++i)
          vertex_dim[i] = SHARED_PREFIX::shared_ptr<nvgraph::Vector<ValueType> >(new Vector<ValueType>(this->num_vertices, stream)); 
    }

    inline void allocateEdgeData(size_t edges_dim, cudaStream_t stream) 
    {
        values_dim.resize(edges_dim);
         for (size_t i = 0; i < values_dim.size(); ++i)
           values_dim[i] = SHARED_PREFIX::shared_ptr<nvgraph::Vector<ValueType> >(new Vector<ValueType>(this->num_edges, stream)); 
    }

    inline void attachVertexData(size_t i, ValueType* data, cudaStream_t stream) 
    {
        if (vertex_dim.size() <= i)
            vertex_dim.resize(i+1);
         vertex_dim[i] = SHARED_PREFIX::shared_ptr<nvgraph::Vector<ValueType> >(new Vector<ValueType>(this->num_vertices, data, stream)); 
    }

    inline void attachEdgeData(size_t i, ValueType* data, cudaStream_t stream) 
    {
         if (values_dim.size() <= i)
            values_dim.resize(i+1);
        values_dim[i] = SHARED_PREFIX::shared_ptr<nvgraph::Vector<ValueType> >(new Vector<ValueType>(this->num_edges, data, stream)); 
    }
    
    inline size_t getNumValues() {
   	 return values_dim.size();
    }

    inline size_t get_num_vertex_dim() const { return vertex_dim.size(); }
    inline size_t get_num_edge_dim() const { return values_dim.size(); }
    inline Vector<ValueType>& get_vertex_dim(size_t v_dim)  { return *vertex_dim[v_dim]; }
    inline Vector<ValueType>& get_edge_dim(size_t e_dim)  { return *values_dim[e_dim]; }
    inline ValueType* get_raw_vertex_dim(size_t v_dim)  { return vertex_dim[v_dim]->raw(); }
    inline ValueType* get_raw_edge_dim(size_t e_dim)  { return values_dim[e_dim]->raw(); }
    inline const Vector<ValueType>& get_vertex_dim(size_t v_dim) const  { return *vertex_dim[v_dim]; }
    inline const Vector<ValueType>& get_edge_dim(size_t e_dim) const { return *values_dim[e_dim]; }
    inline const ValueType* get_raw_vertex_dim(size_t v_dim) const { return vertex_dim[v_dim]->raw(); }
    inline const ValueType* get_raw_edge_dim(size_t e_dim) const { return values_dim[e_dim]->raw(); }
    /*! Extract a \p ValuedCsrGraph from a given dimension of the \p MultiValuedCsrGraph 
     *  \param dim_index Wanted dimension of the \p MultiValuedCsrGraph 
     */
    ValuedCsrGraph<IndexType, ValueType>* get_valued_csr_graph(const size_t dim_index)
    {
        //ValuedCsrGraph<IndexType, ValueType> *v = new ValuedCsrGraph<IndexType, ValueType>(static_cast<nvgraph::CsrGraph<IndexType> >(*this), *values_dim[dim_index]);
        //return *v;
      
        //SHARED_PREFIX::shared_ptr<ValuedCsrGraph<IndexType, ValueType> > svcsr = SHARED_PREFIX::shared_ptr<ValuedCsrGraph<IndexType, ValueType> >(new ValuedCsrGraph<IndexType, ValueType>(static_cast<nvgraph::CsrGraph<IndexType> >(*this), *values_dim[dim_index]));
        //return svcsr; //segfaults

        ///return ValuedCsrGraph<IndexType, ValueType>(static_cast<nvgraph::CsrGraph<IndexType> >(*this), *values_dim[dim_index]);//segfaults
        ValuedCsrGraph<IndexType, ValueType>* pvcsr = new ValuedCsrGraph<IndexType, ValueType>(static_cast<nvgraph::CsrGraph<IndexType> >(*this), *values_dim[dim_index]);
        return pvcsr;
    }



    /*! Assignment from another MultiValuedCsrGraph graph.
     *
     *  \param graph Another MultiValuedCsrGraph
     */
    MultiValuedCsrGraph& operator=(const MultiValuedCsrGraph& graph);
   

    //RESIZE: We should try not to resize MULTI CSR graphs in general for performance reasons

    // SET 
    //Set should be done in a safe way in the API 
    // it is possible to use a cudaMemcpy like : cudaMemcpy(G.get_raw_vertex_dim(1), v_h,           
    //                                           (size_t)(n*sizeof(v_h[0])),           
    //                                            cudaMemcpyHostToDevice);
    
    //Accept method injection
    DEFINE_VISITABLE(IndexType_)

}; // class MultiValuedCsrGraph
}

