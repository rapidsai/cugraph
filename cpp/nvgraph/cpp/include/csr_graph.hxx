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

#include "graph.hxx"
#include <cnmem_shared_ptr.hxx> // interface with CuMem (memory pool lib) for shared ptr

namespace nvgraph
{

/*! A CsrGraph is a graph strored in a CSR data structure.
    It represents an unweighted graph and has storage for row_offsets and column_indices 
 */
template <typename IndexType_>
class CsrGraph : public nvgraph::Graph<IndexType_>
{
public:
    typedef IndexType_ IndexType;

private:
    typedef nvgraph::Graph<IndexType> Parent;

protected:
    /*! Storage for the cuda stream
     */
    cudaStream_t stream_;

    /*! Storage for the row offsets of the CSR data structure.  Also called the "row pointer" array.
     */
    SHARED_PREFIX::shared_ptr<IndexType> row_offsets;

    /*! Storage for the column indices of the CSR data structure.
     */
    SHARED_PREFIX::shared_ptr<IndexType> column_indices;

public:
        
    /*! Construct an empty \p CsrGraph.
     */
    CsrGraph(void) {}

    /*! Destruct an empty \p CsrGraph.
     */
    ~CsrGraph(void) {}

    /*! Construct a \p CsrGraph with a specific shape and number of nonzero entries.
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero graph entries.
     */
    CsrGraph(size_t num_rows, size_t num_entries, cudaStream_t stream, bool external = false)
        : Parent(num_rows, num_entries),
          stream_(stream)
          {
              if (external)
              {
                row_offsets = nullptr;
                column_indices = nullptr;
              }
              else
              {
                row_offsets = allocateDevice<IndexType>((num_rows+1), NULL);
                column_indices = allocateDevice<IndexType>(num_entries, NULL);
              }
          }


    /*! Construct a \p CsrGraph from another graph.
     *
     *  \param CsrGraph Another graph in csr
     */
    CsrGraph(const CsrGraph& gr): 
        Parent(gr),
        row_offsets(gr.row_offsets),
        column_indices(gr.column_indices)
    {}

    /*! Construct a \p CsrGraph from another graph.
     *
     *  \param CsrGraph Another graph in csr
     */
    CsrGraph(const Parent& gr): 
       Parent(gr)
      // row_offsets(allocateDevice<IndexType>((gr.get_num_vertices()+1), NULL)),
      // column_indices(allocateDevice<IndexType>(gr.get_num_edges(), NULL))
    {}

    inline void allocate_row_offsets() 
    {
         row_offsets = allocateDevice<IndexType>(this->get_num_vertices()+1, NULL);
    }
    inline void allocate_column_indices() 
    {
        column_indices = allocateDevice<IndexType>(this->get_num_edges(), NULL);
    }
    inline IndexType* get_raw_row_offsets() { return row_offsets.get(); }
    inline IndexType* get_raw_column_indices() { return column_indices.get(); }
    inline void set_raw_row_offsets(IndexType* ptr) { row_offsets = attachDevicePtr<IndexType>(ptr, stream_); }
    inline void set_raw_column_indices(IndexType* ptr) {column_indices = attachDevicePtr<IndexType>(ptr, stream_); }
    inline const IndexType* get_raw_row_offsets()  const { return row_offsets.get(); }
    inline const IndexType* get_raw_column_indices()  const { return column_indices.get(); }
    inline cudaStream_t get_stream() const { return stream_; }

    /*! Resize graph dimensions and underlying storage
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero graph entries.
     */
    // We should try not to resize CSR graphs in general
    // void resize(const size_t num_rows, const size_t num_entries);

    /*! Swap the contents of two \p CsrGraph objects.
     *
     *  \param graph Another graph in csr 
     */
    void swap(CsrGraph& graph);

    /*! Assignment from another graph.
     *
     *  \param graph Another graph in csr
     */
    CsrGraph& operator=(const CsrGraph& graph);

    //Accept method injection
    DEFINE_VISITABLE(IndexType_)

}; // class CsrGraph
} // end namespace nvgraph

