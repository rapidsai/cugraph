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

#include <csr_graph.hxx>
#include <async_event.hxx>
#include <nvgraph_error.hxx>
#include <nvgraph_vector.hxx>

#include <cuda_runtime.h>

#include <triangles_counting_defines.hxx>

namespace nvgraph
{

namespace triangles_counting
{


typedef enum { TCOUNT_DEFAULT, TCOUNT_BSH, TCOUNT_B2B, TCOUNT_WRP, TCOUNT_THR } TrianglesCountAlgo;


template <typename IndexType>
class TrianglesCount 
{
private:
    //CsrGraph <IndexType>& m_last_graph ;
    AsyncEvent          m_event;
    uint64_t            m_triangles_number;
    spmat_t<IndexType>  m_mat;
    int                 m_dev_id;
    cudaDeviceProp      m_dev_props;

    Vector<IndexType>   m_seq;

    cudaStream_t        m_stream;

    bool m_done;

    void tcount_bsh();
    void tcount_b2b();
    void tcount_wrp();
    void tcount_thr();

public:
    // Simple constructor 
    TrianglesCount(const CsrGraph <IndexType>& graph, cudaStream_t stream = NULL, int device_id = -1);
    // Simple destructor
    ~TrianglesCount();

    NVGRAPH_ERROR count(TrianglesCountAlgo algo = TCOUNT_DEFAULT );
    inline uint64_t get_triangles_count() const {return m_triangles_number;}
};

} // end namespace triangles_counting

} // end namespace nvgraph

