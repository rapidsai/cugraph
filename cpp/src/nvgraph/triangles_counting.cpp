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

#include "include/triangles_counting.hxx"
#include "include/triangles_counting_kernels.hxx"

#include <thrust/sequence.h>

namespace nvgraph
{

namespace triangles_counting
{

template <typename IndexType>
TrianglesCount<IndexType>::TrianglesCount(const CsrGraph <IndexType>& graph, cudaStream_t stream, int device_id)
{
    m_stream = stream;
    m_done = true;
    if (device_id == -1)
        cudaGetDevice(&m_dev_id);
    else
        m_dev_id = device_id;

    cudaGetDeviceProperties(&m_dev_props, m_dev_id);
    cudaCheckError();
    cudaSetDevice(m_dev_id);
    cudaCheckError();

    // fill spmat struct;
    m_mat.nnz = graph.get_num_edges();
    m_mat.N = graph.get_num_vertices();
    m_mat.roff_d = graph.get_raw_row_offsets();
    m_mat.cols_d = graph.get_raw_column_indices();

    m_seq.allocate(m_mat.N, stream);
    create_nondangling_vector(m_mat.roff_d, m_seq.raw(), &(m_mat.nrows), m_mat.N, m_stream); 
    m_mat.rows_d = m_seq.raw();
}

template <typename IndexType>
TrianglesCount<IndexType>::~TrianglesCount()
{
    cudaSetDevice(m_dev_id);
}

template <typename IndexType>
void TrianglesCount<IndexType>::tcount_bsh()
{
//    printf("TrianglesCount: %s\n", __func__); fflush(stdout);
    
    if (m_dev_props.sharedMemPerBlock*8 < (size_t)m_mat.nrows) 
    {
        FatalError("Number of vertices to high to use this kernel!", NVGRAPH_ERR_BAD_PARAMETERS);
    }

    unsigned int    *bmap_d;
    size_t      bmld = DIV_UP(m_mat.N,8*sizeof(*bmap_d));

    bmld = 8ull*DIV_UP(bmld*sizeof(*bmap_d), 8);
    bmld /= sizeof(*bmap_d);
    
    //size_t bmap_sz = sizeof(*bmap_d)*bmld;
    int nblock = m_mat.nrows;

    Vector<uint64_t> ocnt_d(nblock);
    cudaMemset(ocnt_d.raw(), 0, ocnt_d.bytes());
    cudaCheckError();

    tricnt_bsh(nblock, &m_mat, ocnt_d.raw(), bmld, m_stream);

    m_triangles_number = reduce(ocnt_d.raw(), nblock, m_stream);
}

template <typename IndexType>
void TrianglesCount<IndexType>::tcount_b2b()
{

//    printf("TrianglesCount: %s\n", __func__); fflush(stdout);

    // allocate a big enough array for output

    Vector<uint64_t> ocnt_d(m_mat.nrows);
    cudaMemset(ocnt_d.raw(), 0, ocnt_d.bytes());
    cudaCheckError();

    // allocate level 1 bitmap
    Vector<unsigned int> bmapL1_d;
    size_t bmldL1 = DIV_UP(m_mat.N,8*sizeof(*bmapL1_d.raw()));

    // make the size a multiple of 8 bytes, for zeroing in kernel...    
    bmldL1 = 8ull*DIV_UP(bmldL1*sizeof(*bmapL1_d.raw()), 8);
    bmldL1 /= sizeof(*bmapL1_d.raw());

    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    cudaCheckError();

    int nblock = (free_bytes*95/100) / (sizeof(*bmapL1_d.raw())*bmldL1);//@TODO: what?
    nblock = MIN(nblock, m_mat.nrows);

    size_t bmapL1_sz = sizeof(*bmapL1_d.raw())*bmldL1*nblock;

    bmapL1_d.allocate(bmldL1*nblock);
    //cuda 8.0 : memory past 16th GB may not be set with cudaMemset(),
    //CHECK_CUDA(cudaMemset(bmapL1_d, 0, bmapL1_sz));
    myCudaMemset((unsigned long long *)bmapL1_d.raw(), 0ull, bmapL1_sz/8, m_stream);

    // allocate level 0 bitmap
    Vector<unsigned int> bmapL0_d;
    size_t          bmldL0 = DIV_UP(DIV_UP(m_mat.N, BLK_BWL0), 8*sizeof(*bmapL0_d.raw()));

    bmldL0 = 8ull*DIV_UP(bmldL0*sizeof(*bmapL0_d.raw()), 8);
    bmldL0 /= sizeof(*bmapL0_d.raw());

    size_t bmapL0_sz = sizeof(*bmapL0_d.raw())*nblock*bmldL0;
    bmapL0_d.allocate(nblock*bmldL0);

    myCudaMemset((unsigned long long *)bmapL0_d.raw(), 0ull, bmapL0_sz/8, m_stream);
    tricnt_b2b(nblock, &m_mat, ocnt_d.raw(), bmapL0_d.raw(), bmldL0, bmapL1_d.raw(), bmldL1, m_stream);
    m_triangles_number = reduce(ocnt_d.raw(), nblock, m_stream);
}

template <typename IndexType>
void TrianglesCount<IndexType>::tcount_wrp()
{
//    printf("TrianglesCount: %s\n", __func__); fflush(stdout);

    // allocate a big enough array for output
    Vector<uint64_t> ocnt_d;
    size_t ocnt_sz = DIV_UP(m_mat.nrows, (THREADS/32));
    ocnt_d.allocate(ocnt_sz);

    cudaMemset(ocnt_d.raw(), 0, ocnt_d.bytes());
    cudaCheckError();

    Vector<unsigned int> bmap_d;
    size_t      bmld = DIV_UP(m_mat.N,8*sizeof(*bmap_d.raw()));

    // make the size a multiple of 8 bytes, for zeroing in kernel...    
    bmld = 8ull*DIV_UP(bmld*sizeof(*bmap_d.raw()), 8);
    bmld /= sizeof(*bmap_d.raw());

    // number of blocks limited by birmap size
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    cudaCheckError();

    int nblock = (free_bytes*95/100) / (sizeof(*bmap_d.raw())*bmld*(THREADS/32));
    nblock = MIN(nblock, DIV_UP(m_mat.nrows, (THREADS/32)));
    //int maxblocks = props.multiProcessorCount * props.maxThreadsPerMultiProcessor / THREADS;
    //nblock = MIN(nblock, maxblocks);

    size_t bmap_sz = bmld*nblock*(THREADS/32);

    bmap_d.allocate(bmap_sz);
    //CUDA 8.0 memory past 16th GB may not be set with cudaMemset()
    //CHECK_CUDA(cudaMemset(bmap_d, 0, bmap_sz));
    myCudaMemset((unsigned long long *)bmap_d.raw(), 0ull, bmap_sz*sizeof(*bmap_d.raw())/8, m_stream);

    tricnt_wrp(nblock, &m_mat, ocnt_d.raw(), bmap_d.raw(), bmld, m_stream);
    m_triangles_number = reduce(ocnt_d.raw(), nblock, m_stream);
}

template <typename IndexType>
void TrianglesCount<IndexType>::tcount_thr()
{
//    printf("TrianglesCount: %s\n", __func__); fflush(stdout);

    int maxblocks = m_dev_props.multiProcessorCount * m_dev_props.maxThreadsPerMultiProcessor / THREADS;

    int nblock = MIN(maxblocks, DIV_UP(m_mat.nrows,THREADS));

    Vector<uint64_t> ocnt_d(nblock);

    cudaMemset(ocnt_d.raw(), 0, ocnt_d.bytes());
    cudaCheckError();

    tricnt_thr(nblock, &m_mat, ocnt_d.raw(), m_stream);
    m_triangles_number = reduce(ocnt_d.raw(), nblock, m_stream);
}

template <typename IndexType>
NVGRAPH_ERROR TrianglesCount<IndexType>::count(TrianglesCountAlgo algo)
{
//  std::cout << "Starting TrianglesCount::count, Algo=" << algo << "\n";
    switch(algo)
    {
        case TCOUNT_BSH:
            tcount_bsh();
            break;
        case TCOUNT_B2B:
            tcount_b2b();
            break;
        case TCOUNT_WRP:
            tcount_wrp();
            break;
        case TCOUNT_THR:
            tcount_thr();
            break;
        case TCOUNT_DEFAULT:
            {
                double mean_deg = (double)m_mat.nnz / m_mat.nrows;
                if      (mean_deg <  DEG_THR1) tcount_thr();
                else if (mean_deg <  DEG_THR2) tcount_wrp();
                else 
                {
                    const int shMinBlkXSM = 6;
                    if (m_dev_props.sharedMemPerBlock*8/shMinBlkXSM < (size_t)m_mat.N)
                        tcount_b2b();
                    else    
                        tcount_bsh();
                }
            }
            break;
        default:
            FatalError("Bad algorithm specified for triangles counting", NVGRAPH_ERR_BAD_PARAMETERS);
    }
    m_event.record();
    return NVGRAPH_OK;
}

template class TrianglesCount<int>;

} // end namespace triangle counting

} // end namespace nvgraph

