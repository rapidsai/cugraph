#define CUSP_USE_TEXTURE_MEMORY

#include <cusp/dia_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>

#include <thrust/sequence.h>
#include <thrust/fill.h>

#include <iostream>

#include <cusp/detail/device/spmv/csr_vector.h>
#include "../timer.h"

template <bool UseCache, unsigned int THREADS_PER_VECTOR, typename IndexType, typename ValueType>
void perform_spmv(const cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
                  const ValueType * x, 
                        ValueType * y)
{
    const unsigned int VECTORS_PER_BLOCK  = 128 / THREADS_PER_VECTOR;
    const unsigned int THREADS_PER_BLOCK  = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    //const unsigned int MAX_BLOCKS = MAX_THREADS / THREADS_PER_BLOCK;
    const unsigned int MAX_BLOCKS = 16 * 1024;
    const unsigned int NUM_BLOCKS = std::min(MAX_BLOCKS, static_cast<unsigned int>((csr.num_rows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK));
    
    if (UseCache)
        bind_x(x);

    cusp::detail::device::spmv_csr_vector_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR, UseCache> <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> 
        (csr.num_rows,
         thrust::raw_pointer_cast(&csr.row_offsets[0]),
         thrust::raw_pointer_cast(&csr.column_indices[0]),
         thrust::raw_pointer_cast(&csr.values[0]),
         x, y);

    if (UseCache)
        unbind_x(x);
}
 
template <unsigned int ThreadsPerVector, typename IndexType, typename ValueType>
float benchmark_matrix(const cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>& csr)
{
    const size_t num_iterations = 100;

    cusp::array1d<ValueType, cusp::device_memory> x(csr.num_cols);
    cusp::array1d<ValueType, cusp::device_memory> y(csr.num_rows);

    // warmup
    perform_spmv<true, ThreadsPerVector>(csr, thrust::raw_pointer_cast(&x[0]), thrust::raw_pointer_cast(&y[0]));

    // time several SpMV iterations
    timer t;
    for(size_t i = 0; i < num_iterations; i++)
        perform_spmv<true, ThreadsPerVector>(csr, thrust::raw_pointer_cast(&x[0]), thrust::raw_pointer_cast(&y[0]));
    cudaThreadSynchronize();

    float sec_per_iteration = t.seconds_elapsed() / num_iterations;
    float gflops = 2.0 * (csr.num_entries/sec_per_iteration) / 1e9;

    return gflops;
}


template <typename IndexType, typename ValueType>
void make_synthetic_example(const IndexType N, const IndexType D, 
                            cusp::csr_matrix<IndexType, ValueType, cusp::device_memory>& csr)
{
//    // create DIA matrix with D diagonal bands
//    const IndexType NNZ = N * D - (D * (D - 1)) / 2;
//    cusp::dia_matrix<IndexType, ValueType, cusp::host_memory> dia(N, N, NNZ, D, N);
//    thrust::sequence(dia.diagonal_offsets.begin(), dia.diagonal_offsets.end());
//    thrust::fill(dia.values.values.begin(), dia.values.values.end(), 1);

    // create ELL matrix with D diagonals
    cusp::ell_matrix<IndexType, ValueType, cusp::host_memory> ell(N, D, N * D, D);
    for(IndexType i = 0; i < N; i++)
    {
        for(IndexType j = 0; j < D; j++)
        {
            ell.column_indices(i,j) = j;
            ell.values(i,j) = 1.0;
        }
    }

    // convert to CSR
    csr = ell;
}


int main(int argc, char** argv)
{
    typedef int   IndexType;
    typedef float ValueType;
        

    if (argc == 1)
    {
        // matrix varies along rows, # of threads per vector varies along column
        printf("matrix      , nnz per row,       2,       4,       8,      16,      32,\n");

        const IndexType N = 320 * 1000;
        const IndexType max_diagonals = 64;

        for(IndexType D = 1; D <= max_diagonals; D++)
        {
            cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> csr;
            make_synthetic_example(N, D, csr);
            printf("dense_%02d    ,    %8.2f,", (int) D, (float) csr.num_entries / (float) csr.num_rows);
            printf("  %5.4f,", benchmark_matrix< 2, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix< 4, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix< 8, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix<16, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix<32, IndexType, ValueType>(csr));
            printf("\n");
        }
    }
    else
    {
        // matrix varies along rows, # of threads per vector varies along column
        printf("matrix              , nnz per row,       2,       4,       8,      16,      32,\n");

        for(int i = 1; i < argc; i++)
        {
            cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> csr;
            cusp::io::read_matrix_market_file(csr, std::string(argv[i]));
            printf("%20s,    %8.2f,", argv[i], (float) csr.num_entries / (float) csr.num_rows);
            printf("  %5.4f,", benchmark_matrix< 2, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix< 4, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix< 8, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix<16, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix<32, IndexType, ValueType>(csr));
            printf("\n");
        }
    }

    return 0;
}

