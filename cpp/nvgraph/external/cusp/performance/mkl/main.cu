#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/io/matrix_market.h>
#include <cusp/blas/blas.h>
#include <cusp/multiply.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <stdio.h>

#include "mkl.h"
#include "../timer.h"

void spmv_mkl(MKL_INT m, float values[], MKL_INT rowIndex[], MKL_INT columns[], float x[], float y[])
{
    spmv_mkl_float(m, values, rowIndex, columns, x, y);
}

void spmv_mkl(MKL_INT m, double values[], MKL_INT rowIndex[], MKL_INT columns[], double x[], double y[])
{
    spmv_mkl_double(m, values, rowIndex, columns, x, y);
}

template <typename IndexType, typename ValueType>
float benchmark_mkl_spmv(cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>& A)
{
    cusp::array1d<ValueType,cusp::host_memory> x(A.num_cols, 0);
    cusp::array1d<ValueType,cusp::host_memory> y(A.num_rows, 0);

    // warm up
    spmv_mkl(A.num_rows,
             &A.values[0],
             &A.row_offsets[0],
             &A.column_indices[0],
             &x[0],
             &y[0]);

    // benchmark SpMV
    timer t;

    const size_t num_iterations = 500;

    for(size_t i = 0; i < num_iterations; i++)
        spmv_mkl(A.num_rows,
                 &A.values[0],
                 &A.row_offsets[0],
                 &A.column_indices[0],
                 &x[0],
                 &y[0]);


    float time = t.seconds_elapsed() / num_iterations;
    float GFLOPs = (time == 0) ? 0 : (2 * A.num_entries / time) / 1e9;

    //printf("MKL SpMV %8.4f ms ( %5.2f GFLOP/s )\n", 1e3 * time, GFLOPs); 

    return GFLOPs;
}
template <typename IndexType, typename ValueType>
float benchmark_cusp_spmv(cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>& A_host)
{
    cusp::hyb_matrix<IndexType,ValueType,cusp::device_memory> A(A_host);

    cusp::array1d<ValueType,cusp::device_memory> x(A.num_cols, 0);
    cusp::array1d<ValueType,cusp::device_memory> y(A.num_rows, 0);
    
    // warm up
    cusp::multiply(A, x, y);

    // benchmark SpMV
    timer t;

    const size_t num_iterations = 500;

    for(size_t i = 0; i < num_iterations; i++)
        cusp::multiply(A, x, y);

    float time = t.seconds_elapsed() / num_iterations;
    float GFLOPs = (time == 0) ? 0 : (2 * A.num_entries / time) / 1e9;

    //printf("CUSP SpMV %8.4f ms ( %5.2f GFLOP/s )\n", 1e3 * time, GFLOPs); 

    return GFLOPs;
}

template <typename IndexType, typename ValueType>
void benchmark_mkl_cg(cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>& A,
                      cusp::array1d<ValueType, cusp::host_memory>& b,
                      ValueType target_residual)
{
    cusp::array1d<ValueType, cusp::host_memory> x(A.num_rows, 0);

    timer t;

    cg_mkl_double(A.num_rows,
                  &A.values[0],
                  &A.row_offsets[0],
                  &A.column_indices[0],
                  &x[0],
                  &b[0],
                  A.num_rows,
                  ValueType(0),
                  target_residual);

    float time = t.seconds_elapsed();

    printf("MKL CG finished in %8.4f secs\n", time);
}


template <typename IndexType, typename ValueType>
void benchmark_cusp_cg(cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>& A_host,
                       cusp::array1d<ValueType, cusp::host_memory>& b_host,
                       ValueType target_residual)
{
    cusp::hyb_matrix<IndexType,ValueType,cusp::device_memory> A(A_host);

    cusp::array1d<ValueType, cusp::device_memory> x(A.num_rows, 0);
    cusp::array1d<ValueType, cusp::device_memory> b(b_host);

    // set stopping criteria:
    cusp::default_monitor<ValueType> monitor(target_residual/cusp::blas::nrm2(b), A.num_rows);

    timer t;

    // obtain a linear operator from matrix A and call CG
    cusp::krylov::cg(A, x, b, monitor);

    float time = t.seconds_elapsed();

    printf("CUSP CG finished in %8.4f secs\n", time);
}

int main(int argc, char ** argv)
{
    if (argc == 1)
    {
        std::cout << "usage: " << argv[0] << " A.mtx [B.mtx ...]" << std::endl;
        return -1;
    }

    cudaSetDevice(0);

    typedef int    IndexType;
    typedef double ValueType;

    for (int i = 1; i < argc; i++)
    {
        cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> A;
        cusp::io::read_matrix_market_file(A, argv[i]);
    
        printf("%s,%f\n", argv[i], benchmark_mkl_spmv(A));
    
        //std::cout << "loaded matrix with shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n\n";
    
    //    std::cout << "---------- benchmarking SpMV ----------\n";
    //    benchmark_cusp_spmv(A);
    
    //    if (A.num_rows == A.num_cols)
    //    {
    //        cusp::array1d<ValueType, cusp::host_memory> x;
    //        cusp::array1d<ValueType, cusp::host_memory> b;
    
    //        { cusp::array2d<ValueType, cusp::host_memory> temp; cusp::io::read_matrix_market_file(temp, "x.mtx"); temp.values.swap(x); }
    //        { cusp::array2d<ValueType, cusp::host_memory> temp; cusp::io::read_matrix_market_file(temp, "b.mtx"); temp.values.swap(b); }
    //        std::cout << "\n----------- benchmarking CG -----------\n";
    //        // compute residual
    //        cusp::array1d<ValueType, cusp::host_memory> r(A.num_rows,0);
    //        cusp::multiply(A, x, r);
    //        cusp::blas::axpy(b, r, ValueType(-1.0));
    //
    //        ValueType residual_norm = cusp::blas::nrm2(r);
    //        
    //        std::cout << " provided solution has residual norm " << residual_norm << std::endl;
    //
    //        benchmark_mkl_cg(A, b, residual_norm);
    //        benchmark_cusp_cg(A, b, residual_norm);
    //    }
    }

    return 0;
}

