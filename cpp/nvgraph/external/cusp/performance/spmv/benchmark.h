#pragma once

#include <cusp/multiply.h>
#include <cusp/system/cuda/detail/multiply/coo_flat_k.h>
#include <cusp/system/cuda/detail/multiply/csr_scalar.h>

#include "bytes_per_spmv.h"
#include "utility.h"
#include "../timer.h"

#include <string>
#include <iostream>
#include <stdio.h>

const char * BENCHMARK_OUTPUT_FILE_NAME = "benchmark_output.log";

template <typename HostMatrix, typename TestMatrix, typename TestKernel>
float check_spmv(HostMatrix& host_matrix, TestMatrix& test_matrix, TestKernel test_kernel)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = host_matrix.num_rows;
    const IndexType N = host_matrix.num_cols;

    // create host input (x) and output (y) vectors
    cusp::array1d<ValueType,cusp::host_memory> host_x(N);
    cusp::array1d<ValueType,cusp::host_memory> host_y(M);
    //for(IndexType i = 0; i < N; i++) host_x[i] = (rand() % 21) - 10;
    for(IndexType i = 0; i < N; i++) host_x[i] = (int(i % 21) - 10);
    for(IndexType i = 0; i < M; i++) host_y[i] = 0;

    // create test input (x) and output (y) vectors
    cusp::array1d<ValueType, MemorySpace> test_x(host_x.begin(), host_x.end());
    cusp::array1d<ValueType, MemorySpace> test_y(host_y.begin(), host_y.end());

    // compute SpMV on host and device
    cusp::multiply(host_matrix, host_x, host_y);
    test_kernel(test_matrix, test_x, test_y);

    // compare results
    cusp::array1d<ValueType,cusp::host_memory> test_y_copy(test_y.begin(), test_y.end());
    double error = l2_error(M, thrust::raw_pointer_cast(&test_y_copy[0]), thrust::raw_pointer_cast(&host_y[0]));

    return error;
}

template <typename HostMatrix, typename TestMatrix, typename TestKernel>
float check_block_spmv(HostMatrix& host_matrix, TestMatrix& test_matrix, TestKernel test_kernel, size_t num_cols)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = host_matrix.num_rows;
    const IndexType N = host_matrix.num_cols;

    // create host input (x) and output (y) vectors
    cusp::array2d<ValueType,cusp::host_memory> host_x(N, num_cols);
    cusp::array2d<ValueType,cusp::host_memory> host_y(M, num_cols, 0);

    // initialize host_x to random array
    cusp::copy(cusp::random_array<ValueType>(host_x.values.size()), host_x.values);

    // create test input (x) and output (y) vectors
    cusp::array2d<ValueType, MemorySpace> test_x(host_x);
    cusp::array2d<ValueType, MemorySpace> test_y(host_y);

    // compute SpMV on host and device
    cusp::multiply(host_matrix, host_x, host_y);
    test_kernel(test_matrix, test_x, test_y);

    // compare results
    cusp::array2d<ValueType,cusp::host_memory> test_y_copy(test_y);

    ValueType error = 0;
    for(size_t i = 0; i < num_cols; i++)
        error = std::max(error, l2_error(test_y_copy.column(i), host_y.column(i)));

    return error;
}

template <typename TestMatrix, typename TestKernel>
float time_spmv(TestMatrix& test_matrix, TestKernel test_spmv, double seconds = 3.0, size_t min_iterations = 100, size_t max_iterations = 500)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = test_matrix.num_rows;
    const IndexType N = test_matrix.num_cols;

    // create test input (x) and output (y) vectors
    cusp::array1d<ValueType, MemorySpace> test_x(N);
    cusp::array1d<ValueType, MemorySpace> test_y(M);

    // warmup
    timer time_one_iteration;
    test_spmv(test_matrix, test_x, test_y);
    cudaThreadSynchronize();
    double estimated_time = time_one_iteration.seconds_elapsed();

    // determine # of seconds dynamically
    size_t num_iterations;
    if (estimated_time == 0)
        num_iterations = max_iterations;
    else
        num_iterations = std::min(max_iterations, std::max(min_iterations, (size_t) (seconds / estimated_time)) );

    // time several SpMV iterations
    timer t;
    for(size_t i = 0; i < num_iterations; i++)
        test_spmv(test_matrix, test_x, test_y);
    cudaThreadSynchronize();

    float sec_per_iteration = t.seconds_elapsed() / num_iterations;

    return sec_per_iteration;
}

template <typename TestMatrix, typename TestKernel>
float time_spmv_block(TestMatrix& test_matrix, size_t num_cols, TestKernel test_spmv, double seconds = 3.0, size_t min_iterations = 100, size_t max_iterations = 500)
{
    typedef typename TestMatrix::index_type   IndexType; // ASSUME same as HostMatrix::index_type
    typedef typename TestMatrix::value_type   ValueType; // ASSUME same as HostMatrix::value_type
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = test_matrix.num_rows;
    const IndexType N = test_matrix.num_cols;

    // create test input (x) and output (y) vectors
    cusp::array2d<ValueType, MemorySpace, cusp::row_major> test_x(N, num_cols);
    cusp::array2d<ValueType, MemorySpace, cusp::row_major> test_y(M, num_cols);

    // warmup
    timer time_one_iteration;
    test_spmv(test_matrix, test_x, test_y);
    cudaThreadSynchronize();
    double estimated_time = time_one_iteration.seconds_elapsed();

    // determine # of seconds dynamically
    size_t num_iterations;
    if (estimated_time == 0)
        num_iterations = max_iterations;
    else
        num_iterations = std::min(max_iterations, std::max(min_iterations, (size_t) (seconds / estimated_time)) );

    // time several SpMV iterations
    timer t;
    for(size_t i = 0; i < num_iterations; i++)
      test_spmv(test_matrix, test_x, test_y);
    cudaThreadSynchronize();

    float sec_per_iteration = t.seconds_elapsed() / num_iterations;

    return sec_per_iteration;
}

template <typename HostMatrix, typename TestMatrixOnHost, typename TestMatrixOnDevice, typename TestKernel>
void test_spmv(std::string         kernel_name,
               HostMatrix&         host_matrix,
               TestMatrixOnHost&   test_matrix_on_host,
               TestMatrixOnDevice& test_matrix_on_device,
               TestKernel          test_spmv)
{
    float error = check_spmv(host_matrix, test_matrix_on_device, test_spmv);
    float time  = time_spmv(              test_matrix_on_device, test_spmv);
    float gbyte = bytes_per_spmv(test_matrix_on_host);

    float GFLOPs = (time == 0) ? 0 : (2 * host_matrix.num_entries / time) / 1e9;
    float GBYTEs = (time == 0) ? 0 : (gbyte / time)                       / 1e9;

    printf("\t%-20s: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", kernel_name.c_str(), 1e3 * time, GFLOPs, GBYTEs, error);

    //record results to file
    FILE * fid = fopen(BENCHMARK_OUTPUT_FILE_NAME, "a");
    fprintf(fid, "kernel=%s gflops=%f gbytes=%f msec=%f\n", kernel_name.c_str(), GFLOPs, GBYTEs, 1e3 * time);
    fclose(fid);
}

template <typename HostMatrix, typename TestMatrixOnHost, typename TestMatrixOnDevice, typename TestKernel>
void test_spmv_block(std::string         kernel_name,
                     size_t              num_cols,
                     HostMatrix&         host_matrix,
                     TestMatrixOnHost&   test_matrix_on_host,
                     TestMatrixOnDevice& test_matrix_on_device,
                     TestKernel          test_spmv)
{
    std::ostringstream block_string;
    block_string << "(" << num_cols << ")";
    kernel_name += block_string.str();

    float error = check_block_spmv(host_matrix, test_matrix_on_device, test_spmv, num_cols);
    float time  = time_spmv_block(test_matrix_on_device, num_cols, test_spmv);
    float gbyte = bytes_per_spmv_block(test_matrix_on_host, num_cols);

    float GFLOPs = (time == 0) ? 0 : (num_cols * 2 * host_matrix.num_entries / time) / 1e9;
    float GBYTEs = (time == 0) ? 0 : (gbyte / time)                       / 1e9;

    printf("\t%-20s: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %f]\n", kernel_name.c_str(), 1e3 * time, GFLOPs, GBYTEs, error);

    //record results to file
    FILE * fid = fopen(BENCHMARK_OUTPUT_FILE_NAME, "a");
    fprintf(fid, "kernel=%s gflops=%f gbytes=%f msec=%f\n", kernel_name.c_str(), GFLOPs, GBYTEs, 1e3 * time);
    fclose(fid);
}

/////////////////////////////////////////////////////
// These methods test specific formats and kernels //
/////////////////////////////////////////////////////

template <typename HostMatrix>
void test_coo(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(host_matrix);

    // transfer TestMatrix to device
    typedef typename cusp::coo_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType,cusp::device_memory>                DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("coo",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
}

template <typename HostMatrix>
void test_csr(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(host_matrix);

    // transfer csr_matrix to device
    typedef typename cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType,cusp::device_memory>                DeviceArray;
    typedef typename cusp::array2d<ValueType,cusp::device_memory>                DeviceArray2d;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("csr_vector", host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
    test_spmv("csr_scalar", host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::system::cuda::detail::spmv_csr_scalar<DeviceMatrix,DeviceArray,DeviceArray>);

    for(size_t num_cols = 2; num_cols < 64; num_cols *= 2)
      test_spmv_block("csr_block",  num_cols, host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray2d,DeviceArray2d>);
}

template <typename HostMatrix>
void test_dia(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::dia_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host;

    try
    {
        test_matrix_on_host = host_matrix;
    }
    catch (cusp::format_conversion_exception)
    {
        std::cout << "\tRefusing to convert to DIA format" << std::endl;
        return;
    }

    // transfer TestMatrix to device
    typedef typename cusp::dia_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType, cusp::device_memory>               DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("dia",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
}

template <typename HostMatrix>
void test_ell(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::ell_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host;

    try
    {
        test_matrix_on_host = host_matrix;
    }
    catch (cusp::format_conversion_exception)
    {
        std::cout << "\tRefusing to convert to ELL format" << std::endl;
        return;
    }

    // transfer TestMatrix to device
    typedef typename cusp::ell_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType, cusp::device_memory>               DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("ell",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
}

template <typename HostMatrix>
void test_hyb(HostMatrix& host_matrix)
{
    typedef typename HostMatrix::index_type IndexType;
    typedef typename HostMatrix::value_type ValueType;

    // convert HostMatrix to TestMatrix on host
    cusp::hyb_matrix<IndexType, ValueType, cusp::host_memory> test_matrix_on_host(host_matrix);

    // transfer TestMatrix to device
    typedef typename cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix;
    typedef typename cusp::array1d<ValueType, cusp::device_memory>               DeviceArray;
    DeviceMatrix test_matrix_on_device(test_matrix_on_host);

    test_spmv("hyb",     host_matrix, test_matrix_on_host, test_matrix_on_device, cusp::multiply<DeviceMatrix,DeviceArray,DeviceArray>);
}

