#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/blas/blas.h>

#include <cusp/system/cuda/detail/cublas/blas.h>

#include <iostream>
#include <stdio.h>

#include "../timer.h"

template <typename T, typename MemorySpace=cusp::device_memory>
struct test_nrm2
{
    cusp::array1d<T,MemorySpace> x;
    const size_t n;
    cublasHandle_t handle;

    test_nrm2(const size_t n)
        : n(n), x(n)
    {
      if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
      {
        throw cusp::runtime_exception("cublasCreate failed");
      }
    }

    ~test_nrm2()
    {
      cublasDestroy(handle);
    }

    void operator()(void)
    {
        cusp::blas::nrm2(cusp::cuda::par.with(handle), x);
    }

    std::string name(void) const { return (sizeof(T) == 4) ? "snrm2" : "dnrm2"; }

    size_t bytes(void) const { return n * sizeof(T); }
};

template <typename T, typename MemorySpace=cusp::device_memory>
struct test_dot
{
    cusp::array1d<T,MemorySpace> x, y;
    const size_t n;
    cublasHandle_t handle;

    test_dot(const size_t n)
        : n(n), x(n), y(n)
    {
      if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
      {
        throw cusp::runtime_exception("cublasCreate failed");
      }
    }

    ~test_dot()
    {
      cublasDestroy(handle);
    }

    void operator()(void)
    {
        cusp::blas::dot(cusp::cuda::par.with(handle), x, y);
    }

    std::string name(void) const { return (sizeof(T) == 4) ? "sdot" : "ddot"; }

    size_t bytes(void) const { return 2 * n * sizeof(T); }
};

template <typename T, typename MemorySpace=cusp::device_memory>
struct test_axpy
{
    cusp::array1d<T,MemorySpace> x, y;
    const size_t n;
    cublasHandle_t handle;

    test_axpy(const size_t n)
        : n(n), x(n), y(n)
    {
      if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
      {
        throw cusp::runtime_exception("cublasCreate failed");
      }
    }

    ~test_axpy()
    {
      cublasDestroy(handle);
    }

    void operator()(void)
    {
        cusp::blas::axpy(cusp::cuda::par.with(handle), x, y, T(1.0));
    }

    std::string name(void) const { return (sizeof(T) == 4) ? "saxpy" : "daxpy"; }

    size_t bytes(void) const { return 3 * n * sizeof(T); }
};

template <typename T, typename MemorySpace=cusp::device_memory, typename Orientation=cusp::column_major>
struct test_gemm
{
    cusp::array2d<T,MemorySpace,Orientation> A, B;
    const size_t n;
    cublasHandle_t handle;

    test_gemm(const size_t n)
        : n(n), A(n,n), B(n,n)
    {
      if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
      {
        throw cusp::runtime_exception("cublasCreate failed");
      }
    }

    ~test_gemm()
    {
      cublasDestroy(handle);
    }

    void operator()(void)
    {
        cusp::blas::gemm(cusp::cuda::par.with(handle), A, A, B);
    }

    std::string name(void) const { return (sizeof(T) == 4) ? "sgemm" : "dgemm"; }

    size_t bytes(void) const { return A.num_rows * B.num_rows * B.num_cols * sizeof(T); }
};

template <typename Test>
void benchmark(const size_t n, const size_t iterations = 5)
{
    Test test(n);
    test();

    timer t0;
    for(size_t i = 0; i < iterations; i++)
        test();
    float ms = t0.milliseconds_elapsed() / iterations;
    float bw = (test.bytes() / (ms / 1e3)) / 1e9;

    printf("%-10s %6.1f GB/s [ %8.3f ms]\n", test.name().c_str(), bw, ms);
}

int main(int argc, char ** argv)
{

    for (size_t e = 16; e < 27; e++)
    {
        size_t n = 1 << e;
        std::cout << "N = " << n << std::endl;
        benchmark< test_nrm2<float>  >(n);
        benchmark< test_nrm2<double> >(n);
        benchmark< test_dot <float>  >(n);
        benchmark< test_dot <double> >(n);
        benchmark< test_axpy<float>  >(n);
        benchmark< test_axpy<double> >(n);
    }

    printf("\n");

    for (size_t n = 900; n < 1500; n += 100)
    {
        std::cout << "N = " << n << std::endl;
        benchmark< test_gemm<float,  cusp::device_memory, cusp::row_major>    >(n);
        benchmark< test_gemm<float,  cusp::device_memory, cusp::column_major> >(n);
        benchmark< test_gemm<double, cusp::device_memory, cusp::row_major>    >(n);
        benchmark< test_gemm<double, cusp::device_memory, cusp::column_major> >(n);
    }

    return 0;
}

