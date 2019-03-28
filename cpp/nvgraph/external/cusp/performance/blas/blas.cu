#include <cusp/array1d.h>
#include <cusp/blas/blas.h>

#include <iostream>
#include <stdio.h>

#include "../timer.h"

template <typename T, typename MemorySpace=cusp::device_memory>
struct test_nrm2
{
    cusp::array1d<T,MemorySpace> x;
    const size_t n;

    test_nrm2(const size_t n)
        : n(n), x(n) {}

    void operator()(void)
    {
        cusp::blas::nrm2(x);
    }

    std::string name(void) const { return (sizeof(T) == 4) ? "snrm2" : "dnrm2"; }

    size_t bytes(void) const { return n * sizeof(T); }
};

template <typename T, typename MemorySpace=cusp::device_memory>
struct test_dot
{
    cusp::array1d<T,MemorySpace> x, y;
    const size_t n;

    test_dot(const size_t n)
        : n(n), x(n), y(n) {}

    void operator()(void)
    {
        cusp::blas::dot(x, y);
    }

    std::string name(void) const { return (sizeof(T) == 4) ? "sdot" : "ddot"; }

    size_t bytes(void) const { return 2 * n * sizeof(T); }
};

template <typename T, typename MemorySpace=cusp::device_memory>
struct test_axpy
{
    cusp::array1d<T,MemorySpace> x, y;
    const size_t n;

    test_axpy(const size_t n)
        : n(n), x(n), y(n) {}

    void operator()(void)
    {
        cusp::blas::axpy(x, y, T(1.0));
    }

    std::string name(void) const { return (sizeof(T) == 4) ? "saxpy" : "daxpy"; }

    size_t bytes(void) const { return 3 * n * sizeof(T); }
};


template <typename Test>
void benchmark(const size_t n)
{
    Test test(n);
    test();

    timer t0;
    test();
    float ms = t0.milliseconds_elapsed();
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

    return 0;
}

