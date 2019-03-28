#pragma once

#include <cuda.h>
#include <iostream>

#include <thrust/transform_reduce.h>
#include <cusp/functional.h>

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                                \
 cudaError err = call;                                                    \
 if( cudaSuccess != err) {                                                \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
             __FILE__, __LINE__, cudaGetErrorString( err) );              \
     exit(EXIT_FAILURE);                                                  \
 } } while (0)

#define CUDA_SAFE_CALL( call) do {                                        \
 CUDA_SAFE_CALL_NO_SYNC(call);                                            \
 cudaError err = cudaThreadSynchronize();                                 \
 if( cudaSuccess != err) {                                                \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
             __FILE__, __LINE__, cudaGetErrorString( err) );              \
     exit(EXIT_FAILURE);                                                  \
 } } while (0)


void set_device(int device_id)
{
    cudaSetDevice(device_id);
}

void list_devices(void)
{
    int deviceCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
        std::cout << "There is no device supporting CUDA" << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                std::cout << "There is no device supporting CUDA." << std::endl;
            else if (deviceCount == 1)
                std::cout << "There is 1 device supporting CUDA" << std:: endl;
            else
                std::cout << "There are " << deviceCount <<  " devices supporting CUDA" << std:: endl;
        }

        std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"" << std::endl;
        std::cout << "  Major revision number:                         " << deviceProp.major << std::endl;
        std::cout << "  Minor revision number:                         " << deviceProp.minor << std::endl;
        std::cout << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes" << std::endl;
    }
    std::cout << std::endl;
}


template <typename T>
T l2_error(size_t N, const T * a, const T * b)
{
    T numerator   = 0;
    T denominator = 0;
    for(size_t i = 0; i < N; i++)
    {
        numerator   += (a[i] - b[i]) * (a[i] - b[i]);
        denominator += (b[i] * b[i]);
    }

    return numerator/denominator;
}

template <typename Array1d>
typename Array1d::value_type
l2_error(const Array1d& a, const Array1d& b)
{
    typedef typename Array1d::value_type T;

    cusp::array1d<T,cusp::host_memory> diff(b);
    cusp::blas::axpy(a, diff, T(-1));

    T numerator = thrust::transform_reduce(diff.begin(), diff.end(), cusp::square_functor<T>(), T(0), thrust::plus<T>());
    T denominator = thrust::transform_reduce(b.begin(), b.end(), cusp::square_functor<T>(), T(0), thrust::plus<T>());

    return numerator/denominator;
}
