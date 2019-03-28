#include <cuda.h>
#include <thrust/device_vector.h>
#include "../timer.h"

#include <iostream>
#include <iomanip>

struct malloc_free
{
    void operator()(size_t n)
    {
        char * buff = (char *) malloc(n);
        for (size_t i = 0; i < n; i += 4096)
            buff[i] = '\0';
        free(buff);
    }
};

struct cudaMalloc_cudaFree
{
    void operator()(size_t n)
    {
        char * buff;
        cudaMalloc(&buff, n);
        cudaFree(buff);
    }
};

struct thrust_device_malloc_device_free
{
    void operator()(size_t n)
    {
        thrust::device_ptr<char> buff = thrust::device_malloc<char>(n);
        thrust::device_free(buff);
    }
};

struct thrust_device_vector
{
    void operator()(size_t n)
    {
        thrust::device_vector<char> v(n);
    }
};

template <typename TestFunction>
void benchmark(TestFunction f, size_t max_n = (size_t) 1 << 30, size_t num_iterations = 10)
{
    for (size_t n = 1; n <= max_n; n *= 2)
    {
        // warm up
        //f(n);

        // time several iterations
        timer t;
        for(size_t i = 0; i < num_iterations; i++)
        {
            f(n);
        }
        cudaThreadSynchronize();

        float ms = t.milliseconds_elapsed() / num_iterations;
        std::cout << std::setw(12) << n;
        std::cout << std::setw(12) << std::setiosflags(std::ios::fixed) << std::setprecision(5) << ms;
        std::cout << " ms" << std::endl;
    }
}


int main(void)
{
    std::cout << "malloc() & free()" << std::endl;
    benchmark(malloc_free());

    std::cout << "cudaMalloc() & cudaFree()" << std::endl;
    benchmark(cudaMalloc_cudaFree());
    
    std::cout << "thrust::device_malloc & thrust::device_free" << std::endl;
    benchmark(thrust_device_malloc_device_free());

    std::cout << "thrust::device_vector" << std::endl;
    benchmark(thrust_device_vector());

    return 0;
}

