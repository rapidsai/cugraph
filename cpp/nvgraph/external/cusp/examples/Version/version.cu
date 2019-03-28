#include <thrust/version.h>
#include <cusp/version.h>
#include <iostream>

int main(void)
{
    std::cout << "The following libraries were found:" << std::endl;

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVCC
    std::cout << "    CUDA   v" << (CUDA_VERSION / 1000) << "." <<
                                   (CUDA_VERSION % 1000) / 10 << std::endl;
#endif

    std::cout << "    Thrust v" << THRUST_MAJOR_VERSION << "." <<
                                   THRUST_MINOR_VERSION << "." <<
                                   THRUST_SUBMINOR_VERSION << std::endl;

    std::cout << "    Cusp   v" << CUSP_MAJOR_VERSION << "." <<
                                   CUSP_MINOR_VERSION << "." <<
                                   CUSP_SUBMINOR_VERSION << std::endl;

    return 0;
}

