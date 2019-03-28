#include <cusp/array2d.h>
#include <cusp/blas/blas.h>
#include <cusp/print.h>
#include <iostream>

int main(void)
{
    // initialize x vector
    cusp::array1d<float, cusp::host_memory> x(2);
    x[0] = 1;
    x[1] = 2;

    // initialize y vector
    cusp::array1d<float, cusp::host_memory> y(2);
    y[0] = 1;
    y[1] = 2;

    // compute y = alpha * x + y
    cusp::blas::axpy(x,y,2);
    // print y
    cusp::print(y);

    // allocate output vector
    cusp::array1d<float, cusp::host_memory> z(2);
    // compute z = x .* y (element-wise multiplication)
    cusp::blas::xmy(x,y,z);
    // print z
    cusp::print(z);

    // compute the l_2 norm of z in 2 different ways
    std::cout << "|z| = " << cusp::blas::nrm2(z) << std::endl;
    std::cout << "sqrt(z'z) = " << sqrt(cusp::blas::dotc(z,z)) << std::endl;
    // compute the l_1 norm of z (manhattan distance)
    std::cout << "|z|_1 = " << cusp::blas::nrm1(z) << std::endl;
    // compute the largest component of a vector in absolute value
    std::cout << "max(|z_i|) = " << cusp::blas::nrmmax(z) << std::endl;

    return 0;
}
