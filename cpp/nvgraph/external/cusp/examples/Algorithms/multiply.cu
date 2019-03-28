#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

int main(void)
{
    // initialize matrix
    cusp::array2d<float, cusp::host_memory> A(2,2);
    A(0,0) = 10;  A(0,1) = 20;
    A(1,0) = 40;  A(1,1) = 50;

    // initialize input vector
    cusp::array1d<float, cusp::host_memory> x(2);
    x[0] = 1;
    x[1] = 2;

    // allocate output vector
    cusp::array1d<float, cusp::host_memory> y(2);

    // compute y = A * x
    cusp::multiply(A, x, y);

    // print y
    cusp::print(y);

    return 0;
}

