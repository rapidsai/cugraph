#include <cusp/transpose.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

int main(void)
{
    // initialize a 2x3 matrix
    cusp::array2d<float, cusp::host_memory> A(2,3);
    A(0,0) = 10;  A(0,1) = 20;  A(0,2) = 30;
    A(1,0) = 40;  A(1,1) = 50;  A(1,2) = 60;

    // print A
    cusp::print(A);

    // compute the transpose
    cusp::array2d<float, cusp::host_memory> At;
    cusp::transpose(A, At);

    // print A^T
    cusp::print(At);

    return 0;
}

