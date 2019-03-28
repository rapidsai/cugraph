#include <cusp/gallery/poisson.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>

int main(void)
{
    cusp::coo_matrix<int, float, cusp::device_memory> A;

    // create a matrix for a Poisson problem on a 4x4 grid
    cusp::gallery::poisson5pt(A, 4, 4);

    // print matrix
    cusp::print(A);

    return 0;
}

