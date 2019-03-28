#include <cusp/dia_matrix.h>
#include <cusp/print.h>

int main(void)
{
    // allocate storage for (4,3) matrix with 6 nonzeros in 3 diagonals
    cusp::dia_matrix<int,float,cusp::host_memory> A(4,3,6,3);

    // initialize diagonal offsets
    A.diagonal_offsets[0] = -2;
    A.diagonal_offsets[1] =  0;
    A.diagonal_offsets[2] =  1;

    // initialize diagonal values

    // first diagonal
    A.values(0,2) =  0;  // outside matrix
    A.values(1,2) =  0;  // outside matrix
    A.values(2,0) = 40;
    A.values(3,0) = 60;

    // second diagonal
    A.values(0,1) = 10;
    A.values(1,1) =  0;
    A.values(2,1) = 50;
    A.values(3,1) = 50;  // outside matrix

    // third diagonal
    A.values(0,2) = 20;
    A.values(1,2) = 30;
    A.values(2,2) =  0;  // outside matrix
    A.values(3,2) =  0;  // outside matrix

    // A now represents the following matrix
    //    [10 20  0]
    //    [ 0  0 30]
    //    [40  0 50]
    //    [ 0 60  0]

    // print matrix entries
    cusp::print(A);

    return 0;
}

