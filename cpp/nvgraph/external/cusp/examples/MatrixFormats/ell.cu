#include <cusp/ell_matrix.h>
#include <cusp/print.h>

int main(void)
{
    // allocate storage for (4,3) matrix with 6 nonzeros and at most 3 nonzeros per row.
    cusp::ell_matrix<int,float,cusp::host_memory> A(4,3,6,3);

    // X is used to fill unused entries in the matrix
    const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;

    // Initialize A to represent the following matrix
    //    [10  0 20]
    //    [ 0  0  0]
    //    [ 0  0 30]
    //    [40 50 60]

    // first row
    A.column_indices(0,0) = 0; A.values(0,0) = 10;
    A.column_indices(0,1) = 2; A.values(0,1) = 20;  // shifted to leftmost position
    A.column_indices(0,2) = X; A.values(0,2) =  0;  // padding

    // second row
    A.column_indices(1,0) = X; A.values(1,0) =  0;  // padding
    A.column_indices(1,1) = X; A.values(1,1) =  0;  // padding
    A.column_indices(1,2) = X; A.values(1,2) =  0;  // padding

    // third row
    A.column_indices(2,0) = 2; A.values(2,0) = 30;  // shifted to leftmost position
    A.column_indices(2,1) = X; A.values(2,1) =  0;  // padding
    A.column_indices(2,2) = X; A.values(2,2) =  0;  // padding

    // fourth row
    A.column_indices(3,0) = 0; A.values(3,0) = 40;
    A.column_indices(3,1) = 1; A.values(3,1) = 50;
    A.column_indices(3,2) = 2; A.values(3,2) = 60;

    // print matrix entries
    cusp::print(A);

    return 0;
}

