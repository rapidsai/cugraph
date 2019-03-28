#include <cusp/coo_matrix.h>
#include <cusp/print.h>

int main(void)
{
    // allocate storage for (4,3) matrix with 6 nonzeros
    cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);

    // initialize matrix entries on host
    A.row_indices[0] = 0; A.column_indices[0] = 0; A.values[0] = 10;
    A.row_indices[1] = 0; A.column_indices[1] = 2; A.values[1] = 20;
    A.row_indices[2] = 2; A.column_indices[2] = 2; A.values[2] = 30;
    A.row_indices[3] = 3; A.column_indices[3] = 0; A.values[3] = 40;
    A.row_indices[4] = 3; A.column_indices[4] = 1; A.values[4] = 50;
    A.row_indices[5] = 3; A.column_indices[5] = 2; A.values[5] = 60;

    // A now represents the following matrix
    //    [10  0 20]
    //    [ 0  0  0]
    //    [ 0  0 30]
    //    [40 50 60]

    // print matrix entries
    cusp::print(A);

    return 0;
}

