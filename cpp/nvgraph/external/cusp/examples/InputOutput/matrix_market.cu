#include <cusp/io/matrix_market.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>

int main(void)
{
    // create a simple example
    cusp::array2d<float, cusp::host_memory> A(3,4);
    A(0,0) = 10;  A(0,1) =  0;  A(0,2) = 20;  A(0,3) =  0;
    A(1,0) =  0;  A(1,1) = 30;  A(1,2) =  0;  A(1,3) = 40;
    A(2,0) = 50;  A(2,1) = 60;  A(2,2) = 70;  A(2,3) = 80;

    // save A to disk in MatrixMarket format
    cusp::io::write_matrix_market_file(A, "A.mtx");

    // load A from disk into a coo_matrix
    cusp::coo_matrix<int, float, cusp::device_memory> B;
    cusp::io::read_matrix_market_file(B, "A.mtx");

    // print B
    cusp::print(B);

    return 0;
}

