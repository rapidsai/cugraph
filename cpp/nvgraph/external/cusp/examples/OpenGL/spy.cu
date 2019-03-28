#include <cusp/complex.h>

#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/opengl/spy/spy.h>

int main(int argc, char** argv)
{
    cusp::csr_matrix<int,float,cusp::host_memory> A;
    if (argc == 1)
    {
        // no input file was specified, generate an example
        cusp::gallery::poisson5pt(A, 500, 500);
    }
    else if (argc == 2)
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
    }
    cusp::opengl::spy::view_matrix(A);

    return 0;
} // end main

