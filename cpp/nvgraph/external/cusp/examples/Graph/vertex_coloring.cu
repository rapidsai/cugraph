#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/vertex_coloring.h>

int main(int argc, char*argv[])
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int,float,cusp::device_memory> G;

    // create a 2d Poisson problem on a 16x16 mesh
    cusp::gallery::poisson5pt(G, 16, 16);

    // create vector to contain vertex colors
    cusp::array1d<int,cusp::device_memory> colors(G.num_rows, 0);

    // execute vertex coloring on device
    cusp::graph::vertex_coloring(G, colors);

    return 0;
}

