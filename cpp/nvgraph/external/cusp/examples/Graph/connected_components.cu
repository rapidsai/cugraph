#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/connected_components.h>

int main(int argc, char*argv[])
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int,float,cusp::device_memory> G;

    // create a 2d Poisson problem on a 512x512 mesh
    cusp::gallery::poisson5pt(G, 512, 512);

    // create vector to contain component labels
    cusp::array1d<int,cusp::device_memory> components(G.num_rows);

    // execute connected components operation on device
    cusp::graph::connected_components(G, components);

    return 0;
}

