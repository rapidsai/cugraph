#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/pseudo_peripheral.h>

#include <iostream>

int main(int argc, char*argv[])
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int,float,cusp::device_memory> G;

    // create a 2d Poisson problem on a 1024x1024 mesh
    cusp::gallery::poisson5pt(G, 1024, 1024);

    // compute pseudo peripheral vertex on device
    int vertex = cusp::graph::pseudo_peripheral_vertex(G);

    std::cout << "Pseudo-peripheral vertex : " << vertex << std::endl;

    return 0;
}

