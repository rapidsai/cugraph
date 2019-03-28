#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/breadth_first_search.h>

int main(int argc, char*argv[])
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int,float,cusp::device_memory> G;

    // create a 2d Poisson problem on a 16x16 mesh
    cusp::gallery::poisson5pt(G, 16, 16);

    // set the source vertex
    int source = 0;

    // create vector to contain BFS labels
    cusp::array1d<int,cusp::device_memory> labels(G.num_rows);

    // execute BFS traversal on device
    cusp::graph::breadth_first_search(G, source, labels);

    return 0;
}

