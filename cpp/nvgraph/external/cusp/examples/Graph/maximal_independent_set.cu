#include <cusp/graph/maximal_independent_set.h>
#include <cusp/gallery/poisson.h>
#include <cusp/coo_matrix.h>

#include <cstddef>
#include <iostream>

// This example computes a maximal independent set (MIS)
// for a 10x10 grid.  The graph for the 10x10 grid is
// described by the sparsity pattern of a sparse matrix
// corresponding to a 10x10 Poisson problem.
//
// [1] http://en.wikipedia.org/wiki/Maximal_independent_set

int main(void)
{
    size_t N = 10;

    // initialize matrix representing 10x10 grid
    cusp::coo_matrix<int, float, cusp::device_memory> G;
    cusp::gallery::poisson5pt(G, N, N);

    // allocate storage for the MIS
    cusp::array1d<int, cusp::device_memory> stencil(G.num_rows);

    // compute the MIS
    cusp::graph::maximal_independent_set(G, stencil);

    // print MIS as a 2d grid
    std::cout << "maximal independent set (marked with Xs)\n";
    for (size_t i = 0; i < N; i++)
    {
        std::cout << "  ";
        for (size_t j = 0; j < N; j++)
        {
            std::cout << ((stencil[N * i + j]) ? "X" : "0");
        }
        std::cout << "\n";
    }

    return 0;
}

