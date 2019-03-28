#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/maximal_independent_set.h>
#include <cusp/io/matrix_market.h>

#include "../timer.h"

template<typename MemorySpace, typename MatrixType>
void MIS(const MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef cusp::csr_matrix<IndexType,IndexType,MemorySpace> GraphType;

    GraphType G_mis(G);
    cusp::array1d<bool,MemorySpace> stencil(G.num_rows);

    timer t;
    size_t num_mis = cusp::graph::maximal_independent_set(G_mis, stencil);
    std::cout << "MIS time : " << t.milliseconds_elapsed() << " (ms)." << std::endl;
    std::cout << "Number of MIS vertices : " << num_mis << std::endl;
}

int main(int argc, char*argv[])
{
    srand(time(NULL));

    typedef int   IndexType;
    typedef float ValueType;
    typedef cusp::host_memory MemorySpace;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;
    size_t size = 512;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        std::cout << "Generated matrix (poisson5pt) ";
        cusp::gallery::poisson5pt(A, size, size);
    }
    else if (argc == 2)
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
        std::cout << "Read matrix (" << argv[1] << ") ";
    }

    std::cout << "with shape ("  << A.num_rows << "," << A.num_cols << ") and "
              << A.num_entries << " entries" << "\n\n";

    std::cout << " Device ";
    MIS<cusp::device_memory>(A);

    std::cout << " Host ";
    MIS<cusp::host_memory>(A);

    return EXIT_SUCCESS;
}

