#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/vertex_coloring.h>
#include <cusp/io/matrix_market.h>

#include "../timer.h"

template<typename MemorySpace, typename MatrixType>
void coloring(const MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef cusp::csr_matrix<IndexType,IndexType,MemorySpace> GraphType;

    GraphType G_csr(G);
    cusp::array1d<IndexType,MemorySpace> colors(G.num_rows, 0);

    timer t;
    size_t max_color = cusp::graph::vertex_coloring(G_csr, colors);
    std::cout << "Coloring time    : " << t.milliseconds_elapsed() << " (ms)." << std::endl;
    std::cout << "Number of colors : " << max_color << std::endl;

    if(max_color > 0)
    {
      cusp::array1d<IndexType,MemorySpace> color_counts(max_color);
      thrust::sort(colors.begin(), colors.end());
      thrust::reduce_by_key(colors.begin(),
                          colors.end(),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          color_counts.begin());
      cusp::print(color_counts);
    }
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
    coloring<cusp::device_memory>(A);

    std::cout << " Host ";
    coloring<cusp::host_memory>(A);

    return EXIT_SUCCESS;
}

