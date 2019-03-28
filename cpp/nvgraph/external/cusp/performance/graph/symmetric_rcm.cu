#include <cusp/csr_matrix.h>
#include <cusp/permutation_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/symmetric_rcm.h>
#include <cusp/io/matrix_market.h>

#include <thrust/functional.h>
#include "../timer.h"

template<typename MatrixType>
size_t bandwidth(const MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    cusp::coo_matrix<IndexType,ValueType,MemorySpace> G_coo(G);

    cusp::array1d<IndexType, MemorySpace> min_column(G.num_rows, 0);
    cusp::array1d<IndexType, MemorySpace> max_column(G.num_rows, 0);
    cusp::array1d<IndexType, MemorySpace> rowwise_bandwidth(G.num_rows, 0);

    thrust::reduce_by_key(G_coo.row_indices.begin(),
                          G_coo.row_indices.end(),
                          G_coo.column_indices.begin(),
                          thrust::make_discard_iterator(),
                          min_column.begin(),
                          thrust::equal_to<IndexType>(),
                          thrust::minimum<IndexType>());
    thrust::reduce_by_key(G_coo.row_indices.begin(),
                          G_coo.row_indices.end(),
                          G_coo.column_indices.begin(),
                          thrust::make_discard_iterator(),
                          max_column.begin(),
                          thrust::equal_to<IndexType>(),
                          thrust::maximum<IndexType>());

    thrust::transform(max_column.begin(), max_column.end(), min_column.begin(), rowwise_bandwidth.begin(), thrust::minus<IndexType>());
    return *thrust::max_element(rowwise_bandwidth.begin(), rowwise_bandwidth.end()) + 1;
}

template<typename MemorySpace, typename MatrixType>
void RCM(const MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef cusp::csr_matrix<IndexType,IndexType,MemorySpace> GraphType;
    typedef cusp::array1d<IndexType,MemorySpace> Array;

    GraphType G_rcm(G);
    cusp::permutation_matrix<IndexType,MemorySpace> P(G.num_rows);

    timer t;
    cusp::graph::symmetric_rcm(G_rcm, P);
    std::cout << " RCM time : " << t.milliseconds_elapsed() << " (ms)." << std::endl;

    P.symmetric_permute(G_rcm);
    std::cout << " Bandwidth after RCM : " << bandwidth(G_rcm) << std::endl;
}

int main(int argc, char*argv[])
{
    srand(time(NULL));

    typedef int   IndexType;
    typedef float ValueType;
    typedef cusp::device_memory MemorySpace;

    cusp::coo_matrix<IndexType, ValueType, MemorySpace> A;
    size_t size = 1024;

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

    std::cout << "Bandwidth before RCM : " << bandwidth(A) << std::endl;

    std::cout << " Device ";
    RCM<cusp::device_memory>(A);

    std::cout << " Host ";
    RCM<cusp::host_memory>(A);

    return EXIT_SUCCESS;
}

