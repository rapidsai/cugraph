#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/blas/blas.h>

#include <thrust/binary_search.h>

#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <stdio.h>

#include "../timer.h"


template <typename SourceType, typename DestinationType, typename InputType>
float time_conversion(const InputType& A)
{
    unsigned int N = 10;

    SourceType S;

    try
    {
        S = A;
    }
    catch (cusp::format_conversion_exception)
    {
        return -1;
    }

    try
    {
        DestinationType D(S);
    }
    catch (cusp::format_conversion_exception)
    {
        return -1;
    }

    timer t;

    for(unsigned int i = 0; i < N; i++)
        DestinationType D(S);

    return t.milliseconds_elapsed() / N;
}

template <typename SourceType, typename InputType>
void for_each_destination(const InputType& A)
{
    typedef typename SourceType::index_type   I;
    typedef typename SourceType::value_type   V;
    typedef typename SourceType::memory_space M;

    typedef cusp::coo_matrix<I,V,M> COO;
    typedef cusp::csr_matrix<I,V,M> CSR;
    typedef cusp::dia_matrix<I,V,M> DIA;
    typedef cusp::ell_matrix<I,V,M> ELL;
    typedef cusp::hyb_matrix<I,V,M> HYB;

    printf(" %9.2f |", time_conversion<SourceType, COO>(A));
    printf(" %9.2f |", time_conversion<SourceType, CSR>(A));
    printf(" %9.2f |", time_conversion<SourceType, DIA>(A));
    printf(" %9.2f |", time_conversion<SourceType, ELL>(A));
    printf(" %9.2f |", time_conversion<SourceType, HYB>(A));
}

template <typename MemorySpace, typename InputType>
void for_each_source(const InputType& A)
{
    typedef typename InputType::index_type I;
    typedef typename InputType::value_type V;

    typedef cusp::coo_matrix<I,V,MemorySpace> COO;
    typedef cusp::csr_matrix<I,V,MemorySpace> CSR;
    typedef cusp::dia_matrix<I,V,MemorySpace> DIA;
    typedef cusp::ell_matrix<I,V,MemorySpace> ELL;
    typedef cusp::hyb_matrix<I,V,MemorySpace> HYB;

    printf(" From \\ To |    COO    |    CSR    |    DIA    |    ELL    |    HYB    |\n");
    printf("    COO    |");
    for_each_destination<COO>(A);
    printf("\n");
    printf("    CSR    |");
    for_each_destination<CSR>(A);
    printf("\n");
    printf("    DIA    |");
    for_each_destination<DIA>(A);
    printf("\n");
    printf("    ELL    |");
    for_each_destination<ELL>(A);
    printf("\n");
    printf("    HYB    |");
    for_each_destination<HYB>(A);
    printf("\n\n");

    printf(" To COO view |    COO    |    CSR    |    DIA    |    ELL    |    HYB    |\n");
    printf("\t     ");
    printf("| %9.2f ", time_conversion<COO, typename COO::const_coo_view_type>(A));
    printf("| %9.2f ", time_conversion<CSR, typename CSR::const_coo_view_type>(A));
    printf("| %9.2f ", time_conversion<DIA, typename DIA::const_coo_view_type>(A));
    printf("| %9.2f ", time_conversion<ELL, typename ELL::const_coo_view_type>(A));
    printf("| %9.2f |", time_conversion<HYB, typename HYB::const_coo_view_type>(A));
    printf("\n");
}

int main(int argc, char ** argv)
{
    cudaSetDevice(0);

    typedef int    IndexType;
    typedef float  ValueType;

    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> A;

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

    std::cout << "Input matrix has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n\n";

    printf("Host Conversions (milliseconds per conversion)\n");
    for_each_source<cusp::host_memory>(A);

    printf("\n\n");

    printf("Device Conversions (milliseconds per conversion)\n");
    for_each_source<cusp::device_memory>(A);

    return 0;
}

