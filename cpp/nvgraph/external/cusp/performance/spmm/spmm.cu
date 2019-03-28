#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/multiply.h>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cusparse_v2.h>

#include "../timer.h"

cusparseStatus_t status;
cusparseHandle_t handle  = 0;
cusparseMatDescr_t descrA = 0;
cusparseMatDescr_t descrB = 0;
cusparseMatDescr_t descrC = 0;


#define CLEANUP(s)                                   \
do {                                                 \
    printf ("%s\n", s);                              \
    if (descrA)              cusparseDestroyMatDescr(descrA);\
    if (descrB)              cusparseDestroyMatDescr(descrB);\
    if (descrC)              cusparseDestroyMatDescr(descrC);\
    if (handle)             cusparseDestroy(handle); \
    cudaDeviceReset();          \
    fflush (stdout);                                 \
} while (0)

int cusparse_init(void)
{
    /* initialize cusparse library */
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library initialization failed");
        return 1;
    }

    /* create and setup matrix descriptor */
    status = cusparseCreateMatDescr(&descrA);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseCreateMatDescr(&descrB);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseCreateMatDescr(&descrC);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);

    return 0;
}

template <typename MatrixType, typename InputType>
float time_spmm(const InputType& A,
                const InputType& B)
{
    unsigned int N = 10;

    MatrixType A_;
    MatrixType B_;

    try
    {
        A_ = A;
        B_ = B;
    }
    catch (cusp::format_conversion_exception)
    {
        return -1;
    }

    timer t;

    for(unsigned int i = 0; i < N; i++)
    {
        MatrixType C_;
        cusp::multiply(A_, B_, C_);
    }

    return t.milliseconds_elapsed() / N;
}

template <typename MatrixType, typename InputType>
float time_cusparse(const InputType& A,
                    const InputType& B)
{
    if( cusparse_init() )
    {
	throw cusp::runtime_exception("CUSPARSE init failed");
    }

    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    unsigned int N = 10;

    int m = A.num_rows;
    int n = A.num_cols;
    int k = B.num_cols;
    int nnzA = A.num_entries;
    int nnzB = B.num_entries;

    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    MatrixType A_(A);
    MatrixType B_(B);

    int * csrRowPtrA = thrust::raw_pointer_cast(&A_.row_offsets[0]);
    int * csrColIndA = thrust::raw_pointer_cast(&A_.column_indices[0]);
    float * csrValA  = thrust::raw_pointer_cast(&A_.values[0]);

    int * csrRowPtrB = thrust::raw_pointer_cast(&B_.row_offsets[0]);
    int * csrColIndB = thrust::raw_pointer_cast(&B_.column_indices[0]);
    float * csrValB  = thrust::raw_pointer_cast(&B_.values[0]);

    int * csrRowPtrC;
    int * csrColIndC;
    float * csrValC;

    timer t;

    for(unsigned int i = 0; i < N; i++)
    {
        int baseC, nnzC;
        cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
        status = cusparseXcsrgemmNnz(handle, transA, transB, m, n, k,
                                     descrA, nnzA, csrRowPtrA, csrColIndA,
                                     descrB, nnzB, csrRowPtrB, csrColIndB,
                                     descrC, csrRowPtrC, &nnzC );
        if (status != CUSPARSE_STATUS_SUCCESS) {
            CLEANUP("CSR Matrix-Matrix multiplication failed");
            return 1;
        }
        cudaMemcpy(&nnzC , csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, csrRowPtrC  , sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
        cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
        cudaMalloc((void**)&csrValC   , sizeof(float)*nnzC);
        status = cusparseScsrgemm(handle, transA, transB, m, n, k,
                                  descrA, nnzA,
                                  csrValA, csrRowPtrA, csrColIndA,
                                  descrB, nnzB,
                                  csrValB, csrRowPtrB, csrColIndB,
                                  descrC,
                                  csrValC, csrRowPtrC, csrColIndC);

        if (status != CUSPARSE_STATUS_SUCCESS) {
            CLEANUP("CSR Matrix-Matrix multiplication failed");
            return 1;
        }

        cudaFree(csrRowPtrC);
        cudaFree(csrColIndC);
        cudaFree(csrValC);
    }

    return t.milliseconds_elapsed() / N;
}

int main(int argc, char ** argv)
{
    typedef int    IndexType;
    typedef float  ValueType;

    typedef cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> CSRHost;
    typedef cusp::csr_matrix<IndexType,ValueType,cusp::device_memory> CSRDev;
    typedef cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> COO;

    cudaSetDevice(0);

    CSRHost A;
    CSRHost B;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        cusp::gallery::poisson5pt(A, 200, 200);
        cusp::gallery::poisson5pt(B, 200, 200);
    }
    else if (argc == 2)
    {
        // no input file was specified, generate an example
        cusp::io::read_matrix_market_file(A, argv[1]);
        B = A;
    }
    else if (argc == 3)
    {
        // input files were specified, read them from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
        cusp::io::read_matrix_market_file(B, argv[2]);
    }

    std::cout << "Input matrix A has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n";
    std::cout << "             B has shape (" << B.num_rows << "," << B.num_cols << ") and " << B.num_entries << " entries" << "\n\n";

    printf("Host Sparse Matrix-Matrix Multiply (milliseconds per multiplication)\n");
    printf("    Host    | %9.2f\n", time_spmm<CSRHost>(A,B));

    printf("\n\n");

    printf("Device Sparse Matrix-Matrix Multiply (milliseconds per multiplication)\n");
    printf("    Device  | %9.2f\n", time_spmm<COO>(A,B));
    printf(" CUSPARSE   | %9.2f\n", time_cusparse<CSRDev>(A,B));

    return 0;
}

