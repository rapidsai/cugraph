#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkl_rci.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_service.h"
#include "mkl_types.h"

void spmv_mkl_float(MKL_INT m, 
                    float values[],
                    MKL_INT rowIndex[],
                    MKL_INT columns[],
                    float x[],
                    float y[])
{
    char transa = 'n';

    mkl_cspblas_scsrgemv(&transa, &m, values, rowIndex, columns, x, y);
}

void spmv_mkl_double(MKL_INT m, 
                     double values[],
                     MKL_INT rowIndex[],
                     MKL_INT columns[],
                     double x[],
                     double y[])
{
    char transa = 'n';

    mkl_cspblas_dcsrgemv(&transa, &m, values, rowIndex, columns, x, y);
}

int cg_mkl_double(MKL_INT n, 
                  double a[], 
                  MKL_INT ia[],
                  MKL_INT ja[],
                  double solution[],
                  double rhs[],
                  MKL_INT max_iter,
                  double r_tol,
                  double a_tol)
{
	MKL_INT rci_request, itercount, i;

    // parameter arrays for solver
	MKL_INT ipar[128];
    double  dpar[128];

	double euclidean_norm;
    
    // for SpMV
    char tr = 'n';

    double * tmp;
    double * residual;

    tmp      = (double *) malloc(4 * n * sizeof(double));	
    residual = (double *) malloc(n * sizeof(double));

	// initialize the solver
	dcg_init(&n,solution,rhs,&rci_request,ipar,dpar,tmp);

	if (rci_request!=0) goto failure;
    
	ipar[1]=6;                       // output all warnings and errors 
	ipar[4]=max_iter;                // maximum number of iterations
	ipar[7]=1;                       // stop iteration at maximum iterations
	ipar[8]=1;                       // residual stopping test
	ipar[9]=0;                       // request for the user defined stopping test
	dpar[0]=r_tol * r_tol;           // relative residual tolerance
	dpar[1]=a_tol * a_tol;           // absolute residual tolerance

	/*---------------------------------------------------------------------------*/
	/* Check the correctness and consistency of the newly set parameters         */
	/*---------------------------------------------------------------------------*/
	dcg_check(&n,solution,rhs,&rci_request,ipar,dpar,tmp);
	if (rci_request!=0) goto failure;

	/*---------------------------------------------------------------------------*/
	/* Compute the solution by RCI (P)CG solver without preconditioning          */
	/* Reverse Communications starts here                                        */
	/*---------------------------------------------------------------------------*/
rci: dcg(&n,solution,rhs,&rci_request,ipar,dpar,tmp);
    //printf("Residual norm is %e\n", sqrt(dpar[4]));
	/*---------------------------------------------------------------------------*/
	/* If rci_request=0, then the solution was found with the required precision */
	/*---------------------------------------------------------------------------*/
	if (rci_request==0) goto getsln;
	/*---------------------------------------------------------------------------*/
	/* If rci_request=1, then compute the vector A*tmp[0]                        */
	/* and put the result in vector tmp[n]                                       */
	/*---------------------------------------------------------------------------*/
	if (rci_request==1)
	{
        mkl_cspblas_dcsrgemv(&tr, &n, a, ia, ja, tmp, &tmp[n]);
		goto rci;
	}
	/*---------------------------------------------------------------------------*/
	/* If rci_request=anything else, then dcg subroutine failed                  */
	/* to compute the solution vector: solution[n]                               */
	/*---------------------------------------------------------------------------*/
	goto failure;
	/*---------------------------------------------------------------------------*/
	/* Reverse Communication ends here                                           */
	/* Get the current iteration number into itercount                           */
	/*---------------------------------------------------------------------------*/
getsln: dcg_get(&n,solution,rhs,&rci_request,ipar,dpar,tmp,&itercount);

    mkl_cspblas_dcsrgemv(&tr, &n, a, ia, ja, solution, residual);
	for(i=0;i<n;i++) residual[i] -= rhs[i];
    i=1; euclidean_norm=dnrm2(&n,residual,&i);
	
    printf("\nMKL CG reached %e residual in %d iterations\n",euclidean_norm, itercount);

    // release memory
	MKL_FreeBuffers();
    free(tmp);
    free(residual);

    if (itercount <= max_iter && (euclidean_norm * euclidean_norm) < (dpar[0] * dpar[4] + dpar[5]))
    {
//        printf("This example has successfully PASSED through all steps of computation!");
//        printf("\n");
//        printf("(Residual norm is %e)\n", euclidean_norm);
        return 0;
    }
    else
    {
//        printf("This example may have FAILED as either the number of iterations exceeds");
//        printf("\nthe maximum number of iterations %d, or the ", max_iter);
//        printf("computed solution\ndiffers has not sufficiently converged.");
//        printf("(Residual norm is %e), or both.\n", euclidean_norm);
        return 1;
    }
	/*-------------------------------------------------------------------------*/
	/* Release internal MKL memory that might be used for computations         */
	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	/* unless you disable MKL Memory Manager                                   */
	/*-------------------------------------------------------------------------*/
failure: printf("This example FAILED as the solver has returned the ERROR ");
				 printf("code %d", rci_request);
         MKL_FreeBuffers();
         return 1;
}

//int main(void)
//{
//    /* Fill all arrays containing matrix data. */
//    MKL_INT ia[12]={ 0,  2,  5,  8, 11, 14, 17, 20, 23, 26, 29, 31};
//    MKL_INT ja[31]={0,  1,  0,  1,  2,  1,  2,  3,  2,  3,  4,  3,  4,  5,  4,  5,  6, 5,  6,  7,  6,  7,  8,  7,  8,  9,  8,  9, 10,  9, 10};
//    double   a[31]={ 2., -1., -1.,  2., -1., -1.,  2., -1., -1.,  2., -1., -1.,  2., -1., -1.,  2., -1., -1.,  2., -1., -1.,  2., -1., -1.,  2., -1., -1.,  2., -1., -1.,  2.};
//  	
//    //MKL_INT ia[26]={  0,   3,   7,  11,  15,  18,  22,  27,  32,  37,  41,  45,  50, 55,  60,  64,  68,  73,  78,  83,  87,  90,  94,  98, 102, 105};
//    //MKL_INT ja[105]={ 0,  1,  5,  0,  1,  2,  6,  1,  2,  3,  7,  2,  3,  4,  8,  3,  4, 9,  0,  5,  6, 10,  1,  5,  6,  7, 11,  2,  6,  7,  8, 12,  3,  7, 8,  9, 13,  4,  8,  9, 14,  5, 10, 11, 15,  6, 10, 11, 12, 16,  7, 11, 12, 13, 17,  8, 12, 13, 14, 18,  9, 13, 14, 19, 10, 15, 16, 20, 11, 15, 16, 17, 21, 12, 16, 17, 18, 22, 13, 17, 18, 19, 23, 14, 18, 19, 24, 15, 20, 21, 16, 20, 21, 22, 17, 21, 22, 23, 18, 22, 23, 24, 19, 23, 24};
//    //double   a[105]={ 4., -1., -1., -1.,  4., -1., -1., -1.,  4., -1., -1., -1.,  4., -1., -1., -1.,  4., -1., -1.,  4., -1., -1., -1., -1.,  4., -1., -1., -1., -1.,  4., -1., -1., -1., -1.,  4., -1., -1., -1., -1.,        4., -1., -1.,  4., -1., -1., -1., -1.,  4., -1., -1., -1., -1., 4., -1., -1., -1., -1.,  4., -1., -1., -1., -1.,  4., -1., -1.,        4., -1., -1., -1., -1.,  4., -1., -1., -1., -1.,  4., -1., -1., -1., -1.,  4., -1., -1., -1., -1.,  4., -1., -1.,  4., -1., -1., -1.,  4., -1., -1., -1.,  4., -1., -1., -1.,  4., -1., -1., -1.,  4.};
//    
//  	double * rhs;
//  	double * solution;
//
//    MKL_INT n = (sizeof(ia) / sizeof(MKL_INT)) - 1;
//    MKL_INT i;
//    int result;
//	
//    rhs      = (double *) malloc(n * sizeof(double));
//    solution = (double *) malloc(n * sizeof(double));
//
//    for(i=0;i<n;i++) rhs[i]      = rand() % 100;
//	  for(i=0;i<n;i++) solution[i] = 0;
//
//    result = cg_mkl_double(n, a, ia, ja, solution, rhs, n, 1e-5, 0.0);
//
//    free(rhs);
//    free(solution);
//
//    return result;
//}

