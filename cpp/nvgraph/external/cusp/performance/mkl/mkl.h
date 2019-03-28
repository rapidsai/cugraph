#pragma once

#include "mkl_types.h"

void spmv_mkl_float(MKL_INT m, 
                    float values[],
                    MKL_INT rowIndex[],
                    MKL_INT columns[],
                    float x[],
                    float y[]);

void spmv_mkl_double(MKL_INT m, 
                     double values[],
                     MKL_INT rowIndex[],
                     MKL_INT columns[],
                     double x[],
                     double y[]);

int cg_mkl_double(MKL_INT n, 
                  double a[], 
                  MKL_INT ia[],
                  MKL_INT ja[],
                  double solution[],
                  double rhs[],
                  MKL_INT max_iter,
                  double r_tol,
                  double a_tol);

