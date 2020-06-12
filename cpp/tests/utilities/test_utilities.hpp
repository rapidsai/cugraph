/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "functions.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include "mmio.h"
}
#include <nccl.h>

#include <cstdio>
#include <string>

// FIXME: RAFT error handling macros should be used instead
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                               \
  {                                                                                      \
    cudaError_t cudaStatus = call;                                                       \
    if (cudaSuccess != cudaStatus) {                                                     \
      fprintf(stderr,                                                                    \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
              #call,                                                                     \
              __LINE__,                                                                  \
              __FILE__,                                                                  \
              cudaGetErrorString(cudaStatus),                                            \
              cudaStatus);                                                               \
    }                                                                                    \
  }
#endif

// FIXME: RAFT error handling macros should be used instead
#define NCCLCHECK(cmd)                                                                          \
  {                                                                                             \
    ncclResult_t nccl_status = cmd;                                                             \
    if (nccl_status != ncclSuccess) {                                                           \
      printf("NCCL failure %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(nccl_status)); \
      FAIL();                                                                                   \
    }                                                                                           \
  }

#define MPICHECK(cmd)                                                  \
  {                                                                    \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      FAIL();                                                          \
    }                                                                  \
  }

namespace cugraph {
namespace test {

std::string getFileName(const std::string& s)
{
  char sep = '/';

#ifdef _WIN32
  sep = '\\';
#endif

  size_t i = s.rfind(sep, s.length());
  if (i != std::string::npos) { return (s.substr(i + 1, s.length() - i)); }
  return ("");
}

/// Read matrix properties from Matrix Market file
/** Matrix Market file is assumed to be a sparse matrix in coordinate
 *  format.
 *
 *  @param f File stream for Matrix Market file.
 *  @param tg Boolean indicating whether to convert matrix to general
 *  format (from symmetric, Hermitian, or skew symmetric format).
 *  @param t (Output) MM_typecode with matrix properties.
 *  @param m (Output) Number of matrix rows.
 *  @param n (Output) Number of matrix columns.
 *  @param nnz (Output) Number of non-zero matrix entries.
 *  @return Zero if properties were read successfully. Otherwise
 *  non-zero.
 */
template <typename IndexType_>
int mm_properties(FILE* f, int tg, MM_typecode* t, IndexType_* m, IndexType_* n, IndexType_* nnz)
{
  // Read matrix properties from file
  int mint, nint, nnzint;
  if (fseek(f, 0, SEEK_SET)) {
    fprintf(stderr, "Error: could not set position in file\n");
    return -1;
  }
  if (mm_read_banner(f, t)) {
    fprintf(stderr, "Error: could not read Matrix Market file banner\n");
    return -1;
  }
  if (!mm_is_matrix(*t) || !mm_is_coordinate(*t)) {
    fprintf(stderr, "Error: file does not contain matrix in coordinate format\n");
    return -1;
  }
  if (mm_read_mtx_crd_size(f, &mint, &nint, &nnzint)) {
    fprintf(stderr, "Error: could not read matrix dimensions\n");
    return -1;
  }
  if (!mm_is_pattern(*t) && !mm_is_real(*t) && !mm_is_integer(*t) && !mm_is_complex(*t)) {
    fprintf(stderr, "Error: matrix entries are not valid type\n");
    return -1;
  }
  *m   = mint;
  *n   = nint;
  *nnz = nnzint;

  // Find total number of non-zero entries
  if (tg && !mm_is_general(*t)) {
    // Non-diagonal entries should be counted twice
    IndexType_ nnzOld = *nnz;
    *nnz *= 2;

    // Diagonal entries should not be double-counted
    int i;
    int st;
    for (i = 0; i < nnzOld; ++i) {
      // Read matrix entry
      IndexType_ row, col;
      double rval, ival;
      if (mm_is_pattern(*t))
        st = fscanf(f, "%d %d\n", &row, &col);
      else if (mm_is_real(*t) || mm_is_integer(*t))
        st = fscanf(f, "%d %d %lg\n", &row, &col, &rval);
      else  // Complex matrix
        st = fscanf(f, "%d %d %lg %lg\n", &row, &col, &rval, &ival);
      if (ferror(f) || (st == EOF)) {
        fprintf(stderr, "Error: error %d reading Matrix Market file (entry %d)\n", st, i + 1);
        return -1;
      }

      // Check if entry is diagonal
      if (row == col) --(*nnz);
    }
  }

  return 0;
}

/// Read Matrix Market file and convert to COO format matrix
/** Matrix Market file is assumed to be a sparse matrix in coordinate
 *  format.
 *
 *  @param f File stream for Matrix Market file.
 *  @param tg Boolean indicating whether to convert matrix to general
 *  format (from symmetric, Hermitian, or skew symmetric format).
 *  @param nnz Number of non-zero matrix entries.
 *  @param cooRowInd (Output) Row indices for COO matrix. Should have
 *  at least nnz entries.
 *  @param cooColInd (Output) Column indices for COO matrix. Should
 *  have at least nnz entries.
 *  @param cooRVal (Output) Real component of COO matrix
 *  entries. Should have at least nnz entries. Ignored if null
 *  pointer.
 *  @param cooIVal (Output) Imaginary component of COO matrix
 *  entries. Should have at least nnz entries. Ignored if null
 *  pointer.
 *  @return Zero if matrix was read successfully. Otherwise non-zero.
 */
template <typename IndexType_, typename ValueType_>
int mm_to_coo(FILE* f,
              int tg,
              IndexType_ nnz,
              IndexType_* cooRowInd,
              IndexType_* cooColInd,
              ValueType_* cooRVal,
              ValueType_* cooIVal)
{
  // Read matrix properties from file
  MM_typecode t;
  int m, n, nnzOld;
  if (fseek(f, 0, SEEK_SET)) {
    fprintf(stderr, "Error: could not set position in file\n");
    return -1;
  }
  if (mm_read_banner(f, &t)) {
    fprintf(stderr, "Error: could not read Matrix Market file banner\n");
    return -1;
  }
  if (!mm_is_matrix(t) || !mm_is_coordinate(t)) {
    fprintf(stderr, "Error: file does not contain matrix in coordinate format\n");
    return -1;
  }
  if (mm_read_mtx_crd_size(f, &m, &n, &nnzOld)) {
    fprintf(stderr, "Error: could not read matrix dimensions\n");
    return -1;
  }
  if (!mm_is_pattern(t) && !mm_is_real(t) && !mm_is_integer(t) && !mm_is_complex(t)) {
    fprintf(stderr, "Error: matrix entries are not valid type\n");
    return -1;
  }

  // Add each matrix entry in file to COO format matrix
  IndexType_ i;      // Entry index in Matrix Market file
  IndexType_ j = 0;  // Entry index in COO format matrix
  for (i = 0; i < nnzOld; ++i) {
    // Read entry from file
    int row, col;
    double rval, ival;
    int st;
    if (mm_is_pattern(t)) {
      st   = fscanf(f, "%d %d\n", &row, &col);
      rval = 1.0;
      ival = 0.0;
    } else if (mm_is_real(t) || mm_is_integer(t)) {
      st   = fscanf(f, "%d %d %lg\n", &row, &col, &rval);
      ival = 0.0;
    } else  // Complex matrix
      st = fscanf(f, "%d %d %lg %lg\n", &row, &col, &rval, &ival);
    if (ferror(f) || (st == EOF)) {
      fprintf(stderr, "Error: error %d reading Matrix Market file (entry %d)\n", st, i + 1);
      return -1;
    }

    // Switch to 0-based indexing
    --row;
    --col;

    // Record entry
    cooRowInd[j] = row;
    cooColInd[j] = col;
    if (cooRVal != NULL) cooRVal[j] = rval;
    if (cooIVal != NULL) cooIVal[j] = ival;
    ++j;

    // Add symmetric complement of non-diagonal entries
    if (tg && !mm_is_general(t) && (row != col)) {
      // Modify entry value if matrix is skew symmetric or Hermitian
      if (mm_is_skew(t)) {
        rval = -rval;
        ival = -ival;
      } else if (mm_is_hermitian(t)) {
        ival = -ival;
      }

      // Record entry
      cooRowInd[j] = col;
      cooColInd[j] = row;
      if (cooRVal != NULL) cooRVal[j] = rval;
      if (cooIVal != NULL) cooIVal[j] = ival;
      ++j;
    }
  }
  return 0;
}

int read_binary_vector(FILE* fpin, int n, std::vector<float>& val)
{
  size_t is_read1;

  double* t_storage = new double[n];
  is_read1          = fread(t_storage, sizeof(double), n, fpin);
  for (int i = 0; i < n; i++) {
    if (t_storage[i] == DBL_MAX)
      val[i] = FLT_MAX;
    else if (t_storage[i] == -DBL_MAX)
      val[i] = -FLT_MAX;
    else
      val[i] = static_cast<float>(t_storage[i]);
  }
  delete[] t_storage;

  if (is_read1 != (size_t)n) {
    printf("%s", "I/O fail\n");
    return 1;
  }
  return 0;
}

int read_binary_vector(FILE* fpin, int n, std::vector<double>& val)
{
  size_t is_read1;

  is_read1 = fread(&val[0], sizeof(double), n, fpin);

  if (is_read1 != (size_t)n) {
    printf("%s", "I/O fail\n");
    return 1;
  }
  return 0;
}

// FIXME: A similar function could be useful for CSC format
//        There are functions above that operate coo -> csr and coo->csc
/**
 * @tparam
 */
template <typename VT, typename ET, typename WT>
std::unique_ptr<cugraph::experimental::GraphCSR<VT, ET, WT>> generate_graph_csr_from_mm(
  bool& directed, std::string mm_file)
{
  VT number_of_vertices;
  ET number_of_edges;

  FILE* fpin = fopen(mm_file.c_str(), "r");
  EXPECT_NE(fpin, nullptr);

  VT number_of_columns = 0;
  MM_typecode mm_typecode{0};
  EXPECT_EQ(mm_properties<VT>(
              fpin, 1, &mm_typecode, &number_of_vertices, &number_of_columns, &number_of_edges),
            0);
  EXPECT_TRUE(mm_is_matrix(mm_typecode));
  EXPECT_TRUE(mm_is_coordinate(mm_typecode));
  EXPECT_FALSE(mm_is_complex(mm_typecode));
  EXPECT_FALSE(mm_is_skew(mm_typecode));

  directed = !mm_is_symmetric(mm_typecode);

  // Allocate memory on host
  std::vector<VT> coo_row_ind(number_of_edges);
  std::vector<VT> coo_col_ind(number_of_edges);
  std::vector<WT> coo_val(number_of_edges);

  // Read
  EXPECT_EQ((mm_to_coo<VT, WT>(
              fpin, 1, number_of_edges, &coo_row_ind[0], &coo_col_ind[0], &coo_val[0], NULL)),
            0);
  EXPECT_EQ(fclose(fpin), 0);

  cugraph::experimental::GraphCOOView<VT, ET, WT> cooview(
    &coo_row_ind[0], &coo_col_ind[0], &coo_val[0], number_of_vertices, number_of_edges);

  return cugraph::coo_to_csr(cooview);
}

////////////////////////////////////////////////////////////////////////////////
// FIXME: move this code to rapids-core
////////////////////////////////////////////////////////////////////////////////

// Define RAPIDS_DATASET_ROOT_DIR using a preprocessor variable to
// allow for a build to override the default. This is useful for
// having different builds for specific default dataset locations.
#ifndef RAPIDS_DATASET_ROOT_DIR
#define RAPIDS_DATASET_ROOT_DIR "/datasets"
#endif

static const std::string& get_rapids_dataset_root_dir()
{
  static std::string rdrd("");
  // Env var always overrides the value of RAPIDS_DATASET_ROOT_DIR
  if (rdrd == "") {
    const char* envVar = std::getenv("RAPIDS_DATASET_ROOT_DIR");
    rdrd               = (envVar != NULL) ? envVar : RAPIDS_DATASET_ROOT_DIR;
  }
  return rdrd;
}

}  // namespace test
}  // namespace cugraph
