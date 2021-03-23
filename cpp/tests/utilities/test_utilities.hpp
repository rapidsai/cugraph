/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <experimental/graph.hpp>
#include <graph.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <string>
#include <vector>

extern "C" {
#include "mmio.h"
}

namespace cugraph {
namespace test {

std::string getFileName(const std::string& s);

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
int mm_properties(FILE* f, int tg, MM_typecode* t, IndexType_* m, IndexType_* n, IndexType_* nnz);

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
              ValueType_* cooIVal);

// FIXME: A similar function could be useful for CSC format
//        There are functions above that operate coo -> csr and coo->csc
/**
 * @tparam
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<cugraph::GraphCSR<vertex_t, edge_t, weight_t>> generate_graph_csr_from_mm(
  bool& directed, std::string mm_file);

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

// returns a tuple of (rows, columns, weights, number_of_vertices, is_symmetric)
template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           vertex_t,
           bool>
read_edgelist_from_matrix_market_file(raft::handle_t const& handle,
                                      std::string const& graph_file_full_path,
                                      bool test_weighted);

// renumber must be true if multi_gpu is true
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>>
read_graph_from_matrix_market_file(raft::handle_t const& handle,
                                   std::string const& graph_file_full_path,
                                   bool test_weighted,
                                   bool renumber);

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>>
generate_graph_from_rmat_params(raft::handle_t const& handle,
                                size_t scale,
                                size_t edge_factor,
                                double a,
                                double b,
                                double c,
                                uint64_t seed,
                                bool undirected,
                                bool scramble_vertex_ids,
                                bool test_weighted,
                                bool renumber,
                                std::vector<size_t> const& partition_ids,
                                size_t num_partitions);

struct rmat_params_t {
  size_t scale{};
  size_t edge_factor{};
  double a{};
  double b{};
  double c{};
  uint64_t seed{};
  bool undirected{};
  bool scramble_vertex_ids{};
};

struct input_graph_specifier_t {
  enum { MATRIX_MARKET_FILE_PATH, RMAT_PARAMS } tag{};
  std::string graph_file_full_path{};
  rmat_params_t rmat_params{};
};

}  // namespace test
}  // namespace cugraph
