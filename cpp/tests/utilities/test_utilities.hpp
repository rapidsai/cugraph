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

#include <numeric>
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

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>>
generate_graph_from_edgelist(raft::handle_t const& handle,
                             rmm::device_uvector<vertex_t>&& vertices,
                             rmm::device_uvector<vertex_t>&& edgelist_rows,
                             rmm::device_uvector<vertex_t>&& edgelist_cols,
                             rmm::device_uvector<weight_t>&& edgelist_weights,
                             bool is_symmetric,
                             bool test_weighted,
                             bool renumber);

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

class File_Usecase {
 public:
  File_Usecase() = delete;

  File_Usecase(std::string const& graph_file_path)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path_ = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path_ = graph_file_path;
    }
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const
  {
    return read_graph_from_matrix_market_file<vertex_t,
                                              edge_t,
                                              weight_t,
                                              store_transposed,
                                              multi_gpu>(
      handle, graph_file_full_path_, test_weighted, renumber);
  }

 private:
  std::string graph_file_full_path_;
};

class Rmat_Usecase {
 public:
  Rmat_Usecase() = delete;

  Rmat_Usecase(size_t scale,
               size_t edge_factor,
               double a,
               double b,
               double c,
               uint64_t seed,
               bool undirected,
               bool scramble_vertex_ids,
               bool multi_gpu_usecase = false)
    : scale_(scale),
      edge_factor_(edge_factor),
      a_(a),
      b_(b),
      c_(c),
      seed_(seed),
      undirected_(undirected),
      scramble_vertex_ids_(scramble_vertex_ids),
      multi_gpu_usecase_(multi_gpu_usecase)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>
  construct_graph(raft::handle_t const& handle, bool test_weighted, bool renumber = true) const
  {
    std::vector<size_t> partition_ids(1);
    size_t comm_size;

    if (multi_gpu_usecase_) {
      auto& comm           = handle.get_comms();
      comm_size = comm.get_size();
      auto const comm_rank = comm.get_rank();

      partition_ids.resize(multi_gpu ? size_t{1} : static_cast<size_t>(comm_size));

      std::iota(partition_ids.begin(),
                partition_ids.end(),
                multi_gpu ? static_cast<size_t>(comm_rank) : size_t{0});
    } else {
      comm_size = 1;
      partition_ids[0] = size_t{0};
    }

    return generate_graph_from_rmat_params<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      scale_,
      edge_factor_,
      a_,
      b_,
      c_,
      seed_,
      undirected_,
      scramble_vertex_ids_,
      test_weighted,
      renumber,
      partition_ids,
      comm_size);
  }

 private:
  size_t scale_;
  size_t edge_factor_;
  double a_;
  double b_;
  double c_;
  uint64_t seed_;
  bool undirected_;
  bool scramble_vertex_ids_;
  bool multi_gpu_usecase_;
};

}  // namespace test
}  // namespace cugraph
