/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <detail/graph_utils.cuh>
#include <utilities/test_utilities.hpp>

#include <cugraph/graph_functions.hpp>
#include <cugraph/legacy/functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/cudart_utils.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <cstdint>

namespace cugraph {
namespace test {

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
    *nnz *= 2;

    // Diagonal entries should not be double-counted
    int st;
    for (int i = 0; i < nnzint; ++i) {
      // Read matrix entry
      // MTX only supports int for row and col idx
      int row, col;
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
  int i;      // Entry index in Matrix Market file; can only be int in the MTX format
  int j = 0;  // Entry index in COO format matrix; can only be int in the MTX format
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

// FIXME: A similar function could be useful for CSC format
//        There are functions above that operate coo -> csr and coo->csc
/**
 * @tparam
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<cugraph::legacy::GraphCSR<vertex_t, edge_t, weight_t>> generate_graph_csr_from_mm(
  bool& directed, std::string mm_file)
{
  vertex_t number_of_vertices;
  edge_t number_of_edges;

  FILE* fpin = fopen(mm_file.c_str(), "r");
  CUGRAPH_EXPECTS(fpin != nullptr, "fopen (%s) failure.", mm_file.c_str());

  vertex_t number_of_columns = 0;
  MM_typecode mm_typecode{0};
  CUGRAPH_EXPECTS(
    mm_properties<vertex_t>(
      fpin, 1, &mm_typecode, &number_of_vertices, &number_of_columns, &number_of_edges) == 0,
    "mm_properties query failure.");
  CUGRAPH_EXPECTS(mm_is_matrix(mm_typecode), "Invalid input file.");
  CUGRAPH_EXPECTS(mm_is_coordinate(mm_typecode), "Invalid input file.");
  CUGRAPH_EXPECTS(!mm_is_complex(mm_typecode), "Invalid input file.");
  CUGRAPH_EXPECTS(!mm_is_skew(mm_typecode), "Invalid input file.");

  directed = !mm_is_symmetric(mm_typecode);

  // Allocate memory on host
  std::vector<vertex_t> coo_row_ind(number_of_edges);
  std::vector<vertex_t> coo_col_ind(number_of_edges);
  std::vector<weight_t> coo_val(number_of_edges);

  // Read
  CUGRAPH_EXPECTS(
    (mm_to_coo<vertex_t, weight_t>(
      fpin, 1, number_of_edges, &coo_row_ind[0], &coo_col_ind[0], &coo_val[0], NULL)) == 0,
    "file read failure.");
  CUGRAPH_EXPECTS(fclose(fpin) == 0, "fclose failure.");

  cugraph::legacy::GraphCOOView<vertex_t, edge_t, weight_t> cooview(
    &coo_row_ind[0], &coo_col_ind[0], &coo_val[0], number_of_vertices, number_of_edges);

  return cugraph::coo_to_csr(cooview);
}

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<vertex_t>,
           bool>
read_edgelist_from_matrix_market_file(raft::handle_t const& handle,
                                      std::string const& graph_file_full_path,
                                      bool test_weighted,
                                      bool store_transposed,
                                      bool multi_gpu)
{
  MM_typecode mc{};
  vertex_t m{};
  size_t nnz{};

  FILE* file = fopen(graph_file_full_path.c_str(), "r");
  CUGRAPH_EXPECTS(file != nullptr, "fopen (%s) failure.", graph_file_full_path.c_str());

  size_t tmp_m{};
  size_t tmp_k{};
  auto mm_ret = cugraph::test::mm_properties<size_t>(file, 1, &mc, &tmp_m, &tmp_k, &nnz);
  CUGRAPH_EXPECTS(mm_ret == 0, "could not read Matrix Market file properties.");
  m = static_cast<vertex_t>(tmp_m);
  CUGRAPH_EXPECTS(mm_is_matrix(mc) && mm_is_coordinate(mc) && !mm_is_complex(mc) && !mm_is_skew(mc),
                  "invalid Matrix Market file properties.");

  vertex_t number_of_vertices = m;
  bool is_symmetric           = mm_is_symmetric(mc);

  std::vector<vertex_t> h_rows(nnz);
  std::vector<vertex_t> h_cols(nnz);
  std::vector<weight_t> h_weights(nnz);

  mm_ret = cugraph::test::mm_to_coo<vertex_t, weight_t>(
    file, 1, nnz, h_rows.data(), h_cols.data(), h_weights.data(), static_cast<weight_t*>(nullptr));
  CUGRAPH_EXPECTS(mm_ret == 0, "could not read matrix data");

  auto file_ret = fclose(file);
  CUGRAPH_EXPECTS(file_ret == 0, "fclose failure.");

  rmm::device_uvector<vertex_t> d_edgelist_srcs(h_rows.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_edgelist_dsts(h_cols.size(), handle.get_stream());
  auto d_edgelist_weights = test_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(
                                              h_weights.size(), handle.get_stream())
                                          : std::nullopt;

  rmm::device_uvector<vertex_t> d_vertices(number_of_vertices, handle.get_stream());

  raft::update_device(d_edgelist_srcs.data(), h_rows.data(), h_rows.size(), handle.get_stream());
  raft::update_device(d_edgelist_dsts.data(), h_cols.data(), h_cols.size(), handle.get_stream());
  if (d_edgelist_weights) {
    raft::update_device(
      (*d_edgelist_weights).data(), h_weights.data(), h_weights.size(), handle.get_stream());
  }

  thrust::sequence(handle.get_thrust_policy(), d_vertices.begin(), d_vertices.end(), vertex_t{0});

  if (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();

    auto vertex_key_func = cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size};
    d_vertices.resize(
      thrust::distance(d_vertices.begin(),
                       thrust::remove_if(handle.get_thrust_policy(),
                                         d_vertices.begin(),
                                         d_vertices.end(),
                                         [comm_rank, key_func = vertex_key_func] __device__(
                                           auto val) { return key_func(val) != comm_rank; })),
      handle.get_stream());
    d_vertices.shrink_to_fit(handle.get_stream());

    auto edge_key_func = cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
      comm_size, row_comm_size, col_comm_size};
    size_t number_of_local_edges{};
    if (d_edgelist_weights) {
      auto edge_first       = thrust::make_zip_iterator(thrust::make_tuple(
        d_edgelist_srcs.begin(), d_edgelist_dsts.begin(), (*d_edgelist_weights).begin()));
      number_of_local_edges = thrust::distance(
        edge_first,
        thrust::remove_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + d_edgelist_srcs.size(),
          [store_transposed, comm_rank, key_func = edge_key_func] __device__(auto e) {
            auto major = thrust::get<0>(e);
            auto minor = thrust::get<1>(e);
            return store_transposed ? key_func(minor, major) != comm_rank
                                    : key_func(major, minor) != comm_rank;
          }));
    } else {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(d_edgelist_srcs.begin(), d_edgelist_dsts.begin()));
      number_of_local_edges = thrust::distance(
        edge_first,
        thrust::remove_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + d_edgelist_srcs.size(),
          [store_transposed, comm_rank, key_func = edge_key_func] __device__(auto e) {
            auto major = thrust::get<0>(e);
            auto minor = thrust::get<1>(e);
            return store_transposed ? key_func(minor, major) != comm_rank
                                    : key_func(major, minor) != comm_rank;
          }));
    }

    d_edgelist_srcs.resize(number_of_local_edges, handle.get_stream());
    d_edgelist_srcs.shrink_to_fit(handle.get_stream());
    d_edgelist_dsts.resize(number_of_local_edges, handle.get_stream());
    d_edgelist_dsts.shrink_to_fit(handle.get_stream());
    if (d_edgelist_weights) {
      (*d_edgelist_weights).resize(number_of_local_edges, handle.get_stream());
      (*d_edgelist_weights).shrink_to_fit(handle.get_stream());
    }
  }

  return std::make_tuple(std::move(d_edgelist_srcs),
                         std::move(d_edgelist_dsts),
                         std::move(d_edgelist_weights),
                         std::move(d_vertices),
                         is_symmetric);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           std::optional<
             cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                      weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
read_graph_from_matrix_market_file(raft::handle_t const& handle,
                                   std::string const& graph_file_full_path,
                                   bool test_weighted,
                                   bool renumber)
{
  auto [d_edgelist_srcs, d_edgelist_dsts, d_edgelist_weights, d_vertices, is_symmetric] =
    read_edgelist_from_matrix_market_file<vertex_t, weight_t>(
      handle, graph_file_full_path, test_weighted, store_transposed, multi_gpu);

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph(handle);
  std::optional<
    cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::tie(graph, edge_weights, std::ignore, renumber_map) = cugraph::
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(d_vertices),
      std::move(d_edgelist_srcs),
      std::move(d_edgelist_dsts),
      std::move(d_edgelist_weights),
      std::nullopt,
      cugraph::graph_properties_t{is_symmetric, false},
      renumber);

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
}

// explicit instantiations

template int32_t mm_to_coo(FILE* f,
                           int32_t tg,
                           int32_t nnz,
                           int32_t* cooRowInd,
                           int32_t* cooColInd,
                           int32_t* cooRVal,
                           int32_t* cooIVal);

template int32_t mm_to_coo(FILE* f,
                           int32_t tg,
                           int32_t nnz,
                           int32_t* cooRowInd,
                           int32_t* cooColInd,
                           double* cooRVal,
                           double* cooIVal);

template int32_t mm_to_coo(FILE* f,
                           int32_t tg,
                           int32_t nnz,
                           int32_t* cooRowInd,
                           int32_t* cooColInd,
                           float* cooRVal,
                           float* cooIVal);

template std::unique_ptr<cugraph::legacy::GraphCSR<int32_t, int32_t, float>>
generate_graph_csr_from_mm(bool& directed, std::string mm_file);

template std::unique_ptr<cugraph::legacy::GraphCSR<uint32_t, uint32_t, float>>
generate_graph_csr_from_mm(bool& directed, std::string mm_file);

template std::unique_ptr<cugraph::legacy::GraphCSR<int32_t, int32_t, double>>
generate_graph_csr_from_mm(bool& directed, std::string mm_file);

template std::unique_ptr<cugraph::legacy::GraphCSR<int64_t, int64_t, float>>
generate_graph_csr_from_mm(bool& directed, std::string mm_file);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<int32_t>,
                    bool>
read_edgelist_from_matrix_market_file<int32_t, float>(raft::handle_t const& handle,
                                                      std::string const& graph_file_full_path,
                                                      bool test_weighted,
                                                      bool store_transposed,
                                                      bool multi_gpu);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, false, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, true, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, false, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, true, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, false, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, true, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, false, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, true, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_matrix_market_file<int32_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, false, false>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, true, false>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, false, false>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, true, false>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_matrix_market_file<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

}  // namespace test
}  // namespace cugraph
