/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <utilities/test_utilities.hpp>

#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph_functions.hpp>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/remove.h>

#include <cstdint>

namespace cugraph {
namespace test {

namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>>
generate_graph_from_edgelist(raft::handle_t const& handle,
                             rmm::device_uvector<vertex_t>&& vertices,
                             rmm::device_uvector<vertex_t>&& edgelist_rows,
                             rmm::device_uvector<vertex_t>&& edgelist_cols,
                             rmm::device_uvector<weight_t>&& edgelist_weights,
                             bool is_symmetric,
                             bool test_weighted,
                             bool renumber)
{
  CUGRAPH_EXPECTS(renumber, "renumber should be true if multi_gpu is true.");

  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto const comm_rank     = comm.get_rank();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  vertex_t number_of_vertices = static_cast<vertex_t>(vertices.size());

  auto vertex_key_func =
    cugraph::experimental::detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size};
  vertices.resize(thrust::distance(vertices.begin(),
                                   thrust::remove_if(
                                     rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                     vertices.begin(),
                                     vertices.end(),
                                     [comm_rank, key_func = vertex_key_func] __device__(auto val) {
                                       return key_func(val) != comm_rank;
                                     })),
                  handle.get_stream());
  vertices.shrink_to_fit(handle.get_stream());

  auto edge_key_func = cugraph::experimental::detail::compute_gpu_id_from_edge_t<vertex_t>{
    false, comm_size, row_comm_size, col_comm_size};
  size_t number_of_local_edges{};
  if (test_weighted) {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_rows.begin(), edgelist_cols.begin(), edgelist_weights.begin()));
    number_of_local_edges = thrust::distance(
      edge_first,
      thrust::remove_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        edge_first,
                        edge_first + edgelist_rows.size(),
                        [comm_rank, key_func = edge_key_func] __device__(auto e) {
                          auto major = store_transposed ? thrust::get<1>(e) : thrust::get<0>(e);
                          auto minor = store_transposed ? thrust::get<0>(e) : thrust::get<1>(e);
                          return key_func(major, minor) != comm_rank;
                        }));
  } else {
    auto edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(edgelist_rows.begin(), edgelist_cols.begin()));
    number_of_local_edges = thrust::distance(
      edge_first,
      thrust::remove_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        edge_first,
                        edge_first + edgelist_rows.size(),
                        [comm_rank, key_func = edge_key_func] __device__(auto e) {
                          auto major = store_transposed ? thrust::get<1>(e) : thrust::get<0>(e);
                          auto minor = store_transposed ? thrust::get<0>(e) : thrust::get<1>(e);
                          return key_func(major, minor) != comm_rank;
                        }));
  }

  edgelist_rows.resize(number_of_local_edges, handle.get_stream());
  edgelist_rows.shrink_to_fit(handle.get_stream());
  edgelist_cols.resize(number_of_local_edges, handle.get_stream());
  edgelist_cols.shrink_to_fit(handle.get_stream());
  if (test_weighted) {
    edgelist_weights.resize(number_of_local_edges, handle.get_stream());
    edgelist_weights.shrink_to_fit(handle.get_stream());
  }

  // 3. renumber

  rmm::device_uvector<vertex_t> renumber_map_labels(0, handle.get_stream());
  cugraph::experimental::partition_t<vertex_t> partition{};
  vertex_t aggregate_number_of_vertices{};
  edge_t number_of_edges{};
  // FIXME: set do_expensive_check to false once validated
  std::tie(renumber_map_labels, partition, aggregate_number_of_vertices, number_of_edges) =
    cugraph::experimental::renumber_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      vertices.data(),
      static_cast<vertex_t>(vertices.size()),
      store_transposed ? edgelist_cols.data() : edgelist_rows.data(),
      store_transposed ? edgelist_rows.data() : edgelist_cols.data(),
      edgelist_rows.size(),
      false,
      true);
  assert(aggregate_number_of_vertices == number_of_vertices);

  // 4. create a graph

  return std::make_tuple(
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::vector<cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t>>{
        cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t>{
          edgelist_rows.data(),
          edgelist_cols.data(),
          test_weighted ? edgelist_weights.data() : nullptr,
          static_cast<edge_t>(edgelist_rows.size())}},
      partition,
      number_of_vertices,
      number_of_edges,
      cugraph::experimental::graph_properties_t{is_symmetric, false},
      true,
      true),
    std::move(renumber_map_labels));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
    rmm::device_uvector<vertex_t>>>
generate_graph_from_edgelist(raft::handle_t const& handle,
                             rmm::device_uvector<vertex_t>&& vertices,
                             rmm::device_uvector<vertex_t>&& edgelist_rows,
                             rmm::device_uvector<vertex_t>&& edgelist_cols,
                             rmm::device_uvector<weight_t>&& edgelist_weights,
                             bool is_symmetric,
                             bool test_weighted,
                             bool renumber)
{
  vertex_t number_of_vertices = static_cast<vertex_t>(vertices.size());

  // FIXME: set do_expensive_check to false once validated
  auto renumber_map_labels =
    renumber ? cugraph::experimental::renumber_edgelist<vertex_t, edge_t, multi_gpu>(
                 handle,
                 vertices.data(),
                 static_cast<vertex_t>(vertices.size()),
                 store_transposed ? edgelist_cols.data() : edgelist_rows.data(),
                 store_transposed ? edgelist_rows.data() : edgelist_cols.data(),
                 static_cast<edge_t>(edgelist_rows.size()),
                 true)
             : rmm::device_uvector<vertex_t>(0, handle.get_stream());

  // FIXME: set do_expensive_check to false once validated
  return std::make_tuple(
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t>{
        edgelist_rows.data(),
        edgelist_cols.data(),
        test_weighted ? edgelist_weights.data() : nullptr,
        static_cast<edge_t>(edgelist_rows.size())},
      number_of_vertices,
      cugraph::experimental::graph_properties_t{is_symmetric, false},
      renumber ? true : false,
      true),
    std::move(renumber_map_labels));
}

}  // namespace detail

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
                             bool renumber)
{
  return detail::
    generate_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(vertices),
      std::move(edgelist_rows),
      std::move(edgelist_cols),
      std::move(edgelist_weights),
      is_symmetric,
      test_weighted,
      renumber);
}

// explicit instantiations

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, false, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, false, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, true, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, float, true, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, false, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, false, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, true, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int32_t, double, true, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, false, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, false, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, true, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, float, true, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, false, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, false, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, true, false>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int32_t, int64_t, double, true, true>,
                    rmm::device_uvector<int32_t>>
generate_graph_from_edgelist<int32_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_rows,
  rmm::device_uvector<int32_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, false, false>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_edgelist<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_rows,
  rmm::device_uvector<int64_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, false, true>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_edgelist<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_rows,
  rmm::device_uvector<int64_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, true, false>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_edgelist<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_rows,
  rmm::device_uvector<int64_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, float, true, true>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_edgelist<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_rows,
  rmm::device_uvector<int64_t>&& edgelist_cols,
  rmm::device_uvector<float>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, false, false>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_edgelist<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_rows,
  rmm::device_uvector<int64_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, false, true>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_edgelist<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_rows,
  rmm::device_uvector<int64_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, true, false>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_edgelist<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_rows,
  rmm::device_uvector<int64_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

template std::tuple<cugraph::experimental::graph_t<int64_t, int64_t, double, true, true>,
                    rmm::device_uvector<int64_t>>
generate_graph_from_edgelist<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_rows,
  rmm::device_uvector<int64_t>&& edgelist_cols,
  rmm::device_uvector<double>&& edgelist_weights,
  bool is_symmetric,
  bool test_weighted,
  bool renumber);

}  // namespace test
}  // namespace cugraph
