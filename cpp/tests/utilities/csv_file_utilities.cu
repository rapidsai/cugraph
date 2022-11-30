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

#include <utilities/test_utilities.hpp>

#include <detail/graph_utils.cuh>

#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

#include <cstdint>
#include <fstream>

namespace cugraph {
namespace test {
namespace detail {

template <typename vertex_t, typename weight_t>
bool check_symmetric(raft::handle_t const& handle,
                     raft::device_span<vertex_t const> edgelist_srcs,
                     raft::device_span<vertex_t const> edgelist_dsts,
                     std::optional<raft::device_span<weight_t const>> edgelist_weights)
{
  rmm::device_uvector<vertex_t> org_srcs(edgelist_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> org_dsts(edgelist_dsts.size(), handle.get_stream());
  auto org_weights = edgelist_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                          (*edgelist_weights).size(), handle.get_stream())
                                      : std::nullopt;
  rmm::device_uvector<vertex_t> symmetrized_srcs(edgelist_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> symmetrized_dsts(edgelist_dsts.size(), handle.get_stream());
  auto symmetrized_weights = edgelist_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                                  (*edgelist_weights).size(), handle.get_stream())
                                              : std::nullopt;

  thrust::copy(
    handle.get_thrust_policy(), edgelist_srcs.begin(), edgelist_srcs.end(), org_srcs.begin());
  thrust::copy(handle.get_thrust_policy(),
               edgelist_srcs.begin(),
               edgelist_srcs.end(),
               symmetrized_srcs.begin());
  thrust::copy(
    handle.get_thrust_policy(), edgelist_dsts.begin(), edgelist_dsts.end(), org_dsts.begin());
  thrust::copy(handle.get_thrust_policy(),
               edgelist_dsts.begin(),
               edgelist_dsts.end(),
               symmetrized_dsts.begin());
  if (edgelist_weights) {
    thrust::copy(handle.get_thrust_policy(),
                 (*edgelist_weights).begin(),
                 (*edgelist_weights).end(),
                 (*org_weights).begin());
    thrust::copy(handle.get_thrust_policy(),
                 (*edgelist_weights).begin(),
                 (*edgelist_weights).end(),
                 (*symmetrized_weights).begin());
  }

  std::tie(symmetrized_srcs, symmetrized_dsts, symmetrized_weights) =
    symmetrize_edgelist<vertex_t, weight_t, false, false>(handle,
                                                          std::move(symmetrized_srcs),
                                                          std::move(symmetrized_dsts),
                                                          std::move(symmetrized_weights),
                                                          true);

  if (edgelist_weights) {
    auto org_first = thrust::make_zip_iterator(
      thrust::make_tuple(org_srcs.begin(), org_dsts.begin(), (*org_weights).begin()));
    thrust::sort(handle.get_thrust_policy(), org_first, org_first + org_srcs.size());
    auto symmetrized_first = thrust::make_zip_iterator(thrust::make_tuple(
      symmetrized_srcs.begin(), symmetrized_dsts.begin(), (*symmetrized_weights).begin()));
    thrust::sort(
      handle.get_thrust_policy(), symmetrized_first, symmetrized_first + symmetrized_srcs.size());
    return thrust::equal(
      handle.get_thrust_policy(), org_first, org_first + org_srcs.size(), symmetrized_first);
  } else {
    auto org_first =
      thrust::make_zip_iterator(thrust::make_tuple(org_srcs.begin(), org_dsts.begin()));
    thrust::sort(handle.get_thrust_policy(), org_first, org_first + org_srcs.size());
    auto symmetrized_first = thrust::make_zip_iterator(
      thrust::make_tuple(symmetrized_srcs.begin(), symmetrized_dsts.begin()));
    thrust::sort(
      handle.get_thrust_policy(), symmetrized_first, symmetrized_first + symmetrized_srcs.size());
    return thrust::equal(
      handle.get_thrust_policy(), org_first, org_first + org_srcs.size(), symmetrized_first);
  }
}

}  // namespace detail

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           bool>
read_edgelist_from_csv_file(raft::handle_t const& handle,
                            std::string const& graph_file_full_path,
                            bool test_weighted,
                            bool store_transposed,
                            bool multi_gpu)
{
  std::ifstream file(graph_file_full_path);
  CUGRAPH_EXPECTS(file.is_open(), "File open (%s) failure.", graph_file_full_path.c_str());

  file.seekg(0, file.end);
  auto length = file.tellg();
  file.seekg(0, file.beg);

  std::vector<char> buffer(length + 1);

  file.read(buffer.data(), length);
  CUGRAPH_EXPECTS(file, "File read failure.");

  buffer.back() = '\0';  // null termination

  file.close();

  char const* delimiters = ", \t\n";

  std::vector<vertex_t> h_edgelist_srcs{};
  std::vector<vertex_t> h_edgelist_dsts{};
  std::vector<weight_t> h_edgelist_weights{};

  char const* cur = buffer.data();
  size_t num_tokens_this_line{0};
  while (cur) {
    char const* prev = cur;
    cur              = strpbrk(prev, delimiters);
    if (cur) {
      auto token = std::string(prev, cur);
      if (num_tokens_this_line == 0) {  // source
        auto src = stoll(token);
        CUGRAPH_EXPECTS((src >= std::numeric_limits<vertex_t>::lowest()) &&
                          (src <= std::numeric_limits<vertex_t>::max()),
                        "vertex_t overflow.");
        h_edgelist_srcs.push_back(static_cast<vertex_t>(src));
      } else if (num_tokens_this_line == 1) {  // destination
        auto dst = stoll(token);
        CUGRAPH_EXPECTS((dst >= std::numeric_limits<vertex_t>::lowest()) &&
                          (dst <= std::numeric_limits<vertex_t>::max()),
                        "vertex_t overflow.");
        h_edgelist_dsts.push_back(static_cast<vertex_t>(dst));
      } else if (num_tokens_this_line == 2) {  // weight
        auto w = stod(token);
        h_edgelist_weights.push_back(static_cast<weight_t>(w));
      } else {
        CUGRAPH_FAIL("Too many tokens in a line.");
      }
      ++num_tokens_this_line;
      auto num_delimiters = std::strspn(cur, delimiters);
      for (size_t i = 0; i < num_delimiters; ++i) {
        if (*cur == '\n') { num_tokens_this_line = 0; }
      }
      cur += num_delimiters;
    }
  }

  CUGRAPH_EXPECTS(h_edgelist_srcs.size() == h_edgelist_dsts.size(),
                  "Invalid input file contents (# source IDs != # destination IDs).");

  CUGRAPH_EXPECTS(
    (h_edgelist_weights.size() == 0) || (h_edgelist_srcs.size() == h_edgelist_weights.size()),
    "Invalid input file contents (# source IDs != # weights).");
  CUGRAPH_EXPECTS(!test_weighted || (h_edgelist_weights.size() > 0),
                  "test_weighted set but weights are not provided.");

  rmm::device_uvector<vertex_t> d_edgelist_srcs(h_edgelist_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_edgelist_dsts(h_edgelist_dsts.size(), handle.get_stream());
  auto d_edgelist_weights = test_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(
                                              h_edgelist_weights.size(), handle.get_stream())
                                          : std::nullopt;

  raft::update_device(
    d_edgelist_srcs.data(), h_edgelist_srcs.data(), h_edgelist_srcs.size(), handle.get_stream());
  raft::update_device(
    d_edgelist_dsts.data(), h_edgelist_dsts.data(), h_edgelist_dsts.size(), handle.get_stream());
  if (d_edgelist_weights) {
    raft::update_device((*d_edgelist_weights).data(),
                        h_edgelist_weights.data(),
                        h_edgelist_weights.size(),
                        handle.get_stream());
  }

  bool is_symmetric = detail::check_symmetric(
    handle,
    raft::device_span<vertex_t const>(d_edgelist_srcs.data(), d_edgelist_srcs.size()),
    raft::device_span<vertex_t const>(d_edgelist_dsts.data(), d_edgelist_dsts.size()),
    d_edgelist_weights ? std::make_optional<raft::device_span<weight_t const>>(
                           (*d_edgelist_weights).data(), (*d_edgelist_weights).size())
                       : std::nullopt);

  if (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();

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
read_graph_from_csv_file(raft::handle_t const& handle,
                         std::string const& graph_file_full_path,
                         bool test_weighted,
                         bool renumber)
{
  auto [d_edgelist_srcs, d_edgelist_dsts, d_edgelist_weights, is_symmetric] =
    read_edgelist_from_csv_file<vertex_t, weight_t>(
      handle, graph_file_full_path, test_weighted, store_transposed, multi_gpu);

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph(handle);
  std::optional<
    cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::tie(graph, edge_weights, std::ignore, renumber_map) = cugraph::
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::nullopt,
      std::move(d_edgelist_srcs),
      std::move(d_edgelist_dsts),
      std::move(d_edgelist_weights),
      std::nullopt,
      cugraph::graph_properties_t{is_symmetric, false},
      renumber);

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
}

// explicit instantiations

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    bool>
read_edgelist_from_csv_file<int32_t, float>(raft::handle_t const& handle,
                                            std::string const& graph_file_full_path,
                                            bool test_weighted,
                                            bool store_transposed,
                                            bool multi_gpu);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, false, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, true, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, false, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, true, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int32_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, false, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, true, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, false, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, true, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int32_t, int64_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
read_graph_from_csv_file<int32_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, false, false>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_csv_file<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_csv_file<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, true, false>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_csv_file<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_csv_file<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, false, false>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_csv_file<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_csv_file<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, false>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, true, false>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_csv_file<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, true>,
  std::optional<cugraph::edge_property_t<graph_view_t<int64_t, int64_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
read_graph_from_csv_file<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  std::string const& graph_file_full_path,
  bool test_weighted,
  bool renumber);

}  // namespace test
}  // namespace cugraph
