/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "mg_test_utils.h"

#include <utilities/cxxopts.hpp>
#include <utilities/high_res_timer.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>

#include <c_api/graph.hpp>

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

#include <sstream>
#include <vector>

typedef int64_t vertex_t;
typedef int64_t edge_t;
typedef float weight_t;

constexpr double a        = 0.57;
constexpr double b        = 0.19;
constexpr double c        = 0.19;
data_type_id_t vertex_tid = INT64;
data_type_id_t edge_tid   = INT64;
data_type_id_t weight_tid = FLOAT32;
constexpr bool store_transposed{FALSE};
constexpr bool_t with_replacement{FALSE};

int main(int argc, char** argv)
{
  // Set up MPI:
  int comm_rank;
  int comm_size;
  int num_gpus_per_node;
  cudaError_t status;
  int mpi_status;

  cugraph_resource_handle_t* handle = NULL;
  cugraph_error_t* ret_error;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_sample_result_t* result = NULL;
  int prows{1};
  size_t scale{24};
  size_t edge_factor{16};
  uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
  bool clip_and_flip{false};
  size_t num_test_iterations{10};
  std::string fanout_str{"10,25"};
  vertex_t number_of_start_vertices{100};

  try {
    cxxopts::Options options(argv[0], " - uniform neighbor sample benchmark command line options");
    options.allow_unrecognised_options().add_options()(
      "rmm_mode", "RMM allocation mode", cxxopts::value<std::string>()->default_value("pool"))(
      "rmat_scale", "specify R-mat scale", cxxopts::value<size_t>()->default_value("24"))(
      "rmat_edge_factor",
      "specify R-mat edge factor",
      cxxopts::value<size_t>()->default_value("16"))(
      "num_iterations", "number of test iterations", cxxopts::value<size_t>()->default_value("10"))(
      "fan_out",
      "comma separate list of fanouts (e.g. 10,25)",
      cxxopts::value<std::string>()->default_value("10,25"))(
      "prows", "number of rows in GPU grid", cxxopts::value<int>()->default_value("1"))(
      "number_start_vertices",
      "number of starting vertices (batch size)",
      cxxopts::value<vertex_t>()->default_value("100"));

    auto cmd_opts = options.parse(argc, argv);

    scale                    = cmd_opts["rmat_scale"].as<size_t>();
    edge_factor              = cmd_opts["rmat_edge_factor"].as<size_t>();
    prows                    = cmd_opts["prows"].as<int>();
    num_test_iterations      = cmd_opts["num_iterations"].as<size_t>();
    fanout_str               = cmd_opts["fan_out"].as<std::string>();
    number_of_start_vertices = cmd_opts["number_start_vertices"].as<vertex_t>();

  } catch (const cxxopts::OptionException& e) {
    std::cerr << "Error parsing command line options: " << e.what() << std::endl;
    exit(1);
  }

  C_MPI_TRY(MPI_Init(&argc, &argv));
  C_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
  C_MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));
  C_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
  C_CUDA_TRY(cudaSetDevice(comm_rank % num_gpus_per_node));

  raft::handle_t* raft_handle = static_cast<raft::handle_t*>(create_raft_handle(prows));
  handle                      = cugraph_create_resource_handle(raft_handle);

#ifdef AFTER_3601_MERGES
  cugraph::test::enforce_p2p_initialization(raft_handle->get_comms(), raft_handle->get_stream());
  cugraph::test::enforce_p2p_initialization(
    raft_handle->get_subcomm(cugraph::partition_2d::key_naming_t().col_name()),
    raft_handle->get_stream());
#endif

  std::vector<int32_t> fan_out;

  std::stringstream fanout_stream(fanout_str);
  while (fanout_stream.good()) {
    std::string value;
    std::getline(fanout_stream, value, ',');
    fan_out.push_back(std::stoi(value));
  }

  cugraph::test::Rmat_Usecase rmat_usecase(
    scale, edge_factor, a, b, c, seed, false, false, 0, clip_and_flip);

  auto [src, dst, wgt, vertex, undirected] =
    rmat_usecase.construct_edgelist<vertex_t, weight_t>(*raft_handle, true, store_transposed, true);

  int test_ret_value = 0;

  cugraph_graph_t* graph = NULL;

  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  cugraph_graph_properties_t properties;

  properties.is_symmetric  = undirected ? FALSE : TRUE;
  properties.is_multigraph = TRUE;

  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  src_view = cugraph_type_erased_device_array_view_create(src.data(), src.size(), vertex_tid);
  dst_view = cugraph_type_erased_device_array_view_create(dst.data(), dst.size(), vertex_tid);
  wgt_view = cugraph_type_erased_device_array_view_create(wgt->data(), wgt->size(), weight_tid);

  ret_code = cugraph_mg_graph_create(handle,
                                     &properties,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,  // need this...
                                     NULL,  // need this...
                                     store_transposed ? TRUE : FALSE,
                                     src.size(),
                                     FALSE,
                                     &graph,
                                     &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);

  auto graph_view = reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, store_transposed, true>*>(
                      reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->graph_)
                      ->view();
  auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(
    reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->number_map_);

  h_fan_out_view =
    cugraph_type_erased_host_array_view_create(fan_out.data(), fan_out.size(), INT32);

  HighResTimer timer;

  for (size_t i = 0; i < num_test_iterations; ++i) {
    auto start_vertices = cugraph::test::random_ext_vertex_ids<vertex_t, true>(
      *raft_handle, *number_map, number_of_start_vertices, seed);

    d_start_view = cugraph_type_erased_device_array_view_create(
      start_vertices.data(), start_vertices.size(), vertex_tid);

#ifdef AFTER_3601_MERGES
    if (comm_rank == 0) timer.start("cugraph_uniform_neighbor_sample");
#else
    if ((i > 0) && (comm_rank == 0)) timer.start("cugraph_uniform_neighbor_sample");
#endif

    ret_code = cugraph_uniform_neighbor_sample(
      handle, graph, d_start_view, h_fan_out_view, with_replacement, FALSE, &result, &ret_error);

#ifdef AFTER_3601_MERGES
    if (comm_rank == 0) timer.stop();
#else
    if ((i > 0) && (comm_rank == 0)) timer.stop();
#endif

    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

    cugraph_sample_result_free(result);
    cugraph_type_erased_device_array_view_free(d_start_view);
  }

  if (comm_rank == 0) timer.display(std::cout);

  cugraph_type_erased_host_array_view_free(h_fan_out_view);
  cugraph_mg_graph_free(graph);
  cugraph_error_free(ret_error);

  free_raft_handle(raft_handle);

  C_MPI_TRY(MPI_Finalize());

  return ret_code;
}
