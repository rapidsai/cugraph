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

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>

#include <gtest/gtest.h>

void compare(float modularity, float sg_modularity) { ASSERT_FLOAT_EQ(modularity, sg_modularity); }
void compare(double modularity, double sg_modularity)
{
  ASSERT_DOUBLE_EQ(modularity, sg_modularity);
}

////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_CASE_P()
//
struct Louvain_Usecase {
  std::string graph_file_full_path{};
  bool weighted{false};
  size_t max_level;
  double resolution;

  // FIXME:  We really should have a Graph_Testparms_Base class or something
  //         like that which can handle this graph_full_path thing.
  //
  Louvain_Usecase(std::string const& graph_file_path,
                  bool weighted,
                  size_t max_level,
                  double resolution)
    : weighted(weighted), max_level(max_level), resolution(resolution)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
};

////////////////////////////////////////////////////////////////////////////////
// Parameterized test fixture, to be used with TEST_P().  This defines common
// setup and teardown steps as well as common utilities used by each E2E MG
// test.  In this case, each test is identical except for the inputs and
// expected outputs, so the entire test is defined in the run_test() method.
//
class Louvain_MG_Testfixture : public ::testing::TestWithParam<Louvain_Usecase> {
 public:
  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  // Run once for each test instance
  virtual void SetUp() {}
  virtual void TearDown() {}

  // Return the results of running louvain on a single GPU for the dataset in
  // graph_file_path.
  template <typename vertex_t, typename edge_t, typename weight_t>
  std::tuple<int, weight_t, std::vector<vertex_t>> get_sg_results(
    raft::handle_t const& handle,
    std::string const& graph_file_path,
    size_t max_level,
    weight_t resolution)
  {
    // FIXME:  Put this in the Graph test base class
    //         (make the call simpler here)
    auto graph_tuple =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
        handle,
        graph_file_path,
        true,
        false);  // FIXME: should use param.test_weighted instead of true

    auto graph_view     = std::get<0>(graph_tuple).view();
    cudaStream_t stream = handle.get_stream();

    rmm::device_uvector<vertex_t> clustering_v(graph_view.get_number_of_local_vertices(), stream);

    size_t level;
    weight_t modularity;

    std::tie(level, modularity) =
      cugraph::louvain(handle, graph_view, clustering_v.data(), max_level, resolution);

    std::vector<vertex_t> clustering(graph_view.get_number_of_local_vertices());
    raft::update_host(clustering.data(), clustering_v.data(), clustering_v.size(), stream);

    return std::make_tuple(level, modularity, clustering);
  }

  // Compare the results of running louvain on multiple GPUs to that of a
  // single-GPU run for the configuration in param.  Note that MNMG Louvain
  // and single GPU Louvain are ONLY deterministic through a single
  // iteration of the outer loop.  Renumbering of the partitions when coarsening
  // the graph is a function of the number of GPUs in the GPU cluster.
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_test(const Louvain_Usecase& param)
  {
    raft::handle_t handle;

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    const auto& comm = handle.get_comms();

    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) { --row_comm_size; }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    cudaStream_t stream = handle.get_stream();

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, true> mg_graph(handle);
    rmm::device_uvector<vertex_t> d_renumber_map_labels(0, handle.get_stream());

    std::tie(mg_graph, d_renumber_map_labels) =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, true>(
        handle, param.graph_file_full_path, true, false);

    // Each GPU will have a subset of the clustering
    int sg_level;
    weight_t sg_modularity;
    std::vector<vertex_t> sg_clustering;

    // FIXME:  Consider how to test for max_level > 1
    //         perhaps some sort of approximation
    // size_t local_max_level{param.max_level};
    size_t local_max_level{1};

    auto mg_graph_view = mg_graph.view();

    rmm::device_uvector<vertex_t> clustering_v(mg_graph_view.get_number_of_local_vertices(),
                                               stream);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    int level;
    weight_t modularity;

    std::tie(level, modularity) = cugraph::louvain(
      handle, mg_graph_view, clustering_v.data(), local_max_level, param.resolution);

    if (comm_rank == 0) {
      SCOPED_TRACE("compare modularity input: " + param.graph_file_full_path);

      std::tie(sg_level, sg_modularity, sg_clustering) = get_sg_results<vertex_t, edge_t, weight_t>(
        handle, param.graph_file_full_path, local_max_level, param.resolution);

      compare(modularity, sg_modularity);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
TEST_P(Louvain_MG_Testfixture, CheckInt32Int32Float)
{
  run_test<int32_t, int32_t, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(
  simple_test,
  Louvain_MG_Testfixture,
  ::testing::Values(Louvain_Usecase("test/datasets/karate.mtx", true, 100, 1),
                    Louvain_Usecase("test/datasets/smallworld.mtx", true, 100, 1)));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
