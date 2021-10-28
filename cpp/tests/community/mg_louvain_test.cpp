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

#include "mg_louvain_helper.hpp"

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/cudart_utils.h>
#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <gtest/gtest.h>

void compare(float mg_modularity, float sg_modularity)
{
  ASSERT_FLOAT_EQ(mg_modularity, sg_modularity);
}
void compare(double mg_modularity, double sg_modularity)
{
  ASSERT_DOUBLE_EQ(mg_modularity, sg_modularity);
}

////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_SUITE_P()
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

  // Compare the results of MNMG Louvain with the results of running
  // each step of SG Louvain, renumbering the coarsened graphs based
  // on the MNMG renumbering.
  template <typename vertex_t, typename edge_t, typename weight_t>
  void compare_sg_results(raft::handle_t const& handle,
                          std::string const& graph_filename,
                          rmm::device_uvector<vertex_t>& d_renumber_map_gathered_v,
                          cugraph::Dendrogram<vertex_t> const& dendrogram,
                          weight_t resolution,
                          int rank,
                          weight_t mg_modularity)
  {
    auto sg_graph =
      std::make_unique<cugraph::graph_t<vertex_t, edge_t, weight_t, false, false>>(handle);
    rmm::device_uvector<vertex_t> d_clustering_v(0, handle.get_stream());
    weight_t sg_modularity{-1.0};

    if (rank == 0) {
      // Create initial SG graph, renumbered according to the MNMG renumber map

      auto [d_edgelist_rows,
            d_edgelist_cols,
            d_edgelist_weights,
            d_vertices,
            number_of_vertices,
            is_symmetric] =
        cugraph::test::read_edgelist_from_matrix_market_file<vertex_t, weight_t, false, false>(
          handle, graph_filename, true);

      d_clustering_v.resize(d_vertices.size(), handle.get_stream());

      // renumber using d_renumber_map_gathered_v
      cugraph::test::single_gpu_renumber_edgelist_given_number_map(
        handle, d_edgelist_rows, d_edgelist_cols, d_renumber_map_gathered_v);

      std::tie(*sg_graph, std::ignore) =
        cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, false, false>(
          handle,
          std::move(d_vertices),
          std::move(d_edgelist_rows),
          std::move(d_edgelist_cols),
          std::move(d_edgelist_weights),
          cugraph::graph_properties_t{is_symmetric, false},
          false);
    }

    std::for_each(
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(dendrogram.num_levels()),
      [&dendrogram, &sg_graph, &d_clustering_v, &sg_modularity, &handle, resolution, rank](
        size_t i) {
        auto d_dendrogram_gathered_v = cugraph::test::device_gatherv(
          handle, dendrogram.get_level_ptr_nocheck(i), dendrogram.get_level_size_nocheck(i));

        if (rank == 0) {
          auto graph_view = sg_graph->view();

          d_clustering_v.resize(graph_view.get_number_of_vertices(), handle.get_stream());

          std::tie(std::ignore, sg_modularity) =
            cugraph::louvain(handle, graph_view, d_clustering_v.data(), size_t{1}, resolution);

          EXPECT_TRUE(cugraph::test::renumbered_vectors_same(
            handle, d_clustering_v, d_dendrogram_gathered_v));

          sg_graph =
            cugraph::test::coarsen_graph(handle, graph_view, d_dendrogram_gathered_v.data());
        }
      });

    if (rank == 0) compare(mg_modularity, sg_modularity);
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
    while (comm_size % row_comm_size != 0) {
      --row_comm_size;
    }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    cudaStream_t stream = handle.get_stream();

    auto [mg_graph, d_renumber_map_labels] =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, true>(
        handle, param.graph_file_full_path, true, true);

    auto mg_graph_view = mg_graph.view();

    std::unique_ptr<cugraph::Dendrogram<vertex_t>> dendrogram;
    weight_t mg_modularity;

    std::tie(dendrogram, mg_modularity) =
      cugraph::louvain(handle, mg_graph_view, param.max_level, param.resolution);

    SCOPED_TRACE("compare modularity input: " + param.graph_file_full_path);

    auto d_renumber_map_gathered_v = cugraph::test::device_gatherv(
      handle, (*d_renumber_map_labels).data(), (*d_renumber_map_labels).size());

    compare_sg_results<vertex_t, edge_t, weight_t>(handle,
                                                   param.graph_file_full_path,
                                                   d_renumber_map_gathered_v,
                                                   *dendrogram,
                                                   param.resolution,
                                                   comm_rank,
                                                   mg_modularity);
  }
};

////////////////////////////////////////////////////////////////////////////////
TEST_P(Louvain_MG_Testfixture, CheckInt32Int32Float)
{
  run_test<int32_t, int32_t, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Louvain_MG_Testfixture,
  ::testing::Values(Louvain_Usecase("test/datasets/karate.mtx", true, 100, 1),
                    Louvain_Usecase("test/datasets/dolphins.mtx", true, 100, 1)));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
