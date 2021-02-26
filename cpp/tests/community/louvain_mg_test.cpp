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
#include <utilities/mg_test_utilities.hpp>
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <partition_manager.hpp>

#include <gtest/gtest.h>

////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_CASE_P()
//
struct Louvain_Testparams {
  std::string graph_file_full_path{};
  bool weighted{false};
  size_t max_level;
  double resolution;

  // TODO:  We really should have a Graph_Testparms_Base class or something
  //        like that which can handle this graph_full_path thing.
  //
  Louvain_Testparams(std::string const& graph_file_path,
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
class Louvain_MG_Testfixture : public cugraph::test::MG_TestFixture_t,
                               public ::testing::WithParamInterface<Louvain_Testparams> {
 public:
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
    // TODO:  Put this in the Graph test base class
    //        (make the call simpler here)
    auto graph =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false>(
        handle, graph_file_path, true);  // FIXME: should use param.test_weighted instead of true

    auto graph_view     = graph.view();
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
  // single-GPU run for the configuration in param.
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_test(const Louvain_Testparams& param)
  {
    raft::handle_t handle;
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    const auto& comm = handle.get_comms();

    cudaStream_t stream = handle.get_stream();

    // Assuming 2 GPUs which means 1 row, 2 cols. 2 cols = row_comm_size of 2.
    // FIXME: DO NOT ASSUME 2 GPUs, add code to compute prows, pcols
    size_t row_comm_size{2};
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    int my_rank = comm.get_rank();

    // FIXME: graph must be weighted!
    std::unique_ptr<cugraph::experimental::
                      graph_t<vertex_t, edge_t, weight_t, false, true>>  // store_transposed=false,
                                                                         // multi_gpu=true
      mg_graph_ptr{};
    rmm::device_uvector<vertex_t> d_renumber_map_labels(0, handle.get_stream());

    std::tie(mg_graph_ptr, d_renumber_map_labels) = cugraph::test::
      create_graph_for_gpu<vertex_t, edge_t, weight_t, false>  // store_transposed=true
      (handle, param.graph_file_full_path);

    auto mg_graph_view = mg_graph_ptr->view();

    rmm::device_uvector<vertex_t> clustering_v(mg_graph_view.get_number_of_local_vertices(),
                                               stream);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    int level;
    weight_t modularity;

    std::cout << "calling MG louvain" << std::endl;

    std::tie(level, modularity) = cugraph::louvain(
      handle, mg_graph_view, clustering_v.data(), param.max_level, param.resolution);

    std::vector<vertex_t> clustering(mg_graph_view.get_number_of_local_vertices());

    raft::update_host(clustering.data(), clustering_v.data(), clustering_v.size(), stream);

    std::vector<vertex_t> h_renumber_map_labels(mg_graph_view.get_number_of_vertices());
    raft::update_host(h_renumber_map_labels.data(),
                      d_renumber_map_labels.data(),
                      d_renumber_map_labels.size(),
                      stream);

    // Compare MG to SG

    // Each GPU will have a subset of the clustering
    int sg_level;
    weight_t sg_modularity;
    std::vector<vertex_t> sg_clustering;

    std::tie(sg_level, sg_modularity, sg_clustering) = get_sg_results<vertex_t, edge_t, weight_t>(
      handle, param.graph_file_full_path, param.max_level, param.resolution);

    std::cout << "MG:  level = " << level << ", modularity = " << modularity << std::endl;
    raft::print_host_vector("clustering", clustering.data(), clustering.size(), std::cout);

    std::cout << "SG:  level = " << sg_level << ", modularity = " << sg_modularity << std::endl;
    raft::print_host_vector("clustering", sg_clustering.data(), sg_clustering.size(), std::cout);

#if 0
    // For this test, each GPU will have the full set of vertices and
    // therefore the pageranks vectors should be equal in size.
    ASSERT_EQ(h_sg_pageranks.size(), h_mg_pageranks.size());

    auto threshold_ratio = 1e-3;
    auto threshold_magnitude =
      (1.0 / static_cast<result_t>(mg_graph_view.get_number_of_vertices())) *
      threshold_ratio;  // skip comparison for low PageRank verties (lowly ranked vertices)
    auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
      return std::abs(lhs - rhs) <
             std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
    };

    vertex_t mapped_vertex{0};
    for (vertex_t i = 0;
         i + mg_graph_view.get_local_vertex_first() < mg_graph_view.get_local_vertex_last();
         ++i) {
      mapped_vertex = h_renumber_map_labels[i];
      ASSERT_TRUE(nearly_equal(h_mg_pageranks[i], h_sg_pageranks[mapped_vertex]))
        << "MG PageRank value for vertex: " << i << " in rank: " << my_rank
        << " has value: " << h_mg_pageranks[i]
        << " which exceeds the error margin for comparing to SG value: " << h_sg_pageranks[i];
    }
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
TEST_P(Louvain_MG_Testfixture, CheckInt32Int32FloatFloat)
{
  run_test<int32_t, int32_t, float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(
  e2e,
  Louvain_MG_Testfixture,
  ::testing::Values(Louvain_Testparams("test/datasets/karate.mtx", true, 100, 1)
                    // Louvain_Testparams("test/datasets/webbase-1M.mtx", true, 100, 1),
                    ));

// FIXME: Enable proper RMM configuration by using CUGRAPH_TEST_PROGRAM_MAIN().
//        Currently seeing a RMM failure during init, need to investigate.
// CUGRAPH_TEST_PROGRAM_MAIN()
