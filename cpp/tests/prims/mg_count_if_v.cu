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

#include <utilities/high_res_clock.h>
#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/partition_manager.hpp>

#include <cugraph/experimental/graph_view.hpp>
#include <cugraph/prims/any_of_adj_matrix_row.cuh>
#include <cugraph/prims/copy_to_adj_matrix_row_col.cuh>
#include <cugraph/prims/copy_v_transform_reduce_in_out_nbr.cuh>
#include <cugraph/prims/count_if_e.cuh>
#include <cugraph/prims/count_if_v.cuh>
#include <cugraph/prims/reduce_v.cuh>
#include <cugraph/prims/transform_reduce_v.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device.cuh>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

// do the perf measurements
// enabled by command line parameter s'--perf'
//
static int PERF = 0;

template <typename input_usecase_t>
class Tests_MG_CountIfV
  : public ::testing::TestWithParam<input_usecase_t> {
 public:
  Tests_MG_CountIfV() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running Katz Centrality on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(input_usecase_t const &input_usecase)
  {
    // 1. initialize handle

    raft::handle_t handle{};
    HighResClock hr_clock{};

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    auto &comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) { --row_comm_size; }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    // 2. create MG graph

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, true> mg_graph(handle);
    rmm::device_uvector<vertex_t> d_mg_renumber_map_labels(0, handle.get_stream());
    std::tie(mg_graph, d_mg_renumber_map_labels) =
      input_usecase.template construct_graph<vertex_t, edge_t, weight_t, true, true>(
        handle, true, true);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    std::cout<<"Number of local vertices : "<<mg_graph_view.get_number_of_local_vertices()<<"\n";
    std::cout<<"Number of map labels : "<<d_mg_renumber_map_labels.size()<<"\n";

    vertex_t const * data = d_mg_renumber_map_labels.data();

    auto num_vertices = count_if_v(handle,
                                   mg_graph_view,
                                   data,
                                   [] __device__(auto val) { return true; });
    std::cout<<"Count if vertices : "<<num_vertices<<"\n";


    // 3. compute max in-degree

    // 4. run MG Katz Centrality

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG Katz Centrality took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 5. copmare SG & MG results

  }
};

using Tests_MG_CountIfV_File = Tests_MG_CountIfV<cugraph::test::File_Usecase>;
using Tests_MG_CountIfV_Rmat = Tests_MG_CountIfV<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MG_CountIfV_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(GetParam());
}

TEST_P(Tests_MG_CountIfV_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MG_CountIfV_File,
  ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx")));
 // ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
 //                   cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
 //                   cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
 //                   cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx")));

//INSTANTIATE_TEST_SUITE_P(rmat_small_test,
//                         Tests_MG_CountIfV_Rmat,
//                         ::testing::Values(cugraph::test::Rmat_Usecase(
//                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true)));

//INSTANTIATE_TEST_SUITE_P(rmat_large_test,
//                         Tests_MG_CountIfV_Rmat,
//                         ::testing::Values(cugraph::test::Rmat_Usecase(
//                             20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true)));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
