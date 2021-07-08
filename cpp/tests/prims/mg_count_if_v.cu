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
#include <cugraph/prims/count_if_v.cuh>
#include <cuco/detail/hash_functions.cuh>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/count.h>

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

  // Compare the results of count_if_v primitive and thrust count_if on a single GPU
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
    auto [mg_graph, d_mg_renumber_map_labels] =
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

    const int hash_bin_count = 5;

    auto primitive_lambda = [hash_bin_count] __device__(auto val) {
      cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
      return (0 == (hash_func(val) % hash_bin_count)); };

    // 4. run MG count if

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    vertex_t const * data = (*d_mg_renumber_map_labels).data();
    auto vertex_count = count_if_v(handle,
                                   mg_graph_view,
                                   data,
                                   primitive_lambda);

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG count if took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 5. compare SG & MG results
    auto [sg_graph, d_sg_renumber_map_labels] =
      input_usecase.template construct_graph<vertex_t, edge_t, weight_t, true, false>(
        handle, true, false);
    auto sg_graph_view = sg_graph.view();
    auto expected_vertex_count =
      thrust::count_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       thrust::make_counting_iterator(sg_graph_view.get_local_vertex_first()),
                       thrust::make_counting_iterator(sg_graph_view.get_local_vertex_last()),
                                                  primitive_lambda);
    ASSERT_TRUE(expected_vertex_count == vertex_count);
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
  ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                    cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                    cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                    cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx")));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MG_CountIfV_Rmat,
                         ::testing::Values(cugraph::test::Rmat_Usecase(
                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true)));

INSTANTIATE_TEST_SUITE_P(rmat_large_test,
                         Tests_MG_CountIfV_Rmat,
                         ::testing::Values(cugraph::test::Rmat_Usecase(
                             20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true)));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
