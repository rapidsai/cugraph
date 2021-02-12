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
#pragma once

#include <experimental/graph.hpp>

#include <raft/comms/mpi_comms.hpp>

#include <gtest/gtest.h>

namespace cugraph {
namespace test {

// Given a raft handle and a path to a dataset (must be a .mtx file), returns a
// tuple containing:
//  * graph_t instance for the partition accesible from the raft handle
//  * 4-tuple containing renumber info resulting from renumbering the
//    edgelist for the partition
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
std::tuple<
  std::unique_ptr<cugraph::experimental::
                    graph_t<vertex_t, edge_t, weight_t, store_transposed, true>>,  // multi_gpu=true
  rmm::device_uvector<vertex_t>>
create_graph_for_gpu(raft::handle_t& handle, const std::string& graph_file_path);

/**
 * @brief Base test fixture class, responsible for handling common operations
 * needed by all MG tests.
 *
 * It's expected this class will be built out and refactored often as new MG C++
 * tests are added and new patterns evolve.
 *
 * Example:
 * ```
 * class MyTestFixture : public cugraph::test::MG_TestFixture_t<My_Testparams_t> {};
 * ```
 **/

// FIXME: consider moving this to a separate file? (eg. mg_test_fixture.cpp)?

class MG_TestFixture_t : public ::testing::Test {
 public:
  static void SetUpTestCase()
  {
    MPI_TRY(MPI_Init(NULL, NULL));

    int rank, size;
    MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int nGpus;
    CUDA_CHECK(cudaGetDeviceCount(&nGpus));

    ASSERT(
      nGpus >= size, "Number of GPUs are lesser than MPI ranks! ngpus=%d, nranks=%d", nGpus, size);

    CUDA_CHECK(cudaSetDevice(rank));
  }

  static void TearDownTestCase() { MPI_TRY(MPI_Finalize()); }
};

}  // namespace test
}  // namespace cugraph
