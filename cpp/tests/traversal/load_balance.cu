/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "gtest/gtest.h"
#include "utilities/test_utilities.hpp"

#include <raft/handle.hpp>
#include <utilities/error.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include "traversal/opg/load_balance.cuh"

// ============================================================================
// Test Suite
// ============================================================================
typedef struct BFS_Usecase_t {
  std::string config_;     // Path to graph file
  std::string file_path_;  // Complete path to graph using dataset_root_dir
  BFS_Usecase_t(const std::string &config) : config_(config)
  {
    const std::string &rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((config_ != "") && (config_[0] != '/')) {
      file_path_ = rapidsDatasetRootDir + "/" + config_;
    } else {
      file_path_ = config_;
    }
  };
} LB_Usecase;

template <typename VT, typename ET>
struct in_degree {
  ET * in_degree_counter_;
  in_degree(ET * in_degree_counter) : in_degree_counter_(in_degree_counter) {}
  __device__ void operator()(VT src, VT dst) {
    atomicAdd(in_degree_counter_ + dst, 1);
  }
};

template <typename VT, typename ET>
struct in_degree_simple {
  ET * in_degree_counter_;
  in_degree_simple(ET * in_degree_counter) : in_degree_counter_(in_degree_counter) {}
  __device__ void operator()(VT dst) {
    atomicAdd(in_degree_counter_ + dst, 1);
  }
};

class Tests_LB : public ::testing::TestWithParam<LB_Usecase> {
 public:
  Tests_LB() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // VT                 vertex identifier data type
  // ET                 edge identifier data type
  // WT                 edge weight data type
  template <typename VT, typename ET, typename WT>
  void run_current_test(const LB_Usecase &configuration)
  {
    // Step 1: Construction of the graph based on configuration
    //VT number_of_vertices;
    //ET number_of_edges;
    bool directed = false;
    auto csr =
      cugraph::test::generate_graph_csr_from_mm<VT, ET, WT>(directed, configuration.file_path_);
    cudaDeviceSynchronize();
    cugraph::experimental::GraphCSRView<VT, ET, WT> G = csr->view();
    G.prop.directed                                   = directed;

    rmm::device_vector<ET> in_degree_lb(G.number_of_vertices, 0);

    raft::handle_t handle;
    detail::opg::LoadBalanceExecution<VT, ET, WT> lb(handle, G);
    CUDA_TRY(cudaGetLastError());
    in_degree<VT, ET> in_degree_op(in_degree_lb.data().get());
    lb.run(in_degree_op);
    CUDA_TRY(cudaGetLastError());

    cudaStream_t stream = 0;

    //Calculate the in degree of destinations
    rmm::device_vector<ET> gold_indegree(G.number_of_vertices, 0);
    rmm::device_vector<VT> destinations(G.number_of_edges);
    CUDA_TRY(cudaMemcpy(destinations.data().get(), G.indices,
        sizeof(VT) * destinations.size(),
        cudaMemcpyDeviceToDevice));
    thrust::for_each(rmm::exec_policy(stream)->on(stream),
        destinations.begin(), destinations.end(),
        in_degree_simple<VT, ET>(gold_indegree.data().get())
        );

    bool is_result_equal = thrust::equal(rmm::exec_policy(stream)->on(stream),
        in_degree_lb.begin(), in_degree_lb.end(),
        gold_indegree.begin());

    EXPECT_TRUE(is_result_equal);
  }
};

TEST_P(Tests_LB, CheckFP32_SP_COUNTER) { run_current_test<int, int, float>(GetParam()); }

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_LB,
                        ::testing::Values(LB_Usecase("test/datasets/karate.mtx"),
                                          LB_Usecase("test/datasets/polbooks.mtx"),
                                          LB_Usecase("test/datasets/netscience.mtx"),
                                          LB_Usecase("test/datasets/netscience.mtx"),
                                          LB_Usecase("test/datasets/wiki2003.mtx"),
                                          LB_Usecase("test/datasets/wiki-Talk.mtx")));

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}
