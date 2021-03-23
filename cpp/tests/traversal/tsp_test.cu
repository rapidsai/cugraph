/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// TSP solver tests
// Author: Hugo Linsenmaier hlinsenmaier@nvidia.com

/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// TSP solver tests
// Author: Hugo Linsenmaier hlinsenmaier@nvidia.com

#include <utilities/high_res_clock.h>
#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <graph.hpp>

#include <cuda_profiler_api.h>

#include <raft/error.hpp>
#include <raft/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <fstream>
#include <set>
#include <vector>

typedef struct Tsp_Usecase_t {
  std::string tsp_file;
  float ref_cost;
  Tsp_Usecase_t(const std::string& a, const float c)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      tsp_file = rapidsDatasetRootDir + "/" + a;
    } else {
      tsp_file = a;
    }
    ref_cost = c;
  }
  Tsp_Usecase_t& operator=(const Tsp_Usecase_t& rhs)
  {
    tsp_file = rhs.tsp_file;
    ref_cost = rhs.ref_cost;
    return *this;
  }
} Tsp_Usecase;

static std::vector<Tsp_Usecase_t> euc_2d{
  {"tsplib/datasets/a280.tsp", 2579},      {"tsplib/datasets/berlin52.tsp", 7542},
  {"tsplib/datasets/bier127.tsp", 118282}, {"tsplib/datasets/ch130.tsp", 6110},
  {"tsplib/datasets/ch150.tsp", 6528},     {"tsplib/datasets/d1291.tsp", 50801},
  {"tsplib/datasets/d1655.tsp", 62128},    {"tsplib/datasets/d198.tsp", 15780},
  {"tsplib/datasets/d2103.tsp", 80450},    {"tsplib/datasets/d493.tsp", 35002},
  {"tsplib/datasets/d657.tsp", 48912},     {"tsplib/datasets/eil101.tsp", 629},
  {"tsplib/datasets/eil51.tsp", 426},      {"tsplib/datasets/eil76.tsp", 538},
  {"tsplib/datasets/fl1400.tsp", 20127},   {"tsplib/datasets/fl1577.tsp", 22249},
  {"tsplib/datasets/fl417.tsp", 11861},    {"tsplib/datasets/gil262.tsp", 2378},
  {"tsplib/datasets/kroA100.tsp", 21282},  {"tsplib/datasets/kroA150.tsp", 26524},
  {"tsplib/datasets/kroA200.tsp", 29368},  {"tsplib/datasets/kroB100.tsp", 22141},
  {"tsplib/datasets/kroB150.tsp", 26130},  {"tsplib/datasets/kroB200.tsp", 29437},
  {"tsplib/datasets/kroC100.tsp", 20749},  {"tsplib/datasets/kroD100.tsp", 21294},
  {"tsplib/datasets/kroE100.tsp", 22068},  {"tsplib/datasets/lin105.tsp", 14379},
  {"tsplib/datasets/lin318.tsp", 42029},   {"tsplib/datasets/nrw1379.tsp", 56638},
  {"tsplib/datasets/p654.tsp", 34643},     {"tsplib/datasets/pcb1173.tsp", 56892},
  {"tsplib/datasets/pcb442.tsp", 50778},   {"tsplib/datasets/pr1002.tsp", 259045},
  {"tsplib/datasets/pr107.tsp", 44303},    {"tsplib/datasets/pr136.tsp", 96772},
  {"tsplib/datasets/pr144.tsp", 58537},    {"tsplib/datasets/pr152.tsp", 73682},
  {"tsplib/datasets/pr226.tsp", 80369},    {"tsplib/datasets/pr264.tsp", 49135},
  {"tsplib/datasets/pr299.tsp", 48191},    {"tsplib/datasets/pr439.tsp", 107217},
  {"tsplib/datasets/pr76.tsp", 108159},    {"tsplib/datasets/rat195.tsp", 2323},
  {"tsplib/datasets/rat575.tsp", 6773},    {"tsplib/datasets/rat783.tsp", 8806},
  {"tsplib/datasets/rat99.tsp", 1211},     {"tsplib/datasets/rd100.tsp", 7910},
  {"tsplib/datasets/rd400.tsp", 15281},    {"tsplib/datasets/rl1323.tsp", 270199},
  {"tsplib/datasets/st70.tsp", 675},       {"tsplib/datasets/ts225.tsp", 126643},
  {"tsplib/datasets/tsp225.tsp", 3916},    {"tsplib/datasets/u1060.tsp", 224094},
  {"tsplib/datasets/u1432.tsp", 152970},   {"tsplib/datasets/u159.tsp", 42080},
  {"tsplib/datasets/u574.tsp", 36905},     {"tsplib/datasets/u724.tsp", 41910},
  {"tsplib/datasets/vm1084.tsp", 239297},
};

struct Route {
  std::vector<int> cities;
  std::vector<float> x_pos;
  std::vector<float> y_pos;
};

class Tests_Tsp : public ::testing::TestWithParam<Tsp_Usecase> {
 public:
  Tests_Tsp() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  void run_current_test(const Tsp_Usecase& param)
  {
    const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".") +
                          std::string(test_info->name()) + std::string("_") +
                          cugraph::test::getFileName(param.tsp_file) + std::string("_") +
                          ss.str().c_str();

    float tol = 1E-1f;
    HighResClock hr_clock;
    double time_tmp;
    Route input;

    std::cout << "File: " << param.tsp_file.c_str() << "\n";
    int nodes = load_tsp(param.tsp_file.c_str(), &input);

    // Device alloc
    raft::handle_t const handle;
    rmm::device_uvector<int> vertices(static_cast<size_t>(nodes), nullptr);
    rmm::device_uvector<int> route(static_cast<size_t>(nodes), nullptr);
    rmm::device_uvector<float> x_pos(static_cast<size_t>(nodes), nullptr);
    rmm::device_uvector<float> y_pos(static_cast<size_t>(nodes), nullptr);

    int* vtx_ptr   = vertices.data();
    int* d_route   = route.data();
    float* d_x_pos = x_pos.data();
    float* d_y_pos = y_pos.data();

    CUDA_TRY(cudaMemcpy(vtx_ptr, input.cities.data(), sizeof(int) * nodes, cudaMemcpyHostToDevice));
    CUDA_TRY(
      cudaMemcpy(d_x_pos, input.x_pos.data(), sizeof(float) * nodes, cudaMemcpyHostToDevice));
    CUDA_TRY(
      cudaMemcpy(d_y_pos, input.y_pos.data(), sizeof(float) * nodes, cudaMemcpyHostToDevice));

    // Default parameters
    int restarts     = 4096;
    bool beam_search = true;
    int k            = 4;
    int nstart       = 0;
    bool verbose     = false;

    hr_clock.start();
    cudaDeviceSynchronize();
    cudaProfilerStart();

    float final_cost = cugraph::traveling_salesperson(
      handle, vtx_ptr, d_x_pos, d_y_pos, nodes, restarts, beam_search, k, nstart, verbose, d_route);
    cudaProfilerStop();
    cudaDeviceSynchronize();
    hr_clock.stop(&time_tmp);

    std::vector<int> h_route;
    h_route.resize(nodes);
    std::vector<int> h_vertices;
    h_vertices.resize(nodes);
    CUDA_TRY(cudaMemcpy(h_route.data(), d_route, sizeof(int) * nodes, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    CUDA_TRY(cudaMemcpy(h_vertices.data(), vtx_ptr, sizeof(int) * nodes, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    std::cout << "tsp_time: " << time_tmp << " us" << std::endl;
    std::cout << "Ref cost is: " << param.ref_cost << "\n";
    std::cout << "Final cost is: " << final_cost << "\n";
    float err = fabs(final_cost - param.ref_cost);
    err /= param.ref_cost;
    std::cout << "Approximation error is: " << err * 100 << "%\n";
    EXPECT_LE(err, tol);

    // Check route goes through each vertex once
    size_t u_nodes = nodes;
    std::set<int> node_set(h_route.begin(), h_route.end());
    ASSERT_EQ(node_set.size(), u_nodes);

    // Bound check
    int max = *std::max_element(h_vertices.begin(), h_vertices.end());
    int min = *std::min_element(h_vertices.begin(), h_vertices.end());
    EXPECT_GE(*node_set.begin(), min);
    EXPECT_LE(*node_set.rbegin(), max);
  }

 private:
  std::vector<std::string> split(const std::string& s, char delimiter)
  {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
      if (token.size() == 0) continue;
      tokens.push_back(token);
    }
    return tokens;
  }

  // FIXME: At the moment TSP does not accept a graph_t as input and therefore
  // deviates from the standard testing I/O pattern. Once other input types
  // are supported we want to reconcile TSP testing with the rest of cugraph.
  int load_tsp(const char* fname, Route* input)
  {
    std::fstream fs;
    fs.open(fname);
    std::string line;
    std::vector<std::string> tokens;
    int nodes = 0;
    while (std::getline(fs, line) && line.find(':') != std::string::npos) {
      tokens           = split(line, ':');
      auto strip_token = split(tokens[0], ' ')[0];
      if (strip_token == "DIMENSION") nodes = std::stof(tokens[1]);
    }

    while (std::getline(fs, line) && line.find(' ') != std::string::npos) {
      tokens       = split(line, ' ');
      auto city_id = std::stof(tokens[0]);
      auto x       = std::stof(tokens[1]);
      auto y       = std::stof(tokens[2]);
      input->cities.push_back(city_id);
      input->x_pos.push_back(x);
      input->y_pos.push_back(y);
    }
    fs.close();
    assert(nodes == input->cities.size());
    return nodes;
  }
};

TEST_P(Tests_Tsp, CheckFP32_T) { run_current_test(GetParam()); }

INSTANTIATE_TEST_CASE_P(simple_test, Tests_Tsp, ::testing::ValuesIn(euc_2d));
CUGRAPH_TEST_PROGRAM_MAIN()
