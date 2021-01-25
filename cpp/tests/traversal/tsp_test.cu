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

#include <algorithms.hpp>
#include <cmath>
#include <cuda_profiler_api.h>
#include <err.h>
#include <graph.hpp>
#include <raft/error.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_utilities.hpp>

typedef struct Tsp_Usecase_t {
  std::string tsp_file;
  int restarts;
  int ref_cost;
  Tsp_Usecase_t(const std::string& a, const int b, const int c)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      tsp_file = rapidsDatasetRootDir + "/" + a;
    } else {
      tsp_file = a;
    }
    restarts = b;
    ref_cost = c;
  }
  Tsp_Usecase_t& operator=(const Tsp_Usecase_t& rhs)
  {
    tsp_file = rhs.tsp_file;
    restarts = rhs.restarts;
    ref_cost = rhs.ref_cost;
    return *this;
  }
} Tsp_Usecase;

struct Route {
  uint num_packages;
  int *cities;
  float *x;
  float *y;
  float *vol;
  uint *order;
};

class Tests_Tsp : public ::testing::TestWithParam<Tsp_Usecase> {
 public:
  Tests_Tsp() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename T>
  void run_current_test(const Tsp_Usecase& param)
  {
    const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".") +
                          std::string(test_info->name()) + std::string("_") +
                          cugraph::test::getFileName(param.tsp_file) + std::string("_") +
                          ss.str().c_str();

    float tol = 5E-2f;
    HighResClock hr_clock;
    double time_tmp;
    Route input;

    int restarts = param.restarts;
    int nodes = load_tsp(param.tsp_file.c_str(), &input);

    // Device alloc
    raft::handle_t handle;
    rmm::device_uvector<int> vertices(static_cast<size_t>(nodes), nullptr);
    rmm::device_uvector<int> route(static_cast<size_t>(nodes), nullptr);
    rmm::device_uvector<float> x_pos(static_cast<size_t>(nodes), nullptr);
    rmm::device_uvector<float> y_pos(static_cast<size_t>(nodes), nullptr);

    int *vtx_ptr = vertices.data();
    int *d_route   = route.data();
    float *d_x_pos = x_pos.data();
    float *d_y_pos = y_pos.data();

    CUDA_TRY(cudaMemcpy(vtx_ptr, input.cities, sizeof(int) * nodes, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_x_pos, input.x, sizeof(float) * nodes, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_y_pos, input.y, sizeof(float) * nodes, cudaMemcpyHostToDevice));

    // Default parameters
    bool beam_search = true;
    int k          = 4;
    int nstart = 0;
    bool verbose   = false;

    hr_clock.start();
    cudaDeviceSynchronize();
    cudaProfilerStart();

    float final_cost = cugraph::traveling_salesman(handle,
                                                   vtx_ptr,
                                                   d_route,
                                                   d_x_pos,
                                                   d_y_pos,
                                                   nodes,
                                                   restarts,
                                                   beam_search,
                                                   k,
                                                   nstart,
                                                   verbose);
    cudaProfilerStop();
    cudaDeviceSynchronize();
    hr_clock.stop(&time_tmp);
    std::cout << "tsp_time: " << time_tmp << " us" << std::endl;
    std::cout << "Final cost is: " << final_cost << "\n";
    float err = final_cost - param.ref_cost;
    err /= param.ref_cost;
    std::cout << "Approximation error is: " << err * 100 << "%\n";
    EXPECT_LE(err, tol);
  }

 private:
  static int load_tsp(const char *fname, Route *input) {
    int ch, cnt, in1, nodes;
    float in2, in3;
    FILE *f;
    int res;
    char str[256];  // potential for buffer overrun

    f = fopen(fname, "rt");
    if (f == NULL)
      err(1, "could not open file %s\n", fname);

    ch = getc(f);
    while ((ch != EOF) && (ch != '\n')) ch = getc(f);
    ch = getc(f);
    while ((ch != EOF) && (ch != '\n')) ch = getc(f);
    ch = getc(f);
    while ((ch != EOF) && (ch != '\n')) ch = getc(f);

    ch = getc(f);
    while ((ch != EOF) && (ch != ':')) ch = getc(f);
    res = fscanf(f, "%s\n", str);
    nodes = atoi(str);
    input->num_packages = (uint)nodes;

    if (nodes <= 2)
      err(1, "only %d nodes\n", nodes);

    input->cities = (int *)malloc(sizeof(int) * nodes);
    if (!input->cities)
      err(1, "cannot allocate %d city vector\n", nodes);
    input->x = (float *)malloc(sizeof(float) * nodes);
    if (!input->x)
      err(1, "cannot allocate %d xcoords\n", nodes);
    input->y = (float *)malloc(sizeof(float) * nodes);
    if (!input->y)
      err(1, "cannot allocate %d ycoords\n", nodes);

    ch = getc(f);
    while ((ch != EOF) && (ch != '\n')) ch = getc(f);
    res = fscanf(f, "%s\n", str);
    if (strcmp(str, "NODE_COORD_SECTION") != 0)
      err(1, "wrong file format\n");

    cnt = 0;
    while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) {
      input->cities[cnt] = in1;
      input->x[cnt] = in2;
      input->y[cnt] = in3;
      cnt++;
      if (cnt > nodes)
        err(1, "inconsistent data: input too long\n");
      if (cnt != in1)
        err(1, "input line mismatch: expected %d instead of %d\n", cnt, in1);
    }
    if (cnt != nodes)
      err(1, "inconsistent data: read %d instead of %d nodes\n", cnt, nodes);

    res = fscanf(f, "%s", str);
    if (strcmp(str, "EOF") != 0)
      err(1, "didn't see 'EOF' at end of file\n");
    res = res;
    fclose(f);
    return nodes;
  }
};


TEST_P(Tests_Tsp, CheckFP32_T) { run_current_test<float>(GetParam()); }

TEST_P(Tests_Tsp, CheckFP64_T) { run_current_test<double>(GetParam()); }

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_Tsp,
                        ::testing::Values(Tsp_Usecase("tsplib/datasets/tsp225.tsp", 4096, 3916)
                       ));
CUGRAPH_TEST_PROGRAM_MAIN()

