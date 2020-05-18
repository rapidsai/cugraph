/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// strongly connected components tests
// Author: Andrei Schaffer aschaffer@nvidia.com

#include "cuda_profiler_api.h"
#include "gtest/gtest.h"
#include "high_res_clock.h"

#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <algorithm>
#include <iterator>
#include "test_utils.h"

#include <algorithms.hpp>
#include <converters/COOtoCSR.cuh>
#include <graph.hpp>

#include "components/scc_matrix.cuh"
#include "topology/topology.cuh"

// do the perf measurements
// enabled by command line parameter s'--perf'
//
static int PERF = 0;

template <typename T>
using DVector = thrust::device_vector<T>;

namespace {  // un-nammed
struct Usecase {
  explicit Usecase(const std::string& a)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
  }

  const std::string& get_matrix_file(void) const { return matrix_file; }

 private:
  std::string matrix_file;
};

// checker of counts of labels for each component
// expensive, for testing purposes only;
//
// params:
// p_d_labels: device array of labels of size nrows;
// nrows: |V| for graph G(V, E);
// d_v_counts: #labels for each component; (_not_ pre-allocated!)
//
template <typename IndexT>
size_t get_component_sizes(const IndexT* p_d_labels, size_t nrows, DVector<size_t>& d_v_counts)
{
  DVector<IndexT> d_sorted_l(p_d_labels, p_d_labels + nrows);
  thrust::sort(d_sorted_l.begin(), d_sorted_l.end());

  size_t counts =
    thrust::distance(d_sorted_l.begin(), thrust::unique(d_sorted_l.begin(), d_sorted_l.end()));

  IndexT* p_d_srt_l = d_sorted_l.data().get();

  d_v_counts.resize(counts);
  thrust::transform(
    thrust::device,
    d_sorted_l.begin(),
    d_sorted_l.begin() + counts,
    d_v_counts.begin(),
    [p_d_srt_l, counts] __device__(IndexT indx) {
      return thrust::count_if(
        thrust::seq, p_d_srt_l, p_d_srt_l + counts, [indx](IndexT label) { return label == indx; });
    });

  // sort the counts:
  thrust::sort(d_v_counts.begin(), d_v_counts.end());

  return counts;
}
}  // namespace

struct Tests_Strongly_CC : ::testing::TestWithParam<Usecase> {
  Tests_Strongly_CC() {}
  static void SetupTestCase() {}
  static void TearDownTestCase()
  {
    if (PERF) {
      for (unsigned int i = 0; i < strongly_cc_time.size(); ++i) {
        std::cout << strongly_cc_time[i] << std::endl;
      }

      std::cout << "#iterations:\n";
      for (auto&& count : strongly_cc_counts) std::cout << count << std::endl;
    }
  }
  virtual void SetUp() {}
  virtual void TearDown() {}

  static std::vector<double> strongly_cc_time;
  static std::vector<int> strongly_cc_counts;

  void run_current_test(const Usecase& param)
  {
    const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    std::string test_id =
      std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) +
      std::string("_") + getFileName(param.get_matrix_file()) + std::string("_") + ss.str().c_str();

    using ByteT  = unsigned char;
    using IndexT = int;

    IndexT m, k, nnz;
    MM_typecode mc;

    HighResClock hr_clock;
    double time_tmp;

    FILE* fpin = fopen(param.get_matrix_file().c_str(), "r");
    ASSERT_NE(fpin, nullptr) << "fopen (" << param.get_matrix_file().c_str() << ") failure.";

    ASSERT_EQ(mm_properties<IndexT>(fpin, 1, &mc, &m, &k, &nnz), 0)
      << "could not read Matrix Market file properties"
      << "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    size_t nrows = static_cast<size_t>(m);
    size_t n2    = 2 * nrows * nrows;

    ASSERT_TRUE(n2 < prop.totalGlobalMem);

    // Allocate memory on host
    std::vector<IndexT> cooRowInd(nnz);
    std::vector<IndexT> cooColInd(nnz);
    std::vector<IndexT> labels(m);  // for G(V, E), m := |V|
    std::vector<IndexT> verts(m);

    // Read: COO Format
    //
    ASSERT_EQ(
      (mm_to_coo<IndexT, IndexT>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], nullptr, nullptr)), 0)
      << "could not read matrix data"
      << "\n";
    ASSERT_EQ(fclose(fpin), 0);

    cugraph::experimental::GraphCOOView<int, int, float> G_coo(&cooRowInd[0], &cooColInd[0], nullptr, m, nnz);
    auto G_unique = cugraph::coo_to_csr(G_coo);
    cugraph::experimental::GraphCSRView<int, int, float> G = G_unique->view();

    rmm::device_vector<int> d_labels(m);

    size_t count = 0;

    if (PERF) {
      hr_clock.start();
      cugraph::connected_components(
        G, cugraph::cugraph_cc_t::CUGRAPH_STRONG, d_labels.data().get());
      cudaDeviceSynchronize();
      hr_clock.stop(&time_tmp);
      strongly_cc_time.push_back(time_tmp);
    } else {
      cudaProfilerStart();
      cugraph::connected_components(
        G, cugraph::cugraph_cc_t::CUGRAPH_STRONG, d_labels.data().get());
      cudaProfilerStop();
      cudaDeviceSynchronize();
    }
    strongly_cc_counts.push_back(count);

    DVector<size_t> d_counts;
    auto count_labels = get_component_sizes(d_labels.data().get(), nrows, d_counts);
  }
};

std::vector<double> Tests_Strongly_CC::strongly_cc_time;
std::vector<int> Tests_Strongly_CC::strongly_cc_counts;

TEST_P(Tests_Strongly_CC, Strongly_CC) { run_current_test(GetParam()); }

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_Strongly_CC,
  ::testing::Values(
    Usecase("test/datasets/cage6.mtx")  // DG "small" enough to meet SCC GPU memory requirements
    ));

int main(int argc, char** argv)
{
  rmmInitialize(nullptr);
  testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  rmmFinalize();
  return rc;
}
