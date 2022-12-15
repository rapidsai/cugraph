/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <components/legacy/scc_matrix.cuh>
#include <converters/legacy/COOtoCSR.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <rmm/device_vector.hpp>

#include <cuda_profiler_api.h>

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <iterator>

template <typename T>
using DVector = rmm::device_vector<T>;

namespace {  // un-nammed
struct Usecase {
  explicit Usecase(const std::string& a)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
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

// counts number of vertices in each component;
// (of same label);
// potentially expensive, for testing purposes only;
//
// params:
// in: p_d_labels: device array of labels of size nrows;
// in: nrows: |V| for graph G(V, E);
// out: d_v_counts: #labels for each component; (_not_ pre-allocated!)
// return: number of components;
//
template <typename IndexT>
size_t get_component_sizes(const IndexT* p_d_labels,
                           size_t nrows,
                           DVector<size_t>& d_num_vs_per_component)
{
  DVector<IndexT> d_sorted_l(p_d_labels, p_d_labels + nrows);
  thrust::sort(d_sorted_l.begin(), d_sorted_l.end());

  auto pair_it = thrust::reduce_by_key(d_sorted_l.begin(),
                                       d_sorted_l.end(),
                                       thrust::make_constant_iterator<size_t>(1),
                                       thrust::make_discard_iterator(),  // ignore...
                                       d_num_vs_per_component.begin());

  size_t counts = thrust::distance(d_num_vs_per_component.begin(), pair_it.second);

  d_num_vs_per_component.resize(counts);
  return counts;
}

template <typename ByteT, typename IndexT>
DVector<IndexT> byte_matrix_to_int(const DVector<ByteT>& d_adj_byte_matrix)
{
  auto n2 = d_adj_byte_matrix.size();
  thrust::device_vector<IndexT> d_vec_matrix(n2, 0);
  thrust::transform(d_adj_byte_matrix.begin(),
                    d_adj_byte_matrix.end(),
                    d_vec_matrix.begin(),
                    [] __device__(auto byte_v) { return static_cast<IndexT>(byte_v); });
  return d_vec_matrix;
}

}  // namespace

struct Tests_Strongly_CC : ::testing::TestWithParam<Usecase> {
  Tests_Strongly_CC() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase()
  {
    if (cugraph::test::g_perf) {
      for (unsigned int i = 0; i < strongly_cc_time.size(); ++i) {
        std::cout << strongly_cc_time[i] << std::endl;
      }

      std::cout << "#iterations:\n";
      for (auto&& count : strongly_cc_counts)
        std::cout << count << std::endl;
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
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".") +
                          std::string(test_info->name()) + std::string("_") +
                          cugraph::test::getFileName(param.get_matrix_file()) + std::string("_") +
                          ss.str().c_str();

    using ByteT  = unsigned char;
    using IndexT = int;

    IndexT m, k, nnz;
    MM_typecode mc;

    HighResTimer hr_timer{};

    FILE* fpin = fopen(param.get_matrix_file().c_str(), "r");
    ASSERT_NE(fpin, nullptr) << "fopen (" << param.get_matrix_file().c_str() << ") failure.";

    ASSERT_EQ(cugraph::test::mm_properties<IndexT>(fpin, 1, &mc, &m, &k, &nnz), 0)
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
    std::vector<IndexT> labels(nrows);  // for G(V, E), m := |V|
    std::vector<IndexT> verts(nrows);

    // Read: COO Format
    //
    ASSERT_EQ((cugraph::test::mm_to_coo<IndexT, IndexT>(
                fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], nullptr, nullptr)),
              0)
      << "could not read matrix data"
      << "\n";
    ASSERT_EQ(fclose(fpin), 0);

    cugraph::legacy::GraphCOOView<int, int, float> G_coo(
      &cooRowInd[0], &cooColInd[0], nullptr, nrows, nnz);
    auto G_unique                                    = cugraph::coo_to_csr(G_coo);
    cugraph::legacy::GraphCSRView<int, int, float> G = G_unique->view();

    rmm::device_vector<int> d_labels(nrows);

    size_t count = 0;

    if (cugraph::test::g_perf) {
      hr_timer.start("SCC");
      cugraph::connected_components(
        G, cugraph::cugraph_cc_t::CUGRAPH_STRONG, d_labels.data().get());
      cudaDeviceSynchronize();
      auto time_tmp = hr_timer.stop();
      strongly_cc_time.push_back(time_tmp);
    } else {
      cudaProfilerStart();
      cugraph::connected_components(
        G, cugraph::cugraph_cc_t::CUGRAPH_STRONG, d_labels.data().get());
      cudaProfilerStop();
      cudaDeviceSynchronize();
    }
    strongly_cc_counts.push_back(count);

    DVector<size_t> d_counts(nrows);
    auto count_labels = get_component_sizes(d_labels.data().get(), nrows, d_counts);
  }
};

std::vector<double> Tests_Strongly_CC::strongly_cc_time;
std::vector<int> Tests_Strongly_CC::strongly_cc_counts;

TEST_P(Tests_Strongly_CC, Strongly_CC) { run_current_test(GetParam()); }

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_Strongly_CC,
  ::testing::Values(
    Usecase("test/datasets/cage6.mtx")  // DG "small" enough to meet SCC GPU memory requirements
    ));

struct SCCSmallTest : public ::testing::Test {
};

// FIXME: we should take advantage of gtest parameterization over copy-and-paste reuse.
//
TEST_F(SCCSmallTest, CustomGraphSimpleLoops)
{
  using IndexT = int;

  size_t nrows = 5;
  size_t n2    = 2 * nrows * nrows;

  cudaDeviceProp prop;
  int device = 0;
  cudaGetDeviceProperties(&prop, device);

  ASSERT_TRUE(n2 < prop.totalGlobalMem);

  // Allocate memory on host
  std::vector<IndexT> cooRowInd{0, 1, 2, 3, 3, 4};
  std::vector<IndexT> cooColInd{1, 0, 0, 1, 4, 3};
  std::vector<IndexT> labels(nrows);
  std::vector<IndexT> verts(nrows);

  size_t nnz = cooRowInd.size();

  EXPECT_EQ(nnz, cooColInd.size());

  cugraph::legacy::GraphCOOView<int, int, float> G_coo(
    &cooRowInd[0], &cooColInd[0], nullptr, nrows, nnz);
  auto G_unique                                    = cugraph::coo_to_csr(G_coo);
  cugraph::legacy::GraphCSRView<int, int, float> G = G_unique->view();

  rmm::device_vector<IndexT> d_labels(nrows);

  cugraph::connected_components(G, cugraph::cugraph_cc_t::CUGRAPH_STRONG, d_labels.data().get());

  DVector<size_t> d_counts(nrows);
  auto count_components = get_component_sizes(d_labels.data().get(), nrows, d_counts);

  EXPECT_EQ(count_components, static_cast<size_t>(3));

  std::vector<size_t> v_counts(d_counts.size());

  cudaMemcpy(v_counts.data(),
             d_counts.data().get(),
             sizeof(size_t) * v_counts.size(),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  std::vector<size_t> v_counts_exp{2, 1, 2};

  EXPECT_EQ(v_counts, v_counts_exp);
}

TEST_F(SCCSmallTest, /*DISABLED_*/ CustomGraphWithSelfLoops)
{
  using IndexT = int;

  size_t nrows = 5;
  size_t n2    = 2 * nrows * nrows;

  cudaDeviceProp prop;
  int device = 0;
  cudaGetDeviceProperties(&prop, device);

  ASSERT_TRUE(n2 < prop.totalGlobalMem);

  // Allocate memory on host
  std::vector<IndexT> cooRowInd{0, 0, 1, 1, 2, 2, 3, 3, 4};
  std::vector<IndexT> cooColInd{0, 1, 0, 1, 0, 2, 1, 3, 4};
  std::vector<IndexT> labels(nrows);
  std::vector<IndexT> verts(nrows);

  size_t nnz = cooRowInd.size();

  EXPECT_EQ(nnz, cooColInd.size());

  cugraph::legacy::GraphCOOView<int, int, float> G_coo(
    &cooRowInd[0], &cooColInd[0], nullptr, nrows, nnz);
  auto G_unique                                    = cugraph::coo_to_csr(G_coo);
  cugraph::legacy::GraphCSRView<int, int, float> G = G_unique->view();

  rmm::device_vector<IndexT> d_labels(nrows);

  cugraph::connected_components(G, cugraph::cugraph_cc_t::CUGRAPH_STRONG, d_labels.data().get());

  DVector<size_t> d_counts(nrows);
  auto count_components = get_component_sizes(d_labels.data().get(), nrows, d_counts);

  EXPECT_EQ(count_components, static_cast<size_t>(4));

  std::vector<size_t> v_counts(d_counts.size());

  cudaMemcpy(v_counts.data(),
             d_counts.data().get(),
             sizeof(size_t) * v_counts.size(),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  std::vector<size_t> v_counts_exp{2, 1, 1, 1};

  EXPECT_EQ(v_counts, v_counts_exp);
}

TEST_F(SCCSmallTest, SmallGraphWithSelfLoops1)
{
  using IndexT = int;

  size_t nrows = 3;

  std::vector<IndexT> cooRowInd{0, 0, 1, 2};
  std::vector<IndexT> cooColInd{0, 1, 0, 0};

  std::vector<size_t> v_counts_exp{2, 1};

  std::vector<IndexT> labels(nrows);
  std::vector<IndexT> verts(nrows);

  size_t nnz = cooRowInd.size();

  EXPECT_EQ(nnz, cooColInd.size());

  cugraph::legacy::GraphCOOView<int, int, float> G_coo(
    &cooRowInd[0], &cooColInd[0], nullptr, nrows, nnz);
  auto G_unique                                    = cugraph::coo_to_csr(G_coo);
  cugraph::legacy::GraphCSRView<int, int, float> G = G_unique->view();

  rmm::device_vector<IndexT> d_labels(nrows);

  cugraph::connected_components(G, cugraph::cugraph_cc_t::CUGRAPH_STRONG, d_labels.data().get());

  DVector<size_t> d_counts(nrows);
  auto count_components = get_component_sizes(d_labels.data().get(), nrows, d_counts);

  // std::cout << "vertex labels:\n";
  // print_v(d_labels, std::cout);

  decltype(count_components) num_components_exp = 2;

  EXPECT_EQ(count_components, num_components_exp);
}

TEST_F(SCCSmallTest, SmallGraphWithIsolated)
{
  using IndexT = int;

  size_t nrows = 3;

  std::vector<IndexT> cooRowInd{0, 0, 1};
  std::vector<IndexT> cooColInd{0, 1, 0};

  std::vector<size_t> v_counts_exp{2, 1};

  std::vector<IndexT> labels(nrows);
  std::vector<IndexT> verts(nrows);

  size_t nnz = cooRowInd.size();

  EXPECT_EQ(nnz, cooColInd.size());

  // Note: there seems to be a BUG in coo_to_csr() or view()
  // COO format doesn't account for isolated vertices;
  //
  // cugraph::legacy::GraphCOOView<int, int, float> G_coo(&cooRowInd[0], &cooColInd[0], nullptr,
  // nrows, nnz); auto G_unique                            = cugraph::coo_to_csr(G_coo);
  // cugraph::legacy::GraphCSRView<int, int, float> G = G_unique->view();
  //
  //
  // size_t num_vertices = G.number_of_vertices;
  // size_t num_edges    = G.number_of_edges;
  //
  // EXPECT_EQ(num_vertices, nrows); //fails when G was constructed from COO
  // EXPECT_EQ(num_edges, nnz);

  std::vector<IndexT> ro{0, 2, 3, 3};
  std::vector<IndexT> ci{0, 1, 0};

  nnz = ci.size();

  thrust::device_vector<IndexT> d_ro(ro);
  thrust::device_vector<IndexT> d_ci(ci);

  cugraph::legacy::GraphCSRView<int, int, float> G{
    d_ro.data().get(), d_ci.data().get(), nullptr, static_cast<int>(nrows), static_cast<int>(nnz)};

  size_t num_vertices = G.number_of_vertices;
  size_t num_edges    = G.number_of_edges;

  EXPECT_EQ(num_vertices, nrows);
  EXPECT_EQ(num_edges, nnz);

  rmm::device_vector<IndexT> d_labels(nrows);

  cugraph::connected_components(G, cugraph::cugraph_cc_t::CUGRAPH_STRONG, d_labels.data().get());

  DVector<size_t> d_counts(nrows);
  auto count_components = get_component_sizes(d_labels.data().get(), nrows, d_counts);

  // std::cout << "vertex labels:\n";
  // print_v(d_labels, std::cout);

  decltype(count_components) num_components_exp = 2;

  EXPECT_EQ(count_components, num_components_exp);
}

CUGRAPH_TEST_PROGRAM_MAIN()
