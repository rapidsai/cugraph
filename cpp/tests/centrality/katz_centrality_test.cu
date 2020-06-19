#include <thrust/device_ptr.h>
#include <algorithms.hpp>
#include <converters/COOtoCSR.cuh>
#include <fstream>
#include <graph.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include "cuda_profiler_api.h"
#include "gmock/gmock-generated-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "utilities/high_res_clock.h"

#include "utilities/test_utilities.hpp"

std::vector<int> getGoldenTopKIds(std::ifstream& fs_result, int k = 10)
{
  std::vector<int> vec;
  int val;
  int count = 0;
  while (fs_result >> val && ((count++) < k)) { vec.push_back(val); }
  vec.resize(k);
  return vec;
}

std::vector<int> getTopKIds(double* p_katz, int count, int k = 10)
{
  cudaStream_t stream = nullptr;
  rmm::device_vector<int> id(count);
  thrust::sequence(rmm::exec_policy(stream)->on(stream), id.begin(), id.end());
  thrust::sort_by_key(rmm::exec_policy(stream)->on(stream),
                      p_katz,
                      p_katz + count,
                      id.begin(),
                      thrust::greater<double>());
  std::vector<int> topK(k);
  thrust::copy(id.begin(), id.begin() + k, topK.begin());
  return topK;
}

template <typename VT, typename ET, typename WT>
int getMaxDegree(cugraph::experimental::GraphCSRView<VT, ET, WT> const& g)
{
  cudaStream_t stream{nullptr};

  rmm::device_vector<ET> degree_vector(g.number_of_vertices);
  ET* p_degree = degree_vector.data().get();
  g.degree(p_degree, cugraph::experimental::DegreeDirection::OUT);
  ET max_out_degree = thrust::reduce(rmm::exec_policy(stream)->on(stream),
                                     p_degree,
                                     p_degree + g.number_of_vertices,
                                     static_cast<ET>(-1),
                                     thrust::maximum<ET>());
  return max_out_degree;
}

typedef struct Katz_Usecase_t {
  std::string matrix_file;
  std::string result_file;
  Katz_Usecase_t(const std::string& a, const std::string& b)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
    if ((b != "") && (b[0] != '/')) {
      result_file = rapidsDatasetRootDir + "/" + b;
    } else {
      result_file = b;
    }
  }
  Katz_Usecase_t& operator=(const Katz_Usecase_t& rhs)
  {
    matrix_file = rhs.matrix_file;
    result_file = rhs.result_file;
    return *this;
  }
} Katz_Usecase;

class Tests_Katz : public ::testing::TestWithParam<Katz_Usecase> {
 public:
  Tests_Katz() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  void run_current_test(const Katz_Usecase& param)
  {
    FILE* fpin = fopen(param.matrix_file.c_str(), "r");
    ASSERT_NE(fpin, nullptr) << "fopen (" << param.matrix_file << ") failure.";

    std::ifstream fs_result(param.result_file);
    ASSERT_EQ(fs_result.is_open(), true) << "file open (" << param.result_file << ") failure.";

    int m, k;
    int nnz;
    MM_typecode mc;
    ASSERT_EQ(cugraph::test::mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz), 0)
      << "could not read Matrix Market file properties"
      << "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_FALSE(mm_is_complex(mc));
    ASSERT_FALSE(mm_is_skew(mc));

    // Allocate memory on host
    std::vector<int> cooRowInd(nnz), cooColInd(nnz);
    std::vector<int> cooVal(nnz);
    std::vector<double> katz_centrality(m);

    // Read
    ASSERT_EQ((cugraph::test::mm_to_coo<int, int>(
                fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)),
              0)
      << "could not read matrix data"
      << "\n";
    ASSERT_EQ(fclose(fpin), 0);

    cugraph::experimental::GraphCOOView<int, int, float> cooview(
      &cooColInd[0], &cooRowInd[0], nullptr, m, nnz);
    auto csr                                               = cugraph::coo_to_csr(cooview);
    cugraph::experimental::GraphCSRView<int, int, float> G = csr->view();

    rmm::device_vector<double> katz_vector(m);
    double* d_katz = thrust::raw_pointer_cast(katz_vector.data());

    int max_out_degree = getMaxDegree(G);
    double alpha       = 1 / (static_cast<double>(max_out_degree) + 1);

    cugraph::katz_centrality(G, d_katz, alpha, 100, 1e-6, false, true);

    std::vector<int> top10CUGraph = getTopKIds(d_katz, m);
    std::vector<int> top10Golden  = getGoldenTopKIds(fs_result);

    EXPECT_THAT(top10CUGraph, ::testing::ContainerEq(top10Golden));
  }
};

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_Katz,
  ::testing::Values(Katz_Usecase("test/datasets/karate.mtx", "ref/katz/karate.csv"),
                    Katz_Usecase("test/datasets/netscience.mtx", "ref/katz/netscience.csv"),
                    Katz_Usecase("test/datasets/polbooks.mtx", "ref/katz/polbooks.csv"),
                    Katz_Usecase("test/datasets/dolphins.mtx", "ref/katz/dolphins.csv")));

TEST_P(Tests_Katz, Check) { run_current_test(GetParam()); }

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}
