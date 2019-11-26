#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gmock/gmock-generated-matchers.h"
#include "high_res_clock.h"
#include "cuda_profiler_api.h"
#include <cugraph.h>
#include "test_utils.h"
#include <thrust/device_ptr.h>
#include <fstream>

std::vector<int>
getGoldenTopKIds(std::string file, int k = 10) {
  std::vector<int> vec;
  std::ifstream fin(file);
  int val;
  int count = 0;
  while (fin>>val && ((count++) < k)) {
    vec.push_back(val);
  }
  vec.resize(k);
  return vec;
}

std::vector<int>
getTopKIds(gdf_column_ptr katz, int k = 10) {
  int count = katz.get()->size;
  cudaStream_t stream = nullptr;
  rmm::device_vector<int> id(count);
  thrust::sequence(rmm::exec_policy(stream)->on(stream), id.begin(), id.end());
  auto colptr = thrust::device_pointer_cast(static_cast<double*>(katz.get()->data));
  thrust::sort_by_key(rmm::exec_policy(stream)->on(stream),
      colptr, colptr + count, id.begin(), thrust::greater<double>());
  std::vector<int> topK(k);
  thrust::copy(id.begin(), id.begin() + k, topK.begin());
  return topK;
}

int
getMaxDegree(cugraph::Graph * G) {
      cugraph::add_adj_list(G);
      std::vector<int> out_degree(G->numberOfVertices);
      gdf_column_ptr col_out_degree = create_gdf_column(out_degree);
      cugraph::degree(G, col_out_degree.get(), 2);
      auto degreePtr = thrust::device_pointer_cast(static_cast<int*>(col_out_degree.get()->data));
      cudaStream_t stream = nullptr;
      int max_out_degree = thrust::reduce(rmm::exec_policy(stream)->on(stream),
          degreePtr, degreePtr + col_out_degree.get()->size, static_cast<int>(-1), thrust::maximum<int>());
      return max_out_degree;
}

typedef struct Katz_Usecase_t {
  std::string matrix_file;
  std::string result_file;
  Katz_Usecase_t(const std::string& a, const std::string& b) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
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
  Katz_Usecase_t& operator=(const Katz_Usecase_t& rhs) {
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

  void run_current_test(const Katz_Usecase& param) {
       Graph_ptr G{new cugraph::Graph, Graph_deleter};
       gdf_column_ptr col_src, col_dest, col_katz_centrality;

       FILE* fpin = fopen(param.matrix_file.c_str(),"r");
       ASSERT_NE(fpin, nullptr) << "fopen (" << param.matrix_file << ") failure.";

       int m, k;
       int nnz;
       MM_typecode mc;
       ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
       ASSERT_TRUE(mm_is_matrix(mc));
       ASSERT_TRUE(mm_is_coordinate(mc));
       ASSERT_FALSE(mm_is_complex(mc));
       ASSERT_FALSE(mm_is_skew(mc));

       // Allocate memory on host
       std::vector<int> cooRowInd(nnz), cooColInd(nnz);
       std::vector<int> cooVal(nnz);
       std::vector<double> katz_centrality(m);

       // Read
       ASSERT_EQ( (mm_to_coo<int,int>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
       ASSERT_EQ(fclose(fpin),0);

      // gdf columns
      col_src = create_gdf_column(cooRowInd);
      col_dest = create_gdf_column(cooColInd);
      col_katz_centrality = create_gdf_column(katz_centrality);

      cugraph::edge_list_view(G.get(), col_src.get(), col_dest.get(), nullptr);
      int max_out_degree = getMaxDegree(G.get());
      double alpha = 1/(static_cast<double>(max_out_degree) + 1);

      cugraph::katz_centrality(G.get(), col_katz_centrality.get(), alpha, 100, 1e-6, false, true);

      std::vector<int> top10CUGraph = getTopKIds(std::move(col_katz_centrality));
      std::vector<int> top10Golden  = getGoldenTopKIds(param.result_file);

      EXPECT_THAT(top10CUGraph, ::testing::ContainerEq(top10Golden));
  }

};

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Katz,
                        ::testing::Values(  Katz_Usecase("test/datasets/karate.mtx",      "ref/katz/karate.csv"    )
                                           ,Katz_Usecase("test/datasets/netscience.mtx",  "ref/katz/netscience.csv")
                                           ,Katz_Usecase("test/datasets/polbooks.mtx",    "ref/katz/polbooks.csv"  )
                                           ,Katz_Usecase("test/datasets/dolphins.mtx",    "ref/katz/dolphins.csv"  )
                                         )
                       );

TEST_P(Tests_Katz, Check) {
    run_current_test(GetParam());
}

int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}