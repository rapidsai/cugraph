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
getMaxDegree(gdf_graph * G) {
      EXPECT_EQ(gdf_add_adj_list(G), 0);
      std::vector<int> out_degree(G->numberOfVertices);
      gdf_column_ptr col_out_degree = create_gdf_column(out_degree);
      EXPECT_EQ(gdf_degree(G, col_out_degree.get(), 2), 0);
      auto degreePtr = thrust::device_pointer_cast(static_cast<int*>(col_out_degree.get()->data));
      cudaStream_t stream = nullptr;
      int max_out_degree = thrust::reduce(rmm::exec_policy(stream)->on(stream),
          degreePtr, degreePtr + col_out_degree.get()->size, static_cast<int>(-1), thrust::maximum<int>());
      return max_out_degree;
}

typedef struct Katz_Usecase_t {
  std::string matrix_file;
  std::vector<int> topVertices;
  Katz_Usecase_t(const std::string& a, const std::vector<int> &top) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
    topVertices = top;
  }
  Katz_Usecase_t& operator=(const Katz_Usecase_t& rhs) {
    matrix_file = rhs.matrix_file;
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
       gdf_graph_ptr G{new gdf_graph, gdf_graph_deleter};
       gdf_column_ptr col_src, col_dest, col_katz_centrality;
       gdf_error status;

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

      ASSERT_EQ(gdf_edge_list_view(G.get(), col_src.get(), col_dest.get(), nullptr),0);
      int max_out_degree = getMaxDegree(G.get());
      double alpha = 1/(static_cast<double>(max_out_degree) + 1);

      status = gdf_katz_centrality(G.get(), col_katz_centrality.get(), alpha, 100, 1e-6, false, true);
      EXPECT_EQ(status,0);

      std::vector<int> top10CUGraph = getTopKIds(std::move(col_katz_centrality));
      std::vector<int> top10Golden  = param.topVertices;

      for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(top10CUGraph[i], top10Golden[i]);
      }
  }

};

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Katz,
                        ::testing::Values(  Katz_Usecase("test/datasets/karate.mtx",      {33 ,0 ,32 ,2 ,1 ,31 ,3 ,8 ,13 ,23}    )
                                           ,Katz_Usecase("test/datasets/netscience.mtx",  {33 ,1429 ,1430 ,1431 ,645 ,1432 ,1433 ,1434 ,1435 ,1436} )
                                           ,Katz_Usecase("test/datasets/polbooks.mtx",    {8 ,12 ,84 ,3 ,73 ,72 ,66 ,30 ,11 ,47}  )
                                           ,Katz_Usecase("test/datasets/dolphins.mtx",    {14 ,37 ,45 ,33 ,51 ,29 ,20 ,40 ,50 ,38}  )
                                         )
                       );

TEST_P(Tests_Katz, Check) {
    run_current_test(GetParam());
}

int main(int argc, char **argv)  {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
