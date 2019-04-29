// -*-c++-*-

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

// coo2csr tests
// Author: Chuck Hastings charlesh@nvidia.com

#include <cudf.h>
#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "COOtoCSR.cuh"
#include "cuda_profiler_api.h"
#include <cugraph.h>
#include "test_utils.h"

#ifdef HAS_RMM
#include <rmm_utils.h>
template<typename T>
using VectorT = thrust::device_vector<T, rmm_allocator<T>>;
#else
template<typename T>
using VectorT = thrust::device_vector<T>;
#endif

// do the perf measurements
// enabled by command line parameter s'--perf'
static int PERF = 0;

template<typename IndexT>
size_t checker(size_t n,
	       const VectorT<IndexT>& di,
	       const VectorT<IndexT>& dj,
	       const IndexT* p_row_offsets,
	       const IndexT* p_col_indices)
{
  size_t nnz{di.size()};
  size_t n1{n+1};

  VectorT<size_t> d_not_founds(nnz, 0);
  const IndexT* pI = di.data().get();
  const IndexT* pJ = dj.data().get();

  thrust::transform(thrust::device,
		    thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nnz),
		    d_not_founds.begin(),
		    [p_row_offsets, p_col_indices, pI, pJ, n1, nnz] __device__ (size_t indx){
		      auto row = pI[indx];
		      auto col = pJ[indx];

		      auto start_pos = p_row_offsets[row];
		      auto after_last_pos = p_row_offsets[row+1];

		      auto begin = p_col_indices + start_pos;
		      auto end = p_col_indices + after_last_pos;

		      auto it = thrust::find(thrust::seq, begin, end, col);
		      return (it == end );
		    });

  size_t errorCount = thrust::reduce(d_not_founds.begin(), d_not_founds.end());
  return errorCount;
}

class Tests_Coo2csr : public ::testing::TestWithParam<std::string> {
public:
  Tests_Coo2csr() {  }
  static void SetupTestCase() {  }
  static void TearDownTestCase() { 
    if (PERF) {
     for (unsigned int i = 0; i < coo2csr_time.size(); ++i) {
      std::cout <<  coo2csr_time[i] << std::endl;
     }
    } 
  }
  virtual void SetUp() {  }
  virtual void TearDown() {  }

  static std::vector<double> coo2csr_time;

  void run_current_test(const std::string& param) {
    const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss; 
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".") + std::string(test_info->name()) + std::string("_") + getFileName(param)+ std::string("_") + ss.str().c_str();
    cudaStream_t stream{nullptr};

    int m, k, nnz;
    MM_typecode mc;
     
    HighResClock hr_clock;
    double time_tmp;

    FILE* fpin = fopen(param.c_str(),"r");
    if (fpin == nullptr) {
      perror("fopen error:");
      std::cerr << "   could not open file " << param.c_str() << std::endl;
    }

    ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_FALSE(mm_is_complex(mc));
    ASSERT_FALSE(mm_is_skew(mc));

    // Allocate memory on host
    std::vector<int> cooRowInd(nnz), cooColInd(nnz);
    std::vector<float> cooVal(nnz);

    // Read
    ASSERT_EQ( (mm_to_coo<int,float>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
    ASSERT_EQ(fclose(fpin),0);

    CSR_Result<int> result;

    int *d_row_indices = nullptr;
    int *d_col_indices = nullptr;

    ASSERT_EQ(cudaMalloc(&d_row_indices, sizeof(int) * nnz), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_col_indices, sizeof(int) * nnz), cudaSuccess);

    cudaMemcpy(d_row_indices, &cooRowInd[0], sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, &cooColInd[0], sizeof(int) * nnz, cudaMemcpyHostToDevice);

    if (PERF)
      hr_clock.start();
    else
      cudaProfilerStart();

    cudaDeviceSynchronize();
    
    //ASSERT_EQ(ConvertCOOtoCSR_segmented_sort(d_row_indices, d_col_indices, nnz, result), GDF_SUCCESS);
    ASSERT_EQ(ConvertCOOtoCSR(d_row_indices, d_col_indices, nnz, result), GDF_SUCCESS);
    if (PERF) {
      cudaDeviceSynchronize();
      hr_clock.stop(&time_tmp);
      coo2csr_time.push_back(time_tmp);
      std::cout << "file = " << param << ", nnz = " << nnz << ", convert time = " << time_tmp << std::endl;
    } else {
      cudaProfilerStop();
      cudaDeviceSynchronize();
    }

    //
    // Check result
    //
    VectorT<int> di(nnz);
    VectorT<int> dj(nnz);

    thrust::copy(cooRowInd.begin(), cooRowInd.end(), di.begin());
    thrust::copy(cooColInd.begin(), cooColInd.end(), dj.begin());

    int errorCount = checker((size_t) nnz, di, dj, result.rowOffsets, result.colIndices);

    EXPECT_EQ(errorCount, 0);

    ALLOC_FREE_TRY(result.rowOffsets, stream);
    ALLOC_FREE_TRY(result.colIndices, stream);
    ALLOC_FREE_TRY(d_row_indices, stream);
    ALLOC_FREE_TRY(d_col_indices, stream);
  }
};
 
std::vector<double> Tests_Coo2csr::coo2csr_time;

TEST_P(Tests_Coo2csr, Coo2Csr) {
    run_current_test(GetParam());
}

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test, Tests_Coo2csr, 
			::testing::Values("/datasets/networks/karate.mtx"
					  ,"/datasets/golden_data/graphs/cit-Patents.mtx"
					  ,"/datasets/golden_data/graphs/ljournal-2008.mtx"
					  ,"/datasets/golden_data/graphs/webbase-1M.mtx"
					  ,"/datasets/golden_data/graphs/web-BerkStan.mtx"
					  ,"/datasets/golden_data/graphs/web-Google.mtx"
					  ,"/datasets/golden_data/graphs/wiki-Talk.mtx"

					  //   Test for really big graph
					  //,"/datasets/chuck/graph500-scale25-ef16_adj.mmio"
					  ));


int main(int argc, char **argv)  {
    srand(42);
    ::testing::InitGoogleTest(&argc, argv);
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--perf") == 0)
            PERF = 1;
    }

  return RUN_ALL_TESTS();
}


