// This is gtest application that contains all of the C API tests. Parameters:
// nvgraph_capi_tests [--perf] [--stress-iters N] [--gtest_filter=NameFilterPatter]
// It also accepts any other gtest (1.7.0) default parameters.
// Right now this application contains:
// 1) Sanity Check tests - tests on simple examples with known answer (or known behaviour)
// 2) Correctness checks tests - tests on real graph data, uses reference algorithm
//    (CPU code for SrSPMV and python scripts for other algorithms, see
//    python scripts here: //sw/gpgpu/nvgraph/test/ref/) with reference results, compares those two.
//    It also measures performance of single algorithm C API call, enf enabled (see below)
// 3) Corner cases tests - tests with some bad inputs, bad parameters, expects library to handle
//    it gracefully
// 4) Stress tests - makes sure that library result is persistent throughout the library usage
//    (a lot of C API calls). Also makes some assumptions and checks on memory usage during
//    this test.
//
// We can control what tests to launch by using gtest filters. For example:
// Only sanity tests:
//    ./nvgraph_capi_tests_traversal --gtest_filter=*Sanity*
// And, correspondingly:
//    ./nvgraph_capi_tests_traversal --gtest_filter=*Correctness*
//    ./nvgraph_capi_tests_traversal --gtest_filter=*Corner*
//    ./nvgraph_capi_tests_traversal --gtest_filter=*Stress*
// Or, combination:
//    ./nvgraph_capi_tests_traversal --gtest_filter=*Sanity*:*Correctness*
//
// Performance reports are provided in the ERIS format and disabled by default.
// Could be enabled by adding '--perf' to the command line. I added this parameter to vlct
//
// Parameter '--stress-iters N', which gives multiplier (not an absolute value) for the number of launches for stress tests
//

#include <utility>

#include "gtest/gtest.h"

#include "nvgraph_test_common.h"

#include "valued_csr_graph.hxx"
#include "readMatrix.hxx"
#include "nvgraphP.h"
#include "nvgraph.h"
#include <nvgraph_experimental.h>  // experimental header, contains hidden API entries, can be shared only under special circumstances without reveling internal things

#include "stdlib.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <sstream>
#include <cstdint>
#include <math.h>
#include "cuda_profiler_api.h"

// do the perf measurements, enabled by command line parameter '--perf'
static int PERF = 0;

// minimum vertices in the graph to perform perf measurements
#define PERF_ROWS_LIMIT 10000

// number of repeats = multiplier/num_vertices
#define Traversal_ITER_MULTIPLIER     30000000

template<typename T>
struct nvgraph_Const;

template<>
struct nvgraph_Const<int>
{
	static const cudaDataType_t Type = CUDA_R_32I;
	static const int inf;
};
const int nvgraph_Const<int>::inf = INT_MAX;

typedef struct Traversal_Usecase_t
{
	std::string graph_file;
	int source_vert;
	size_t n;
	size_t nnz;
	bool useMask;
	bool undirected;

	Traversal_Usecase_t(const std::string& a,
	                    int source,
	                    size_t _n,
	                    size_t _nnz,
	                    bool _useMask = false,
	                    bool _undirected = false) :
			source_vert(source), n(_n), nnz(_nnz), useMask(_useMask), undirected(_undirected) {
		graph_file = a;
	};

	Traversal_Usecase_t& operator=(const Traversal_Usecase_t& rhs)
												{
		graph_file = rhs.graph_file;
		source_vert = rhs.source_vert;
		n = rhs.n;
		nnz = rhs.nnz;
		useMask = rhs.useMask;
		return *this;
	}
} Traversal_Usecase;

//// Traversal tests

class NVGraphCAPITests_2d_bfs: public ::testing::TestWithParam<Traversal_Usecase> {
public:
	NVGraphCAPITests_2d_bfs() :
			handle(NULL) {
	}

	static void SetupTestCase() {
	}
	static void TearDownTestCase() {
	}
	virtual void SetUp() {
		if (handle == NULL) {
			char* nvgraph_gpus = getenv("NVGRAPH_GPUS");
			if (nvgraph_gpus)
				printf("Value of NVGRAPH_GPUS=%s\n", nvgraph_gpus);
			else
				printf("Value of NVGRAPH_GPUS is null\n");
			std::vector<int32_t> gpus;
			int32_t dummy;
			std::stringstream ss(nvgraph_gpus);
			while (ss >> dummy){
				gpus.push_back(dummy);
				if (ss.peek() == ',')
					ss.ignore();
			}
			printf("There were %d devices found: ", (int)gpus.size());
			for (int i = 0; i < gpus.size(); i++)
				std::cout << gpus[i] << "  ";
			std::cout << "\n";

			devices = (int32_t*) malloc(sizeof(int32_t) * gpus.size());
			for (int i = 0; i < gpus.size(); i++)
				devices[i] = gpus[i];
			numDevices = gpus.size();

			status = nvgraphCreateMulti(&handle, numDevices, devices);
			ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		}
	}
	virtual void TearDown() {
		if (handle != NULL) {
			status = nvgraphDestroy(handle);
			ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
			handle = NULL;
			if (devices)
				free(devices);
		}
	}
	nvgraphStatus_t status;
	nvgraphHandle_t handle;
	int32_t *devices;
	int32_t numDevices;

	template<typename EdgeT>
	void run_current_test(const Traversal_Usecase& param) {
		const ::testing::TestInfo* const test_info =
				::testing::UnitTest::GetInstance()->current_test_info();
		std::stringstream ss;
		ss << param.source_vert;
		std::string test_id = std::string(test_info->test_case_name()) + std::string(".")
				+ std::string(test_info->name()) + std::string("_") + getFileName(param.graph_file)
				+ std::string("_") + ss.str().c_str();

		nvgraphTopologyType_t topo = NVGRAPH_2D_32I_32I;
		nvgraphStatus_t status;

		// Read in graph from network file
		std::vector<int32_t> sources;
		std::vector<int32_t> destinations;
		readNetworkFile(param.graph_file.c_str(), param.nnz, sources, destinations);

		// Create graph handle
		nvgraphGraphDescr_t g1 = NULL;
		status = nvgraphCreateGraphDescr(handle, &g1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		// set up graph
		int n = param.n;
		int nnz = param.nnz;
		int blockN = std::max(2,(int)ceil(sqrt(numDevices)));
		std::cout << "Using " << blockN << " as block N\n";

		nvgraph2dCOOTopology32I_st topology = { n, nnz, &sources[0], &destinations[0], CUDA_R_32I,
		NULL, blockN, devices, numDevices, NVGRAPH_DEFAULT };
		status = nvgraphSetGraphStructure(handle, g1, (void*) &topology, topo);

		// set up graph data
		std::vector<int> calculated_distances_res(n);
		std::vector<int> calculated_predecessors_res(n);

		int source_vert = param.source_vert;
		std::cout << "Starting from vertex: " << source_vert << "\n";
		cudaProfilerStart();
		status = nvgraph2dBfs(handle,
										g1,
										source_vert,
										&calculated_distances_res[0],
										&calculated_predecessors_res[0]);
		cudaProfilerStop();
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		cudaDeviceSynchronize();

		if (PERF && n > PERF_ROWS_LIMIT)	{
			double start, stop;
			start = second();
			int repeat = 30;
			for (int i = 0; i < repeat; i++) {
				status = nvgraph2dBfs(handle,
												g1,
												source_vert,
												&calculated_distances_res[0],
												&calculated_predecessors_res[0]);
				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
			}
			cudaDeviceSynchronize();
			stop = second();
			printf("&&&& PERF Time_%s %10.8f -ms\n",
						test_id.c_str(),
						1000.0 * (stop - start) / repeat);
		}

		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//Checking distances
		int visitedCount = 0;
		for (int i = 0; i < n; ++i) {
			if (calculated_distances_res[i] != -1)
				visitedCount++;
		}
		std::cout << "There were " << visitedCount << " vertices visited.\n";

		status = nvgraphDestroyGraphDescr(handle, g1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}
};

TEST_P(NVGraphCAPITests_2d_bfs, CheckResult) {
	run_current_test<float>(GetParam());
}

INSTANTIATE_TEST_CASE_P(CorrectnessCheck,
								NVGraphCAPITests_2d_bfs,
								::testing::Values(
										Traversal_Usecase("/mnt/nvgraph_test_data/Rmat100Mvertices2Bedges.net", 3, 100000000, 2000000000)
								));

int main(int argc, char **argv)
			{

	for (int i = 0; i < argc; i++)
			{
		if (strcmp(argv[i], "--perf") == 0)
			PERF = 1;
	}
	srand(42);
	::testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}
