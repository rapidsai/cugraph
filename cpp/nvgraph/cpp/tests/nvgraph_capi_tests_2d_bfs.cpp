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
#include <cstdint>
#include <math.h>

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

static std::string ref_data_prefix = "";
static std::string graph_data_prefix = "";

// iterations for stress tests = this multiplier * iterations for perf tests
static int STRESS_MULTIPLIER = 10;

void offsetsToIndices(std::vector<int>& offsets, std::vector<int>& indices) {
	int nnz = offsets.back();
	indices.resize(nnz);
	int n = offsets.size() - 1;
	for (int row = 0; row < n; row++) {
		for (int pos = offsets[row]; pos < offsets[row + 1]; pos++)
			indices[pos] = row;
	}
}

bool enough_device_memory(int n, int nnz, size_t add)
									{
	size_t mtotal, mfree;
	cudaMemGetInfo(&mfree, &mtotal);
	if (mfree > add + sizeof(int) * (4 * n)) //graph + pred + distances + 2n (working data)
		return true;
	return false;
}

std::string convert_to_local_path(const std::string& in_file)
												{
	std::string wstr = in_file;
	if ((wstr != "dummy") & (wstr != ""))
			{
		std::string prefix;
		if (graph_data_prefix.length() > 0)
				{
			prefix = graph_data_prefix;
		}
		else
		{
#ifdef _WIN32
			//prefix = "C:\\mnt\\eris\\test\\matrices_collection\\";
			prefix = "Z:\\matrices_collection\\";
			std::replace(wstr.begin(), wstr.end(), '/', '\\');
#else
			prefix = "/mnt/nvgraph_test_data/";
#endif
		}
		wstr = prefix + wstr;
	}
	return wstr;
}

void ref_bfs(int n,
					int nnz,
					int *rowPtr,
					int *colInd,
					int *mask,
					int source_vertex,
					int *distances) {
	for (int i = 0; i != n; ++i)
		distances[i] = -1;

	std::queue<int> q;
	q.push(source_vertex);
	distances[source_vertex] = 0;

	while (!q.empty()) {
		int u = q.front();
		q.pop();

		for (int iCol = rowPtr[u]; iCol != rowPtr[u + 1]; ++iCol) {
			if (mask && !mask[iCol])
				continue;
			int v = colInd[iCol];
			if (distances[v] == -1) { //undiscovered
				distances[v] = distances[u] + 1;
				q.push(v);
			}
		}

	}
}

typedef struct Traversal_Usecase_t
{
	std::string graph_file;
	int source_vert;
	bool useMask;
	bool undirected;

	Traversal_Usecase_t(const std::string& a, int b, bool _useMask = false, bool _undirected = false) :
			source_vert(b), useMask(_useMask), undirected(_undirected) {
		graph_file = convert_to_local_path(a);
	}
	;

	Traversal_Usecase_t& operator=(const Traversal_Usecase_t& rhs)
												{
		graph_file = rhs.graph_file;
		source_vert = rhs.source_vert;
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
			while (ss >> dummy) {
				gpus.push_back(dummy);
				if (ss.peek() == ',')
					ss.ignore();
			}
			printf("There were %d devices found: ", (int) gpus.size());
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

		FILE* fpin = fopen(param.graph_file.c_str(), "rb");
		ASSERT_TRUE(fpin != NULL)<< "Cannot read input graph file: " << param.graph_file << std::endl;
		int n, nnz;
		//Read a network in amgx binary format
		ASSERT_EQ(read_header_amgx_csr_bin(fpin, n, nnz), 0);
		std::vector<int> read_row_ptr(n + 1), read_col_ind(nnz);
		std::vector<EdgeT> csr_read_val(nnz);
		std::cout << getFileName(param.graph_file) << " Vertices: " << n << " Edges: " << nnz << "\n";
		ASSERT_EQ(read_data_amgx_csr_bin(fpin, n, nnz, read_row_ptr, read_col_ind, csr_read_val), 0);
		fclose(fpin);
		std::vector<int> row_ind;
		offsetsToIndices(read_row_ptr, row_ind);

		std::vector<int> csr_mask(nnz, 1);

		if (param.useMask) {
			//Generating a mask
			//Should be improved
			for (int i = 0; i < nnz; i += 2)
				csr_mask[i] = 0;
		}

		if (!enough_device_memory(n, nnz, sizeof(int) * (read_row_ptr.size() + read_col_ind.size())))
													{
			std::cout << "[  WAIVED  ] " << test_info->test_case_name() << "." << test_info->name()
					<< std::endl;
			return;
		}

		nvgraphGraphDescr_t g1 = NULL;
		status = nvgraphCreateGraphDescr(handle, &g1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		// set up graph
		int32_t blockN = std::max(2,(int)ceil(sqrt(numDevices)));
		nvgraph2dCOOTopology32I_st topology = { n, nnz, &row_ind[0], &read_col_ind[0], CUDA_R_32I,
		NULL, numDevices, devices, blockN, NVGRAPH_DEFAULT };
		status = nvgraphSetGraphStructure(handle, g1, (void*) &topology, topo);

		// set up graph data
		std::vector<int> calculated_distances_res(n);
		std::vector<int> calculated_predecessors_res(n);

//		if (param.useMask) {
//			status = nvgraphAllocateEdgeData(handle, g1, numsets_e, type_e);
//			ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//		}

		int source_vert = param.source_vert;

		if (param.useMask) {
			//if we need to use a mask
			//Copying mask into graph

			//status = nvgraphSetEdgeData(handle, g1, &csr_mask[0], 0);
			//ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
			//nvgraphTraversalSetEdgeMaskIndex(&traversal_param, 0);
		}

		status = nvgraph2dBfs(handle,
										g1,
										source_vert,
										&calculated_distances_res[0],
										&calculated_predecessors_res[0]);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		cudaDeviceSynchronize();

		if (PERF && n > PERF_ROWS_LIMIT)
		{
			double start, stop;
			start = second();
			int repeat = 30;
			for (int i = 0; i < repeat; i++)
					{
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

		// check with reference
		std::vector<int> expected_distances_res(n);
		ref_bfs(n,
					nnz,
					&read_row_ptr[0],
					&read_col_ind[0],
					&csr_mask[0],
					source_vert,
					&expected_distances_res[0]);
		//Checking distances
//		int wrong = 0;
//		for (int i = 0; i < n; i++) {
//			if (expected_distances_res[i] != calculated_distances_res[i]) {
//				wrong++;
//				std::cout << "Error at " << i << " expected " << expected_distances_res[i] << " actual "
//						<< calculated_distances_res[i] << "\n";
//			}
//		}
//		std::cout << wrong << "/" << n << " distances are incorrect.\n";
		for (int i = 0; i < n; ++i)
				{
			ASSERT_EQ(expected_distances_res[i], calculated_distances_res[i])<< "Wrong distance from source in row #" << i << " graph " << param.graph_file << " source_vert=" << source_vert<< "\n";
		}

		//Checking predecessors
		for (int i = 0; i < n; ++i) {
			if (calculated_predecessors_res[i] != -1) {
				ASSERT_EQ(expected_distances_res[i], expected_distances_res[calculated_predecessors_res[i]] + 1)<< "Wrong predecessor in row #" << i << " graph " << param.graph_file << " source_vert=" << source_vert<< "\n";
			} else {
				ASSERT_TRUE(expected_distances_res[i] == 0 || expected_distances_res[i] == -1) << "Wrong predecessor in row #" << i << " graph " << param.graph_file << " source_vert=" << source_vert<< "\n";
			}
		}

		status = nvgraphDestroyGraphDescr(handle, g1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}
};

TEST_P(NVGraphCAPITests_2d_bfs, CheckResult) {
	run_current_test<float>(GetParam());
}			/// Few sanity checks.

class NVGraphCAPITests_2d_bfs_Sanity: public ::testing::Test {
public:
	nvgraphStatus_t status;
	nvgraphHandle_t handle;
	nvgraphTopologyType_t topo;
	int n;
	int nnz;
	nvgraphGraphDescr_t g1;
	int32_t* devices;
	int32_t numDevices;

	NVGraphCAPITests_2d_bfs_Sanity() :
			handle(NULL) {
	}

	static void SetupTestCase() {
	}
	static void TearDownTestCase() {
	}
	virtual void SetUp() {
		topo = NVGRAPH_2D_32I_32I;
		nvgraphStatus_t status;
		if (handle == NULL) {
			char* nvgraph_gpus = getenv("NVGRAPH_GPUS");
			if (nvgraph_gpus)
				printf("Value of NVGRAPH_GPUS=%s\n", nvgraph_gpus);
			else
				printf("Value of NVGRAPH_GPUS is null\n");
			std::vector<int32_t> gpus;
			int32_t dummy;
			std::stringstream ss(nvgraph_gpus);
			while (ss >> dummy) {
				gpus.push_back(dummy);
				if (ss.peek() == ',')
					ss.ignore();
			}
			printf("There were %d devices found: ", (int) gpus.size());
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
		}
	}

	template<typename EdgeT>
	void prepare_and_run(nvgraph2dCOOTopology32I_st& topo_st, int* expected)
								{
		g1 = NULL;
		status = nvgraphCreateGraphDescr(handle, &g1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		// set up graph
		n = topo_st.nvertices;
		nnz = topo_st.nedges;
		status = nvgraphSetGraphStructure(handle, g1, (void*) &topo_st, topo);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		int source_vert = 0;

		// Call BFS
		std::vector<int> calculated_dist(n);
		std::vector<int> calculated_pred(n);
		status = nvgraph2dBfs(handle, g1, source_vert, &calculated_dist[0], &calculated_pred[0]);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		// Check results against reference implementation
		for (int row = 0; row < n; row++) {
			int reference_res = (int) expected[row];
			int nvgraph_res = (int) calculated_dist[row];
			ASSERT_EQ(reference_res, nvgraph_res);
		}

		status = nvgraphDestroyGraphDescr(handle, g1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}

// cycle graph, shortest path = vertex number
	template<typename EdgeT>
	void run_cycle_test()
	{
		n = 1024;
		nnz = n;
		std::vector<int> offsets(n + 1), neighborhood(n);
		for (int i = 0; i < n; i++) {
			offsets[i] = i;
			neighborhood[i] = (i + 1) % n;
		}
		offsets[n] = n;
		std::vector<int> expected_res(n, nvgraph_Const<int>::inf);
		for (int i = 0; i < n; i++) {
			expected_res[i] = i;
		}
		int32_t blockN = std::max(2,(int)ceil(sqrt(numDevices)));
		nvgraph2dCOOTopology32I_st topology = { n, nnz, &offsets[0], &neighborhood[0], CUDA_R_32I,
		NULL, numDevices, devices, blockN, NVGRAPH_DEFAULT };

		prepare_and_run<EdgeT>(topology, &expected_res[0]);
		free(devices);
	}

	template<typename EdgeT>
	void run_cycle_test_undirected()
	{
		n = 16;
		nnz = n * 2;
		std::vector<int> offsets(n + 1), neighborhood(nnz);
		for (int i = 0; i < n; i++) {
			offsets[i] = i * 2;
			neighborhood[i * 2] = (i - 1 + n) % n;
			neighborhood[i * 2 + 1] = (i + 1 + n) % n;
		}
		offsets[n] = nnz;
		std::vector<int> expected_res(n, nvgraph_Const<int>::inf);
		for (int i = 0; i < n; i++) {
			expected_res[i] = i;
		}
		int32_t blockN = std::max(2,(int)ceil(sqrt(numDevices)));
		nvgraph2dCOOTopology32I_st topology = { n, nnz, &offsets[0], &neighborhood[0], CUDA_R_32I,
		NULL, numDevices, devices, blockN, NVGRAPH_DEFAULT };

		prepare_and_run<EdgeT>(topology, &expected_res[0]);
		free(devices);
	}

	template<typename EdgeT>
	void run_block_skip_test() {
		n = 10;
		nnz = 4;
		int rowIndices[4] = { 0, 1, 5, 6 };
		int columnIndices[4] = { 1, 5, 6, 3 };
		int expected[10] = { 0, 1, -1, 4, -1, 2, 3, -1, -1, -1 };
		int32_t blockN = std::max(2,(int)ceil(sqrt(numDevices)));
		nvgraph2dCOOTopology32I_st topology = { n, nnz, rowIndices, columnIndices, CUDA_R_32I,
		NULL, numDevices, devices, blockN, NVGRAPH_DEFAULT };
		prepare_and_run<EdgeT>(topology, expected);
		free(devices);
	}

	template<typename EdgeT>
	void run_multi_path_test() {
		n = 10;
		nnz = 6;
		int rowIndices[6] = { 0, 0, 1, 5, 5, 6 };
		int columnIndices[6] = { 1, 5, 6, 6, 9, 9 };
		int expected[10] = { 0, 1, -1, -1, -1, 1, 2, -1, -1, 2 };
		int32_t blockN = std::max(2,(int)ceil(sqrt(numDevices)));
		nvgraph2dCOOTopology32I_st topology = { n, nnz, rowIndices, columnIndices, CUDA_R_32I,
		NULL, numDevices, devices, blockN, NVGRAPH_DEFAULT };
		prepare_and_run<EdgeT>(topology, expected);
		free(devices);
	}

};

TEST_F(NVGraphCAPITests_2d_bfs_Sanity, SanityCycle)
{
	run_cycle_test<float>();
}

TEST_F(NVGraphCAPITests_2d_bfs_Sanity, BlockSkip) {
	run_block_skip_test<float>();
}

TEST_F(NVGraphCAPITests_2d_bfs_Sanity, MultiPath) {
	run_multi_path_test<float>();
}

//	class NVGraphCAPITests_Traversal_Stress: public ::testing::TestWithParam<Traversal_Usecase> {
//	public:
//		NVGraphCAPITests_Traversal_Stress() :
//		handle(NULL) {
//		}
//
//		static void SetupTestCase() {
//		}
//		static void TearDownTestCase() {
//		}
//		virtual void SetUp() {
//			//const ::testing::TestInfo* const test_info =::testing::UnitTest::GetInstance()->current_test_info();
//			//printf("We are in test %s of test case %s.\n", test_info->name(), test_info->test_case_name());
//			if (handle == NULL) {
//				int* devices = (int*)malloc(sizeof(int) * 2);
//				devices[0] = 0;
//				devices[1] = 1;
//				status = nvgraphCreateMulti(&handle, 2, devices);
//				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//				free(devices);
//			}
//		}
//		virtual void TearDown() {
//			if (handle != NULL) {
//				status = nvgraphDestroy(handle);
//				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//				handle = NULL;
//			}
//		}
//		nvgraphStatus_t status;
//		nvgraphHandle_t handle;
//
//		template<typename EdgeT>
//		void run_current_test(const Traversal_Usecase& param)
//		{
//			nvgraphTopologyType_t topo = NVGRAPH_2D_32I_32I;
//
//			nvgraphStatus_t status;
//
//			FILE* fpin = fopen(param.graph_file.c_str(), "rb");
//			ASSERT_TRUE(fpin != NULL)<< "Cannot read input graph file: " << param.graph_file << std::endl;
//			int n, nnz;
//			//Read a network in amgx binary format and the bookmark of dangling nodes
//			ASSERT_EQ(read_header_amgx_csr_bin(fpin, n, nnz), 0);
//			std::vector<int> read_row_ptr(n + 1), read_col_ind(nnz);
//			std::vector<EdgeT> read_val(nnz);
//			ASSERT_EQ(read_data_amgx_csr_bin(fpin, n, nnz, read_row_ptr, read_col_ind, read_val), 0);
//			fclose(fpin);
//
//			nvgraphGraphDescr_t g1 = NULL;
//			status = nvgraphCreateGraphDescr(handle, &g1);
//			ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//
//			// set up graph
//			int* devices = (int*)malloc(sizeof(int) * 2);
//			devices[0] = 0;
//			devices[1] = 1;
//			nvgraph2dCOOTopology32I_st topology = {n, nnz, &read_row_ptr[0], &read_col_ind[0], CUDA_R_32I, NULL, 2, devices, 2, NVGRAPH_DEFAULT};
//			status = nvgraphSetGraphStructure(handle, g1, (void*) &topology, topo);
//			free(devices);
//
//			std::vector<int> calculated_res(n);
//			int source_vert = param.source_vert;
//
//			// run
//			int repeat = 2;//std::max((int)(((float)(Traversal_ITER_MULTIPLIER)*STRESS_MULTIPLIER)/(3*n)), 1);
//
//			std::vector<int> calculated_res1(n), calculated_res_mid1(n), calculated_res_last(n);
//			std::vector<int> calculated_res2(n), calculated_res_mid2(n);
//			size_t free_mid = 0, free_last = 0, total = 0;
//			for (int i = 0; i < repeat; i++)	{
//				status = nvgraphTraversal( handle,
//						g1,
//						NVGRAPH_TRAVERSAL_BFS,
//						&source_vert,
//						traversal_param);
//				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//
//				// all of those should be equal
//				if (i == 0)
//				{
//					status = nvgraphGetVertexData(handle,
//							g1,
//							(void *) &calculated_res1[0],
//							traversal_distances_index);
//					ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//					status = nvgraphGetVertexData(handle,
//							g1,
//							(void *) &calculated_res2[0],
//							traversal_predecessors_index);
//					ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//				}
//				else
//				{
//					status = nvgraphGetVertexData(handle,
//							g1,
//							(void *) &calculated_res_mid1[0],
//							traversal_distances_index);
//					ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//					status = nvgraphGetVertexData(handle,
//							g1,
//							(void *) &calculated_res_mid2[0],
//							traversal_predecessors_index);
//					ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//
//					for (int row = 0; row < n; row++)
//					{
//						ASSERT_EQ(calculated_res1[row], calculated_res_mid1[row])<< "Difference in result in distances for row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" << i;
//						// predecessors could be different since multiple shortest paths are possible
//						//ASSERT_EQ(calculated_res2[row], calculated_res_mid2[row]) << "Difference in result in predecessors for row #" << row << " graph " << param.graph_file << " for iterations #0 and iteration #" <<  i;
//					}
//				}
//
//				if (i == std::min(50, (int) (repeat / 2)))
//				{
//					cudaMemGetInfo(&free_mid, &total);
//				}
//				if (i == repeat - 1)
//				{
//					status = nvgraphGetVertexData(handle,
//							g1,
//							(void *) &calculated_res_last[0],
//							traversal_distances_index);
//					ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//					cudaMemGetInfo(&free_last, &total);
//				}
//			}
//
//			ASSERT_LE(free_mid, free_last)<< "Memory difference between iteration #" << std::min(50, (int)(repeat/2)) << " and last iteration is " << (double)(free_last-free_mid)/1e+6 << "MB";
//
//			status = nvgraphDestroyGraphDescr(handle, g1);
//			ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
//		}
//	};
//
//	TEST_P(NVGraphCAPITests_Traversal_Stress, Stress)
//	{
//		run_current_test<float>(GetParam());
//	}

// instatiation of the performance/correctness checks 

INSTANTIATE_TEST_CASE_P(CorrectnessCheck,
								NVGraphCAPITests_2d_bfs,
								//                                  graph FILE                                                  source vert #    file with expected result (in binary?)
								::testing::Values(
								Traversal_Usecase("graphs/cage/cage13_T.mtx.bin", 0)
								, Traversal_Usecase("graphs/cage/cage13_T.mtx.bin", 10)
								, Traversal_Usecase("graphs/cage/cage14_T.mtx.bin", 0)
								, Traversal_Usecase("graphs/cage/cage14_T.mtx.bin", 10)
								, Traversal_Usecase("graphs/small/small.bin", 0)
								, Traversal_Usecase("graphs/small/small.bin", 0)
								, Traversal_Usecase("graphs/small/small.bin", 3)
								, Traversal_Usecase("graphs/dblp/dblp.bin", 0, false, true)
								, Traversal_Usecase("graphs/dblp/dblp.bin", 100, false, true)
								, Traversal_Usecase("graphs/dblp/dblp.bin", 1000, false, true)
								, Traversal_Usecase("graphs/dblp/dblp.bin", 100000, false, true)
								, Traversal_Usecase("graphs/Wikipedia/2003/wiki2003.bin", 0)
								, Traversal_Usecase("graphs/Wikipedia/2003/wiki2003.bin", 100)
								, Traversal_Usecase("graphs/Wikipedia/2003/wiki2003.bin", 10000)
								, Traversal_Usecase("graphs/Wikipedia/2003/wiki2003.bin", 100000)
								, Traversal_Usecase("graphs/Wikipedia/2011/wiki2011.bin", 1)
								, Traversal_Usecase("graphs/Wikipedia/2011/wiki2011.bin", 1000)
								, Traversal_Usecase("dimacs10/road_usa_T.mtx.bin", 100)
								, Traversal_Usecase("graphs/Twitter/twitter.bin", 0)
								, Traversal_Usecase("graphs/Twitter/twitter.bin", 100)
								, Traversal_Usecase("graphs/Twitter/twitter.bin", 10000)
								, Traversal_Usecase("graphs/Twitter/twitter.bin", 3000000)
								, Traversal_Usecase("dimacs10/hugebubbles-00020_T.mtx.bin", 100000)
								//					/// instances using mask
//					, Traversal_Usecase("graphs/small/small.bin", 0, true)
//					, Traversal_Usecase("graphs/small/small.bin", 0, true)
//					, Traversal_Usecase("graphs/small/small.bin", 3, true)
//					, Traversal_Usecase("graphs/dblp/dblp.bin", 0, true)
//					, Traversal_Usecase("graphs/dblp/dblp.bin", 100, true)
//					, Traversal_Usecase("graphs/dblp/dblp.bin", 1000, true)
//					, Traversal_Usecase("graphs/dblp/dblp.bin", 100000, true)
//					, Traversal_Usecase("graphs/Wikipedia/2003/wiki2003.bin", 0, true)
								)

								);

//	INSTANTIATE_TEST_CASE_P(StressTest,
//			NVGraphCAPITests_Traversal_Stress,
//			::testing::Values(
//					Traversal_Usecase("graphs/Wikipedia/2003/wiki2003.bin", 0)
//			)
//	);

int main(int argc, char **argv)
			{

	for (int i = 0; i < argc; i++)
			{
		if (strcmp(argv[i], "--perf") == 0)
			PERF = 1;
		if (strcmp(argv[i], "--stress-iters") == 0)
			STRESS_MULTIPLIER = atoi(argv[i + 1]);
		if (strcmp(argv[i], "--ref-data-dir") == 0)
			ref_data_prefix = std::string(argv[i + 1]);
		if (strcmp(argv[i], "--graph-data-dir") == 0)
			graph_data_prefix = std::string(argv[i + 1]);
	}
	srand(42);
	::testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}
