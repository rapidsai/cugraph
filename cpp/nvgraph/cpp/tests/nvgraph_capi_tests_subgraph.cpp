#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <iterator>
#include <fstream>
#include <cassert>
#include <sstream>
#include <string>
#include <cstdio>

#include "gtest/gtest.h"
#include "valued_csr_graph.hxx"
#include "nvgraphP.h"
#include "nvgraph.h"

static std::string ref_data_prefix = "";
static std::string graph_data_prefix = "";

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

//annonymus:
namespace {

	class file_read_error
	{
	public:
		file_read_error(const std::string& msg) :
				msg_(msg)
		{
			msg_ = std::string("File read error: ") + msg;
		}
		~file_read_error() {
		}

		const std::string& what() const {
			return (msg_);
		}
	private:
		std::string msg_;
	};

	template<typename Vector>
	void fill_extraction_data(const std::string& fname,
										Vector& g_row_offsets,
										Vector& g_col_indices,
										Vector& aggregates,
										Vector& cg_row_offsets,
										Vector& cg_col_indices)
										{
		typedef typename Vector::value_type T;
		std::ifstream m_stream(fname.c_str(), std::ifstream::in);
		std::string line;

		if (!m_stream.is_open())
		{
			throw file_read_error(fname);
		}

		bool keep_going = !std::getline(m_stream, line).eof();

		//debug:
		//std::cout<<line<<std::endl;

		if (!keep_going)
			return;

		char c;
		int g_nrows = 0;
		int g_nnz = 0;
		std::sscanf(line.c_str(), "%c: nrows=%d, nnz=%d", &c, &g_nrows, &g_nnz);

		//debug:
		//std::cout<<c<<","<<g_nrows<<","<<g_nnz<<"\n";
		int n_entries = g_nrows + 1;
		g_row_offsets.reserve(n_entries);

		//ignore next line:
		//
		if (!std::getline(m_stream, line))
			return;

		//read G row_offsets:
		for (int i = 0; (i < n_entries) && keep_going; ++i)
				{
			T value(0);

			keep_going = !std::getline(m_stream, line).eof();
			std::stringstream ss(line);
			ss >> value;
			g_row_offsets.push_back(value);
		}

		//ignore next 2 lines:
		//
		if (!std::getline(m_stream, line) || !std::getline(m_stream, line))
			return;

		g_col_indices.reserve(g_nnz);

		//read G col_indices:
		for (int i = 0; (i < g_nnz) && keep_going; ++i)
				{
			T value(0);

			keep_going = !std::getline(m_stream, line).eof();
			std::stringstream ss(line);
			ss >> value;
			g_col_indices.push_back(value);
		}

		//ignore next line:
		//
		if (!std::getline(m_stream, line))
			return;

		//remove the following for extraction:
		//{
		if (!std::getline(m_stream, line))
			return;
		int n_aggs = 0;
		std::sscanf(line.c_str(), "aggregate: size=%d", &n_aggs);

		//assert( n_aggs == g_nrows );//not true for subgraph extraction!

		aggregates.reserve(n_aggs);

		//read aggregate:
		for (int i = 0; (i < n_aggs) && keep_going; ++i)
				{
			T value(0);

			keep_going = !std::getline(m_stream, line).eof();
			std::stringstream ss(line);
			ss >> value;
			aggregates.push_back(value);
		}
		//} end remove code for extraction

		if (!keep_going || !std::getline(m_stream, line))
			return;
		int cg_nrows = 0;
		int cg_nnz = 0;
		std::sscanf(line.c_str(), "result %c: nrows=%d, nnz=%d", &c, &cg_nrows, &cg_nnz);

		//debug:
		//std::cout<<c<<","<<cg_nrows<<","<<cg_nnz<<"\n";

		//
		//m_stream.close();//not really needed...destructor handles this
		//return;

		n_entries = cg_nrows + 1;
		cg_row_offsets.reserve(n_entries);

		//ignore next line:
		//
		if (!std::getline(m_stream, line))
			return;

		//read G row_offsets:
		for (int i = 0; (i < n_entries) && keep_going; ++i)
				{
			T value(0);

			keep_going = !std::getline(m_stream, line).eof();
			std::stringstream ss(line);
			ss >> value;
			cg_row_offsets.push_back(value);
		}

		//ignore next 2 lines:
		//
		if (!std::getline(m_stream, line) || !std::getline(m_stream, line))
			return;

		cg_col_indices.reserve(cg_nnz);

		//read G col_indices:
		for (int i = 0; (i < cg_nnz) && keep_going; ++i)
				{
			T value(0);

			keep_going = !std::getline(m_stream, line).eof();
			std::stringstream ss(line);
			ss >> value;
			cg_col_indices.push_back(value);
		}

		m_stream.close();  //not really needed...destructor handles this
	}

	template<typename Vector>
	bool check_diffs(const Vector& v1, const Vector& v2)
							{
		typedef typename Vector::value_type T;

		Vector v(v1.size(), 0);
		std::transform(v1.begin(), v1.end(),
							v2.begin(),
							v.begin(),
							std::minus<T>());

		if (std::find_if(v.begin(), v.end(), std::bind2nd(std::not_equal_to<T>(), 0)) != v.end())
			return true;
		else
			return false;
	}

//check if sort(delta(r1)) == sort(delta(r2))
//where delta(r)={r[i+1]-r[i] | i <- [0..|r|-1]}
//
	template<typename Vector>
	bool check_delta_invariant(const Vector& r1, const Vector& r2)
										{
		typedef typename Vector::value_type T;

		size_t sz = r1.size();
		assert(sz == r2.size());

		Vector d1(sz - 1);

		std::transform(r1.begin() + 1, r1.end(),
							r1.begin(),
							d1.begin(),
							std::minus<int>());

		Vector d2(sz - 1);

		std::transform(r2.begin() + 1, r2.end(),
							r2.begin(),
							d2.begin(),
							std::minus<int>());

		std::sort(d1.begin(), d1.end());
		std::sort(d2.begin(), d2.end());

		return (d1 == d2);
	}
}

class NvgraphCAPITests_SubgraphCSR: public ::testing::Test {
public:
	NvgraphCAPITests_SubgraphCSR() :
			nvgraph_handle(NULL), initial_graph(NULL) {
	}

protected:
	static void SetupTestCase()
	{
	}
	static void TearDownTestCase()
	{
	}
	virtual void SetUp()
	{
		if (nvgraph_handle == NULL) {
			status = nvgraphCreate(&nvgraph_handle);
			ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		}
		// set up graph
		status = nvgraphCreateGraphDescr(nvgraph_handle, &initial_graph);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		nvgraphCSRTopology32I_st topoData;
		topoData.nvertices = 5;
		topoData.nedges = 9;
		int neighborhood[] = { 0, 2, 3, 5, 7, 9 };
		int edgedest[] = { 1, 3, 3, 1, 4, 0, 2, 2, 4 };
		topoData.source_offsets = neighborhood;
		topoData.destination_indices = edgedest;
		status = nvgraphSetGraphStructure(	nvgraph_handle,
														initial_graph,
														(void*) &topoData,
														NVGRAPH_CSR_32);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		// set up graph data
		size_t numsets = 2;
		float vertexvals0[] = { 0.1f, 0.15893e-20f, 1e27f, 13.2f, 0.f };
		float vertexvals1[] = { 13., 322.64, 1e28, -1.4, 22.3 };
		void* vertexptr[] = { (void*) vertexvals0, (void*) vertexvals1 };
		cudaDataType_t type_v[] = { CUDA_R_32F, CUDA_R_32F };
		float edgevals0[] = { 0.1f, 0.9153e-20f, 0.42e27f, 185.23, 1e21f, 15.6f, 215.907f, 912.2f,
				0.2f };
		float edgevals1[] = { 13., 322.64, 1e28, 197534.2, 0.1, 0.425e-5, 5923.4, 0.12e-12, 52. };
		void* edgeptr[] = { (void*) edgevals0, (void*) edgevals1 };
		cudaDataType_t type_e[] = { CUDA_R_32F, CUDA_R_32F };

		status = nvgraphAllocateVertexData(nvgraph_handle, initial_graph, numsets, type_v);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphSetVertexData(nvgraph_handle, initial_graph, (void *) vertexptr[0], 0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphSetVertexData(nvgraph_handle, initial_graph, (void *) vertexptr[1], 1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphAllocateEdgeData(nvgraph_handle, initial_graph, numsets, type_e);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphSetEdgeData(nvgraph_handle, initial_graph, (void *) edgeptr[0], 0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphSetEdgeData(nvgraph_handle, initial_graph, (void *) edgeptr[1], 1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//save data - those will be available in the tests directly
		graph_neigh.assign(neighborhood, neighborhood + topoData.nvertices + 1);
		graph_edged.assign(edgedest, edgedest + topoData.nedges);
		graph_vvals0.assign(vertexvals0, vertexvals0 + topoData.nvertices);
		graph_vvals1.assign(vertexvals1, vertexvals1 + topoData.nvertices);
		graph_evals0.assign(edgevals0, edgevals0 + topoData.nedges);
		graph_evals1.assign(edgevals1, edgevals1 + topoData.nedges);
	}
	virtual void TearDown()
	{
		// destroy graph
		status = nvgraphDestroyGraphDescr(nvgraph_handle, initial_graph);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		// release library
		if (nvgraph_handle != NULL) {
			status = nvgraphDestroy(nvgraph_handle);
			ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
			nvgraph_handle = NULL;
		}
	}
	nvgraphStatus_t status;
	nvgraphHandle_t nvgraph_handle;
	nvgraphGraphDescr_t initial_graph;

	std::vector<int> graph_neigh;
	std::vector<int> graph_edged;
	std::vector<float> graph_vvals0;
	std::vector<float> graph_vvals1;
	std::vector<float> graph_evals0;
	std::vector<float> graph_evals1;
};

class NvgraphCAPITests_SubgCSR_Isolated: public ::testing::Test {
public:
	NvgraphCAPITests_SubgCSR_Isolated() :
			nvgraph_handle(NULL), initial_graph(NULL) {
	}

protected:
	static void SetupTestCase()
	{
	}
	static void TearDownTestCase()
	{
	}
	virtual void SetUp()
	{
		if (nvgraph_handle == NULL) {
			status = nvgraphCreate(&nvgraph_handle);
			ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		}
		// set up graph
		status = nvgraphCreateGraphDescr(nvgraph_handle, &initial_graph);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		/*
		 *  here is the graph we'll test with:
		 *     0 -> 2
		 *     1 -> 3
		 *
		 *  Extracting the subgraph that uses vertices 0, 1, 3 will get
		 *  a graph with 3 vertices and 1 edge... and that edge won't
		 *  use vertex id 0.  Pre bug fix the resulting graph is a single
		 *  edge:  0 -> 3  which does not even exist in the original graph.
		 */
		nvgraphCSRTopology32I_st topoData;
		std::vector<int> v_neighborhood { 0, 1, 2, 2, 2 };
		std::vector<int> v_edgedest{ 2, 3 };

		topoData.nvertices = v_neighborhood.size();
		topoData.nedges = v_edgedest.size();

		topoData.source_offsets = v_neighborhood.data();
		topoData.destination_indices = v_edgedest.data();
		status = nvgraphSetGraphStructure(	nvgraph_handle,
														initial_graph,
														(void*) &topoData,
														NVGRAPH_CSR_32);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		graph_neigh = v_neighborhood;
		graph_edged = v_edgedest;
	}
	virtual void TearDown()
	{
		// destroy graph
		status = nvgraphDestroyGraphDescr(nvgraph_handle, initial_graph);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		// release library
		if (nvgraph_handle != NULL) {
			status = nvgraphDestroy(nvgraph_handle);
			ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
			nvgraph_handle = NULL;
		}
	}
	nvgraphStatus_t status;
	nvgraphHandle_t nvgraph_handle;
	nvgraphGraphDescr_t initial_graph;

	std::vector<int> graph_neigh;
	std::vector<int> graph_edged;
};

TEST_F(NvgraphCAPITests_SubgCSR_Isolated, CSRSubgraphVertices_Bug60)
{
  nvgraphStatus_t status;
  nvgraphGraphDescr_t temp_graph2 = NULL;

  {
    status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph2);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    //int vertices[] = { 0, 1, 17 };
    int vertices[] = { 0, 1, 3 };
    status = nvgraphExtractSubgraphByVertex(nvgraph_handle,
                                initial_graph,
                                temp_graph2,
                              	vertices,
                                sizeof(vertices) / sizeof(vertices[0]));
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    nvgraphCSRTopology32I_st tData;
    int tData_source_offsets[3], tData_destination_indices[3];
    tData.source_offsets = tData_source_offsets;
    tData.destination_indices = tData_destination_indices;
    nvgraphTopologyType_t TT;
    status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph2, (void*) &tData, &TT);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(TT, NVGRAPH_CSR_32);
    ASSERT_EQ(tData.nvertices, 3);
    ASSERT_EQ(tData.nedges, 1);

    // check structure
    ASSERT_EQ(tData.source_offsets[0], 0);
    ASSERT_EQ(tData.source_offsets[1], 0);
    ASSERT_EQ(tData.source_offsets[2], 1);
    ASSERT_EQ(tData.source_offsets[3], 1);
    ASSERT_EQ(tData.destination_indices[0], 2);

    status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph2);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
  }
}

TEST_F(NvgraphCAPITests_SubgraphCSR, CSRSubgraphVertices_Sanity)
{
	nvgraphStatus_t status;
	nvgraphGraphDescr_t temp_graph1 = NULL, temp_graph2 = NULL;

	float getVvals0[4];
	float getVvals1[4];
	float getEvals0[4];
	float getEvals1[4];

	{
		status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph2);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		int vertices[] = { 2, 4 };
		status = nvgraphExtractSubgraphByVertex(	nvgraph_handle,
																initial_graph,
																temp_graph2,
																vertices,
																2);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		nvgraphCSRTopology32I_st tData;
		int tData_source_offsets[3], tData_destination_indices[3];
		tData.source_offsets = tData_source_offsets;
		tData.destination_indices = tData_destination_indices;
		nvgraphTopologyType_t TT;
		status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph2, (void*) &tData, &TT);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		ASSERT_EQ(TT, NVGRAPH_CSR_32);
		ASSERT_EQ(tData.nvertices, 2);
		ASSERT_EQ(tData.nedges, 3);

		status = nvgraphGetVertexData(nvgraph_handle, temp_graph2, (void *) getVvals0, 0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetVertexData(nvgraph_handle, temp_graph2, (void *) getVvals1, 1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetEdgeData(nvgraph_handle, temp_graph2, (void *) getEvals0, 0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetEdgeData(nvgraph_handle, temp_graph2, (void *) getEvals1, 1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		// we are extracting two vertices, but we are not sure which of them will be #0 and which will be #1
		// we are comparing vertex values to determine that and handle both cases
		if (getVvals0[0] == graph_vvals0[vertices[0]])
				//vertex #0 in new graph - vertex #2 in old graph
				//vertex #1 in new graph - vertex #4 in old graph
				{
			// check that vertex values are extracted correctly
			ASSERT_EQ(getVvals0[0], graph_vvals0[vertices[0]]);
			ASSERT_EQ(getVvals1[0], graph_vvals1[vertices[0]]);
			ASSERT_EQ(getVvals0[1], graph_vvals0[vertices[1]]);
			ASSERT_EQ(getVvals1[1], graph_vvals1[vertices[1]]);
			// check that edge values are extracted correctly
			ASSERT_EQ(getEvals0[0], graph_evals0[4]);
			ASSERT_EQ(getEvals0[1], graph_evals0[7]);
			ASSERT_EQ(getEvals0[2], graph_evals0[8]);
			ASSERT_EQ(getEvals1[0], graph_evals1[4]);
			ASSERT_EQ(getEvals1[1], graph_evals1[7]);
			ASSERT_EQ(getEvals1[2], graph_evals1[8]);
			// Check structure
			ASSERT_EQ(tData.source_offsets[0], 0);
			ASSERT_EQ(tData.source_offsets[1], 1);
			ASSERT_EQ(tData.source_offsets[2], 3);
			ASSERT_EQ(tData.destination_indices[0], 1);
			ASSERT_EQ(tData.destination_indices[1], 0);
			ASSERT_EQ(tData.destination_indices[2], 1);
		}

		//vertex #0 in new graph - vertex #4 in old graph
		//vertex #1 in new graph - vertex #2 in old graph
		else
		{
			// check that vertex values are extracted correctly
			ASSERT_EQ(getVvals0[0], graph_vvals0[vertices[1]]);
			ASSERT_EQ(getVvals0[1], graph_vvals0[vertices[0]]);
			ASSERT_EQ(getVvals1[0], graph_vvals1[vertices[1]]);
			ASSERT_EQ(getVvals1[1], graph_vvals1[vertices[0]]);
			// check that edge values are extracted correctly
			ASSERT_EQ(getEvals0[0], graph_evals0[7]);
			ASSERT_EQ(getEvals0[1], graph_evals0[8]);
			ASSERT_EQ(getEvals0[2], graph_evals0[4]);
			ASSERT_EQ(getEvals1[0], graph_evals1[7]);
			ASSERT_EQ(getEvals1[1], graph_evals1[8]);
			ASSERT_EQ(getEvals1[2], graph_evals1[4]);
			// check structure
			ASSERT_EQ(tData.source_offsets[0], 0);
			ASSERT_EQ(tData.source_offsets[1], 2);
			ASSERT_EQ(tData.source_offsets[2], 3);
			ASSERT_EQ(tData.destination_indices[0], 0);
			ASSERT_EQ(tData.destination_indices[1], 1);
			ASSERT_EQ(tData.destination_indices[2], 0);
		}

		status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph2);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}

	//@TODO: how to check extracting by multiple vertices? do we preserve order of vertices/edges?
	//@TODO: this would make sense only if vertices order is perserved in the extracted subgraph
	int vertices[4] = { 0, 1, 3, 4 };
	status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
	ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	status = nvgraphExtractSubgraphByVertex(nvgraph_handle, initial_graph, temp_graph1, vertices, 3);
	ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	/*size_t nverts1 = 0, nedges1 = 0;
	 int neighborget[5];
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(nverts1, 4);
	 status = nvgraphGetGraphNedges(nvgraph_handle, temp_graph1, &nedges1);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(nedges1, 4);

	 // check structure:
	 status = nvgraphGetGraphNeighborhood(nvgraph_handle, temp_graph1, neighborget);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(neighborget[0], 0);
	 ASSERT_EQ(neighborget[1], 2);
	 ASSERT_EQ(neighborget[2], 3);
	 ASSERT_EQ(neighborget[3], 4);
	 ASSERT_EQ(neighborget[4], 4);

	 int edgeget[4];
	 status = nvgraphGetGraphEdgeDest( nvgraph_handle, temp_graph1, edgeget);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(edgeget[0], 1);
	 ASSERT_EQ(edgeget[1], 3);
	 ASSERT_EQ(edgeget[2], 3);
	 ASSERT_EQ(edgeget[3], 0);

	 // check values
	 status = nvgraphGetVertexData(nvgraph_handle, temp_graph1, (void *)getVvals0, 0);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(getVvals0[0], vertexvals0[vertices[0]]);
	 ASSERT_EQ(getVvals0[1], vertexvals0[vertices[1]]);
	 ASSERT_EQ(getVvals0[2], vertexvals0[vertices[2]]);
	 ASSERT_EQ(getVvals0[3], vertexvals0[vertices[3]]);
	 status = nvgraphGetVertexData(nvgraph_handle, temp_graph1, (void *)getVvals1, 1);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(getVvals1[0], vertexvals1[vertices[0]]);
	 ASSERT_EQ(getVvals1[1], vertexvals1[vertices[1]]);
	 ASSERT_EQ(getVvals1[2], vertexvals1[vertices[2]]);
	 ASSERT_EQ(getVvals1[3], vertexvals1[vertices[3]]);

	 status = nvgraphGetEdgeData(nvgraph_handle, temp_graph1, (void *)getEvals0, 0);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(getEvals0[0], edgevals0[0]);
	 ASSERT_EQ(getEvals0[1], edgevals0[1]);
	 ASSERT_EQ(getEvals0[2], edgevals0[2]);
	 ASSERT_EQ(getEvals0[3], edgevals0[6]);
	 status = nvgraphGetEdgeData(nvgraph_handle, temp_graph1, (void *)getEvals1, 1);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(getEvals1[0], edgevals1[0]);
	 ASSERT_EQ(getEvals1[1], edgevals1[1]);
	 ASSERT_EQ(getEvals1[2], edgevals1[2]);
	 ASSERT_EQ(getEvals1[3], edgevals1[6]);*/

	status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
	ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}

TEST_F(NvgraphCAPITests_SubgraphCSR, CSRSubgraphVertices_CornerCases)
{
	nvgraphStatus_t status;

	nvgraphGraphDescr_t temp_graph1 = NULL, temp_graph2 = NULL;
	float getVvals0[4];
	float getVvals1[4];
	float getEvals0[4];
	float getEvals1[4];
// failures
	{
		int vertices[2] = { 1, 3 };
		status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		// bad library nvgraph_handle
		status = nvgraphExtractSubgraphByEdge(NULL, initial_graph, temp_graph1, vertices, 1);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		// bad descriptor 1
		status = nvgraphExtractSubgraphByEdge(nvgraph_handle, temp_graph2, temp_graph1, vertices, 1);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		// bad descriptor 2
		status = nvgraphExtractSubgraphByEdge(	nvgraph_handle,
															initial_graph,
															temp_graph2,
															vertices,
															1);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		// NULL pointer
		status = nvgraphExtractSubgraphByEdge(	nvgraph_handle,
															initial_graph,
															temp_graph1,
															(int*) NULL,
															1);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		// extract zero vertices - failure expected
		status = nvgraphExtractSubgraphByVertex(	nvgraph_handle,
																initial_graph,
																temp_graph1,
																vertices,
																0);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		// extracting vertices more than in original graph - failure expected
		int too_many_vertices[] = { 0, 1, 2, 3, 4, 5, 10, 15 };
		status = nvgraphExtractSubgraphByVertex(	nvgraph_handle,
																initial_graph,
																temp_graph1,
																too_many_vertices,
																8);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		// unexisting indices - failure expected
		int bad_vertices[] = { -1, 2, 15 };
		status = nvgraphExtractSubgraphByVertex(	nvgraph_handle,
																initial_graph,
																temp_graph1,
																bad_vertices,
																3);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}

	// Not connected vertices
	{
		int vertices[] = { 0, 2 };
		status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphExtractSubgraphByVertex(	nvgraph_handle,
																initial_graph,
																temp_graph1,
																vertices,
																2);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		nvgraphCSRTopology32I_st tData;
		tData.source_offsets = NULL;
		tData.destination_indices = NULL;
		nvgraphTopologyType_t TT;
		status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph1, (void*) &tData, &TT);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		ASSERT_EQ(TT, NVGRAPH_CSR_32);
		ASSERT_EQ(tData.nvertices, 2);
		ASSERT_EQ(tData.nedges, 0);

		status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}

	// extract vertex that has edge to itself
	{
		int vertices[] = { 4 };
		status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphExtractSubgraphByVertex(	nvgraph_handle,
																initial_graph,
																temp_graph1,
																vertices,
																1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		nvgraphCSRTopology32I_st tData;
		tData.source_offsets = NULL;
		tData.destination_indices = NULL;
		status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph1, (void*) &tData, NULL);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		ASSERT_EQ(tData.nvertices, 1);
		ASSERT_EQ(tData.nedges, 1);

		status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph1, (void*) &tData, NULL);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetVertexData(nvgraph_handle, temp_graph1, (void *) getVvals0, 0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetVertexData(nvgraph_handle, temp_graph1, (void *) getVvals1, 1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetEdgeData(nvgraph_handle, temp_graph1, (void *) getEvals0, 0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetEdgeData(nvgraph_handle, temp_graph1, (void *) getEvals1, 1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		ASSERT_EQ(getVvals0[0], graph_vvals0[vertices[0]]);
		ASSERT_EQ(getVvals1[0], graph_vvals1[vertices[0]]);
		ASSERT_EQ(getEvals0[0], graph_evals0[graph_evals0.size() - 1]);
		ASSERT_EQ(getEvals1[0], graph_evals1[graph_evals0.size() - 1]);

		status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}

	// extract whole graph
	{
		int vertices[] = { 0, 1, 2, 3, 4 };
		status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphExtractSubgraphByVertex(	nvgraph_handle,
																initial_graph,
																temp_graph1,
																vertices,
																5);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		nvgraphCSRTopology32I_st tData;
		tData.source_offsets = NULL;
		tData.destination_indices = NULL;
		status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph1, (void*) &tData, NULL);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		ASSERT_EQ(tData.nvertices, (int )graph_vvals0.size());
		ASSERT_EQ(tData.nedges, (int )graph_evals0.size());

		status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}
}

TEST_F(NvgraphCAPITests_SubgraphCSR, CSRSubgraphEdges_Sanity)
{
	nvgraphStatus_t status;

	nvgraphGraphDescr_t temp_graph1 = NULL, temp_graph2 = NULL;
	status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
	ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

	float getVvals0[4];
	float getVvals1[4];
	float getEvals0[4];
	float getEvals1[4];

	// for all edges: try to extract graph using only 1 edge
	{
		for (int r = 0; r < (int) graph_vvals0.size() /* == nvertices */; r++)
				{
			for (int e = graph_neigh[r]; e < graph_neigh[r + 1]; e++)
					{
				status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph2);
				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
				status = nvgraphExtractSubgraphByEdge(	nvgraph_handle,
																	initial_graph,
																	temp_graph2,
																	&e,
																	1);
				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

				nvgraphCSRTopology32I_st tData;
				tData.source_offsets = NULL;
				tData.destination_indices = NULL;
				status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph2, (void*) &tData, NULL);
				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
				status = nvgraphGetVertexData(nvgraph_handle, temp_graph2, (void *) getVvals0, 0);
				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
				status = nvgraphGetVertexData(nvgraph_handle, temp_graph2, (void *) getVvals1, 1);
				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
				status = nvgraphGetEdgeData(nvgraph_handle, temp_graph2, (void *) getEvals0, 0);
				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
				status = nvgraphGetEdgeData(nvgraph_handle, temp_graph2, (void *) getEvals1, 1);
				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

				// check structure - should always be 1 edge and 2 vertices, special case for the last edge, because it is from vertex #5 to itself
				if (e != (int) graph_evals0.size() - 1)
						{
					// check structure
					ASSERT_EQ(tData.nvertices, 2)<< "Row : " << r << ", Edge : " << e;
					ASSERT_EQ(tData.nedges, 1) << "Row : " << r << ", Edge : " << e;
					// check vertex data
					ASSERT_TRUE((getVvals0[0] == graph_vvals0[r]) || (getVvals0[0] == graph_vvals0[graph_edged[e]])) << getVvals0[0] << " " << graph_vvals0[r] << " " << graph_vvals0[graph_edged[e]];
					ASSERT_TRUE((getVvals0[1] == graph_vvals0[r]) || (getVvals0[1] == graph_vvals0[graph_edged[e]])) << getVvals0[1] << " " << graph_vvals0[r] << " " << graph_vvals0[graph_edged[e]];
					ASSERT_TRUE(getVvals0[0] != getVvals0[1]) << getVvals0[0] << " " << getVvals0[1];
					ASSERT_TRUE((getVvals1[0] == graph_vvals1[r]) || (getVvals1[0] == graph_vvals1[graph_edged[e]])) << getVvals1[0] << " " << graph_vvals1[r] << " " << graph_vvals1[graph_edged[e]];
					ASSERT_TRUE((getVvals1[1] == graph_vvals1[r]) || (getVvals1[1] == graph_vvals1[graph_edged[e]])) << getVvals1[1] << " " << graph_vvals1[r] << " " << graph_vvals1[graph_edged[e]];
					ASSERT_TRUE(getVvals1[0] != getVvals1[1]) << getVvals1[0] << " " << getVvals1[1];
				}
				else // special case for the last edge - from last vertex to itself
				{
					// check structure
					ASSERT_EQ(tData.nvertices, 1) << "Row : " << r << ", Edge : " << e;
					ASSERT_EQ(tData.nedges, 1) << "Row : " << r << ", Edge : " << e;
					// check vertex data
					ASSERT_TRUE(getVvals0[0] == graph_vvals0[r]) << getVvals0[0] << " " << graph_vvals0[r];
					ASSERT_TRUE(getVvals1[0] == graph_vvals1[r]) << getVvals1[0] << " " << graph_vvals1[r];
				}
				// check edge data
				ASSERT_EQ(getEvals0[0], graph_evals0[e])<< getEvals0[0] << " " << graph_evals0[e];
				ASSERT_EQ(getEvals1[0], graph_evals1[e])<< getEvals1[0] << " " << graph_evals1[e];

				status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph2);
				ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
			}
		}
	}

	//@TODO: we need somehow check extraction by multiple edges
	//@TODO: this would make sense only if vertices order is perserved in the extracted subgraph
	int edges[2] = { 1, 3 };
	status = nvgraphExtractSubgraphByEdge(nvgraph_handle, initial_graph, temp_graph1, edges, 2);
	ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	/*size_t nverts1 = 0, nedges1 = 0;
	 status = nvgraphGetGraphNvertices(nvgraph_handle, temp_graph1, &nverts1);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(nverts1, 3);
	 status = nvgraphGetGraphNedges(nvgraph_handle, temp_graph1, &nedges1);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(nedges1, 2);

	 // check structure:
	 int neighborget[4];
	 status = nvgraphGetGraphNeighborhood(nvgraph_handle, temp_graph1, neighborget);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(neighborget[0], 0);
	 ASSERT_EQ(neighborget[1], 1);
	 ASSERT_EQ(neighborget[2], 2);
	 ASSERT_EQ(neighborget[3], 2);
	 int edgeget[2];
	 status = nvgraphGetGraphEdgeDest( nvgraph_handle, temp_graph1, edgeget);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(edgeget[0], 2);
	 ASSERT_EQ(edgeget[1], 0);

	 status = nvgraphGetVertexData(nvgraph_handle, temp_graph1, (void *)getVvals0, 0);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(getVvals0[0], vertexvals0[0]);
	 ASSERT_EQ(getVvals0[1], vertexvals0[2]);
	 ASSERT_EQ(getVvals0[2], vertexvals0[3]);
	 status = nvgraphGetVertexData(nvgraph_handle, temp_graph1, (void *)getVvals1, 1);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(getVvals1[0], vertexvals1[0]);
	 ASSERT_EQ(getVvals1[1], vertexvals1[2]);
	 ASSERT_EQ(getVvals1[2], vertexvals1[3]);

	 status = nvgraphGetEdgeData(nvgraph_handle, temp_graph1, (void *)getEvals0, 0);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(getEvals0[0], edgevals0[edges[0]]);
	 ASSERT_EQ(getEvals0[1], edgevals0[edges[1]]);
	 status = nvgraphGetEdgeData(nvgraph_handle, temp_graph1, (void *)getEvals1, 1);
	 ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	 ASSERT_EQ(getEvals1[0], edgevals1[edges[0]]);
	 ASSERT_EQ(getEvals1[1], edgevals1[edges[1]]);*/

	status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
	ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}

TEST_F(NvgraphCAPITests_SubgraphCSR, CSRSubgraphEdges_CornerCases)
{
	nvgraphStatus_t status;

	nvgraphGraphDescr_t temp_graph1 = NULL, temp_graph2 = NULL;

// expected failures
	{
		int edges[2] = { 1, 3 };
		status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		// bad library nvgraph_handle
		status = nvgraphExtractSubgraphByEdge(NULL, initial_graph, temp_graph2, edges, 1);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
		// bad descriptor 1
		status = nvgraphExtractSubgraphByEdge(nvgraph_handle, temp_graph2, temp_graph1, edges, 1);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
		// bad descriptor 2
		status = nvgraphExtractSubgraphByEdge(nvgraph_handle, initial_graph, temp_graph2, edges, 1);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
		// NULL pointer
		status = nvgraphExtractSubgraphByEdge(	nvgraph_handle,
															initial_graph,
															temp_graph1,
															(int*) NULL,
															1);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		// extract zero edges - failure expected
		status = nvgraphExtractSubgraphByEdge(nvgraph_handle, initial_graph, temp_graph1, edges, 0);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		// bad edge number - in the C API we ask array consist of existing col_indices
		int bad_edge[1] = { -10 };
		status = nvgraphExtractSubgraphByEdge(	nvgraph_handle,
															initial_graph,
															temp_graph1,
															bad_edge,
															1);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		// more edges than exists in the graph - in the C API we ask array consist of existing col_indices
		int too_many_edges[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		status = nvgraphExtractSubgraphByEdge(	nvgraph_handle,
															initial_graph,
															temp_graph1,
															too_many_edges,
															10);
		ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

		status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}

	// not connected edges, which should create not connected graph
	{
		int edges[2] = { 0, 8 };
		status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphExtractSubgraphByEdge(nvgraph_handle, initial_graph, temp_graph1, edges, 2);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		nvgraphCSRTopology32I_st tData;
		tData.source_offsets = NULL;
		tData.destination_indices = NULL;
		status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph1, (void*) &tData, NULL);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		// we extracting 2 edges: one between two vertices and another is from third vertex to itself
		ASSERT_EQ(tData.nvertices, 3);
		ASSERT_EQ(tData.nedges, 2);

		status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}

	// triangle.
	{
		int edges[2] = { 0, 2 };
		status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphExtractSubgraphByEdge(nvgraph_handle, initial_graph, temp_graph1, edges, 2);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		nvgraphCSRTopology32I_st tData;
		tData.source_offsets = NULL;
		tData.destination_indices = NULL;
		status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph1, (void*) &tData, NULL);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		// we extracting 2 edges, expecting new graph have 3 vertices and only 2 edges
		ASSERT_EQ(tData.nvertices, 3);
		ASSERT_EQ(tData.nedges, 2);

		status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}

	// extract by edge to the self
	{
		int edges[1] = { 8 };
		status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphExtractSubgraphByEdge(nvgraph_handle, initial_graph, temp_graph1, edges, 1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		nvgraphCSRTopology32I_st tData;
		tData.source_offsets = NULL;
		tData.destination_indices = NULL;
		status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph1, (void*) &tData, NULL);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		// we extracting 1 edge to the vertex itself, expecting new graph have only 1 vertex and 1 edge
		ASSERT_EQ(tData.nvertices, 1);
		ASSERT_EQ(tData.nedges, 1);

		status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}
}

TEST_F(NvgraphCAPITests_SubgraphCSR, CSRContractionNetworkX)
{
	nvgraphStatus_t status;

	try {
		nvgraphGraphDescr_t netx_graph = NULL;
		nvgraphGraphDescr_t extracted_graph = NULL;

		status = nvgraphCreateGraphDescr(nvgraph_handle, &netx_graph);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		status = nvgraphCreateGraphDescr(nvgraph_handle, &extracted_graph);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		std::string fname(convert_to_local_path("graphs/networkx/extr_test.dat"));

		std::vector<int> g_row_offsets;
		std::vector<int> g_col_indices;

		std::vector<int> aggregates;
		std::vector<int> cg_row_offsets;
		std::vector<int> cg_col_indices;

		fill_extraction_data(fname,
									g_row_offsets,
									g_col_indices,
									aggregates,
									cg_row_offsets,
									cg_col_indices);

		//std::cout<<"********* step 1: \n";

		ASSERT_EQ(g_row_offsets.empty(), false);
		ASSERT_EQ(g_col_indices.empty(), false);
		ASSERT_EQ(aggregates.empty(), false);
		ASSERT_EQ(cg_row_offsets.empty(), false);
		ASSERT_EQ(cg_col_indices.empty(), false);

		//std::cout<<"********* step 1.1: \n";

		ASSERT_EQ(g_col_indices.size(), g_row_offsets.back());
		ASSERT_EQ(cg_col_indices.size(), cg_row_offsets.back());

		//std::cout<<"********* step 1.2: \n";

		nvgraphCSRTopology32I_st topoData;
		topoData.nvertices = g_row_offsets.size() - 1;    //last is nnz
		topoData.nedges = g_col_indices.size();

		//std::cout<<"(n,m):"<<topoData.nvertices
		//         <<", "<<topoData.nedges<<std::endl;

		topoData.source_offsets = &g_row_offsets[0];
		topoData.destination_indices = &g_col_indices[0];

		//std::cout<<"********* step 1.3: \n";

		status = nvgraphSetGraphStructure(nvgraph_handle,
														netx_graph,
														(void*) &topoData,
														NVGRAPH_CSR_32);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//std::cout<<"********* step 2: \n";

		size_t numsets = 1;

		std::vector<float> vdata(topoData.nvertices, 1.);
		void* vptr[] = { (void*) &vdata[0] };
		cudaDataType_t type_v[] = { CUDA_R_32F };

		std::vector<float> edata(topoData.nedges, 1.);
		void* eptr[] = { (void*) &edata[0] };
		cudaDataType_t type_e[] = { CUDA_R_32F };

		status = nvgraphAllocateVertexData(nvgraph_handle,
														netx_graph,
														numsets,
														type_v);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//std::cout<<"********* step 3: \n";

		status = nvgraphSetVertexData(nvgraph_handle,
												netx_graph,
												(void *) vptr[0],
												0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//std::cout<<"********* step 4: \n";

		status = nvgraphAllocateEdgeData(nvgraph_handle,
													netx_graph,
													numsets,
													type_e);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//std::cout<<"********* step 5: \n";

		status = nvgraphSetEdgeData(nvgraph_handle,
												netx_graph,
												(void *) eptr[0],
												0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//std::cout<<"********* step 6: \n";

		status = nvgraphExtractSubgraphByVertex(nvgraph_handle,
																netx_graph,
																extracted_graph,
																&aggregates[0],
																aggregates.size());
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//std::cout<<"********* step 7: \n";

		nvgraphCSRTopology32I_st tData;
		tData.source_offsets = NULL;
		tData.destination_indices = NULL;

		//1st time to get nvertices and nedges
		//
		status = nvgraphGetGraphStructure(nvgraph_handle, extracted_graph, (void*) &tData, NULL);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//std::cout<<"********* step 8: \n";

		int cgnv = cg_row_offsets.size() - 1;
		int cgne = cg_col_indices.size();
		ASSERT_EQ(tData.nvertices, cgnv);
		ASSERT_EQ(tData.nedges, cgne);

		//std::cout<<"********* step 9: \n";

		std::vector<int> cgro(cgnv + 1, 0);
		std::vector<int> cgci(cgne, 0);

		tData.source_offsets = &cgro[0];
		tData.destination_indices = &cgci[0];

		//2nd time to get row_offsets and column_indices
		//
		status = nvgraphGetGraphStructure(nvgraph_handle, extracted_graph, (void*) &tData, NULL);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		//std::cout << "cg row_offsets:\n";
		//std::copy(cgro.begin(), cgro.end(),
		//          std::ostream_iterator<int>(std::cout,"\n"));

		//std::cout << "cg col_indices:\n";
		//std::copy(cgci.begin(), cgci.end(),
		//          std::ostream_iterator<int>(std::cout,"\n"));

		//PROBLEM: might differ due to different vertex numbering
		//
		///ASSERT_EQ(check_diffs(cg_row_offsets, cgro), false);
		///ASSERT_EQ(check_diffs(cg_col_indices, cgci), false);

		//this is one invariant we can check, besides vector sizes:
		//
		ASSERT_EQ(check_delta_invariant(cg_row_offsets, cgro), true);

		//std::cout<<"********* step 10: \n";

		status = nvgraphDestroyGraphDescr(nvgraph_handle, extracted_graph);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		status = nvgraphDestroyGraphDescr(nvgraph_handle, netx_graph);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
	}
	catch (const file_read_error& ex)
	{
		std::cout << "Exception: " << ex.what() << ", waiving the test\n";
		const ::testing::TestInfo* const test_info =
				::testing::UnitTest::GetInstance()->current_test_info();
		std::cout << "[  WAIVED  ] " << test_info->test_case_name() << "." << test_info->name()
				<< std::endl;
		return;
	}
	catch (const std::exception& ex)
	{
		// dump exception:
		ASSERT_TRUE(false)<< "Exception: " << ex.what();
	}
	catch(...)
	{
		ASSERT_TRUE(false) << "Exception: Unknown";
	}
}

int main(int argc, char **argv)
			{
	::testing::InitGoogleTest(&argc, argv);
	for (int i = 0; i < argc; i++)
			{
		if (strcmp(argv[i], "--ref-data-dir") == 0)
			ref_data_prefix = std::string(argv[i + 1]);
		if (strcmp(argv[i], "--graph-data-dir") == 0)
			graph_data_prefix = std::string(argv[i + 1]);
	}
	return RUN_ALL_TESTS();
}
