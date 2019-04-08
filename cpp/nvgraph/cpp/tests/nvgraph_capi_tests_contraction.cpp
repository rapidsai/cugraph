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

//annonymus:
namespace{
template<typename Vector>
void fill_contraction_data(const std::string& fname,
                           Vector& g_row_offsets,
                           Vector& g_col_indices,
                           Vector& aggregates,
                           Vector& cg_row_offsets,
                           Vector& cg_col_indices)
{
  typedef typename Vector::value_type T;
  std::ifstream m_stream(fname.c_str(), std::ifstream::in);
  std::string line;

  if( !m_stream.is_open() )
    {
      std::stringstream ss;
      ss<<"ERROR: Could not open file: "<<fname;
      throw std::runtime_error(ss.str().c_str());
    }

  bool keep_going = !std::getline(m_stream, line).eof();

  //debug:
  //std::cout<<line<<std::endl;

  if( !keep_going )
    return;

  char c;
  int g_nrows=0;
  int g_nnz=0;
  std::sscanf(line.c_str(),"%c: nrows=%d, nnz=%d",&c, &g_nrows, &g_nnz);

  //debug:
  //std::cout<<c<<","<<g_nrows<<","<<g_nnz<<"\n";
  int n_entries = g_nrows+1;
  g_row_offsets.reserve(n_entries);

  //ignore next line:
  //
  if( !std::getline(m_stream, line) ) return;

  //read G row_offsets:
  for(int i=0;(i<n_entries) && keep_going;++i)
    {
      T value(0);
      
      keep_going = !std::getline(m_stream, line).eof();
      std::stringstream ss(line);
      ss >> value;
      g_row_offsets.push_back(value);
    }

  //ignore next 2 lines:
  //
  if( !std::getline(m_stream, line) || !std::getline(m_stream, line) ) return;

  g_col_indices.reserve(g_nnz);

  //read G col_indices:
  for(int i=0;(i<g_nnz) && keep_going;++i)
    {
      T value(0);
      
      keep_going = !std::getline(m_stream, line).eof();
      std::stringstream ss(line);
      ss >> value;
      g_col_indices.push_back(value);
    }

  //ignore next line:
  //
  if( !std::getline(m_stream, line) ) return;

  //remove the following for extraction:
  //{
  if( !std::getline(m_stream, line) ) return;
  int n_aggs = 0;
  std::sscanf(line.c_str(),"aggregate: size=%d",&n_aggs);

  //assert( n_aggs == g_nrows );//not true for subgraph extraction!

  aggregates.reserve(n_aggs);

  //read aggregate:
  for(int i=0;(i<n_aggs) && keep_going;++i)
    {
      T value(0);
      
      keep_going = !std::getline(m_stream, line).eof();
      std::stringstream ss(line);
      ss >> value;
      aggregates.push_back(value);
    }
  //} end remove code for extraction
  
  if( !keep_going || !std::getline(m_stream, line) ) return;
  int cg_nrows=0;
  int cg_nnz=0;
  std::sscanf(line.c_str(),"result %c: nrows=%d, nnz=%d",&c, &cg_nrows, &cg_nnz);

  //debug:
  std::cout<<c<<","<<cg_nrows<<","<<cg_nnz<<"\n";

  //
  //m_stream.close();//not really needed...destructor handles this
  //return;

  
  n_entries = cg_nrows+1;
  cg_row_offsets.reserve(n_entries);

  //ignore next line:
  //
  if( !std::getline(m_stream, line) ) return;

  //read G row_offsets:
  for(int i=0;(i<n_entries) && keep_going;++i)
    {
      T value(0);
      
      keep_going = !std::getline(m_stream, line).eof();
      std::stringstream ss(line);
      ss >> value;
      cg_row_offsets.push_back(value);
    }

  //ignore next 2 lines:
  //
  if( !std::getline(m_stream, line) || !std::getline(m_stream, line) ) return;

  cg_col_indices.reserve(cg_nnz);

  //read G col_indices:
  for(int i=0;(i<cg_nnz) && keep_going;++i)
    {
      T value(0);
      
      keep_going = !std::getline(m_stream, line).eof();
      std::stringstream ss(line);
      ss >> value;
      cg_col_indices.push_back(value);
    }
  

  m_stream.close();//not really needed...destructor handles this
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

  if( std::find_if(v.begin(), v.end(), std::bind2nd(std::not_equal_to<T>(), 0)) != v.end() )
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
  assert( sz == r2.size() );

  Vector d1(sz-1);

  std::transform(r1.begin()+1, r1.end(),
                 r1.begin(),
                 d1.begin(),
                 std::minus<int>());

  Vector d2(sz-1);

  std::transform(r2.begin()+1, r2.end(),
                 r2.begin(),
                 d2.begin(),
                 std::minus<int>());

  std::sort(d1.begin(), d1.end());
  std::sort(d2.begin(), d2.end());

  return (d1 == d2);
}
}


class NvgraphCAPITests_ContractionCSR : public ::testing::Test {
  public:
    NvgraphCAPITests_ContractionCSR() : nvgraph_handle(NULL), initial_graph(NULL) {}

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
        int neighborhood[] = {0, 2, 3, 5, 7, 9};     //row_offsets
        int edgedest[] = {1, 3, 3, 1, 4, 0, 2, 2, 4};//col_indices
        topoData.source_offsets = neighborhood;
        topoData.destination_indices = edgedest;
        status = nvgraphSetGraphStructure(nvgraph_handle, initial_graph,(void*) &topoData, NVGRAPH_CSR_32);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        // set up graph data
        size_t numsets = 2;
        float vertexvals0[] = {0.1f, 0.15893e-20f, 1e27f, 13.2f, 0.f};
        float vertexvals1[] = {13., 322.64, 1e28, -1.4, 22.3};
        void*  vertexptr[] = {(void*)vertexvals0, (void*)vertexvals1};
        cudaDataType_t type_v[] = {CUDA_R_32F, CUDA_R_32F};
        float edgevals0[] = {0.1f, 0.9153e-20f, 0.42e27f, 185.23, 1e21f, 15.6f, 215.907f, 912.2f, 0.2f};
        float edgevals1[] = {13., 322.64, 1e28, 197534.2, 0.1, 0.425e-5, 5923.4, 0.12e-12, 52.};
        void*  edgeptr[] = {(void*)edgevals0, (void*)edgevals1};
        cudaDataType_t type_e[] = {CUDA_R_32F, CUDA_R_32F};

        status = nvgraphAllocateVertexData(nvgraph_handle, initial_graph, numsets, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(nvgraph_handle, initial_graph, (void *)vertexptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(nvgraph_handle, initial_graph, (void *)vertexptr[1], 1 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphAllocateEdgeData(nvgraph_handle, initial_graph, numsets, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(nvgraph_handle, initial_graph, (void *)edgeptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(nvgraph_handle, initial_graph, (void *)edgeptr[1], 1 );
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
        if (nvgraph_handle != NULL)
        {
          status = nvgraphDestroyGraphDescr(nvgraph_handle, initial_graph);
          ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
          nvgraph_handle = NULL;
        }
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
 
TEST_F(NvgraphCAPITests_ContractionCSR, CSRContractionTestCreation)
{
    nvgraphStatus_t status;
    nvgraphGraphDescr_t temp_graph1 = NULL;//, temp_graph2 = NULL;

    {
        status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph1);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        //size_t numaggregates = 3;
        size_t szaggregates = 5;
        int aggregates[] = {0, 1, 1, 0, 2};

        //exception is being dumped by GTEST after [RUN]!
        //so try-catch is not needed and it doesn't help with that
        //
        try{
          int mult = 0;
          int sum = 1;
          status = nvgraphContractGraph(nvgraph_handle, initial_graph, temp_graph1,
                                        aggregates, 
                                        szaggregates, 
                                        (nvgraphSemiringOps_t)mult,
                                        (nvgraphSemiringOps_t)sum,
                                        (nvgraphSemiringOps_t)mult,
                                        (nvgraphSemiringOps_t)sum,
                                        0);//unused
        }
        catch( const std::exception& ex )
        {
          // dump exception:
          std::cerr<< "Exception:"<<ex.what()<<std::endl;//nope, but exception is being dumped by GTEST after [RUN]!
          
          //ASSERT_STREQ( "Exception:", ex.what() );//nope...
        }
        catch(...)
        {
          std::cerr<< "Exception: Unknown"<<std::endl;//nope, but exception is being dumped by GTEST after [RUN]!
          
          //ASSERT_STREQ( "Exception:", "Unknown" );//nope...
        }
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        		
		nvgraphCSRTopology32I_st tData;
		tData.source_offsets=NULL;
		tData.destination_indices=NULL;
		status = nvgraphGetGraphStructure(nvgraph_handle, temp_graph1, (void*) &tData, NULL);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

		const int nv = 3;
		const int ne = 7;

		ASSERT_EQ(tData.nvertices, nv);
		ASSERT_EQ(tData.nedges, ne);

		float  getVvals0[nv];
		float  getVvals1[nv];
		float  getEvals0[ne];
		float  getEvals1[ne];

		status = nvgraphGetVertexData(nvgraph_handle, temp_graph1, (void *)getVvals0, 0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetVertexData(nvgraph_handle, temp_graph1, (void *)getVvals1, 1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetEdgeData(nvgraph_handle, temp_graph1, (void *)getEvals0, 0);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
		status = nvgraphGetEdgeData(nvgraph_handle, temp_graph1, (void *)getEvals1, 1);
		ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
    
    status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph1);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}

TEST_F(NvgraphCAPITests_ContractionCSR, CSRContractionNegative)
{
    nvgraphStatus_t status;
    
    {
        nvgraphGraphDescr_t temp_graph2 = NULL;
        status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph2);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        size_t szaggregates = 3;
        int aggregates[] = {0, 1, 2};//this should fail because size of aggregates should match n_vertices of original graph

        //exception is being dumped by GTEST after [RUN]!
        //so try-catch is not needed and it doesn't help with that
        //
        try{
          int mult = 0;
          int sum = 1;
          status = nvgraphContractGraph(nvgraph_handle, initial_graph, temp_graph2,
                                        aggregates, 
                                        szaggregates, 
                                        (nvgraphSemiringOps_t)mult,
                                        (nvgraphSemiringOps_t)sum,
                                        (nvgraphSemiringOps_t)mult,
                                        (nvgraphSemiringOps_t)sum,
                                        0);//unused
        }
        catch( const std::exception& ex )
        {
          // dump exception:
          std::cerr<< "Exception:"<<ex.what()<<std::endl;//nope, but exception is being dumped by GTEST after [RUN]!
          
          //ASSERT_STREQ( "Exception:", ex.what() );//nope...
        }
        catch(...)
        {
          std::cerr<< "Exception: Unknown"<<std::endl;//nope, but exception is being dumped by GTEST after [RUN]!
          
          //ASSERT_STREQ( "Exception:", "Unknown" );//nope...
        }

        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

        status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph2);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }

    {
        nvgraphGraphDescr_t temp_graph2 = NULL;
        status = nvgraphCreateGraphDescr(nvgraph_handle, &temp_graph2);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        size_t szaggregates = 5;
        int aggregates[] = {0, 0, 1, 1, 3};//this should fail because not whole range [0..max(aggregates[])] is covered

        //exception is being dumped by GTEST after [RUN]!
        //so try-catch is not needed and it doesn't help with that
        //
        try{
          int mult = 0;
          int sum = 1;
          status = nvgraphContractGraph(nvgraph_handle, initial_graph, temp_graph2,
                                        aggregates, 
                                        szaggregates, 
                                        (nvgraphSemiringOps_t)mult,
                                        (nvgraphSemiringOps_t)sum,
                                        (nvgraphSemiringOps_t)mult,
                                        (nvgraphSemiringOps_t)sum,
                                        0);//unused
        }
        catch( const std::exception& ex )
        {
          // dump exception:
          std::cerr<< "Exception:"<<ex.what()<<std::endl;//nope, but exception is being dumped by GTEST after [RUN]!
          
          //ASSERT_STREQ( "Exception:", ex.what() );//nope...
        }
        catch(...)
        {
          std::cerr<< "Exception: Unknown"<<std::endl;//nope, but exception is being dumped by GTEST after [RUN]!
          
          //ASSERT_STREQ( "Exception:", "Unknown" );//nope...
        }

        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

        status = nvgraphDestroyGraphDescr(nvgraph_handle, temp_graph2);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
}

TEST_F(NvgraphCAPITests_ContractionCSR, CSRContractionNetworkX)
{
    nvgraphStatus_t status;
    
    try{
        nvgraphGraphDescr_t netx_graph       = NULL;
        nvgraphGraphDescr_t contracted_graph = NULL;
        
        status = nvgraphCreateGraphDescr(nvgraph_handle, &netx_graph);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        
        status = nvgraphCreateGraphDescr(nvgraph_handle, &contracted_graph);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        std::string fname("/mnt/nvgraph_test_data/graphs/networkx/ctr_test.dat");

        std::vector<int> g_row_offsets;
        std::vector<int> g_col_indices;

        std::vector<int> aggregates;
        std::vector<int> cg_row_offsets;
        std::vector<int> cg_col_indices;

        fill_contraction_data(fname,
                              g_row_offsets,
                              g_col_indices,
                              aggregates,
                              cg_row_offsets,
                              cg_col_indices);

        //std::cout<<"********* step 1: \n";

        ASSERT_EQ( g_row_offsets.empty(), false);
        ASSERT_EQ( g_col_indices.empty(), false);
        ASSERT_EQ(    aggregates.empty(), false);
        ASSERT_EQ(cg_row_offsets.empty(), false);
        ASSERT_EQ(cg_col_indices.empty(), false);

        //std::cout<<"********* step 1.1: \n";

        ASSERT_EQ( g_col_indices.size(),  g_row_offsets.back() );
        ASSERT_EQ( cg_col_indices.size(), cg_row_offsets.back());

        //std::cout<<"********* step 1.2: \n";

        nvgraphCSRTopology32I_st topoData;
        topoData.nvertices = g_row_offsets.size()-1;//last is nnz
        topoData.nedges = g_col_indices.size();

        //std::cout<<"(n,m):"<<topoData.nvertices
        //         <<", "<<topoData.nedges<<std::endl;

        topoData.source_offsets      = &g_row_offsets[0];
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
        void* vptr[] =  {(void*) &vdata[0]};
        cudaDataType_t type_v[] = {CUDA_R_32F};

        std::vector<float> edata(topoData.nedges,    1.);
        void* eptr[] =  {(void*) &edata[0]};
        cudaDataType_t type_e[] = {CUDA_R_32F};

        status = nvgraphAllocateVertexData(nvgraph_handle,
                                           netx_graph,
                                           numsets,
                                           type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        //std::cout<<"********* step 3: \n";
        
        status = nvgraphSetVertexData(nvgraph_handle,
                                      netx_graph,
                                      (void *)vptr[0],
                                      0 );
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
                                    (void *)eptr[0],
                                    0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        //std::cout<<"********* step 6: \n";

        int mult = 0;
        int sum = 1;
        status = nvgraphContractGraph(nvgraph_handle,
                                      netx_graph,
                                      contracted_graph,
                                      &aggregates[0], 
                                      aggregates.size(), 
                                      (nvgraphSemiringOps_t)mult,
                                      (nvgraphSemiringOps_t)sum,
                                      (nvgraphSemiringOps_t)mult,
                                      (nvgraphSemiringOps_t)sum,
                                      0);//unused

        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        //std::cout<<"********* step 7: \n";

        nvgraphCSRTopology32I_st tData;
		tData.source_offsets=NULL;
		tData.destination_indices=NULL;

        //1st time to get nvertices and nedges
        //
		status = nvgraphGetGraphStructure(nvgraph_handle, contracted_graph, (void*) &tData, NULL);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        //std::cout<<"********* step 8: \n";

        int cgnv = cg_row_offsets.size()-1;
        int cgne = cg_col_indices.size();
        ASSERT_EQ(tData.nvertices, cgnv);
		ASSERT_EQ(tData.nedges, cgne);

        //std::cout<<"********* step 9: \n";
        
        std::vector<int> cgro(cgnv+1, 0);
        std::vector<int> cgci(cgne, 0);

        tData.source_offsets = &cgro[0];
		tData.destination_indices = &cgci[0];

        //2nd time to get row_offsets and column_indices
        //
        status = nvgraphGetGraphStructure(nvgraph_handle, contracted_graph, (void*) &tData, NULL);
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
        ASSERT_EQ( check_delta_invariant( cg_row_offsets, cgro ), true);

        //std::cout<<"********* step 10: \n";

        status = nvgraphDestroyGraphDescr(nvgraph_handle, contracted_graph);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphDestroyGraphDescr(nvgraph_handle, netx_graph);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
    catch( const std::exception& ex )
      {
        // dump exception:
        std::cerr<< "Exception:"<<ex.what()<<std::endl;
      }
    catch(...)
      {
        std::cerr<< "Exception: Unknown"<<std::endl;
      }
}

int main(int argc, char **argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
