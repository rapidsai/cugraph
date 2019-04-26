#include "gtest/gtest.h"
#include "valued_csr_graph.hxx"
#include "nvgraphP.h"
#include "nvgraph.h"
#include <cstring>
class NvgraphAPITest : public ::testing::Test {
  public:
    NvgraphAPITest() : handle(NULL) {}

  protected:
    static void SetupTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        if (handle == NULL) {
            status = nvgraphCreate(&handle);
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
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    cudaStream_t *stream;
};


nvgraphCSRTopology32I_st topoData;
void createTopo()
{
//    nvgraphStatus_t mystatus;
    topoData.nvertices = 4;
    topoData.nedges = 5;
    int offsets[6];  //{0,1,3,4,5,5};
    offsets[0] = 0;
    offsets[1] = 1;
    offsets[2] = 3;
    offsets[3] = 4;
    offsets[4] = 5;
    offsets[5] = 5;
    topoData.source_offsets= offsets;

    int neighborhood[5];
    neighborhood[0]=0;
    neighborhood[1]=2;
    neighborhood[2]=3;
    neighborhood[3]=4;
    neighborhood[4]=4;

    topoData.destination_indices = neighborhood;

};
 
TEST_F(NvgraphAPITest,NvgraphCreateDestroy)
{
}


TEST_F(NvgraphAPITest,NvgraphStatusGetString )
{
  
        const char *ret_status_str;
        nvgraphStatus_t status = NVGRAPH_STATUS_SUCCESS; 
        ret_status_str = nvgraphStatusGetString( status);
        const std::string success_str = "Success";
    
        ASSERT_EQ( ret_status_str, success_str);

}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailNotInit)
{
//        nvgraphStatus_t status;
        //status = nvgraphDestroy( handle);
        const std::string not_init_str = "nvGRAPH not initialized";
        const char *ret_status_str;
        ret_status_str = nvgraphStatusGetString(NVGRAPH_STATUS_NOT_INITIALIZED); 
        ASSERT_EQ( ret_status_str, not_init_str);
}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailAllocFailed) 
{ 
        const char *ret_status_str;
        const std::string alloc_failed = "nvGRAPH alloc failed";
        ret_status_str = nvgraphStatusGetString(NVGRAPH_STATUS_ALLOC_FAILED);
        ASSERT_EQ( ret_status_str, alloc_failed);
}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailInvalidValue) 
{ 
        const char *ret_status_str;
        const std::string invalid_value = "nvGRAPH invalid value";
        ret_status_str = nvgraphStatusGetString(NVGRAPH_STATUS_INVALID_VALUE);
        ASSERT_EQ( ret_status_str, invalid_value);
}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailArchMismatch) 
{ 
        const char *ret_status_str;
        const std::string arch_mismatch = "nvGRAPH arch mismatch";
        ret_status_str = nvgraphStatusGetString(NVGRAPH_STATUS_ARCH_MISMATCH);
        ASSERT_EQ( ret_status_str, arch_mismatch);
}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailMappingError) 
{ 
        const char *ret_status_str;
        const std::string mapping_error = "nvGRAPH mapping error";
        ret_status_str = nvgraphStatusGetString(NVGRAPH_STATUS_MAPPING_ERROR);
        ASSERT_EQ( ret_status_str, mapping_error);
}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailExecFailed) 
{ 
        const char *ret_status_str;
        const std::string exec_failed = "nvGRAPH execution failed";
        ret_status_str = nvgraphStatusGetString(NVGRAPH_STATUS_EXECUTION_FAILED);
        ASSERT_EQ( ret_status_str, exec_failed);
}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailInternalError) 
{ 
        const char *ret_status_str;
        const std::string internal_error = "nvGRAPH internal error";
        ret_status_str = nvgraphStatusGetString(NVGRAPH_STATUS_INTERNAL_ERROR);
        ASSERT_EQ( ret_status_str, internal_error);
}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailTypeNotSupported) 
{ 
        const char *ret_status_str;
        const std::string type_not_supported = "nvGRAPH type not supported";
        ret_status_str = nvgraphStatusGetString(NVGRAPH_STATUS_TYPE_NOT_SUPPORTED);
        ASSERT_EQ( ret_status_str, type_not_supported);
}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailGraphTypeNotSupported) 
{ 
        const char *ret_status_str;
        const std::string type_not_supported = "nvGRAPH graph type not supported";
        ret_status_str = nvgraphStatusGetString(NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED);
        ASSERT_EQ( ret_status_str, type_not_supported);
}

TEST_F(NvgraphAPITest,NvgraphStatusGetStringFailUnknownNvgraphStatus) 
{
        const char *ret_status_str;
        const std::string unknown_nvgraph_status = "Unknown nvGRAPH Status";
        ret_status_str = nvgraphStatusGetString((nvgraphStatus_t)11);
        ASSERT_EQ( ret_status_str, unknown_nvgraph_status);
}

TEST_F(NvgraphAPITest,NvgraphCreateGraphDescr)
{
        nvgraphGraphDescr_t G=NULL;
        status = nvgraphCreateGraphDescr(handle, &G);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}

TEST_F(NvgraphAPITest,NvgraphCreateDestroyGraphDescr)
{
        nvgraphGraphDescr_t G=NULL;
        status = nvgraphCreateGraphDescr(handle, &G);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphDestroyGraphDescr(handle, G);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}

TEST_F(NvgraphAPITest,NvgraphCreateDestroyGraphDescr_CornerCases)
{
    nvgraphGraphDescr_t G = NULL;
    status = nvgraphDestroyGraphDescr(handle, G);
    ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
}

TEST_F(NvgraphAPITest,NvgraphGraphDescrSetCSRTopology)
{
    nvgraphGraphDescr_t descrG=NULL;
    status = nvgraphCreateGraphDescr(handle, &descrG);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    nvgraphCSRTopology32I_st topoData;
    topoData.nvertices = 0;
    topoData.nedges = 0;
    topoData.source_offsets = NULL;
    topoData.destination_indices = NULL;

    // Bad topology, missing all entries, should fail
    status=nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

    topoData.nvertices = 4;
    topoData.nedges = 4;

    // Bad topology, missing all offsets and indices, should fail
    status=nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

    int offsets[6];  //{0,1,3,4,5,5};
    offsets[0] = 0;
    offsets[1] = 1;
    offsets[2] = 3;
    offsets[3] = 4;
    offsets[4] = 5;
    offsets[5] = 5;
    topoData.source_offsets= offsets;
         
    // Bad topology, missing destination_indices, should fail
    status=nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

    int indices[4];
    indices[0] = 1;
    indices[0] = 2;
    indices[0] = 3;
    indices[0] = 4;
    topoData.destination_indices = indices;
    // Should be ok now
    status=nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    status = nvgraphDestroyGraphDescr(handle, descrG);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}


TEST_F(NvgraphAPITest,NvgraphGraphDescrSetGetTopologyCSR)
{
    nvgraphGraphDescr_t descrG=NULL;
    status = nvgraphCreateGraphDescr(handle, &descrG);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
  
    // 1, 0, 0, 0, 0, 0, 0
    // 0, 1, 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0, 0, 0
    // 1, 0, 0, 0, 0, 0, 1
    // 1, 1, 1, 0, 0, 0, 0
    // 0, 0, 0, 0, 0, 0, 0
    // 1, 1, 1, 0, 0, 0, 1
    // indptr=[0  1  2  2  4  7  7  11] // 8
    // indices=[0  1  0  6  0  1  2  0  1  2  6] // 11
    // n=7
    // nnz=11
    int rowPtr[] = {0, 1, 2, 2, 4, 7, 7, 11};
    int colInd[] = {0, 1, 0, 6, 0, 1, 2, 0, 1, 2, 6};
    
    nvgraphCSRTopology32I_st topoData;
    topoData.nedges = 11; // nnz
    topoData.nvertices = 7; // n
    topoData.source_offsets = rowPtr;
    topoData.destination_indices = colInd;

    status=nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    
    status=nvgraphGetGraphStructure(handle, descrG, NULL, NULL);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    // Check TType return value
    nvgraphTopologyType_t TType;
    status=nvgraphGetGraphStructure(handle, descrG, NULL, &TType);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(NVGRAPH_CSR_32, TType);

    // Check topoGet nedges and nvertices
    nvgraphCSRTopology32I_st topoDataGet;
    topoDataGet.nvertices=0;
    topoDataGet.nedges=0;
    topoDataGet.source_offsets=NULL;
    topoDataGet.destination_indices=NULL;
    status=nvgraphGetGraphStructure(handle, descrG, (void *)&topoDataGet, NULL);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(topoData.nvertices, topoDataGet.nvertices);
    ASSERT_EQ(topoData.nedges, topoDataGet.nedges);

    // Check topoGet nedges, nvertices and offsets
    topoDataGet.nvertices=0;
    topoDataGet.nedges=0;
    int rowPtrGet[8];
    rowPtrGet[0]=0;
    rowPtrGet[1]=0;
    rowPtrGet[2]=0;
    rowPtrGet[3]=0;
    rowPtrGet[4]=0;
    rowPtrGet[5]=0;
    rowPtrGet[6]=0;
    rowPtrGet[7]=0;
    topoDataGet.source_offsets=rowPtrGet;
    topoDataGet.destination_indices=NULL;
    status=nvgraphGetGraphStructure(handle, descrG, (void *)&topoDataGet, NULL);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(topoData.nvertices, topoDataGet.nvertices);
    ASSERT_EQ(topoData.nedges, topoDataGet.nedges);
    ASSERT_EQ(rowPtr[0], rowPtrGet[0]);
    ASSERT_EQ(rowPtr[1], rowPtrGet[1]);
    ASSERT_EQ(rowPtr[2], rowPtrGet[2]);
    ASSERT_EQ(rowPtr[3], rowPtrGet[3]);
    ASSERT_EQ(rowPtr[4], rowPtrGet[4]);
    ASSERT_EQ(rowPtr[5], rowPtrGet[5]);
    ASSERT_EQ(rowPtr[6], rowPtrGet[6]);
    ASSERT_EQ(rowPtr[7], rowPtrGet[7]);

    // Check topoGet
    topoDataGet.nvertices=0;
    topoDataGet.nedges=0;
    rowPtrGet[0]=0;
    rowPtrGet[1]=0;
    rowPtrGet[2]=0;
    rowPtrGet[3]=0;
    rowPtrGet[4]=0;
    rowPtrGet[5]=0;
    rowPtrGet[6]=0;
    rowPtrGet[7]=0;
    int colIndGet[11];
    colIndGet[0]=0;
    colIndGet[1]=0;
    colIndGet[2]=0;
    colIndGet[3]=0;
    colIndGet[4]=0;
    colIndGet[5]=0;
    colIndGet[6]=0;
    colIndGet[7]=0;
    colIndGet[8]=0;
    colIndGet[9]=0;
    colIndGet[10]=0;
    topoDataGet.source_offsets=rowPtrGet;
    topoDataGet.destination_indices=colIndGet;
    status=nvgraphGetGraphStructure(handle, descrG, (void *)&topoDataGet, NULL);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(topoData.nvertices, topoDataGet.nvertices);
    ASSERT_EQ(topoData.nedges, topoDataGet.nedges);
    ASSERT_EQ(rowPtr[0], rowPtrGet[0]);
    ASSERT_EQ(rowPtr[1], rowPtrGet[1]);
    ASSERT_EQ(rowPtr[2], rowPtrGet[2]);
    ASSERT_EQ(rowPtr[3], rowPtrGet[3]);
    ASSERT_EQ(rowPtr[4], rowPtrGet[4]);
    ASSERT_EQ(rowPtr[5], rowPtrGet[5]);
    ASSERT_EQ(rowPtr[6], rowPtrGet[6]);
    ASSERT_EQ(rowPtr[7], rowPtrGet[7]);
    ASSERT_EQ(colInd[0], colIndGet[0]);
    ASSERT_EQ(colInd[1], colIndGet[1]);
    ASSERT_EQ(colInd[2], colIndGet[2]);
    ASSERT_EQ(colInd[3], colIndGet[3]);
    ASSERT_EQ(colInd[4], colIndGet[4]);
    ASSERT_EQ(colInd[5], colIndGet[5]);
    ASSERT_EQ(colInd[6], colIndGet[6]);
    ASSERT_EQ(colInd[7], colIndGet[7]);
    ASSERT_EQ(colInd[8], colIndGet[8]);
    ASSERT_EQ(colInd[9], colIndGet[9]);
    ASSERT_EQ(colInd[10], colIndGet[10]);

    // Check all
    TType=NVGRAPH_CSC_32;
    topoDataGet.nvertices=0;
    topoDataGet.nedges=0;
    rowPtrGet[0]=0;
    rowPtrGet[1]=0;
    rowPtrGet[2]=0;
    rowPtrGet[3]=0;
    rowPtrGet[4]=0;
    rowPtrGet[5]=0;
    rowPtrGet[6]=0;
    rowPtrGet[7]=0;
    colIndGet[0]=0;
    colIndGet[1]=0;
    colIndGet[2]=0;
    colIndGet[3]=0;
    colIndGet[4]=0;
    colIndGet[5]=0;
    colIndGet[6]=0;
    colIndGet[7]=0;
    colIndGet[8]=0;
    colIndGet[9]=0;
    colIndGet[10]=0;
    topoDataGet.source_offsets=rowPtrGet;
    topoDataGet.destination_indices=colIndGet;
    status=nvgraphGetGraphStructure(handle, descrG, (void *)&topoDataGet, &TType);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(NVGRAPH_CSR_32, TType);
    ASSERT_EQ(topoData.nvertices, topoDataGet.nvertices);
    ASSERT_EQ(topoData.nedges, topoDataGet.nedges);
    ASSERT_EQ(rowPtr[0], rowPtrGet[0]);
    ASSERT_EQ(rowPtr[1], rowPtrGet[1]);
    ASSERT_EQ(rowPtr[2], rowPtrGet[2]);
    ASSERT_EQ(rowPtr[3], rowPtrGet[3]);
    ASSERT_EQ(rowPtr[4], rowPtrGet[4]);
    ASSERT_EQ(rowPtr[5], rowPtrGet[5]);
    ASSERT_EQ(rowPtr[6], rowPtrGet[6]);
    ASSERT_EQ(rowPtr[7], rowPtrGet[7]);
    ASSERT_EQ(colInd[0], colIndGet[0]);
    ASSERT_EQ(colInd[1], colIndGet[1]);
    ASSERT_EQ(colInd[2], colIndGet[2]);
    ASSERT_EQ(colInd[3], colIndGet[3]);
    ASSERT_EQ(colInd[4], colIndGet[4]);
    ASSERT_EQ(colInd[5], colIndGet[5]);
    ASSERT_EQ(colInd[6], colIndGet[6]);
    ASSERT_EQ(colInd[7], colIndGet[7]);
    ASSERT_EQ(colInd[8], colIndGet[8]);
    ASSERT_EQ(colInd[9], colIndGet[9]);
    ASSERT_EQ(colInd[10], colIndGet[10]);

    status = nvgraphDestroyGraphDescr(handle,descrG);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}

TEST_F(NvgraphAPITest,NvgraphGraphDescrSetGetTopologyCSC)
{
    nvgraphGraphDescr_t descrG=NULL;
    status = nvgraphCreateGraphDescr(handle, &descrG);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
  
    // 1, 0, 0, 0, 0, 0, 0
    // 0, 1, 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0, 0, 0
    // 1, 0, 0, 0, 0, 0, 1
    // 1, 1, 1, 0, 0, 0, 0
    // 0, 0, 0, 0, 0, 0, 0
    // 1, 1, 1, 0, 0, 0, 1
    // offsets=[0  4  7  9  9  9  9  11]
    // indices=[0  3  4  6  1  4  6  4  6  3  6]
    // n=7
    // nnz=11
    int rowInd[] = {0, 3, 4, 6, 1, 4, 6, 4, 6, 3, 6};
    int colPtr[] = {0, 4, 7, 9, 9, 9, 9, 11};
    
    nvgraphCSCTopology32I_st topoData;
    topoData.nedges = 11; // nnz
    topoData.nvertices = 7; // n
    topoData.destination_offsets = colPtr;
    topoData.source_indices = rowInd;

    status=nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    status=nvgraphGetGraphStructure(handle, descrG, NULL, NULL);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    // Check TType return value
    nvgraphTopologyType_t TType;
    status=nvgraphGetGraphStructure(handle, descrG, NULL, &TType);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(NVGRAPH_CSR_32, TType);

    // Check topoGet nedges and nvertices
    nvgraphCSCTopology32I_st topoDataGet;
    topoDataGet.nvertices=0;
    topoDataGet.nedges=0;
    topoDataGet.destination_offsets=NULL;
    topoDataGet.source_indices=NULL;
    status=nvgraphGetGraphStructure(handle, descrG, (void *)&topoDataGet, NULL);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(topoData.nvertices, topoDataGet.nvertices);
    ASSERT_EQ(topoData.nedges, topoDataGet.nedges);

    // Check topoGet nedges, nvertices and offsets
    topoDataGet.nvertices=0;
    topoDataGet.nedges=0;
    int colPtrGet[8];
    colPtrGet[0]=0;
    colPtrGet[1]=0;
    colPtrGet[2]=0;
    colPtrGet[3]=0;
    colPtrGet[4]=0;
    colPtrGet[5]=0;
    colPtrGet[6]=0;
    colPtrGet[7]=0;
    topoDataGet.destination_offsets=colPtrGet;
    topoDataGet.source_indices=NULL;
    status=nvgraphGetGraphStructure(handle, descrG, (void *)&topoDataGet, NULL);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(topoData.nvertices, topoDataGet.nvertices);
    ASSERT_EQ(topoData.nedges, topoDataGet.nedges);
    ASSERT_EQ(colPtr[0], colPtrGet[0]);
    ASSERT_EQ(colPtr[1], colPtrGet[1]);
    ASSERT_EQ(colPtr[2], colPtrGet[2]);
    ASSERT_EQ(colPtr[3], colPtrGet[3]);
    ASSERT_EQ(colPtr[4], colPtrGet[4]);
    ASSERT_EQ(colPtr[5], colPtrGet[5]);
    ASSERT_EQ(colPtr[6], colPtrGet[6]);
    ASSERT_EQ(colPtr[7], colPtrGet[7]);

    // Check topoGet
    topoDataGet.nvertices=0;
    topoDataGet.nedges=0;
    colPtrGet[0]=0;
    colPtrGet[1]=0;
    colPtrGet[2]=0;
    colPtrGet[3]=0;
    colPtrGet[4]=0;
    colPtrGet[5]=0;
    colPtrGet[6]=0;
    colPtrGet[7]=0;
    int rowIndGet[11];
    rowIndGet[0]=0;
    rowIndGet[1]=0;
    rowIndGet[2]=0;
    rowIndGet[3]=0;
    rowIndGet[4]=0;
    rowIndGet[5]=0;
    rowIndGet[6]=0;
    rowIndGet[7]=0;
    rowIndGet[8]=0;
    rowIndGet[9]=0;
    rowIndGet[10]=0;
    topoDataGet.destination_offsets=colPtrGet;
    topoDataGet.source_indices=rowIndGet;
    status=nvgraphGetGraphStructure(handle, descrG, (void *)&topoDataGet, NULL);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(topoData.nvertices, topoDataGet.nvertices);
    ASSERT_EQ(topoData.nedges, topoDataGet.nedges);
    ASSERT_EQ(colPtr[0], colPtrGet[0]);
    ASSERT_EQ(colPtr[1], colPtrGet[1]);
    ASSERT_EQ(colPtr[2], colPtrGet[2]);
    ASSERT_EQ(colPtr[3], colPtrGet[3]);
    ASSERT_EQ(colPtr[4], colPtrGet[4]);
    ASSERT_EQ(colPtr[5], colPtrGet[5]);
    ASSERT_EQ(colPtr[6], colPtrGet[6]);
    ASSERT_EQ(colPtr[7], colPtrGet[7]);
    ASSERT_EQ(rowInd[0], rowIndGet[0]);
    ASSERT_EQ(rowInd[1], rowIndGet[1]);
    ASSERT_EQ(rowInd[2], rowIndGet[2]);
    ASSERT_EQ(rowInd[3], rowIndGet[3]);
    ASSERT_EQ(rowInd[4], rowIndGet[4]);
    ASSERT_EQ(rowInd[5], rowIndGet[5]);
    ASSERT_EQ(rowInd[6], rowIndGet[6]);
    ASSERT_EQ(rowInd[7], rowIndGet[7]);
    ASSERT_EQ(rowInd[8], rowIndGet[8]);
    ASSERT_EQ(rowInd[9], rowIndGet[9]);
    ASSERT_EQ(rowInd[10], rowIndGet[10]);

    // Check all
    TType=NVGRAPH_CSC_32;
    topoDataGet.nvertices=0;
    topoDataGet.nedges=0;
    colPtrGet[0]=0;
    colPtrGet[1]=0;
    colPtrGet[2]=0;
    colPtrGet[3]=0;
    colPtrGet[4]=0;
    colPtrGet[5]=0;
    colPtrGet[6]=0;
    colPtrGet[7]=0;
    rowIndGet[0]=0;
    rowIndGet[1]=0;
    rowIndGet[2]=0;
    rowIndGet[3]=0;
    rowIndGet[4]=0;
    rowIndGet[5]=0;
    rowIndGet[6]=0;
    rowIndGet[7]=0;
    rowIndGet[8]=0;
    rowIndGet[9]=0;
    rowIndGet[10]=0;
    topoDataGet.destination_offsets=colPtrGet;
    topoDataGet.source_indices=rowIndGet;
    status=nvgraphGetGraphStructure(handle, descrG, (void *)&topoDataGet, &TType);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    ASSERT_EQ(NVGRAPH_CSR_32, TType);
    ASSERT_EQ(topoData.nvertices, topoDataGet.nvertices);
    ASSERT_EQ(topoData.nedges, topoDataGet.nedges);
    ASSERT_EQ(colPtr[0], colPtrGet[0]);
    ASSERT_EQ(colPtr[1], colPtrGet[1]);
    ASSERT_EQ(colPtr[2], colPtrGet[2]);
    ASSERT_EQ(colPtr[3], colPtrGet[3]);
    ASSERT_EQ(colPtr[4], colPtrGet[4]);
    ASSERT_EQ(colPtr[5], colPtrGet[5]);
    ASSERT_EQ(colPtr[6], colPtrGet[6]);
    ASSERT_EQ(colPtr[7], colPtrGet[7]);
    ASSERT_EQ(rowInd[0], rowIndGet[0]);
    ASSERT_EQ(rowInd[1], rowIndGet[1]);
    ASSERT_EQ(rowInd[2], rowIndGet[2]);
    ASSERT_EQ(rowInd[3], rowIndGet[3]);
    ASSERT_EQ(rowInd[4], rowIndGet[4]);
    ASSERT_EQ(rowInd[5], rowIndGet[5]);
    ASSERT_EQ(rowInd[6], rowIndGet[6]);
    ASSERT_EQ(rowInd[7], rowIndGet[7]);
    ASSERT_EQ(rowInd[8], rowIndGet[8]);
    ASSERT_EQ(rowInd[9], rowIndGet[9]);
    ASSERT_EQ(rowInd[10], rowIndGet[10]);

    status = nvgraphDestroyGraphDescr(handle,descrG);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}

TEST_F(NvgraphAPITest,NvgraphGraphDescrSetGetVertexDataSingleFloat)
{
    typedef float T;

    nvgraphGraphDescr_t descrG=NULL;
    status = nvgraphCreateGraphDescr(handle, &descrG);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
 
    /* Create topology before we load data */
    createTopo();

    status = nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        T *vertexvals;
        vertexvals = (T *) malloc(4*sizeof(T));
        vertexvals[0]=0.1;
        vertexvals[1]=2.0;
        vertexvals[2]=3.14;
        vertexvals[3]=0;

//        size_t numsets=1;

        cudaDataType_t type_v[1] = {sizeof(T) > 4 ? CUDA_R_64F : CUDA_R_32F};

        status = nvgraphAllocateVertexData(handle, descrG, 1, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphSetVertexData(handle, descrG, (void *)vertexvals, 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        T *getvals;
        getvals = (T *)malloc(4*sizeof(T));       
       
        status = nvgraphGetVertexData(handle, descrG, (void *)getvals, 0);
        ASSERT_EQ( getvals[0], vertexvals[0]);
        ASSERT_EQ( getvals[1], vertexvals[1]);
        ASSERT_EQ( getvals[2], vertexvals[2]);
        ASSERT_EQ( getvals[3], vertexvals[3]);

        free(vertexvals);
        free(getvals);
  
        status = nvgraphDestroyGraphDescr(handle,descrG);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}


TEST_F(NvgraphAPITest,NvgraphSetGetVertexDataSingleDouble)
{
    typedef double T;

        nvgraphGraphDescr_t descrG=NULL;
        status = nvgraphCreateGraphDescr(handle, &descrG);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    /* Create topology before we load data */
    createTopo();
    status = nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        T *vertexvals;
        vertexvals = (T *) malloc(4*sizeof(T));
        vertexvals[0]=0.1;
        vertexvals[1]=2.0;
        vertexvals[2]=3.14;
        vertexvals[3]=0;

//        size_t numsets=1;

        cudaDataType_t type_v[1] = {sizeof(T) > 4 ? CUDA_R_64F : CUDA_R_32F};

        status = nvgraphAllocateVertexData(handle, descrG, 1, type_v);
//        nvgraph::Graph<int> *G = static_cast<nvgraph::Graph<int>*> (descrG->graph_handle);

        //status = nvgraphSetVertexData(handle, descrG, (void **)&vertexvals, numsets, type_v );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetVertexData(handle, descrG, (void *)vertexvals, 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        T *getvals;
        getvals = (T *)malloc(4*sizeof(T));       
       
        status = nvgraphGetVertexData(handle, descrG, (void *)getvals, 0);
        ASSERT_EQ( getvals[0], vertexvals[0]);
        ASSERT_EQ( getvals[1], vertexvals[1]);
        ASSERT_EQ( getvals[2], vertexvals[2]);
        ASSERT_EQ( getvals[3], vertexvals[3]);

        free(vertexvals);
        free(getvals);
  
        status = nvgraphDestroyGraphDescr(handle,descrG);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}

TEST_F(NvgraphAPITest,NvgraphSetGetVertexData_CornerCases)
{
    nvgraphGraphDescr_t descrG=NULL;
    status = nvgraphCreateGraphDescr(handle, &descrG);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    /* Create topology before we load data */
    createTopo();
    status = nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        double vertexvals0[2] = {0.1, 1e21};
        float vertexvals1[2] = {0.1f, 1e21f};
        void* vertexptr[2] = {(void*) vertexvals0, (void*)vertexvals1};

        size_t numsets=2;

        cudaDataType_t type_v[2] = {CUDA_R_64F, CUDA_R_32F};
        status = nvgraphAllocateVertexData(handle, descrG, 1,  type_v);

        status = nvgraphSetVertexData(NULL, descrG, (void *)vertexptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSetVertexData(handle, NULL, (void *)vertexptr[0], 0 );
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSetVertexData(handle, descrG, NULL, numsets );
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        // probably should be a success
        status = nvgraphSetVertexData(handle, descrG, (void **)&vertexptr, 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphSetVertexData(handle, descrG, (void **)&vertexptr, numsets );
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

        {
            // type mismatch
//            double edge_data0 = 0.;
//            float edge_data1 =1.;
//            void* edge_ptr_bad[] = {(void*)&edge_data0, (void*)&edge_data1};
//            cudaDataType_t type_bad[2] = {CUDA_R_32F, CUDA_R_32F};
            //status = nvgraphSetEdgeData(handle, descrG, (void **)edge_ptr_bad, numsets );
            ASSERT_NE(NVGRAPH_STATUS_SUCCESS, status);
        }

        float getdoublevals0[2];
//        double getdoublevals1[2];
        status = nvgraphGetVertexData(NULL, descrG, (void *)getdoublevals0, 0);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphGetVertexData(handle, NULL, (void *)getdoublevals0, 0);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphGetVertexData(handle, descrG, (void *)NULL, 0);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphGetVertexData(handle, descrG, (void *)getdoublevals0, 10);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
  
        status = nvgraphDestroyGraphDescr(handle,descrG);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}


TEST_F(NvgraphAPITest,NvgraphSetGetVertexDataMulti)
{
        nvgraphGraphDescr_t descrG=NULL;
        status = nvgraphCreateGraphDescr(handle, &descrG);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    /* Create topology data */
    createTopo();
    status = nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

//        size_t numsets=3;
        cudaDataType_t type_v[3] = {CUDA_R_32F, CUDA_R_64F, CUDA_R_32F};

        void **vertexvals;
        vertexvals = (void **)malloc(3*sizeof( void * ));
        vertexvals[0] = (float *) malloc(4*sizeof(float));
        ((float *)vertexvals[0])[0]=0.1;
        ((float *)vertexvals[0])[1]=2.0;
        ((float *)vertexvals[0])[2]=3.14;
        ((float *)vertexvals[0])[3]=0;

        vertexvals[1] = (double *)malloc(4*sizeof(double));
        ((double *)vertexvals[1])[0]=1.1e-10;
        ((double *)vertexvals[1])[1]=2.0e20;
        ((double *)vertexvals[1])[2]=3.14e-26;
        ((double *)vertexvals[1])[3]=0.34e3;

        vertexvals[2] = (float *)malloc(4*sizeof(float));
        ((float *)vertexvals[2])[0]=1.1e-1;
        ((float *)vertexvals[2])[1]=2.0e2;
        ((float *)vertexvals[2])[2]=3.14e-2;
        ((float *)vertexvals[2])[3]=0.34e6;

        status = nvgraphAllocateVertexData(handle, descrG, 1,  type_v);

        float *getfloatvals;
        getfloatvals = (float *)malloc(4*sizeof(float));       
       
        status = nvgraphSetVertexData(handle, descrG, (void *)vertexvals[0], 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphGetVertexData(handle, descrG, (void *)getfloatvals, 0);
        float *float_data=((float *)vertexvals[0]);
        ASSERT_EQ( (float)getfloatvals[0], float_data[0]);
        ASSERT_EQ( (float)getfloatvals[1], float_data[1]);
        ASSERT_EQ( (float)getfloatvals[2], float_data[2]);
        ASSERT_EQ( (float)getfloatvals[3], float_data[3]);

        double *getdoublevals;
        getdoublevals = (double *)malloc(4*sizeof(double));       

        status = nvgraphSetVertexData(handle, descrG, (void *)vertexvals[1], 1);

        status = nvgraphGetVertexData(handle, descrG, (void *)getdoublevals, 1);
//        double *double_data=((double *)vertexvals[1]);
        //ASSERT_EQ( (double)getdoublevals[0], double_data[0]);
        //ASSERT_EQ( (double)getdoublevals[1], double_data[1]);
        //ASSERT_EQ( (double)getdoublevals[2], double_data[2]);
        //ASSERT_EQ( (double)getdoublevals[3], double_data[3]);
     
        free(vertexvals[0]); 
        free(vertexvals[1]); 
        free(vertexvals[2]); 
        free(vertexvals);
        free(getfloatvals);
        free(getdoublevals);
  
        status = nvgraphDestroyGraphDescr(handle,descrG);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}


TEST_F(NvgraphAPITest,NvgraphSetGetEdgeDataSingleFloat)
{
    typedef float T;

        nvgraphGraphDescr_t descrG=NULL;
        status = nvgraphCreateGraphDescr(handle, &descrG);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    /* Create topology */
    createTopo();
    status = nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        T *edgevals;
        edgevals = (T *) malloc(5*sizeof(T));
        edgevals[0]=0.1;
        edgevals[1]=2.0;
        edgevals[2]=3.14;
        edgevals[3]=0;
        edgevals[4]=10101.10101;

//        size_t numsets=1;

        cudaDataType_t type_v[1] = {sizeof(T) > 4 ? CUDA_R_64F : CUDA_R_32F};

        status = nvgraphAllocateEdgeData(handle, descrG, 1,  type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, descrG, (void *)edgevals, 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        T *getvals;
        getvals = (T *)malloc(5*sizeof(T));       

        status = nvgraphGetEdgeData(handle, descrG, (void *)getvals, 0);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    ASSERT_EQ( getvals[0], edgevals[0]);
    ASSERT_EQ( getvals[1], edgevals[1]);
    ASSERT_EQ( getvals[2], edgevals[2]);
    ASSERT_EQ( getvals[3], edgevals[3]);
    ASSERT_EQ( getvals[4], edgevals[4]);
 
        free(edgevals);
        free(getvals);
  
        status = nvgraphDestroyGraphDescr(handle,descrG);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}


TEST_F(NvgraphAPITest,NvgraphSetGetEdgeDataSingleDouble)
{
    typedef double T;

    nvgraphGraphDescr_t descrG=NULL;
    status = nvgraphCreateGraphDescr(handle, &descrG);  
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    /* Create topology */
    createTopo();
    status = nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    

        T *edgevals;
        edgevals = (T *) malloc(5*sizeof(T));
        edgevals[0]=0.1;
        edgevals[1]=2.0;
        edgevals[2]=3.14;
        edgevals[3]=0;
        edgevals[4]=10101.10101;

//        size_t numsets=1;

        cudaDataType_t type_v[1] = {sizeof(T) > 4 ? CUDA_R_64F : CUDA_R_32F};

        status = nvgraphAllocateEdgeData(handle, descrG, 1, type_v);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(handle, descrG, (void *)edgevals, 0 );
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        T *getvals;
        getvals = (T *)malloc(5*sizeof(T));       
        status = nvgraphGetEdgeData(handle, descrG, (void *)getvals, 0);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    ASSERT_EQ( getvals[0], edgevals[0]);
    ASSERT_EQ( getvals[1], edgevals[1]);
    ASSERT_EQ( getvals[2], edgevals[2]);
    ASSERT_EQ( getvals[3], edgevals[3]);
    ASSERT_EQ( getvals[4], edgevals[4]);
 
        free(edgevals);
        free(getvals);
  
        status = nvgraphDestroyGraphDescr(handle,descrG);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}


TEST_F(NvgraphAPITest,NvgraphSetGetEdgeData_CornerCases)
{
        nvgraphGraphDescr_t descrG=NULL;
        status = nvgraphCreateGraphDescr(handle, &descrG);  
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    /* Create topology */
    createTopo();
    status = nvgraphSetGraphStructure(handle, descrG, (void *)&topoData, NVGRAPH_CSR_32);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        double edgevals0[1] = {0.1};
        float edgevals1[1] = {0.1f};
        void* edgeptr[2] = {(void*) edgevals0, (void*)edgevals1};

//        size_t numsets=2;

        cudaDataType_t type_e[2] = {CUDA_R_64F, CUDA_R_32F};
  
        status = nvgraphAllocateEdgeData(handle, descrG, 1, type_e);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphSetEdgeData(NULL, descrG, edgeptr, 0);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSetEdgeData(handle, NULL, edgeptr, 0);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        status = nvgraphSetEdgeData(handle, descrG, NULL, 0);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        //status = nvgraphSetEdgeData(handle, descrG, edgeptr, 0);
        //ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);

        {
            // type mismatch
//            double vertexvals0[2] = {0.1, 1e21};
//            float vertexvals1[2] = {0.1f, 1e21f};
//            void* vertexptr_bad[2] = {(void*) vertexvals0, (void*)vertexvals1};
        
//            cudaDataType_t type_bad[2] = {CUDA_R_32F, CUDA_R_32F};
            //status = nvgraphSetVertexData(handle, descrG, (void **)vertexptr_bad, numsets, type_bad );
            ASSERT_NE(NVGRAPH_STATUS_SUCCESS, status);
        }

//        float getdoublevals0[2];
//        double getdoublevals1[2];
        //status = nvgraphGetEdgeData(NULL, descrG, (void *)getdoublevals0, 0);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        //status = nvgraphGetEdgeData(handle, NULL, (void *)getdoublevals0, 0);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        //status = nvgraphGetEdgeData(handle, descrG, NULL, 0);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
        //status = nvgraphGetEdgeData(handle, descrG, (void *)getdoublevals0, 10);
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
  
        status = nvgraphDestroyGraphDescr(handle,descrG);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
}

int main(int argc, char **argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
