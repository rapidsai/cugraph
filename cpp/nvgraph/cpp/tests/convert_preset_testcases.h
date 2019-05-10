#include <nvgraph.h>


// //                                  Simple Conversion Matrices (1)
// //-------------------------------------------------------------------------------------
// // Matrix A
// // 0.0  0.0  0.2  0.0  1.0
// // 0.3  0.7  0.0  1.2  0.0
// // 0.0  0.0  0.0  0.0  0.0
// // 0.0  0.0  8.6  0.0  0.0
// // 0.0  0.0  0.0  0.0  0.986410984960948401569841
// //
// // n            = 5;
// // m            = 5;
// // nnz          = 7;
// // csrVal = {0.2, 1.0, 0.3, 0.7, 1.2, 8.6, 0.986410984960948401569841};
// // csrColInd = {2, 4, 0, 1, 3, 2, 4};
// // csrRowPtr = {0, 2, 5, 5, 6, 7};
// //
// // cscVal = {0.3, 0.7, 0.2, 8.6, 1.2, 1.0, 0.986410984960948401569841};
// // cscRowInc = {1, 1, 0, 3, 1, 0, 4};
// // cscColPtr = {0, 1, 2, 4, 5, 7};
// //
// // COOSourceVal = {0.2, 1.0, 0.3, 0.7, 1.2, 8.6, 0.986410984960948401569841};
// // COOSourceRowInc = {0, 0, 1, 1, 1, 3, 4};
// // COOSourceColInc = {2, 4, 0, 1, 3, 2, 4};
// //
// // COODestVal = {0.3, 0.7, 0.2, 8.6, 1.2, 1.0, 0.986410984960948401569841};
// // COODestRowInc = {1, 1, 0, 3, 1, 0, 4};
// // COODestColInc = {0, 1, 2, 2, 3, 4, 4};
// //-------------------------------------------------------------------------------------
#define SIMPLE_TEST_1_N 5
#define SIMPLE_TEST_1_NNZ 7

int SIMPLE_CSR_SOURCE_OFFSETS[SIMPLE_TEST_1_N+1]      = {0, 2, 5, 5, 6, 7}; // rowPtr
int SIMPLE_CSR_DESTINATION_INDICES[SIMPLE_TEST_1_NNZ] = {2, 4, 0 ,1 ,3 ,2 ,4}; // colInd

int SIMPLE_CSC_SOURCE_INDICES[SIMPLE_TEST_1_NNZ]      = {1, 1, 0, 3, 1, 0, 4}; // rowInc
int SIMPLE_CSC_DESTINATION_OFFSETS[SIMPLE_TEST_1_N+1] = {0, 1, 2, 4, 5, 7}; // colPtr

int SIMPLE_COOS_SOURCE_INDICES[SIMPLE_TEST_1_NNZ]      = {0, 0, 1, 1, 1, 3, 4}; // row
int SIMPLE_COOS_DESTINATION_INDICES[SIMPLE_TEST_1_NNZ] = {2, 4, 0, 1, 3, 2, 4}; // col

int SIMPLE_COOD_SOURCE_INDICES[SIMPLE_TEST_1_NNZ]      = {1, 1, 0, 3, 1, 0, 4}; // row
int SIMPLE_COOD_DESTINATION_INDICES[SIMPLE_TEST_1_NNZ] = {0, 1, 2, 2, 3, 4, 4}; //col

int SIMPLE_COOU_SOURCE_INDICES[SIMPLE_TEST_1_NNZ]      = {4, 1, 0, 3, 0, 1, 1}; // row
int SIMPLE_COOU_DESTINATION_INDICES[SIMPLE_TEST_1_NNZ] = {4, 1, 2, 2, 4, 3, 0}; //col

const double SIMPLE_CSR_EDGE_DATA[SIMPLE_TEST_1_NNZ]  = {0.2, 1.0, 0.3, 0.7, 1.2, 8.6, 0.986410984960948401569841};
const double SIMPLE_CSC_EDGE_DATA[SIMPLE_TEST_1_NNZ]  = {0.3, 0.7, 0.2, 8.6, 1.2, 1.0, 0.986410984960948401569841};

const double SIMPLE_COOS_EDGE_DATA[SIMPLE_TEST_1_NNZ]  = {0.2, 1.0, 0.3, 0.7, 1.2, 8.6, 0.986410984960948401569841};
const double SIMPLE_COOD_EDGE_DATA[SIMPLE_TEST_1_NNZ]  = {0.3, 0.7, 0.2, 8.6, 1.2, 1.0, 0.986410984960948401569841};
const double SIMPLE_COOU_EDGE_DATA[SIMPLE_TEST_1_NNZ]  = {0.986410984960948401569841, 0.7, 0.2, 8.6, 1.0, 1.2, 0.3};


nvgraphCSRTopology32I_st simpleCsrTopo = {
    SIMPLE_TEST_1_N,
    SIMPLE_TEST_1_NNZ,
    SIMPLE_CSR_SOURCE_OFFSETS,
    SIMPLE_CSR_DESTINATION_INDICES
};
nvgraphCSCTopology32I_st simpleCscTopo = {
    SIMPLE_TEST_1_N,
    SIMPLE_TEST_1_NNZ,
    SIMPLE_CSC_DESTINATION_OFFSETS,
    SIMPLE_CSC_SOURCE_INDICES
};
nvgraphCOOTopology32I_st simpleCooSourceTopo = {
    SIMPLE_TEST_1_N,
    SIMPLE_TEST_1_NNZ,
    SIMPLE_COOS_SOURCE_INDICES,
    SIMPLE_COOS_DESTINATION_INDICES,
    NVGRAPH_SORTED_BY_SOURCE
};
nvgraphCOOTopology32I_st simpleCooDestTopo = {
    SIMPLE_TEST_1_N,
    SIMPLE_TEST_1_NNZ,
    SIMPLE_COOD_SOURCE_INDICES,
    SIMPLE_COOD_DESTINATION_INDICES,
    NVGRAPH_SORTED_BY_DESTINATION
};
nvgraphCOOTopology32I_st simpleCooUnsortedTopo = {
    SIMPLE_TEST_1_N,
    SIMPLE_TEST_1_NNZ,
    SIMPLE_COOU_SOURCE_INDICES,
    SIMPLE_COOU_DESTINATION_INDICES,
    NVGRAPH_UNSORTED
};

// //-------------------------------------------------------------------------------------

struct presetTestContainer_st{
    nvgraphCSRTopology32I_st* csrTopo;
    nvgraphCSCTopology32I_st* cscTopo;
    nvgraphCOOTopology32I_st*  coosTopo; // source
    nvgraphCOOTopology32I_st*  coodTopo; // dest
    nvgraphCOOTopology32I_st*  coouTopo; // unsorted
    const void* csrEdgeData;
    const void* cscEdgeData;
    const void* coosEdgeData;
    const void* coodEdgeData;
    const void* coouEdgeData;
};
typedef struct presetTestContainer_st *presetTestContainer_t;


// Hold all test data in one container
presetTestContainer_st simpleTest1 = {
    &simpleCsrTopo,
    &simpleCscTopo,
    &simpleCooSourceTopo,
    &simpleCooDestTopo,
    &simpleCooUnsortedTopo,
    SIMPLE_CSR_EDGE_DATA,
    SIMPLE_CSC_EDGE_DATA,
    SIMPLE_COOS_EDGE_DATA,
    SIMPLE_COOD_EDGE_DATA,
    SIMPLE_COOU_EDGE_DATA
};

//-------------------------------------------------------------------------------------
// Add your preset tests here
presetTestContainer_st presetTests[] = {simpleTest1};
