#include <vector>
// #include "boost/tuple/tuple.hpp"
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <limits>

#include "gtest/gtest.h"
#include "nvgraph.h"
#include <valued_csr_graph.hxx>
#include <multi_valued_csr_graph.hxx>
#include <nvgraphP.h>  // private header, contains structures, and potentially other things, used in the public C API that should never be exposed.

#include "convert_preset_testcases.h"

#define DEBUG_MSG std::cout      << "-----------> " << __FILE__ << " " << __LINE__ << std::endl;
#define DEBUG_VAR(var) std::cout << "-----------> " << __FILE__ << " " << __LINE__ << ": " << #var"=" << var << std::endl;


typedef enum
{
    CSR_32 = 0,
    CSC_32 = 1,
    COO_DEFAULT_32 = 2,
    COO_UNSORTED_32 = 3,
    COO_SOURCE_32 = 4,
    COO_DESTINATION_32 = 5
} testTopologyType_t;

// ref functions taken from cuSparse
template <typename T_ELEM>
void ref_csr2csc (int m, int n, int nnz, const T_ELEM *csrVals, const int *csrRowptr, const int *csrColInd, T_ELEM *cscVals, int *cscRowind, int *cscColptr, int base=0){
    int i,j, row, col, index;
    int * counters;
    T_ELEM val;

    /* early return */
    if ((m <= 0) || (n <= 0) || (nnz <= 0)){
        return;
    }

    /* build compressed column pointers */
    memset(cscColptr, 0, (n+1)*sizeof(cscColptr[0]));
    cscColptr[0]=base;
    for (i=0; i<nnz; i++){
        cscColptr[1+csrColInd[i]-base]++;
    }
    for(i=0; i<n; i++){
        cscColptr[i+1]+=cscColptr[i];
    }

    /* expand row indecis and copy them and values into csc arrays according to permutation */
    counters = (int *)malloc(n*sizeof(counters[0]));
    memset(counters, 0, n*sizeof(counters[0]));
    for (i=0; i<m; i++){
        for (j=csrRowptr[i]; j<csrRowptr[i+1]; j++){
            row = i+base;
            col = csrColInd[j-base];

            index=cscColptr[col-base]-base+counters[col-base];
            counters[col-base]++;

            cscRowind[index]=row;

            if(csrVals!=NULL || cscVals!=NULL){
                val = csrVals[j-base];
                cscVals[index]  = val;
            }
        }
    }
    free(counters);
}

// Not from cusparse (nvbug: 1762491)
static void ref_coo2csr(const int *cooRowindx, int nnz, int m, int *csrRowPtr, int base=0){

    memset(csrRowPtr, 0, sizeof(int)*(m+1) ); // Fill csrRowPtr with zeros
    for (int i=0; i<nnz; i++){ // fill csrRowPtr with number of nnz per row
        int idx = cooRowindx[i]-base;
        csrRowPtr[idx]++;
    }

    int t = base; // total sum
    for(int i=0; i<m; i++){
        int temp = csrRowPtr[i];
        csrRowPtr[i] = t;
        t += temp;
    }
    csrRowPtr[m] = nnz + base; // last element is trivial
}

void ref_csr2coo(const int *csrRowindx, int nnz, int m, int *cooRowindx){
    int base;

    cooRowindx[0] = csrRowindx[0];
    base = csrRowindx[0];

    for( int j = 0; j < m; j++) {
        int colStart = csrRowindx[j] - base;
        int colEnd   = csrRowindx[j+1]  - base;
        int rowNnz   = colEnd - colStart;

        for ( int i = 0; i < rowNnz; i++) {
            cooRowindx[colStart+i] = j + base;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////
// sort by row/col functions (not from cusparse)
////////////////////////////////////////////////////////////////////////////////////////////////
struct comparator{
    const std::vector<int>& values;
    comparator(const std::vector<int>& val_vec): values(val_vec) {}
    bool operator()(int n, int m){
        return values[n] < values[m];
    }
};

template<typename T>
void getSortPermutation(const std::vector<T>& minorOrder, const std::vector<T>& majorOrder, std::vector<int>& p){
    int n = majorOrder.size();
    p.clear();
    p.reserve(n);
    for(int i=0; i < n; ++i)
        p.push_back(i);

    std::stable_sort(p.begin(), p.end(), comparator(minorOrder)); // first "minor" sort
    std::stable_sort(p.begin(), p.end(), comparator(majorOrder)); // second "major" sort
}

template<typename T>
void ref_cooSortBySource(int n,
    const T *srcData, const int *srcRow, const int *srcCol,
    T *dstData, int *dstRow, int *dstCol){

    std::vector<int> srcR(srcRow, srcRow + n);
    std::vector<int> srcC(srcCol, srcCol + n);
    std::vector<int> p(n, 0);
    getSortPermutation(srcC, srcR, p); // sort p according to srcC

    for (int i=0; i<n ; i++) {
        dstRow[i]=srcRow[p[i]];
        dstCol[i]=srcCol[p[i]];
        dstData[i]=srcData[p[i]];
    }
}

template<typename T>
void ref_cooSortByDestination(int nnz,
    const T *srcData, const int *srcRow, const int *srcCol,
    T *dstData, int *dstRow, int *dstCol){
    ref_cooSortBySource(nnz, srcData, srcCol, srcRow, dstData, dstCol, dstRow);
}
////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
// Random generators
////////////////////////////////////////////////////////////////////////////////////////////////
void randomArray(int n, void* arr, cudaDataType_t *dataType){
    if(*dataType==CUDA_R_32F){
        float* a = (float*)arr;
        for(int i=0; i<n; ++i)
            a[i] = (float)rand()/(rand()+1); // don't divide by 0.
    } else if(*dataType==CUDA_R_64F) {
        double* a = (double*)arr;
        for(int i=0; i<n; ++i)
            a[i] = (double)rand()/(rand()+1); // don't divide by 0.
    } else {
        FAIL();
    }
}

void randomCOOGenerator( int *rowInd, int *colInd, int *nnz, int n,
                        int maxPerRow, int maxjump, int max_nnz) {

    int nnzCounter = 0;
    for(int row = 0 ; row<n && nnzCounter<max_nnz; row++){
        int elementsPerRow = 0;
        int col = 0;
        while( elementsPerRow<maxPerRow && nnzCounter<max_nnz ){
            int jump = (rand() % maxjump) +1;
            col += jump;
            if (col >= n)
                break;
            rowInd[nnzCounter] = row;
            colInd[nnzCounter] = col;
            nnzCounter++;
            elementsPerRow++;
        }
    }
    *nnz = nnzCounter;
}

void randomCsrGenerator( int *rowPtr, int *colInd, int *nnz, int n,
                         int maxPerRow, int maxjump, int max_nnz) {

    int *rowInd = (int*)malloc (sizeof(int)*max_nnz);
    randomCOOGenerator(rowInd, colInd, nnz, n, maxPerRow, maxjump, max_nnz);
    ref_coo2csr(rowInd, *nnz, n, rowPtr);
    free(rowInd);
}

typedef enum{
    HOST       = 0,
    DEVICE     = 1
} addressSpace_t;


class NVGraphAPIConvertTest : public ::testing::Test {
  public:
    nvgraphStatus_t status;
    nvgraphHandle_t handle;

    NVGraphAPIConvertTest() : handle(NULL) {}

    // static void SetupTestCase() {}
    // static void TearDownTestCase() {}
    virtual void SetUp() {
        if (handle == NULL) {
            status = nvgraphCreate(&handle);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }
        srand (time(NULL));
    }
    virtual void TearDown() {
        if (handle != NULL) {
            status =  nvgraphDestroy(handle);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
            handle = NULL;
        }
    }

    // CPU conversion (reference)
    template <typename T>
    static void refConvert(nvgraphTopologyType_t srcTType, void *srcTopology, const T *srcEdgeData,
                           nvgraphTopologyType_t dstTType, void *dstTopology, T *dstEdgeData){

        // Trust me, this a 100 times better than nested ifs.
        if(srcTType==NVGRAPH_CSR_32 && dstTType==NVGRAPH_CSR_32){                                                 // CSR2CSR
            nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t >(srcTopology);
            nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t >(dstTopology);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            memcpy(dstEdgeData, srcEdgeData, sizeof(T)*srcT->nedges);
            memcpy(dstT->source_offsets, srcT->source_offsets, sizeof(int)*(srcT->nvertices+1) );
            memcpy(dstT->destination_indices, srcT->destination_indices, sizeof(int)*(srcT->nedges) );

        } else if(srcTType==NVGRAPH_CSR_32 && dstTType==NVGRAPH_CSC_32) {                                         // CSR2CSC
            nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t >(srcTopology);
            nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t >(dstTopology);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            ref_csr2csc<T> (srcT->nvertices, srcT->nvertices, srcT->nedges,
                srcEdgeData, srcT->source_offsets, srcT->destination_indices,
                dstEdgeData, dstT->source_indices, dstT->destination_offsets);

        } else if(srcTType==NVGRAPH_CSR_32 && dstTType==NVGRAPH_COO_32) {                                         // CSR2COO
            nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t >(srcTopology);
            nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t >(dstTopology);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            if(dstT->tag==NVGRAPH_DEFAULT || dstT->tag==NVGRAPH_UNSORTED || dstT->tag==NVGRAPH_SORTED_BY_SOURCE){
                ref_csr2coo(srcT->source_offsets, srcT->nedges, srcT->nvertices, dstT->source_indices);
                memcpy(dstT->destination_indices, srcT->destination_indices, sizeof(int)*(srcT->nedges) );
                memcpy(dstEdgeData, srcEdgeData, sizeof(T)*(srcT->nedges) );
            } else if (dstT->tag==NVGRAPH_SORTED_BY_DESTINATION) {
                int* tmp=(int*)malloc(sizeof(int)*(dstT->nedges) );
                // Step 1: Convert to COO Source
                ref_csr2coo(srcT->source_offsets, srcT->nedges, srcT->nvertices, tmp);
                // Step 2: Convert to COO Dest
                ref_cooSortByDestination(srcT->nedges,
                    srcEdgeData, tmp, srcT->destination_indices,
                    dstEdgeData, dstT->source_indices, dstT->destination_indices);
                free(tmp);
            } else {
                FAIL();
            }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        } else if(srcTType==NVGRAPH_CSC_32 && dstTType==NVGRAPH_CSR_32) {                                         // CSC2CSR
            nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t >(srcTopology);
            nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t >(dstTopology);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            ref_csr2csc<T> (srcT->nvertices, srcT->nvertices, srcT->nedges,
                srcEdgeData, srcT->destination_offsets, srcT->source_indices,
                dstEdgeData, dstT->destination_indices, dstT->source_offsets);

        } else if(srcTType==NVGRAPH_CSC_32 && dstTType==NVGRAPH_CSC_32) {                                         // CSC2CSC
            nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t >(srcTopology);
            nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t >(dstTopology);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            memcpy(dstT->destination_offsets, srcT->destination_offsets, sizeof(int)*(srcT->nvertices+1) );
            memcpy(dstT->source_indices, srcT->source_indices, sizeof(int)*(srcT->nedges) );
            memcpy(dstEdgeData, srcEdgeData, sizeof(T)*(srcT->nedges) );

        } else if(srcTType==NVGRAPH_CSC_32 && dstTType==NVGRAPH_COO_32) {                                         // CSC2COO
            nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t >(srcTopology);
            nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t >(dstTopology);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            if(dstT->tag==NVGRAPH_SORTED_BY_SOURCE){
                int* tmp = (int*)malloc(sizeof(int)*(dstT->nedges));
                // Step 1: Convert to COO Dest
                ref_csr2coo(srcT->destination_offsets, srcT->nedges, srcT->nvertices, tmp);
                // Step 2: Convert to COO Source
                ref_cooSortBySource(srcT->nedges,
                    srcEdgeData, srcT->source_indices, tmp,
                    dstEdgeData, dstT->source_indices, dstT->destination_indices);
                free(tmp);
            } else if (dstT->tag==NVGRAPH_DEFAULT || dstT->tag==NVGRAPH_UNSORTED || dstT->tag==NVGRAPH_SORTED_BY_DESTINATION) {
                ref_csr2coo(srcT->destination_offsets, srcT->nedges, srcT->nvertices, dstT->destination_indices);
                memcpy(dstT->source_indices, srcT->source_indices, sizeof(int)*(srcT->nedges) );
                memcpy(dstEdgeData, srcEdgeData, sizeof(T)*(srcT->nedges) );
            } else {
                FAIL();
            }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        } else if(srcTType==NVGRAPH_COO_32 && dstTType==NVGRAPH_CSR_32) {                                         // COO2CSR
            nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t >(srcTopology);
            nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t >(dstTopology);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            if(srcT->tag==NVGRAPH_SORTED_BY_SOURCE){
                ref_coo2csr(srcT->source_indices, srcT->nedges, srcT->nvertices, dstT->source_offsets);
                memcpy(dstT->destination_indices, srcT->destination_indices, sizeof(int)*(srcT->nedges) );
                memcpy(dstEdgeData, srcEdgeData, sizeof(T)*(srcT->nedges) );

            } else if(srcT->tag==NVGRAPH_SORTED_BY_DESTINATION || srcT->tag==NVGRAPH_DEFAULT || srcT->tag==NVGRAPH_UNSORTED){
                int *tmp = (int*)malloc(sizeof(int)*(srcT->nedges) );
                // Step 1: convert to COO Dest
                ref_cooSortBySource(srcT->nedges,
                    srcEdgeData, srcT->source_indices, srcT->destination_indices,
                    dstEdgeData, tmp, dstT->destination_indices);
                // Step 1: convert to CSC
                ref_coo2csr(tmp, srcT->nedges, srcT->nvertices, dstT->source_offsets);
                free(tmp);
            } else {
                FAIL();
            }
        } else if(srcTType==NVGRAPH_COO_32 && dstTType==NVGRAPH_CSC_32) {                                         // COO2CSC
            nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t >(srcTopology);
            nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t >(dstTopology);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            if(srcT->tag==NVGRAPH_SORTED_BY_SOURCE || srcT->tag==NVGRAPH_DEFAULT || srcT->tag==NVGRAPH_UNSORTED){
                int *tmp = (int*)malloc(sizeof(int)*srcT->nedges);
                // Step 1: convert to COO dest
                ref_cooSortByDestination(srcT->nedges,
                    srcEdgeData, srcT->source_indices, srcT->destination_indices,
                    dstEdgeData, dstT->source_indices, tmp);
                // Step 1: convert to CSC
                ref_coo2csr(tmp, srcT->nedges, srcT->nvertices, dstT->destination_offsets);
                free(tmp);
            } else if(srcT->tag==NVGRAPH_SORTED_BY_DESTINATION) {
                ref_coo2csr(srcT->destination_indices, srcT->nedges, srcT->nvertices, dstT->destination_offsets);
                memcpy(dstT->source_indices, srcT->source_indices, sizeof(int)*(srcT->nedges) );
                memcpy(dstEdgeData, srcEdgeData, sizeof(T)*(srcT->nedges) );
            } else {
                FAIL();
            }
        } else if(srcTType==NVGRAPH_COO_32 && dstTType==NVGRAPH_COO_32) {                                         // COO2COO
            nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t >(srcTopology);
            nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t >(dstTopology);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            if(srcT->tag==dstT->tag || dstT->tag==NVGRAPH_DEFAULT || dstT->tag==NVGRAPH_UNSORTED) {
                memcpy(dstT->source_indices, srcT->source_indices, sizeof(int)*(srcT->nedges) );
                memcpy(dstT->destination_indices, srcT->destination_indices, sizeof(int)*(srcT->nedges) );
                memcpy(dstEdgeData, srcEdgeData, sizeof(T)*srcT->nedges);
            } else if(dstT->tag==NVGRAPH_SORTED_BY_SOURCE) {
                ref_cooSortBySource(srcT->nedges,
                    srcEdgeData, srcT->source_indices, srcT->destination_indices,
                    dstEdgeData, dstT->source_indices, dstT->destination_indices);
            } else if(dstT->tag==NVGRAPH_SORTED_BY_DESTINATION) {
                ref_cooSortByDestination(srcT->nedges,
                    srcEdgeData, srcT->source_indices, srcT->destination_indices,
                    dstEdgeData, dstT->source_indices, dstT->destination_indices);
            } else {
                FAIL();
            }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        } else {
            FAIL();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Topology  Helper functions
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // The function must be void
    static void topoGetN(testTopologyType_t TType, void *topo, int* n){
        int result=0;
        if(TType==CSR_32){
            nvgraphCSRTopology32I_t t = static_cast<nvgraphCSRTopology32I_t >(topo);
            result = t->nvertices;
        }
        else if(TType==CSC_32){
            nvgraphCSCTopology32I_t t = static_cast<nvgraphCSCTopology32I_t >(topo);
            result = t->nvertices;
        }
        else if(TType==COO_SOURCE_32 || TType==COO_DESTINATION_32 || TType==COO_UNSORTED_32 || TType==COO_DEFAULT_32){
            nvgraphCOOTopology32I_t t = static_cast<nvgraphCOOTopology32I_t >(topo);
            result = t->nvertices;
        }
        else{
            FAIL();
        }
        *n=result;
    }

    // The function must be void
    static void topoGetNNZ(testTopologyType_t TType, void *topo, int*n){
        int result=0;
        if(TType==CSR_32){
            nvgraphCSRTopology32I_t t = static_cast<nvgraphCSRTopology32I_t >(topo);
            result = t->nedges;
        }
        else if(TType==CSC_32){
            nvgraphCSCTopology32I_t t = static_cast<nvgraphCSCTopology32I_t >(topo);
            result = t->nedges;
        }
        else if(TType==COO_SOURCE_32 || TType==COO_DESTINATION_32 || TType==COO_UNSORTED_32 || TType==COO_DEFAULT_32){
            nvgraphCOOTopology32I_t t = static_cast<nvgraphCOOTopology32I_t >(topo);
            result = t->nedges;
        }
        else{
            FAIL();
        }
        *n=result;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocation/de-allocation functions
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    static void allocateTopo(void **topoPtr, testTopologyType_t TType, int n, int nnz, addressSpace_t aSpace){
        if(TType==CSR_32){
            *topoPtr=(nvgraphCSRTopology32I_t)malloc(sizeof(nvgraphCSRTopology32I_st));
            nvgraphCSRTopology32I_t p = static_cast<nvgraphCSRTopology32I_t >(*topoPtr);

            if(aSpace==HOST){
                p->source_offsets = (int*)malloc(sizeof(int)*(n+1));
                p->destination_indices = (int*)malloc(sizeof(int)*(nnz));
            } else if(aSpace==DEVICE){
                cudaMalloc((void**)&(p->source_offsets), sizeof(int)*(n+1));
                cudaMalloc((void**)&(p->destination_indices), sizeof(int)*(nnz));
            } else {
                FAIL();
            }
            p->nvertices = n;
            p->nedges = nnz;
        }
        else if(TType==CSC_32){
            *topoPtr=(nvgraphCSCTopology32I_t)malloc(sizeof(nvgraphCSCTopology32I_st));
            nvgraphCSCTopology32I_t p = static_cast<nvgraphCSCTopology32I_t >(*topoPtr);

            if(aSpace==HOST){
                p->destination_offsets = (int*)malloc(sizeof(int)*(n+1));
                p->source_indices = (int*)malloc(sizeof(int)*(nnz));
            } else if(aSpace==DEVICE){
                cudaMalloc((void**)&(p->destination_offsets), sizeof(int)*(n+1));
                cudaMalloc((void**)&(p->source_indices), sizeof(int)*(nnz));
            } else {
                FAIL();
            }
            p->nvertices = n;
            p->nedges = nnz;
        }
        else if(TType==COO_SOURCE_32 || TType==COO_DESTINATION_32 || TType==COO_UNSORTED_32 || TType==COO_DEFAULT_32){
            *topoPtr=(nvgraphCOOTopology32I_t)malloc(sizeof(nvgraphCOOTopology32I_st));
            nvgraphCOOTopology32I_t p = static_cast<nvgraphCOOTopology32I_t >(*topoPtr);

            if(aSpace==HOST){
                p->source_indices = (int*)malloc(sizeof(int)*(nnz));
                p->destination_indices = (int*)malloc(sizeof(int)*(nnz));
            } else if(aSpace==DEVICE){
                cudaMalloc((void**)&(p->source_indices), sizeof(int)*(nnz));
                cudaMalloc((void**)&(p->destination_indices), sizeof(int)*(nnz));
            } else {
                FAIL();
            }
            p->nvertices = n;
            p->nedges = nnz;

            if(TType==COO_SOURCE_32)
                p->tag=NVGRAPH_SORTED_BY_SOURCE;
            else if(TType==COO_DESTINATION_32)
                p->tag=NVGRAPH_SORTED_BY_DESTINATION;
            else if(TType==COO_UNSORTED_32)
                p->tag=NVGRAPH_UNSORTED;
            else if(TType==COO_DEFAULT_32)
                p->tag=NVGRAPH_DEFAULT;
            else
                FAIL();
        } else {
            FAIL();
        }
    }

    static void deAllocateTopo(void* topo, testTopologyType_t TType, addressSpace_t aSpace){
        if(topo==NULL)
            return;

        void *rowPtr, *colPtr;
        if(TType==CSR_32){
            nvgraphCSRTopology32I_t p = static_cast<nvgraphCSRTopology32I_t >(topo);
            rowPtr = p->source_offsets;
            colPtr = p->destination_indices;
            free(p);
        }
        else if(TType==CSC_32){
            nvgraphCSCTopology32I_t p = static_cast<nvgraphCSCTopology32I_t >(topo);
            rowPtr = p->source_indices;
            colPtr = p->destination_offsets;
            free(p);
        }
        else if(TType==COO_SOURCE_32 || TType==COO_DESTINATION_32 || TType==COO_UNSORTED_32 || TType==COO_DEFAULT_32){
            nvgraphCOOTopology32I_t p = static_cast<nvgraphCOOTopology32I_t >(topo);
            rowPtr = p->source_indices;
            colPtr = p->destination_indices;
            free(p);
        } else {
            FAIL();
        }

        if(aSpace==HOST){
            free(rowPtr);
            free(colPtr);
        } else if (aSpace==DEVICE){
            cudaFree(rowPtr);
            cudaFree(colPtr);
        } else {
            FAIL();
        }
    }

    static void cpyTopo(void *dst, void *src, testTopologyType_t TType, enum cudaMemcpyKind kind=cudaMemcpyDefault){

        int *srcRow=NULL, *srcCol=NULL;
        int *dstRow=NULL, *dstCol=NULL;
        int rowSize=0, colSize=0;
        if(TType==CSR_32) {
            nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t >(src);
            nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t >(dst);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            rowSize = srcT->nvertices+1; colSize = srcT->nedges;
            srcRow = srcT->source_offsets; dstRow = dstT->source_offsets;
            srcCol = srcT->destination_indices; dstCol = dstT->destination_indices;
        } else if(TType==CSC_32) {
            nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t >(src);
            nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t >(dst);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            rowSize = srcT->nedges; colSize = srcT->nvertices+1;
            srcRow = srcT->source_indices; dstRow = dstT->source_indices;
            srcCol = srcT->destination_offsets; dstCol = dstT->destination_offsets;
        } else if(TType==COO_SOURCE_32 || TType==COO_DESTINATION_32 || TType==COO_UNSORTED_32 || TType==COO_DEFAULT_32) {
            nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t >(src);
            nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t >(dst);
            dstT->nvertices = srcT->nvertices;
            dstT->nedges = srcT->nedges;
            dstT->tag = srcT->tag;
            rowSize = srcT->nedges; colSize = srcT->nedges;
            srcRow = srcT->source_indices; dstRow = dstT->source_indices;
            srcCol = srcT->destination_indices; dstCol = dstT->destination_indices;
        } else {
            FAIL();
        }

        ASSERT_EQ(cudaSuccess, cudaMemcpy(dstRow, srcRow, sizeof(int)*rowSize, kind));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dstCol, srcCol, sizeof(int)*colSize, kind));
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Comparison functions
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    static void cmpArray(T* ref, addressSpace_t refSapce, T* dst, addressSpace_t dstSpace, int n){

        T *_refData=NULL, *_dstData=NULL; if(refSapce==DEVICE){
            _refData = (T*)malloc(sizeof(T)*n);
            cudaMemcpy(_refData, ref, sizeof(T)*n, cudaMemcpyDefault);
        } else {
            _refData = ref;
        }

        if(dstSpace==DEVICE){
            _dstData = (T*)malloc(sizeof(T)*n);
            cudaMemcpy(_dstData, dst, sizeof(T)*n, cudaMemcpyDefault);
        } else {
            _dstData = dst;
        }
        std::vector<T> refData;
        std::vector<T> dstData;
        refData.assign(_refData, _refData + n);
        dstData.assign(_dstData, _dstData + n);

        for(int i=0; i<refData.size(); ++i)
            ASSERT_EQ(refData[i], dstData[i]);
        // ASSERT_EQ(refData, dstData);


        if(refSapce==DEVICE)  free(_refData);
        if(dstSpace==DEVICE)  free(_dstData);
    }

    static void cmpTopo(nvgraphTopologyType_t TType, void *refTopology, addressSpace_t refSpace, void *dstTopology, addressSpace_t dstSpace){

        int *_refRows=NULL, *_refCols=NULL;
        int *_dstRows=NULL, *_dstCols=NULL;
        int *refRowsHost=NULL, *refColsHost=NULL;
        int *dstRowsHost=NULL, *dstColsHost=NULL;

        int rowSize=0, colSize=0;
        if(TType==NVGRAPH_CSR_32){
            nvgraphCSRTopology32I_t _refTopology = static_cast<nvgraphCSRTopology32I_t >(refTopology);
            nvgraphCSRTopology32I_t _dstTopology = static_cast<nvgraphCSRTopology32I_t >(dstTopology);
            ASSERT_EQ( _refTopology->nvertices, _dstTopology->nvertices);
            ASSERT_EQ( _refTopology->nedges, _dstTopology->nedges);
            _refRows = _refTopology->source_offsets;
            _refCols = _refTopology->destination_indices;
            _dstRows = _dstTopology->source_offsets;
            _dstCols = _dstTopology->destination_indices;
            colSize = _refTopology->nedges;
            rowSize = _refTopology->nvertices + 1;
        }
        else if(TType==NVGRAPH_CSC_32){
            nvgraphCSCTopology32I_t _refTopology = static_cast<nvgraphCSCTopology32I_t >(refTopology);
            nvgraphCSCTopology32I_t _dstTopology = static_cast<nvgraphCSCTopology32I_t >(dstTopology);
            ASSERT_EQ( _refTopology->nvertices, _dstTopology->nvertices);
            ASSERT_EQ( _refTopology->nedges, _dstTopology->nedges);
            _refRows = _refTopology->source_indices;
            _refCols = _refTopology->destination_offsets;
            _dstRows = _dstTopology->source_indices;
            _dstCols = _dstTopology->destination_offsets;
            colSize = _refTopology->nvertices + 1;
            rowSize = _refTopology->nedges;
        }
        else if(TType==NVGRAPH_COO_32){
            nvgraphCOOTopology32I_t _refTopology = static_cast<nvgraphCOOTopology32I_t >(refTopology);
            nvgraphCOOTopology32I_t _dstTopology = static_cast<nvgraphCOOTopology32I_t >(dstTopology);
            ASSERT_EQ( _refTopology->nvertices, _dstTopology->nvertices);
            ASSERT_EQ( _refTopology->nedges, _dstTopology->nedges);
            ASSERT_EQ( _refTopology->tag, _dstTopology->tag);
            _refRows = _refTopology->source_indices;
            _refCols = _refTopology->destination_indices;
            _dstRows = _dstTopology->source_indices;
            _dstCols = _dstTopology->destination_indices;
            colSize = _refTopology->nedges;
            rowSize = _refTopology->nedges;
        }
        else{
            FAIL();
        }

        if(refSpace==DEVICE){
            refRowsHost = (int*)malloc(sizeof(int)*rowSize);
            refColsHost = (int*)malloc(sizeof(int)*colSize);
            cudaMemcpy(refRowsHost, _refRows, sizeof(int)*rowSize, cudaMemcpyDefault);
            cudaMemcpy(refColsHost, _refCols, sizeof(int)*colSize, cudaMemcpyDefault);
        } else {
            refRowsHost = _refRows;
            refColsHost = _refCols;
        }

        if(dstSpace==DEVICE){
            dstRowsHost = (int*)malloc(sizeof(int)*rowSize);
            dstColsHost = (int*)malloc(sizeof(int)*colSize);
            cudaMemcpy(dstRowsHost, _dstRows, sizeof(int)*rowSize, cudaMemcpyDefault);
            cudaMemcpy(dstColsHost, _dstCols, sizeof(int)*colSize, cudaMemcpyDefault);
        } else {
            dstRowsHost = _dstRows;
            dstColsHost = _dstCols;
        }
        std::vector<int> refRows, refCols;
        std::vector<int> dstRows, dstCols;
        refRows.assign(refRowsHost, refRowsHost + rowSize);
        refCols.assign(refColsHost, refColsHost + colSize);
        dstRows.assign(dstRowsHost, dstRowsHost + rowSize);
        dstCols.assign(dstColsHost, dstColsHost + colSize);

        ASSERT_EQ(refRows, dstRows);
        ASSERT_EQ(refCols, dstCols);
        if(refSpace==DEVICE) {
            free(refRowsHost);
            free(refColsHost);
        }
        if(dstSpace==DEVICE){
            free(dstRowsHost);
            free(dstColsHost);
        }
    }

    static nvgraphTopologyType_t testType2nvGraphType(testTopologyType_t type){
        if(type==CSR_32)
            return NVGRAPH_CSR_32;
        else if(type==CSC_32)
            return NVGRAPH_CSC_32;
        else
            return NVGRAPH_COO_32;
    }

    static nvgraphTag_t testType2tag(testTopologyType_t type){

        if(type==COO_SOURCE_32)
            return NVGRAPH_SORTED_BY_SOURCE;
        else if(type==COO_DESTINATION_32)
            return NVGRAPH_SORTED_BY_DESTINATION;
        else if(type==COO_UNSORTED_32)
            return NVGRAPH_UNSORTED;
        else
            return NVGRAPH_DEFAULT;
    }

};

// Compares the convesion result from and to preset values (Used primary for simple test, and to validate reference convsrsion).
class PresetTopology : public NVGraphAPIConvertTest,
                       public ::testing::WithParamInterface<std::tr1::tuple< cudaDataType_t,                // dataType
                                                                             testTopologyType_t,            // srcTopoType
                                                                             testTopologyType_t,            // dstTopoType
                                                                             presetTestContainer_st> > {    // prestTestContainer
  public:
    // Reference (CPU) conversion check
    template <typename T>
    static void refPrestConvertTest(testTopologyType_t srcTestTopoType, void *srcTopology, const double *srcEdgeData,
                                    testTopologyType_t dstTestTopoType, void *refTopology, const double *refEdgeData){

        int srcN=0, srcNNZ=0;
        int refN=0, refNNZ=0;
        topoGetN(srcTestTopoType, srcTopology, &srcN);
        topoGetNNZ(srcTestTopoType, srcTopology, &srcNNZ);
        topoGetN(dstTestTopoType, refTopology, &refN);
        topoGetNNZ(dstTestTopoType, refTopology, &refNNZ);

        // Allocate result Topology
        T *dstEdgeDataT = (T*)malloc(sizeof(T)*refNNZ);
        void *dstTopology=NULL;
        allocateTopo(&dstTopology, dstTestTopoType, refN, refNNZ, HOST);
        //////////////////////////////////////////////////

        // Convert host edge data to template type
        T *srcEdgeDataT = (T*)malloc(sizeof(T)*srcNNZ);
        T *refEdgeDataT = (T*)malloc(sizeof(T)*refNNZ);
        const double *pT=(const double*)srcEdgeData;
        for(int i=0; i<srcNNZ; ++i)
            srcEdgeDataT[i]=(T)pT[i];
        pT=(const double*)refEdgeData;
        for(int i=0; i<refNNZ; ++i)
            refEdgeDataT[i]=(T)pT[i];
        //////////////////////////////////////////////////
        nvgraphTopologyType_t srcTType, dstTType;
        srcTType = testType2nvGraphType(srcTestTopoType);
        dstTType = testType2nvGraphType(dstTestTopoType);
        refConvert(srcTType, srcTopology, srcEdgeDataT, dstTType, dstTopology, dstEdgeDataT);
        cmpTopo(dstTType, refTopology, HOST, dstTopology, HOST);
        cmpArray(refEdgeDataT, HOST, dstEdgeDataT, HOST, refNNZ);

        free(srcEdgeDataT);
        free(refEdgeDataT);
        free(dstEdgeDataT);
        deAllocateTopo(dstTopology, dstTestTopoType, HOST);
    }

    // nvgraph conversion test
    template <typename T>
    void nvgraphPresetConvertTest(testTopologyType_t srcTestTopoType, void *srcTopologyHst, const double *srcEdgeDataHst, cudaDataType_t *dataType,
                                  testTopologyType_t dstTestTopoType, void *refTopologyHst, const double *refEdgeDataHst){

        int srcN=0, srcNNZ=0;
        int refN=0, refNNZ=0;
        topoGetN(srcTestTopoType, srcTopologyHst, &srcN);
        topoGetNNZ(srcTestTopoType, srcTopologyHst, &srcNNZ);
        topoGetN(dstTestTopoType, refTopologyHst, &refN);
        topoGetNNZ(dstTestTopoType, refTopologyHst, &refNNZ);

        // Allocate topoplogies in device memory
        void *srcTopologyDv=NULL, *dstTopologyDv=NULL;
        allocateTopo(&srcTopologyDv, srcTestTopoType, refN, refNNZ, DEVICE);
        allocateTopo(&dstTopologyDv, dstTestTopoType, refN, refNNZ, DEVICE);
        cpyTopo(srcTopologyDv, srcTopologyHst, srcTestTopoType, cudaMemcpyHostToDevice); // Copy src topology to device
        //////////////////////////////////////////////////

        // Convert host edge data to template type
        T *srcEdgeDataHstT = (T*)malloc(sizeof(T)*srcNNZ);
        T *refEdgeDataHstT = (T*)malloc(sizeof(T)*refNNZ);
        const double *pT=(const double*)srcEdgeDataHst;
        for(int i=0; i<srcNNZ; ++i)
            srcEdgeDataHstT[i]=(T)pT[i];
        pT=(const double*)refEdgeDataHst;
        for(int i=0; i<refNNZ; ++i)
            refEdgeDataHstT[i]=(T)pT[i];
        //////////////////////////////////////////////////

        // Allocate edge data in device memory
        T *srcEdgeDataDvT, *dstEdgeDataDvT;
        ASSERT_EQ(cudaSuccess, cudaMalloc((void**)&srcEdgeDataDvT, sizeof(T)*srcNNZ));
        ASSERT_EQ(cudaSuccess, cudaMalloc((void**)&dstEdgeDataDvT, sizeof(T)*refNNZ));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(srcEdgeDataDvT, srcEdgeDataHstT, sizeof(T)*srcNNZ, cudaMemcpyDefault)); // Copy edge data to device
        //////////////////////////////////////////////////

        nvgraphTopologyType_t srcTType, dstTType;
        srcTType = testType2nvGraphType(srcTestTopoType);
        dstTType = testType2nvGraphType(dstTestTopoType);
        status = nvgraphConvertTopology(handle,
                                srcTType, srcTopologyDv, srcEdgeDataDvT, dataType,
                                dstTType, dstTopologyDv, dstEdgeDataDvT);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        cmpTopo(dstTType, refTopologyHst, HOST, dstTopologyDv, DEVICE);
        cmpArray(refEdgeDataHstT, HOST, dstEdgeDataDvT, DEVICE, refNNZ);

        free(srcEdgeDataHstT);
        free(refEdgeDataHstT);
        ASSERT_EQ(cudaSuccess, cudaFree(srcEdgeDataDvT));
        ASSERT_EQ(cudaSuccess, cudaFree(dstEdgeDataDvT));
        deAllocateTopo(srcTopologyDv, srcTestTopoType, DEVICE);
        deAllocateTopo(dstTopologyDv, dstTestTopoType, DEVICE);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Helper functions
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    static void getTestData(testTopologyType_t TType, void **topo, const void **edgeData, presetTestContainer_st prestTestContainer){
        if(TType==CSR_32){
            *topo = prestTestContainer.csrTopo;
            *edgeData = prestTestContainer.csrEdgeData;
        } else if(TType==CSC_32) {
            *topo = prestTestContainer.cscTopo;
            *edgeData = prestTestContainer.cscEdgeData;
        } else if(TType==COO_SOURCE_32) {
            *topo = prestTestContainer.coosTopo;
            *edgeData = prestTestContainer.coosEdgeData;
        } else if(TType==COO_DESTINATION_32) {
            *topo = prestTestContainer.coodTopo;
            *edgeData = prestTestContainer.coodEdgeData;
        } else if(TType==COO_UNSORTED_32) {
            *topo = prestTestContainer.coouTopo;
            *edgeData = prestTestContainer.coouEdgeData;
        } else if(TType==COO_DEFAULT_32) {
            *topo = prestTestContainer.coouTopo;
            *edgeData = prestTestContainer.coouEdgeData;
        } else {
            FAIL();
        }
    }

};


TEST_P(PresetTopology, referenceValidation) {

    cudaDataType_t dataType = std::tr1::get<0>(GetParam());
    testTopologyType_t srcTestTopoType = std::tr1::get<1>(GetParam());
    testTopologyType_t dstTestTopoType = std::tr1::get<2>(GetParam());
    presetTestContainer_st prestTestContainer = std::tr1::get<3>(GetParam());

    if(dstTestTopoType==COO_UNSORTED_32)
        return;

    void *srcTopology=NULL, *refTopology=NULL;
    const void *srcEdgeData=NULL, *refEdgeData=NULL;
    this->getTestData(srcTestTopoType, &srcTopology, &srcEdgeData, prestTestContainer);
    this->getTestData(dstTestTopoType, &refTopology, &refEdgeData, prestTestContainer);

    if(dataType==CUDA_R_32F) {
        this->refPrestConvertTest<float>(srcTestTopoType, srcTopology, (const double*)srcEdgeData,
                                         dstTestTopoType, refTopology, (const double*)refEdgeData);
    } else if (dataType==CUDA_R_64F) {
        this->refPrestConvertTest<double>(srcTestTopoType, srcTopology, (const double*)srcEdgeData,
                                          dstTestTopoType, refTopology, (const double*)refEdgeData);
    } else {
        FAIL();
    }
}


TEST_P(PresetTopology, nvgraphConvertTopology) {

    cudaDataType_t dataType = std::tr1::get<0>(GetParam());
    testTopologyType_t srcTestTopoType = std::tr1::get<1>(GetParam());
    testTopologyType_t dstTestTopoType = std::tr1::get<2>(GetParam());
    presetTestContainer_st prestTestContainer = std::tr1::get<3>(GetParam());

    if(dstTestTopoType==COO_UNSORTED_32)
        return;

    void *srcTopology=NULL, *refTopology=NULL;
    const void *srcEdgeData=NULL, *refEdgeData=NULL;
    this->getTestData(srcTestTopoType, &srcTopology, &srcEdgeData, prestTestContainer);
    this->getTestData(dstTestTopoType, &refTopology, &refEdgeData, prestTestContainer);

    if(dataType==CUDA_R_32F){
        this->nvgraphPresetConvertTest<float>( srcTestTopoType, srcTopology, (const double*)srcEdgeData, &dataType,
                                               dstTestTopoType, refTopology, (const double*)refEdgeData);
    } else if (dataType==CUDA_R_64F) {
        this->nvgraphPresetConvertTest<double>( srcTestTopoType, srcTopology, (const double*)srcEdgeData, &dataType,
                                                dstTestTopoType, refTopology, (const double*)refEdgeData);
    } else {
        FAIL();
    }
}



class RandomTopology : public NVGraphAPIConvertTest,
                       public ::testing::WithParamInterface<std::tr1::tuple< cudaDataType_t,            // dataType
                                                                             testTopologyType_t,        // srcTopoType
                                                                             testTopologyType_t,        // dstTopoType
                                                                             int,                       // n
                                                                             int> > {                   // nnz
  public:
    virtual void SetUp() {
        NVGraphAPIConvertTest::SetUp();
    }
    // nvgraph conversion check
    template <typename T>
    void nvgraphTopologyConvertTest(testTopologyType_t srcTestTopoType, void *srcTopologyHst, const double *srcEdgeDataHst,
                                    cudaDataType_t *dataType, testTopologyType_t dstTestTopoType){
        int srcN=0, srcNNZ=0;
        topoGetN(srcTestTopoType, srcTopologyHst, &srcN);
        topoGetNNZ(srcTestTopoType, srcTopologyHst, &srcNNZ);

        // Allocate result space in host memory
        T *refResultEdgeDataT=(T*)malloc(sizeof(T)*srcNNZ);
        void *refResultTopologyHst=NULL;
        allocateTopo(&refResultTopologyHst, dstTestTopoType, srcN, srcNNZ, HOST);
        //////////////////////////////////////////////////

        // Allocate topologies space in device memory
        void *srcTopologyDv=NULL, *resultTopologyDv=NULL;
        T *resultEdgeData=NULL;
        ASSERT_EQ(cudaSuccess, cudaMalloc( (void**)&resultEdgeData, sizeof(T)*srcNNZ) );
        allocateTopo(&srcTopologyDv, srcTestTopoType, srcN, srcNNZ, DEVICE);
        allocateTopo(&resultTopologyDv, dstTestTopoType, srcN, srcNNZ, DEVICE);
        cpyTopo(srcTopologyDv, srcTopologyHst, srcTestTopoType, cudaMemcpyHostToDevice); // Copy src topology to device
        //////////////////////////////////////////////////

        // Convert host edge data to template type
        T *srcEdgeDataHstT = (T*)malloc(sizeof(T)*srcNNZ);
        const double *pT=(const double*)srcEdgeDataHst;
        for(int i=0; i<srcNNZ; ++i)
            srcEdgeDataHstT[i]=(T)pT[i];
        //////////////////////////////////////////////////

        // Allocate edge data in device memory
        T *srcEdgeDataDvT, *resultEdgeDataDvT;
        ASSERT_EQ(cudaSuccess, cudaMalloc((void**)&srcEdgeDataDvT, sizeof(T)*srcNNZ));
        ASSERT_EQ(cudaSuccess, cudaMalloc((void**)&resultEdgeDataDvT, sizeof(T)*srcNNZ));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(srcEdgeDataDvT, srcEdgeDataHstT, sizeof(T)*srcNNZ, cudaMemcpyDefault)); // Copy edge data to device
        //////////////////////////////////////////////////

        nvgraphTopologyType_t srcTType, dstTType;
        srcTType = testType2nvGraphType(srcTestTopoType);
        dstTType = testType2nvGraphType(dstTestTopoType);
        refConvert(srcTType, srcTopologyHst, srcEdgeDataHstT, dstTType, refResultTopologyHst, refResultEdgeDataT); // Get reference result
        status = nvgraphConvertTopology(handle,
                                srcTType, srcTopologyDv, srcEdgeDataDvT, dataType,
                                dstTType, resultTopologyDv, resultEdgeDataDvT);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        cmpTopo(dstTType, refResultTopologyHst, HOST, resultTopologyDv, DEVICE);
        cmpArray(refResultEdgeDataT, HOST, resultEdgeDataDvT, DEVICE, srcNNZ);

        free(refResultEdgeDataT);
        free(srcEdgeDataHstT);
        ASSERT_EQ(cudaSuccess, cudaFree(resultEdgeData));
        ASSERT_EQ(cudaSuccess, cudaFree(srcEdgeDataDvT));
        ASSERT_EQ(cudaSuccess, cudaFree(resultEdgeDataDvT));
        deAllocateTopo(refResultTopologyHst, dstTestTopoType, HOST);
        deAllocateTopo(srcTopologyDv, srcTestTopoType, DEVICE);
        deAllocateTopo(resultTopologyDv, dstTestTopoType, DEVICE);
    }


    // nvgraph conversion check
    template <typename T>
    void nvgraphGraphConvertTest(testTopologyType_t srcTestTopoType, void *srcTopologyHst, const double *srcEdgeDataHst,
                                 cudaDataType_t *dataType, testTopologyType_t dstTestTopoType){
        int srcN=0, srcNNZ=0;
        topoGetN(srcTestTopoType, srcTopologyHst, &srcN);
        topoGetNNZ(srcTestTopoType, srcTopologyHst, &srcNNZ);

        // Allocate result space in host memory
        T *refResultEdgeDataT=(T*)malloc(sizeof(T)*srcNNZ);
        void *refResultTopologyHst=NULL;
        allocateTopo(&refResultTopologyHst, dstTestTopoType, srcN, srcNNZ, HOST);
        //////////////////////////////////////////////////

        // Allocate topologies space in device memory
        void *srcTopologyDv=NULL, *resultTopologyDv=NULL;
        T *resultEdgeData=NULL;
        ASSERT_EQ(cudaSuccess, cudaMalloc( (void**)&resultEdgeData, sizeof(T)*srcNNZ) );
        allocateTopo(&srcTopologyDv, srcTestTopoType, srcN, srcNNZ, DEVICE);
        allocateTopo(&resultTopologyDv, dstTestTopoType, srcN, srcNNZ, DEVICE);
        cpyTopo(srcTopologyDv, srcTopologyHst, srcTestTopoType, cudaMemcpyHostToDevice); // Copy src topology to device
        //////////////////////////////////////////////////

        // Convert host edge data to template type
        T *srcEdgeDataHstT = (T*)malloc(sizeof(T)*srcNNZ);
        const double *pT=(const double*)srcEdgeDataHst;
        for(int i=0; i<srcNNZ; ++i)
            srcEdgeDataHstT[i]=(T)pT[i];
        //////////////////////////////////////////////////

        // Allocate edge data in device memory
        T *srcEdgeDataDvT, *resultEdgeDataDvT;
        ASSERT_EQ(cudaSuccess, cudaMalloc((void**)&srcEdgeDataDvT, sizeof(T)*srcNNZ));
        ASSERT_EQ(cudaSuccess, cudaMalloc((void**)&resultEdgeDataDvT, sizeof(T)*srcNNZ));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(srcEdgeDataDvT, srcEdgeDataHstT, sizeof(T)*srcNNZ, cudaMemcpyDefault)); // Copy edge data to device
        //////////////////////////////////////////////////

        nvgraphTopologyType_t srcTType, dstTType;
        srcTType = testType2nvGraphType(srcTestTopoType);
        dstTType = testType2nvGraphType(dstTestTopoType);
        refConvert(srcTType, srcTopologyHst, srcEdgeDataHstT, dstTType, refResultTopologyHst, refResultEdgeDataT); // Get reference result
        status = nvgraphConvertTopology(handle,
                                srcTType, srcTopologyDv, srcEdgeDataDvT, dataType,
                                dstTType, resultTopologyDv, resultEdgeDataDvT);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        cmpTopo(dstTType, refResultTopologyHst, HOST, resultTopologyDv, DEVICE);
        cmpArray(refResultEdgeDataT, HOST, resultEdgeDataDvT, DEVICE, srcNNZ);

        free(refResultEdgeDataT);
        free(srcEdgeDataHstT);
        ASSERT_EQ(cudaSuccess, cudaFree(resultEdgeData));
        ASSERT_EQ(cudaSuccess, cudaFree(srcEdgeDataDvT));
        ASSERT_EQ(cudaSuccess, cudaFree(resultEdgeDataDvT));
        deAllocateTopo(refResultTopologyHst, dstTestTopoType, HOST);
        deAllocateTopo(srcTopologyDv, srcTestTopoType, DEVICE);
        deAllocateTopo(resultTopologyDv, dstTestTopoType, DEVICE);
    }
};


TEST_P(RandomTopology, nvgraphConvertTopology) {

    cudaDataType_t dataType = std::tr1::get<0>(GetParam());
    testTopologyType_t srcTestTopoType = std::tr1::get<1>(GetParam());
    testTopologyType_t dstTestTopoType = std::tr1::get<2>(GetParam());
    int n = std::tr1::get<3>(GetParam());
    int max_nnz = std::tr1::get<4>(GetParam());
    int maxJump = (rand() % n)+1;
    int maxPerRow = (rand() % max_nnz)+1;
    int nnz;

    void *srcTopology;
    allocateTopo(&srcTopology, srcTestTopoType, n, max_nnz, HOST);
    if(srcTestTopoType==CSR_32) {
        nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t >(srcTopology);
        randomCsrGenerator( srcT->source_offsets, srcT->destination_indices, &nnz, n,
                            maxPerRow, maxJump, max_nnz);
        srcT->nedges = nnz;
    } else if(srcTestTopoType==CSC_32) {
        nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t >(srcTopology);
        randomCsrGenerator( srcT->destination_offsets, srcT->source_indices, &nnz, n,
                            maxPerRow, maxJump, max_nnz);
        srcT->nedges = nnz;
    } else if(srcTestTopoType==COO_SOURCE_32) {
        nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t >(srcTopology);
        randomCOOGenerator( srcT->source_indices, srcT->destination_indices, &nnz, n,
                            maxPerRow, maxJump, max_nnz);
        srcT->nedges = nnz;
    } else if(srcTestTopoType==COO_DESTINATION_32 || srcTestTopoType==COO_UNSORTED_32 || srcTestTopoType==COO_DEFAULT_32) {
        // Unsorted and default to have COO_dest sorting. (sorted is a special case of unsorted array)
        nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t >(srcTopology);
        randomCOOGenerator( srcT->destination_indices, srcT->source_indices, &nnz, n,
                            maxPerRow, maxJump, max_nnz);
        srcT->nedges = nnz;
    } else {
        FAIL();
    }

    double *srcEdgeData = (double*)malloc(sizeof(double)*nnz);
    for(int i=0; i<nnz; ++i)
        srcEdgeData[i]=(double)rand()/(rand()+1); // don't divide by zero

    if(dataType==CUDA_R_32F){
        this->nvgraphTopologyConvertTest<float> (srcTestTopoType, srcTopology, srcEdgeData, &dataType, dstTestTopoType);
    } else if (dataType==CUDA_R_64F) {
        this->nvgraphTopologyConvertTest<double> (srcTestTopoType, srcTopology, srcEdgeData, &dataType, dstTestTopoType);
    } else {
        FAIL();
    }
    deAllocateTopo(srcTopology, srcTestTopoType, HOST);
    free(srcEdgeData);
}


class RandomGraph : public NVGraphAPIConvertTest,
                    public ::testing::WithParamInterface<std::tr1::tuple< cudaDataType_t,             // dataType
                                                                          testTopologyType_t,         // srcTopoType
                                                                          testTopologyType_t,         // dstTopoType
                                                                          int,                        // n
                                                                          int> > {                    // nnz
  public:
    nvgraphGraphDescr_t srcGrDesc, dstGrDesc, refGrDesc;
    void *srcEdgeData, *dstEdgeData, *refEdgeData;
    void *srcVertexData, *dstVertexData, *refVertexData;
    void *srcTopology, *refTopology;
    nvgraphTopologyType_t srcTopoType, dstTopoType;
    testTopologyType_t srcTestTopoType, dstTestTopoType;
    virtual void SetUp() {
        NVGraphAPIConvertTest::SetUp();
        status = nvgraphCreateGraphDescr(handle, &srcGrDesc);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphCreateGraphDescr(handle, &dstGrDesc);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphCreateGraphDescr(handle, &refGrDesc);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        srcEdgeData = NULL;
        dstEdgeData = NULL;
        refEdgeData = NULL;
        srcVertexData = NULL;
        dstVertexData = NULL;
        refVertexData = NULL;

        srcTopology = NULL;
        refTopology = NULL;
    }
    virtual void TearDown() {
        if(srcGrDesc!=NULL){
            status = nvgraphDestroyGraphDescr(handle, srcGrDesc);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }
        if(dstGrDesc!=NULL){
            status = nvgraphDestroyGraphDescr(handle, dstGrDesc);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }
        if(refGrDesc!=NULL){
            status = nvgraphDestroyGraphDescr(handle, refGrDesc);
            ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        }
        free(srcEdgeData);
        free(dstEdgeData);
        free(refEdgeData);
        free(srcVertexData);
        free(dstVertexData);
        free(refVertexData);
        deAllocateTopo(srcTopology, srcTestTopoType, HOST);
        deAllocateTopo(refTopology, dstTestTopoType, HOST);
        NVGraphAPIConvertTest::TearDown();
    }
};

TEST_P(RandomGraph, nvgraphConvertGraph) {

    cudaDataType_t dataType = std::tr1::get<0>(GetParam());
    srcTestTopoType = std::tr1::get<1>(GetParam());
    dstTestTopoType = std::tr1::get<2>(GetParam());
    int n = std::tr1::get<3>(GetParam());
    int max_nnz = std::tr1::get<4>(GetParam());
    int maxJump = (rand() % n)+1;
    int maxPerRow = (rand() % max_nnz)+1;
    int nnz;

    nvgraphTopologyType_t srcTopoType, dstTopoType;
    srcTopoType = testType2nvGraphType(srcTestTopoType);
    dstTopoType = testType2nvGraphType(dstTestTopoType);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Prepare input graph
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    allocateTopo(&srcTopology, srcTestTopoType, n, max_nnz, HOST);
    if(srcTestTopoType==CSR_32) {
        nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t >(srcTopology);
        randomCsrGenerator( srcT->source_offsets, srcT->destination_indices, &nnz, n,
                            maxPerRow, maxJump, max_nnz);
        srcT->nedges = nnz;
    } else if(srcTestTopoType==CSC_32) {
        nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t >(srcTopology);
        randomCsrGenerator( srcT->destination_offsets, srcT->source_indices, &nnz, n,
                            maxPerRow, maxJump, max_nnz);
        srcT->nedges = nnz;
    } else if(srcTestTopoType==COO_SOURCE_32) {
        nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t >(srcTopology);
        randomCOOGenerator( srcT->source_indices, srcT->destination_indices, &nnz, n,
                            maxPerRow, maxJump, max_nnz);
        srcT->nedges = nnz;
    } else if(srcTestTopoType==COO_DESTINATION_32 || srcTestTopoType==COO_UNSORTED_32 || srcTestTopoType==COO_DEFAULT_32) {
        // Unsorted and default to have COO_dest sorting. (sorted is a special case of unsorted array)
        nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t >(srcTopology);
        randomCOOGenerator( srcT->destination_indices, srcT->source_indices, &nnz, n,
                            maxPerRow, maxJump, max_nnz);
        srcT->nedges = nnz;
    } else {
        FAIL();
    }

    status = nvgraphSetGraphStructure(handle, srcGrDesc, srcTopology, srcTopoType);
    if(srcTopoType==NVGRAPH_CSR_32 || srcTopoType==NVGRAPH_CSC_32){
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    } else if (srcTopoType==NVGRAPH_COO_32){ // COO graph is not supported
        ASSERT_EQ(NVGRAPH_STATUS_TYPE_NOT_SUPPORTED, status);
        return;
    } else {
        FAIL();
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Prepeate data arrays
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    if(dataType==CUDA_R_32F){
        srcEdgeData = malloc(sizeof(float)*nnz);
        dstEdgeData = malloc(sizeof(float)*nnz);
        refEdgeData = malloc(sizeof(float)*nnz);
        srcVertexData = malloc(sizeof(float)*n);
        dstVertexData = malloc(sizeof(float)*n);
        refVertexData = malloc(sizeof(float)*n);
    }
    else if (dataType==CUDA_R_64F){
        srcEdgeData = malloc(sizeof(double)*nnz);
        dstEdgeData = malloc(sizeof(double)*nnz);
        refEdgeData = malloc(sizeof(double)*nnz);
        srcVertexData = malloc(sizeof(double)*n);
        dstVertexData = malloc(sizeof(double)*n);
        refVertexData = malloc(sizeof(double)*n);
    } else
        FAIL();

    if(srcEdgeData==NULL || dstEdgeData==NULL || refEdgeData==NULL)
        FAIL();
    if(srcVertexData==NULL || dstVertexData==NULL || refVertexData==NULL)
        FAIL();
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Prepare reference graph
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    allocateTopo(&refTopology, dstTestTopoType, n, nnz, HOST);
    if(dataType==CUDA_R_32F)
        refConvert( srcTopoType, srcTopology, (float*)srcEdgeData,
                    dstTopoType, refTopology, (float*)refEdgeData ); // We don't care about edgeData
    else if (dataType==CUDA_R_64F)
        refConvert( srcTopoType, srcTopology, (double*)srcEdgeData,
                    dstTopoType, refTopology, (double*)refEdgeData ); // We don't care about edgeData
    else
        FAIL();
    status = nvgraphSetGraphStructure(handle, refGrDesc, refTopology, dstTopoType);
    if( dstTopoType==NVGRAPH_CSR_32 || dstTopoType==NVGRAPH_CSC_32){
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    } else if (dstTopoType==NVGRAPH_COO_32) { // We don't support COO graphs
        ASSERT_EQ(NVGRAPH_STATUS_TYPE_NOT_SUPPORTED, status);
        return;
    } else {
        FAIL();
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill graph with vertex and edge data
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    size_t edgeDataDim = (rand() % 11); // up to 10 edgeData sets
    std::vector<cudaDataType_t> edgeDataType(edgeDataDim);
    std::fill (edgeDataType.begin(), edgeDataType.end(), dataType);
    status = nvgraphAllocateEdgeData( handle, srcGrDesc, edgeDataDim, edgeDataType.data());
    if(edgeDataDim==0)
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
    else
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    status = nvgraphAllocateEdgeData( handle, refGrDesc, edgeDataDim, edgeDataType.data());
    if(edgeDataDim==0)
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
    else
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    for(size_t i=0; i<edgeDataDim; ++i){
        randomArray(nnz, srcEdgeData, &dataType);
        // src Graph
        status = nvgraphSetEdgeData(handle, srcGrDesc, srcEdgeData, i);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        // ref Graph (not the fastest approach, but I'm too lazy to do the permutation approach)
        if(dataType==CUDA_R_32F)
            refConvert( srcTopoType, srcTopology, (float*)srcEdgeData,
                        dstTopoType, refTopology, (float*)refEdgeData );
        else if (dataType==CUDA_R_64F)
            refConvert( srcTopoType, srcTopology, (double*)srcEdgeData,
                        dstTopoType, refTopology, (double*)refEdgeData );
        else
            FAIL();
        status = nvgraphSetEdgeData(handle, refGrDesc, refEdgeData, i);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }


    size_t vertexDataDim = (rand() % 6); // up to 5 vertexData sets
    std::vector<cudaDataType_t> vertexDataType(vertexDataDim);
    std::fill (vertexDataType.begin(), vertexDataType.end(), dataType);
    status = nvgraphAllocateVertexData( handle, srcGrDesc, vertexDataDim, vertexDataType.data());
    if(vertexDataDim==0)
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
    else
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    status = nvgraphAllocateVertexData( handle, refGrDesc, vertexDataDim, vertexDataType.data());
    if(vertexDataDim==0)
        ASSERT_EQ(NVGRAPH_STATUS_INVALID_VALUE, status);
    else
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    for(size_t i=0; i<vertexDataDim; ++i){
        randomArray(n, srcVertexData, &dataType);
        // src Graph
        status = nvgraphSetVertexData(handle, srcGrDesc, srcVertexData, i);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

        status = nvgraphSetVertexData(handle, refGrDesc, srcVertexData, i);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    // Convert Graph
    status = nvgraphConvertGraph(handle, srcGrDesc, dstGrDesc, dstTopoType);
    ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);

    // ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // // Compare
    // ///////////////////////////////////////////////////////////////////////////////////////////////////////
    int ref_nvertices, ref_nedges, dst_nvertices, dst_nedges;
    int *dstOffset, *dstInd, *refOffset, *refInd;
    if(dataType==CUDA_R_32F){
        nvgraph::MultiValuedCsrGraph<int, float> *refMCSRG = static_cast<nvgraph::MultiValuedCsrGraph<int, float>*> (refGrDesc->graph_handle);
        ref_nvertices = static_cast<int>(refMCSRG->get_num_vertices());
        ref_nedges = static_cast<int>(refMCSRG->get_num_edges());
        refOffset = refMCSRG->get_raw_row_offsets();
        refInd = refMCSRG->get_raw_column_indices();

        nvgraph::MultiValuedCsrGraph<int, float> *dstMCSRG = static_cast<nvgraph::MultiValuedCsrGraph<int, float>*> (dstGrDesc->graph_handle);
        dst_nvertices = static_cast<int>(dstMCSRG->get_num_vertices());
        dst_nedges = static_cast<int>(dstMCSRG->get_num_edges());
        dstOffset = dstMCSRG->get_raw_row_offsets();
        dstInd = dstMCSRG->get_raw_column_indices();
    } else if (dataType==CUDA_R_64F) {
        nvgraph::MultiValuedCsrGraph<int, double> *refMCSRG = static_cast<nvgraph::MultiValuedCsrGraph<int, double>*> (refGrDesc->graph_handle);
        ref_nvertices = static_cast<int>(refMCSRG->get_num_vertices());
        ref_nedges = static_cast<int>(refMCSRG->get_num_edges());
        refOffset = refMCSRG->get_raw_row_offsets();
        refInd = refMCSRG->get_raw_column_indices();

        nvgraph::MultiValuedCsrGraph<int, double> *dstMCSRG = static_cast<nvgraph::MultiValuedCsrGraph<int, double>*> (dstGrDesc->graph_handle);
        dst_nvertices = static_cast<int>(dstMCSRG->get_num_vertices());
        dst_nedges = static_cast<int>(dstMCSRG->get_num_edges());
        dstOffset = dstMCSRG->get_raw_row_offsets();
        dstInd = dstMCSRG->get_raw_column_indices();
    } else
        FAIL();

    ASSERT_EQ(ref_nvertices, dst_nvertices);
    ASSERT_EQ(ref_nedges, dst_nedges);
    cmpArray(refOffset, DEVICE, dstOffset, DEVICE, n+1);
    cmpArray(refInd, DEVICE, dstInd, DEVICE, nnz);

    for(size_t i=0; i<edgeDataDim; ++i){
        status = nvgraphGetEdgeData(handle, refGrDesc, refEdgeData, i);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphGetEdgeData(handle, dstGrDesc, dstEdgeData, i);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        if(dataType==CUDA_R_32F)
            cmpArray((float*)refEdgeData, HOST, (float*)dstEdgeData, HOST, nnz);
        else if (dataType==CUDA_R_64F)
            cmpArray((double*)refEdgeData, HOST, (double*)dstEdgeData, HOST, nnz);
        else
            FAIL();
    }

    for(size_t i=0; i<vertexDataDim; ++i){
        status = nvgraphGetVertexData(handle, refGrDesc, refVertexData, i);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        status = nvgraphGetVertexData(handle, dstGrDesc, dstVertexData, i);
        ASSERT_EQ(NVGRAPH_STATUS_SUCCESS, status);
        if(dataType==CUDA_R_32F)
            cmpArray((float*)refVertexData, HOST, (float*)dstVertexData, HOST, n);
        else if (dataType==CUDA_R_64F)
            cmpArray((double*)refVertexData, HOST, (double*)dstVertexData, HOST, n);
        else
            FAIL();
    }
}

cudaDataType_t DATA_TYPES[] = {CUDA_R_32F, CUDA_R_64F};
testTopologyType_t SRC_TOPO_TYPES[] = {CSR_32, CSC_32, COO_SOURCE_32, COO_DESTINATION_32, COO_UNSORTED_32};
testTopologyType_t DST_TOPO_TYPES[] = {CSR_32, CSC_32, COO_SOURCE_32, COO_DESTINATION_32, COO_UNSORTED_32};
int ns[] = {10, 100, 1000, 50000, 100000, 200000, 300000, 456179, 500000, 1000000};
int nnzs[] = {10, 100, 1000, 25000, 28943, 50000, 100000, 200000};

INSTANTIATE_TEST_CASE_P(PresetTopologyConvertTest, PresetTopology,
                        ::testing::Combine(
                        ::testing::ValuesIn(DATA_TYPES),        // dataType
                        ::testing::ValuesIn(SRC_TOPO_TYPES),    // srcTopoType
                        ::testing::ValuesIn(DST_TOPO_TYPES),    // dstTopoType
                        ::testing::ValuesIn(presetTests)        // testData
                            ));

INSTANTIATE_TEST_CASE_P(RandomTopologyConvertTest, RandomTopology,
                        ::testing::Combine(
                        ::testing::ValuesIn(DATA_TYPES),        // dataType
                        ::testing::ValuesIn(SRC_TOPO_TYPES),    // srcTopoType
                        ::testing::ValuesIn(DST_TOPO_TYPES),    // dstTopoType
                        ::testing::ValuesIn(ns),                // n
                        ::testing::ValuesIn(nnzs)               // nnz
                            ));

INSTANTIATE_TEST_CASE_P(RandomGraphConvertTest, RandomGraph,
                        ::testing::Combine(
                        ::testing::ValuesIn(DATA_TYPES),        // dataType
                        ::testing::ValuesIn(SRC_TOPO_TYPES),    // srcTopoType
                        ::testing::ValuesIn(DST_TOPO_TYPES),    // dstTopoType
                        ::testing::ValuesIn(ns),                // n
                        ::testing::ValuesIn(nnzs)               // nnz
                            ));

int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
