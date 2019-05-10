#include <stdio.h>
#include <stddef.h>
#include <iostream>
#include <stdlib.h> 
#include <vector>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>

#include "mmio.h"

#include "mm_host.hxx"
#include "nerstrand.h"


static double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}


int main(int argc, const char **argv) 
{

    int m, n, nnz;
    double start, stop,r_mod;
    cid_t n_clusters;
    MM_typecode mc;
    if (argc != 3)
    {
        std::cout<<"Usage : ./nerstrand_bench <graph> <number of clusters>"<<std::endl;
        exit(0);
    }
    FILE* fpin = fopen(argv[1],"r");
    n_clusters = atoi(argv[2]);
    
    mm_properties<int>(fpin, 1, &mc, &m, &n, &nnz) ;
    
    // Allocate memory on host
    std::vector<int> cooRowIndA(nnz);
    std::vector<int> cooColIndA(nnz);
    std::vector<double> cooValA(nnz);
    std::vector<int> csrRowPtrA(n+1);
    std::vector<int> csrColIndA(nnz);
    std::vector<double> csrValA(nnz);

    
    mm_to_coo<int,double>(fpin, 1, nnz, &cooRowIndA[0], &cooColIndA[0], &cooValA[0],NULL) ;
    coo2csr<int,double> (n, nnz, &cooValA[0],  &cooRowIndA[0],  &cooColIndA[0], &csrValA[0], &csrColIndA[0],&csrRowPtrA[0]);
    fclose(fpin);   

    vtx_t nerstrand_n = static_cast<vtx_t>(n);
    std::vector<adj_t> nerstrand_csrRowPtrA(csrRowPtrA.begin(), csrRowPtrA.end());
    std::vector<vtx_t> nerstrand_csrColIndA(csrColIndA.begin(), csrColIndA.end());
    std::vector<wgt_t> nerstrand_csrValA(csrValA.begin(), csrValA.end());
    std::vector<cid_t> clustering(n);

    start = second();
    start = second();
    #pragma omp_parallel
    {
    int nerstrand_status = nerstrand_cluster_kway(&nerstrand_n, &nerstrand_csrRowPtrA[0],&nerstrand_csrColIndA[0], &nerstrand_csrValA[0], &n_clusters, &clustering[0], &r_mod);
    if (nerstrand_status != NERSTRAND_SUCCESS) 
        std::cout<<"nerstrand execution failed"<<std::endl;
    

    }
        stop = second();

    std::cout<<r_mod<<","<<stop-start<<std::endl;
}