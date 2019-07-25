/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _NVGRAPH_H_
#define _NVGRAPH_H_

#include <stddef.h>
#include <stdint.h>

#include <rmm/rmm.h>

#include "library_types.h"


#define NVG_CUDA_TRY(T) {\
                         if (T != cudaSuccess)\
                             return NVGRAPH_STATUS_ALLOC_FAILED;\
             }

// This is a gap filler, and should be replaced with a RAPIDS-wise error handling mechanism.
#define NVG_RMM_TRY(T) {\
                         if (T != RMM_SUCCESS)\
                             return NVGRAPH_STATUS_ALLOC_FAILED;\
             }

#ifndef NVGRAPH_API
#ifdef _WIN32
#define NVGRAPH_API __stdcall
#else
#define NVGRAPH_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /* nvGRAPH status type returns */
    typedef enum
    {
        NVGRAPH_STATUS_SUCCESS = 0,
        NVGRAPH_STATUS_NOT_INITIALIZED = 1,
        NVGRAPH_STATUS_ALLOC_FAILED = 2,
        NVGRAPH_STATUS_INVALID_VALUE = 3,
        NVGRAPH_STATUS_ARCH_MISMATCH = 4,
        NVGRAPH_STATUS_MAPPING_ERROR = 5,
        NVGRAPH_STATUS_EXECUTION_FAILED = 6,
        NVGRAPH_STATUS_INTERNAL_ERROR = 7,
        NVGRAPH_STATUS_TYPE_NOT_SUPPORTED = 8,
        NVGRAPH_STATUS_NOT_CONVERGED = 9,
        NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED = 10

    } nvgraphStatus_t;

    const char* nvgraphStatusGetString(nvgraphStatus_t status);

    /* Opaque structure holding nvGRAPH library context */
    struct nvgraphContext;
    typedef struct nvgraphContext *nvgraphHandle_t;

    /* Opaque structure holding the graph descriptor */
    struct nvgraphGraphDescr;
    typedef struct nvgraphGraphDescr *nvgraphGraphDescr_t;

    /* Semi-ring types */
    typedef enum
    {
        NVGRAPH_PLUS_TIMES_SR = 0,
        NVGRAPH_MIN_PLUS_SR = 1,
        NVGRAPH_MAX_MIN_SR = 2,
        NVGRAPH_OR_AND_SR = 3,
    } nvgraphSemiring_t;

    /* Topology types */
    typedef enum
    {
        NVGRAPH_CSR_32 = 0,
        NVGRAPH_CSC_32 = 1,
        NVGRAPH_COO_32 = 2,
        NVGRAPH_2D_32I_32I = 3,
        NVGRAPH_2D_64I_32I = 4
    } nvgraphTopologyType_t;

    typedef enum
    {
        NVGRAPH_DEFAULT = 0,  // Default is unsorted.
        NVGRAPH_UNSORTED = 1,  //
        NVGRAPH_SORTED_BY_SOURCE = 2,  // CSR
        NVGRAPH_SORTED_BY_DESTINATION = 3   // CSC
    } nvgraphTag_t;

    typedef enum
    {
        NVGRAPH_MULTIPLY = 0,
        NVGRAPH_SUM = 1,
        NVGRAPH_MIN = 2,
        NVGRAPH_MAX = 3
    } nvgraphSemiringOps_t;

    typedef enum
    {
        NVGRAPH_MODULARITY_MAXIMIZATION = 0, //maximize modularity with Lanczos solver
        NVGRAPH_BALANCED_CUT_LANCZOS = 1, //minimize balanced cut with Lanczos solver
        NVGRAPH_BALANCED_CUT_LOBPCG = 2 //minimize balanced cut with LOPCG solver
    } nvgraphSpectralClusteringType_t;

    struct SpectralClusteringParameter {
        int n_clusters; //number of clusters
        int n_eig_vects; // //number of eigenvectors
        nvgraphSpectralClusteringType_t algorithm; // algorithm to use
        float evs_tolerance; // tolerance of the eigensolver
        int evs_max_iter; // maximum number of iterations of the eigensolver
        float kmean_tolerance; // tolerance of kmeans
        int kmean_max_iter; // maximum number of iterations of kemeans
        void * opt; // optional parameter that can be used for preconditioning in the future
    };

    typedef enum
    {
        NVGRAPH_MODULARITY, // clustering score telling how good the clustering is compared to random assignment.
        NVGRAPH_EDGE_CUT,  // total number of edges between clusters.
        NVGRAPH_RATIO_CUT // sum for all clusters of the number of edges going outside of the cluster divided by the number of vertex inside the cluster
    } nvgraphClusteringMetric_t;

    struct nvgraphCSRTopology32I_st {
        int nvertices; // n+1
        int nedges; // nnz
        int *source_offsets; // rowPtr
        int *destination_indices; // colInd
    };
    typedef struct nvgraphCSRTopology32I_st *nvgraphCSRTopology32I_t;

    struct nvgraphCSCTopology32I_st {
        int nvertices; // n+1
        int nedges; // nnz
        int *destination_offsets; // colPtr
        int *source_indices; // rowInd
    };
    typedef struct nvgraphCSCTopology32I_st *nvgraphCSCTopology32I_t;

    struct nvgraphCOOTopology32I_st {
        int nvertices; // n+1
        int nedges; // nnz
        int *source_indices; // rowInd
        int *destination_indices; // colInd
        nvgraphTag_t tag;
    };
    typedef struct nvgraphCOOTopology32I_st *nvgraphCOOTopology32I_t;

    struct nvgraph2dCOOTopology32I_st {
        int nvertices;
        int nedges;
        int *source_indices;             // Row Indices
        int *destination_indices;    // Column Indices
        cudaDataType_t valueType;    // The type of values being given.
        void *values;                    // Pointer to array of values.
        int numDevices;                 // Gives the number of devices to be used.
        int *devices;                     // Array of device IDs to use.
        int blockN;                         // Specifies the value of n for an n x n matrix decomposition.
        nvgraphTag_t tag;
    };
    typedef struct nvgraph2dCOOTopology32I_st *nvgraph2dCOOTopology32I_t;

    /* Return properties values for the nvGraph library, such as library version */
    nvgraphStatus_t NVGRAPH_API nvgraphGetProperty(libraryPropertyType type, int *value);

    /* Open the library and create the handle */
    nvgraphStatus_t NVGRAPH_API nvgraphCreate(nvgraphHandle_t *handle);
    nvgraphStatus_t NVGRAPH_API nvgraphCreateMulti(nvgraphHandle_t *handle,
                                                   int numDevices,
                                                   int* devices);

    /*  Close the library and destroy the handle  */
    nvgraphStatus_t NVGRAPH_API nvgraphDestroy(nvgraphHandle_t handle);

    /* Create an empty graph descriptor */
    nvgraphStatus_t NVGRAPH_API nvgraphCreateGraphDescr(nvgraphHandle_t handle,
                                                        nvgraphGraphDescr_t *descrG);

    /* Destroy a graph descriptor */
    nvgraphStatus_t NVGRAPH_API nvgraphDestroyGraphDescr(nvgraphHandle_t handle,
                                                         nvgraphGraphDescr_t descrG);

    /* Set size, topology data in the graph descriptor  */
    nvgraphStatus_t NVGRAPH_API nvgraphSetGraphStructure(nvgraphHandle_t handle,
                                                         nvgraphGraphDescr_t descrG,
                                                         void* topologyData,
                                                         nvgraphTopologyType_t TType);

    /* Query size and topology information from the graph descriptor */
    nvgraphStatus_t NVGRAPH_API nvgraphGetGraphStructure(nvgraphHandle_t handle,
                                                         nvgraphGraphDescr_t descrG,
                                                         void* topologyData,
                                                         nvgraphTopologyType_t* TType);

    /* Allocate numsets vectors of size V representing Vertex Data and attached them the graph.
     * settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type */
    nvgraphStatus_t NVGRAPH_API nvgraphAllocateVertexData(nvgraphHandle_t handle,
                                                          nvgraphGraphDescr_t descrG,
                                                          size_t numsets,
                                                          cudaDataType_t *settypes);

    /* Allocate numsets vectors of size E representing Edge Data and attached them the graph.
     * settypes[i] is the type of vector #i, currently all Vertex and Edge data should have the same type */
    nvgraphStatus_t NVGRAPH_API nvgraphAllocateEdgeData(nvgraphHandle_t handle,
                                                        nvgraphGraphDescr_t descrG,
                                                        size_t numsets,
                                                        cudaDataType_t *settypes);

    /* Update the vertex set #setnum with the data in *vertexData, sets have 0-based index
     *  Conversions are not supported so nvgraphTopologyType_t should match the graph structure */
    nvgraphStatus_t NVGRAPH_API nvgraphSetVertexData(nvgraphHandle_t handle,
                                                     nvgraphGraphDescr_t descrG,
                                                     void *vertexData,
                                                     size_t setnum);

    /* Copy the edge set #setnum in *edgeData, sets have 0-based index
     *  Conversions are not supported so nvgraphTopologyType_t should match the graph structure */
    nvgraphStatus_t NVGRAPH_API nvgraphGetVertexData(nvgraphHandle_t handle,
                                                     nvgraphGraphDescr_t descrG,
                                                     void *vertexData,
                                                     size_t setnum);

    /* Convert the edge data to another topology
     */
    nvgraphStatus_t NVGRAPH_API nvgraphConvertTopology(nvgraphHandle_t handle,
                                                       nvgraphTopologyType_t srcTType,
                                                       void *srcTopology,
                                                       void *srcEdgeData,
                                                       cudaDataType_t *dataType,
                                                       nvgraphTopologyType_t dstTType,
                                                       void *dstTopology,
                                                       void *dstEdgeData);

    /* Convert graph to another structure
     */
    nvgraphStatus_t NVGRAPH_API nvgraphConvertGraph(nvgraphHandle_t handle,
                                                    nvgraphGraphDescr_t srcDescrG,
                                                    nvgraphGraphDescr_t dstDescrG,
                                                    nvgraphTopologyType_t dstTType);

    /* Update the edge set #setnum with the data in *edgeData, sets have 0-based index
     */
    nvgraphStatus_t NVGRAPH_API nvgraphSetEdgeData(nvgraphHandle_t handle,
                                                   nvgraphGraphDescr_t descrG,
                                                   void *edgeData,
                                                   size_t setnum);

    /* Copy the edge set #setnum in *edgeData, sets have 0-based index
     */
    nvgraphStatus_t NVGRAPH_API nvgraphGetEdgeData(nvgraphHandle_t handle,
                                                   nvgraphGraphDescr_t descrG,
                                                   void *edgeData,
                                                   size_t setnum);

    /* create a new graph by extracting a subgraph given a list of vertices
     */
    nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByVertex(nvgraphHandle_t handle,
                                                               nvgraphGraphDescr_t descrG,
                                                               nvgraphGraphDescr_t subdescrG,
                                                               int *subvertices,
                                                               size_t numvertices);
    /* create a new graph by extracting a subgraph given a list of edges
     */
    nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByEdge(nvgraphHandle_t handle,
                                                             nvgraphGraphDescr_t descrG,
                                                             nvgraphGraphDescr_t subdescrG,
                                                             int *subedges,
                                                             size_t numedges);

    /* nvGRAPH Semi-ring sparse matrix vector multiplication
     */
    nvgraphStatus_t NVGRAPH_API nvgraphSrSpmv(nvgraphHandle_t handle,
                                              const nvgraphGraphDescr_t descrG,
                                              const size_t weight_index,
                                              const void *alpha,
                                              const size_t x_index,
                                              const void *beta,
                                              const size_t y_index,
                                              const nvgraphSemiring_t SR);

    /* Helper struct for Traversal parameters
     */
    typedef struct {
        size_t pad[128];
    } nvgraphTraversalParameter_t;

    /* Initializes traversal parameters with default values
     */
    nvgraphStatus_t NVGRAPH_API nvgraphTraversalParameterInit(nvgraphTraversalParameter_t *param);

    /* Stores/retrieves index of a vertex data where target distances will be stored
     */
    nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetDistancesIndex(nvgraphTraversalParameter_t *param,
                                                                                        const size_t value);

    nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetDistancesIndex(const nvgraphTraversalParameter_t param,
                                                                                        size_t *value);

    /* Stores/retrieves index of a vertex data where path predecessors will be stored
     */
    nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetPredecessorsIndex(nvgraphTraversalParameter_t *param,
                                                                     const size_t value);

    nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetPredecessorsIndex(const nvgraphTraversalParameter_t param,
                                                                     size_t *value);

    /* Stores/retrieves index of an edge data which tells traversal algorithm whether path can go through an edge or not
     */
    nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetEdgeMaskIndex(nvgraphTraversalParameter_t *param,
                                                                 const size_t value);

    nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetEdgeMaskIndex(const nvgraphTraversalParameter_t param,
                                                                 size_t *value);

    /* Stores/retrieves flag that tells an algorithm whether the graph is directed or not
     */
    nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetUndirectedFlag(nvgraphTraversalParameter_t *param,
                                                                  const size_t value);

    nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetUndirectedFlag(const nvgraphTraversalParameter_t param,
                                                                  size_t *value);

    /* Stores/retrieves 'alpha' and 'beta' parameters for BFS traversal algorithm
     */
    nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetAlpha(nvgraphTraversalParameter_t *param,
                                                         const size_t value);

    nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetAlpha(const nvgraphTraversalParameter_t param,
                                                         size_t *value);

    nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetBeta(nvgraphTraversalParameter_t *param,
                                                        const size_t value);

    nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetBeta(const nvgraphTraversalParameter_t param,
                                                        size_t *value);

//Traversal available
    typedef enum {
        NVGRAPH_TRAVERSAL_BFS = 0
    } nvgraphTraversal_t;

    /* nvGRAPH Traversal API
     * Compute a traversal of the graph from a single vertex using algorithm specified by traversalT parameter
     */
    nvgraphStatus_t NVGRAPH_API nvgraphTraversal(nvgraphHandle_t handle,
                                                 const nvgraphGraphDescr_t descrG,
                                                 const nvgraphTraversal_t traversalT,
                                                 const int *source_vert,
                                                 const nvgraphTraversalParameter_t params);

    /**
     * CAPI Method for calling 2d BFS algorithm.
     * @param handle Nvgraph context handle.
     * @param descrG Graph handle (must be 2D partitioned)
     * @param source_vert The source vertex ID
     * @param distances Pointer to memory allocated to store the distances.
     * @param predecessors Pointer to memory allocated to store the predecessors
     * @return Status code.
     */
    nvgraphStatus_t NVGRAPH_API nvgraph2dBfs(nvgraphHandle_t handle,
                                             const nvgraphGraphDescr_t descrG,
                                             const int32_t source_vert,
                                             int32_t* distances,
                                             int32_t* predecessors);

    /* nvGRAPH Single Source Shortest Path (SSSP)
     * Calculate the shortest path distance from a single vertex in the graph to all other vertices.
     */
    nvgraphStatus_t NVGRAPH_API nvgraphSssp(nvgraphHandle_t handle,
                                            const nvgraphGraphDescr_t descrG,
                                            const size_t weight_index,
                                            const int *source_vert,
                                            const size_t sssp_index);

    /* nvGRAPH WidestPath
     * Find widest path potential from source_index to every other vertices.
     */
    nvgraphStatus_t NVGRAPH_API nvgraphWidestPath(nvgraphHandle_t handle,
                                                  const nvgraphGraphDescr_t descrG,
                                                  const size_t weight_index,
                                                  const int *source_vert,
                                                  const size_t widest_path_index);

    /* nvGRAPH PageRank
     * Find PageRank for each vertex of a graph with a given transition probabilities, a bookmark vector of dangling vertices, and the damping factor.
     */
    nvgraphStatus_t NVGRAPH_API nvgraphPagerank(nvgraphHandle_t handle,
                                                const nvgraphGraphDescr_t descrG,
                                                const size_t weight_index,
                                                const void *alpha,
                                                const size_t bookmark_index,
                                                const int has_guess,
                                                const size_t pagerank_index,
                                                const float tolerance,
                                                const int max_iter);

    /* nvGRAPH contraction
     * given array of agregates contract graph with
     * given (Combine, Reduce) operators for Vertex Set
     * and Edge Set;
     */
    nvgraphStatus_t NVGRAPH_API nvgraphContractGraph(nvgraphHandle_t handle,
                                                     nvgraphGraphDescr_t descrG,
                                                     nvgraphGraphDescr_t contrdescrG,
                                                     int *aggregates,
                                                     size_t numaggregates,
                                                     nvgraphSemiringOps_t VertexCombineOp,
                                                     nvgraphSemiringOps_t VertexReduceOp,
                                                     nvgraphSemiringOps_t EdgeCombineOp,
                                                     nvgraphSemiringOps_t EdgeReduceOp,
                                                     int flag);

    /* nvGRAPH spectral clustering
     * given a graph and solver parameters of struct SpectralClusteringParameter,
     * assign vertices to groups such as
     * intra-group connections are strong and/or inter-groups connections are weak
     * using spectral technique.
     */
    nvgraphStatus_t NVGRAPH_API nvgraphSpectralClustering(nvgraphHandle_t handle,
                                                          const nvgraphGraphDescr_t graph_descr,
                                                          const size_t weight_index,
                                                          const struct SpectralClusteringParameter *params,
                                                          int* clustering,
                                                          void* eig_vals,
                                                          void* eig_vects);

    /* nvGRAPH analyze clustering
     * Given a graph, a clustering, and a metric
     * compute the score that measures the clustering quality according to the metric.
     */
    nvgraphStatus_t NVGRAPH_API nvgraphAnalyzeClustering(nvgraphHandle_t handle,
                                                         const nvgraphGraphDescr_t graph_descr,
                                                         const size_t weight_index,
                                                         const int n_clusters,
                                                         const int* clustering,
                                                         nvgraphClusteringMetric_t metric,
                                                         float * score);

    /* nvGRAPH Triangles counting
     * count number of triangles (cycles of size 3) formed by graph edges
     */
    nvgraphStatus_t NVGRAPH_API nvgraphTriangleCount(nvgraphHandle_t handle,
                                                     const nvgraphGraphDescr_t graph_descr,
                                                     uint64_t* result);

    /* nvGRAPH Louvain implementation
     */
    nvgraphStatus_t NVGRAPH_API nvgraphLouvain(cudaDataType_t index_type,
                                               cudaDataType_t val_type,
                                               const size_t num_vertex,
                                               const size_t num_edges,
                                               void* csr_ptr,
                                               void* csr_ind,
                                               void* csr_val,
                                               int weighted,
                                               int has_init_cluster,
                                               void* init_cluster,
                                               void* final_modularity,
                                               void* best_cluster_vec,
                                               void* num_level);


    /* nvGRAPH Jaccard implementation
     */
    nvgraphStatus_t NVGRAPH_API nvgraphJaccard(cudaDataType_t index_type,
                                               cudaDataType_t val_type,
                                               const size_t n,
                                               const size_t e,
                                               void* csr_ptr,
                                               void *csr_ind,
                                               void* csr_val,
                                               int weighted,
                                               void* v,
                                               void* gamma,
                                               void* weight_j);

    /* nvGRAPH attach structure
     * Warp external device data into a nvgraphGraphDescr_t
     * Warning : this data remain owned by the user
     */
    nvgraphStatus_t NVGRAPH_API nvgraphAttachGraphStructure(nvgraphHandle_t handle,
                                                            nvgraphGraphDescr_t descrG,
                                                            void* topologyData,
                                                            nvgraphTopologyType_t TT);

    /* nvGRAPH attach Vertex Data
     * Warp external device data into a vertex dim
     * Warning : this data remain owned by the user
     */
    nvgraphStatus_t NVGRAPH_API nvgraphAttachVertexData(nvgraphHandle_t handle,
                                                        nvgraphGraphDescr_t descrG,
                                                        size_t setnum,
                                                        cudaDataType_t settype,
                                                        void *vertexData);

    /* nvGRAPH attach Edge Data
     * Warp external device data into an edge dim
     * Warning : this data remain owned by the user
     */
    nvgraphStatus_t NVGRAPH_API nvgraphAttachEdgeData(nvgraphHandle_t handle,
                                                      nvgraphGraphDescr_t descrG,
                                                      size_t setnum,
                                                      cudaDataType_t settype,
                                                      void *edgeData);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NVGRAPH_H_ */
