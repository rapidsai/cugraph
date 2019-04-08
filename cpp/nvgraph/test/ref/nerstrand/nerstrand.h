/**
 * @file nerstrand.h
 * @brief Main header for for nerstrand 
 * @author Dominique LaSalle <lasalle@cs.umn.edu>
 * Copyright 2013, Regents of the University of Minnesota
 * @version 1
 * @date 2014-01-27
 */




#ifndef NERSTRAND_H
#define NERSTRAND_H




#include <stdint.h>
#include <float.h>
#include <unistd.h>




/******************************************************************************
* VERSION *********************************************************************
******************************************************************************/


#define NERSTRAND_VER_MAJOR 0
#define NERSTRAND_VER_MINOR 5
#define NERSTRAND_VER_SUBMINOR 0




/******************************************************************************
* TYPES ***********************************************************************
******************************************************************************/


#ifndef NERSTRAND_GRAPH_TYPES_DEFINED
#ifdef NERSTRAND_64BIT_VERTICES
typedef uint64_t vtx_t;
#else
typedef uint32_t vtx_t;
#endif
#ifdef NERSTRAND_64BIT_EDGES
typedef uint64_t adj_t;
#else
typedef uint32_t adj_t;
#endif
#ifdef NERSTRAND_DOUBLE_WEIGHTS
typedef double wgt_t;
#else
typedef float wgt_t;
#endif
#endif /* NERSTRAND_GRAPH_TYPES_DEFINED */


#ifdef NERSTRAND_64BIT_CLUSTERS
typedef uint64_t cid_t;
#else
typedef uint32_t cid_t;
#endif




/******************************************************************************
* ENUMS ***********************************************************************
******************************************************************************/


typedef enum nerstrand_error_t {
  NERSTRAND_SUCCESS = 1,
  NERSTRAND_ERROR_INVALIDOPTIONS,
  NERSTRAND_ERROR_INVALIDINPUT,
  NERSTRAND_ERROR_NOTENOUGHMEMORY,
  NERSTRAND_ERROR_UNIMPLEMENTED,
  NERSTRAND_ERROR_UNKNOWN
} nerstrand_error_t;


typedef enum nerstrand_option_t {
  NERSTRAND_OPTION_HELP,
  NERSTRAND_OPTION_NCLUSTERS,
  NERSTRAND_OPTION_NTHREADS,
  NERSTRAND_OPTION_SEED,
  NERSTRAND_OPTION_NRUNS,
  NERSTRAND_OPTION_NREFPASS,
  NERSTRAND_OPTION_NINITSOLUTIONS,
  NERSTRAND_OPTION_AGGTYPE,
  NERSTRAND_OPTION_CONTYPE,
  NERSTRAND_OPTION_SPATYPE,
  NERSTRAND_OPTION_DISTYPE,
  NERSTRAND_OPTION_REFTYPE,
  NERSTRAND_OPTION_INITYPE,
  NERSTRAND_OPTION_PARTYPE,
  NERSTRAND_OPTION_VERBOSITY,
  NERSTRAND_OPTION_AGG_RATE,
  NERSTRAND_OPTION_CNVTXS_PER_CLUSTER,
  NERSTRAND_OPTION_MAXREFMOVES,
  NERSTRAND_OPTION_TIME,
  NERSTRAND_OPTION_MODSTATS,
  NERSTRAND_OPTION_ICSTATS,
  NERSTRAND_OPTION_LBSTATS,
  NERSTRAND_OPTION_AGGSTATS,
  NERSTRAND_OPTION_REFSTATS,
  NERSTRAND_OPTION_SUPERNODE_RATIO,
  NERSTRAND_OPTION_STOPRATIO,
  NERSTRAND_OPTION_STOPCONDITION,
  NERSTRAND_OPTION_DEGREE_WEIGHT,
  NERSTRAND_OPTION_BLOCKSIZE,
  NERSTRAND_OPTION_DISTRIBUTION,
  NERSTRAND_OPTION_RESTEP,
  __NERSTRAND_OPTION_TERM
} nerstrand_option_t;


typedef enum nerstrand_parttype_t {
  NERSTRAND_PARTITION_KWAY,
  NERSTRAND_PARTITION_ANYWAY
} nerstrand_parttype_t;


typedef enum nerstrand_aggtype_t {
  NERSTRAND_AGGREGATE_RM,
  NERSTRAND_AGGREGATE_SHEM,
  NERSTRAND_AGGREGATE_AGM,
  NERSTRAND_AGGREGATE_AGH,
  NERSTRAND_AGGREGATE_RC,
  NERSTRAND_AGGREGATE_FC,
  NERSTRAND_AGGREGATE_AGC
} nerstrand_aggtype_t;


typedef enum nerstrand_sparsifytype_t {
  NERSTRAND_SPARSIFY_NONE,
  NERSTRAND_SPARSIFY_RANDOM,
  NERSTRAND_SPARSIFY_LIGHT,
  NERSTRAND_SPARSIFY_HEAVY,
  NERSTRAND_SPARSIFY_DEGREE
} nerstrand_sparsifytype_t;


typedef enum nerstrand_edgeremovaltype_t {
  NERSTRAND_EDGEREMOVAL_DROP,
  NERSTRAND_EDGEREMOVAL_LOOP,
  NERSTRAND_EDGEREMOVAL_DISTRIBUTE,
  NERSTRAND_EDGEREMOVAL_PHANTOM
} nerstrand_edgeremovaltype_t;


typedef enum nerstrand_ictype_t {
  NERSTRAND_INITIAL_CLUSTERING_BFS,
  NERSTRAND_INITIAL_CLUSTERING_RANDOM,
  NERSTRAND_INITIAL_CLUSTERING_SEED,
  NERSTRAND_INITIAL_CLUSTERING_NEWMAN,
  NERSTRAND_INITIAL_CLUSTERING_LP,
  NERSTRAND_INITIAL_CLUSTERING_GROW,
  NERSTRAND_INITIAL_CLUSTERING_GROWKL,
  NERSTRAND_INITIAL_CLUSTERING_VTX,
  NERSTRAND_INITIAL_CLUSTERING_RVTX
} nerstrand_ictype_t;


typedef enum nerstrand_contype_t {
  NERSTRAND_CONTRACT_SUM
} nerstrand_contype_t;


typedef enum nerstrand_projtype_t {
  NERSTRAND_PROJECT_DIRECT,
  NERSTRAND_PROJECT_SPARSE
} nerstrand_projtype_t;


typedef enum nerstrand_reftype_t {
  NERSTRAND_REFINEMENT_GREEDY,
  NERSTRAND_REFINEMENT_RANDOM
} nerstrand_reftype_t;


typedef enum nerstrand_verbosity_t {
  NERSTRAND_VERBOSITY_MINIMUM=10,
  NERSTRAND_VERBOSITY_LOW=20,
  NERSTRAND_VERBOSITY_MEDIUM=30,
  NERSTRAND_VERBOSITY_HIGH=40,
  NERSTRAND_VERBOSITY_MAXIMUM=50
} nerstrand_verbosity_t;


typedef enum nerstrand_stopcondition_t {
  NERSTRAND_STOPCONDITION_EDGES,
  NERSTRAND_STOPCONDITION_VERTICES,
  NERSTRAND_STOPCONDITION_SIZE
} nerstrand_stopcondition_t;


typedef enum nerstrand_distribution_t {
  NERSTRAND_DISTRIBUTION_BLOCK,
  NERSTRAND_DISTRIBUTION_CYCLIC,
  NERSTRAND_DISTRIBUTION_BLOCKCYCLIC
} nerstrand_distribution_t;




/******************************************************************************
* CONSTANTS *******************************************************************
******************************************************************************/


static const size_t NERSTRAND_NOPTIONS = __NERSTRAND_OPTION_TERM;
static const double NERSTRAND_VAL_OFF = -DBL_MAX;




/******************************************************************************
* FUNCTION PROTOTYPES *********************************************************
******************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Allocate and initialize a set of options for use with the
 * nerstrand_cluster_explicit() function.
 *
 * @return The allocated and initialized options. 
 */
double * nerstrand_init_options(void);


/**
 * @brief Generate a clustering of a graph with a speficied set of options. 
 *
 * @param r_nvtxs A pointer to the number of vertices in the graph.
 * @param xadj The start of the adjacency list of each vertex.
 * @param adjncy The vertex at the far end of each edge, indexed by xadj.
 * @param adjwgt The weight of each edge, indexed by xadj.
 * @param options The options array specifying the parameters for generating
 * the clustering.
 * @param r_nclusters A pointer to the number of clusters.
 * @param cid The cluster assignment for each vertex.
 * @param r_mod A pointer to the modularity of the generated clustering.
 *
 * @return NERSTRAND_SUCCESS unless an error is encountered. 
 */
int nerstrand_cluster_explicit(
    vtx_t const * r_nvtxs,
    adj_t const * xadj,
    vtx_t const * adjncy,
    wgt_t const * adjwgt,
    double const * options,
    cid_t * r_nclusters,
    cid_t * cid,
    double * r_mod);


/**
 * @brief Generate a clustering of a graph with specified number of clusters. 
 *
 * @param r_nvtxs A pointer to the number of vertices in the graph.
 * @param xadj The start of the adjacency list of each vertex.
 * @param adjncy The vertex at the far end of each edge, indexed by xadj.
 * @param adjwgt The weight of each edge, indexed by xadj.
 * @param r_nclusters A pointer to the number of clusters.
 * @param cid The cluster assignment for each vertex.
 * @param r_mod A pointer to the modularity of the generated clustering.
 *
 * @return NERSTRAND_SUCCESS unless an error is encountered. 
 */
int nerstrand_cluster_kway(
    vtx_t const * r_nvtxs, 
    adj_t const * xadj, 
    vtx_t const * adjncy, 
    wgt_t const * adjwgt, 
    cid_t const * r_nclusters, 
    cid_t * cid, 
    double * r_mod);


/**
 * @brief Generate a clustering of a graph with an unspecified number of
 * clusters. 
 *
 * @param r_nvtxs A pointer to the number of vertices in the graph.
 * @param xadj The start of the adjacency list of each vertex.
 * @param adjncy The vertex at the far end of each edge, indexed by xadj.
 * @param adjwgt The weight of each edge, indexed by xadj.
 * @param r_nclusters A pointer to the number of clusters.
 * @param cid The cluster assignment for each vertex.
 * @param r_mod A pointer to the modularity of the generated clustering.
 *
 * @return NERSTRAND_SUCCESS unless an error is encountered. 
 */
int nerstrand_cluster_anyway(
    vtx_t const * r_nvtxs, 
    adj_t const * xadj, 
    vtx_t const * adjncy, 
    wgt_t const * adjwgt, 
    cid_t * r_nclusters, 
    cid_t * cid, 
    double * r_mod);




#ifdef __cplusplus
}
#endif


#endif
