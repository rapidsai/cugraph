#include <nvgraph_gdf.h>
#include <thrust/device_vector.h>
#include <ctime>

//RMM:
//

#include <rmm_utils.h>

template<typename T>
using Vector = thrust::device_vector<T, rmm_allocator<T>>;

gdf_error nvgraph2gdf_error(nvgraphStatus_t nvg_stat)
{
  switch (nvg_stat) {
    case NVGRAPH_STATUS_SUCCESS:
      return GDF_SUCCESS;
    case NVGRAPH_STATUS_NOT_INITIALIZED:
      return GDF_INVALID_API_CALL;
    case NVGRAPH_STATUS_INVALID_VALUE:
      return GDF_INVALID_API_CALL;
    case NVGRAPH_STATUS_TYPE_NOT_SUPPORTED:
      return GDF_UNSUPPORTED_DTYPE;
    case NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED:
      return GDF_INVALID_API_CALL;
    default:
      return GDF_CUDA_ERROR;
  }
}

gdf_error nvgraph2gdf_error_verbose(nvgraphStatus_t nvg_stat)
{
  switch (nvg_stat) {
    case NVGRAPH_STATUS_NOT_INITIALIZED:
      std::cerr << "nvGRAPH not initialized"; return GDF_CUDA_ERROR;
    case NVGRAPH_STATUS_ALLOC_FAILED:
      std::cerr << "nvGRAPH alloc failed"; return GDF_CUDA_ERROR;
    case NVGRAPH_STATUS_INVALID_VALUE:
      std::cerr << "nvGRAPH invalid value"; return GDF_CUDA_ERROR;
    case NVGRAPH_STATUS_ARCH_MISMATCH:
      std::cerr << "nvGRAPH arch mismatch"; return GDF_CUDA_ERROR;
    case NVGRAPH_STATUS_MAPPING_ERROR:
      std::cerr << "nvGRAPH mapping error"; return GDF_CUDA_ERROR;
    case NVGRAPH_STATUS_EXECUTION_FAILED:
      std::cerr << "nvGRAPH execution failed"; return GDF_CUDA_ERROR;
    case NVGRAPH_STATUS_INTERNAL_ERROR:
      std::cerr << "nvGRAPH internal error"; return GDF_CUDA_ERROR;
    case NVGRAPH_STATUS_TYPE_NOT_SUPPORTED:
      std::cerr << "nvGRAPH type not supported"; return GDF_CUDA_ERROR;
    case NVGRAPH_STATUS_NOT_CONVERGED:
      std::cerr << "nvGRAPH algorithm failed to converge"; return GDF_CUDA_ERROR;
    case NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED:
      std::cerr << "nvGRAPH graph type not supported"; return GDF_CUDA_ERROR;
    default:
      std::cerr << "Unknown nvGRAPH Status"; return GDF_CUDA_ERROR;
  }
}


#ifdef VERBOSE
#define NVG_TRY(call)                     \
{                                         \
  if ((call)!=NVGRAPH_STATUS_SUCCESS)   \
    return nvgraph2gdf_error_verbose((call)); \
}
#else
#define NVG_TRY(call)                     \
{                                         \
  if ((call)!=NVGRAPH_STATUS_SUCCESS)   \
    return nvgraph2gdf_error((call)); \
}
#endif

gdf_error gdf_createGraph_nvgraph(nvgraphHandle_t nvg_handle, gdf_graph* gdf_G, nvgraphGraphDescr_t* nvgraph_G, bool use_transposed) {

  // check input
  GDF_REQUIRE(!((gdf_G->edgeList == nullptr) &&
               (gdf_G->adjList == nullptr) && 
               (gdf_G->transposedAdjList == nullptr)), 
                GDF_INVALID_API_CALL);
  nvgraphTopologyType_t TT; ;
  cudaDataType_t settype;
  // create an nvgraph graph handle
  NVG_TRY(nvgraphCreateGraphDescr(nvg_handle, nvgraph_G));
  // setup nvgraph variables
  if (use_transposed) {
     // convert edgeList to transposedAdjList
    if (gdf_G->transposedAdjList == nullptr) {
      GDF_TRY(gdf_add_transpose(gdf_G));
    }
    // using exiting transposedAdjList if it exisits and if adjList is missing 
    TT = NVGRAPH_CSC_32;
    nvgraphCSCTopology32I_st topoData;
    topoData.nvertices = gdf_G->transposedAdjList->offsets->size -1;
    topoData.nedges = gdf_G->transposedAdjList->indices->size;
    topoData.destination_offsets = (int *) gdf_G->transposedAdjList->offsets->data;
    topoData.source_indices = (int *) gdf_G->transposedAdjList->indices->data;
    // attach the transposed adj list
    NVG_TRY(nvgraphAttachGraphStructure(nvg_handle, *nvgraph_G, (void *)&topoData, TT));
      //attach edge values
    if (gdf_G->transposedAdjList->edge_data) {
      switch (gdf_G->transposedAdjList->edge_data->dtype) {
        case GDF_FLOAT32:   
          settype = CUDA_R_32F;
          NVG_TRY(nvgraphAttachEdgeData(nvg_handle, *nvgraph_G, 0, settype, (float *) gdf_G->transposedAdjList->edge_data->data));
        case GDF_FLOAT64:   
          settype = CUDA_R_64F;
          NVG_TRY(nvgraphAttachEdgeData(nvg_handle, *nvgraph_G, 0, settype, (double *) gdf_G->transposedAdjList->edge_data->data));
        default: return GDF_UNSUPPORTED_DTYPE;
      }
    }
  }
  else {
    // convert edgeList to adjList
    if (gdf_G->adjList == nullptr) {
      GDF_TRY(gdf_add_adj_list(gdf_G));
    }
    TT = NVGRAPH_CSR_32;
    nvgraphCSRTopology32I_st topoData;
    topoData.nvertices = gdf_G->adjList->offsets->size -1;
    topoData.nedges = gdf_G->adjList->indices->size;
    topoData.source_offsets = (int *) gdf_G->adjList->offsets->data;
    topoData.destination_indices = (int *) gdf_G->adjList->indices->data;
    
    // attach adj list
    NVG_TRY(nvgraphAttachGraphStructure(nvg_handle, *nvgraph_G, (void *)&topoData, TT));
      //attach edge values
    if (gdf_G->adjList->edge_data) {
      switch (gdf_G->adjList->edge_data->dtype) {
        case GDF_FLOAT32:   
          settype = CUDA_R_32F;
          NVG_TRY(nvgraphAttachEdgeData(nvg_handle, *nvgraph_G, 0, settype, (float *) gdf_G->adjList->edge_data->data));
        case GDF_FLOAT64:   
          settype = CUDA_R_64F;
          NVG_TRY(nvgraphAttachEdgeData(nvg_handle, *nvgraph_G, 0, settype, (double *) gdf_G->adjList->edge_data->data));
        default: return GDF_UNSUPPORTED_DTYPE;
      }
    }
  }
  return GDF_SUCCESS;
}

gdf_error gdf_sssp_nvgraph(gdf_graph *gdf_G, 
                 const int *source_vert,
                 gdf_column *sssp_distances) {


  std::clock_t start;
  GDF_REQUIRE( gdf_G != nullptr , GDF_INVALID_API_CALL ); 
  GDF_REQUIRE( *source_vert >= 0 , GDF_INVALID_API_CALL ); 
  GDF_REQUIRE( *source_vert < sssp_distances->size , GDF_INVALID_API_CALL ); 
  GDF_REQUIRE( sssp_distances != nullptr , GDF_INVALID_API_CALL ); 
  GDF_REQUIRE( sssp_distances->data != nullptr , GDF_INVALID_API_CALL ); 
  GDF_REQUIRE( !sssp_distances->valid , GDF_VALIDITY_UNSUPPORTED );          
  GDF_REQUIRE( sssp_distances->size > 0 , GDF_INVALID_API_CALL ); 

  // init nvgraph
  // TODO : time this call
  nvgraphHandle_t nvg_handle = 0;
  nvgraphGraphDescr_t nvgraph_G = 0;
  cudaDataType_t settype;

  start = std::clock();
  NVG_TRY(nvgraphCreate(&nvg_handle));
  GDF_TRY(gdf_createGraph_nvgraph(nvg_handle, gdf_G, &nvgraph_G, true));
  std::cout << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << ","; // in ms

  int sssp_index = 0;
  int weight_index = 0;
  Vector<float> d_val;

  //RMM:
  //
  cudaStream_t stream{nullptr};
  rmm_temp_allocator allocator(stream);
    
  start = std::clock();
  if (gdf_G->transposedAdjList->edge_data == nullptr) {
    // use a fp32 vector  [1,...,1]
    settype = CUDA_R_32F;
    d_val.resize(gdf_G->transposedAdjList->indices->size);
    thrust::fill(thrust::cuda::par(allocator).on(stream),
                 d_val.begin(), d_val.end(), 1.0);
    NVG_TRY(nvgraphAttachEdgeData(nvg_handle, nvgraph_G, weight_index, settype, (void *) thrust::raw_pointer_cast(d_val.data())));
  }
  else {
    switch (gdf_G->transposedAdjList->edge_data->dtype) {
     case GDF_FLOAT32:   
       settype = CUDA_R_32F;
     case GDF_FLOAT64:   
       settype = CUDA_R_64F;
     default: return GDF_UNSUPPORTED_DTYPE;
    } 
  }

  NVG_TRY(nvgraphAttachVertexData(nvg_handle, nvgraph_G, 0, settype, sssp_distances->data ));
  std::cout << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << ","; // in ms
  start = std::clock();

  NVG_TRY(nvgraphSssp(nvg_handle, nvgraph_G, weight_index, source_vert, sssp_index));
  std::cout << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << ","; // in ms
  start = std::clock();

  NVG_TRY(nvgraphDestroyGraphDescr(nvg_handle, nvgraph_G));
  NVG_TRY(nvgraphDestroy(nvg_handle));
  std::cout << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) <<std::endl; // in ms


  return GDF_SUCCESS;
}
