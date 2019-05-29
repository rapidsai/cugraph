#include "weak_cc.cuh"

#include "utilities/graph_utils.cuh"
#include "utilities/error_utils.h"
#include <cugraph.h>
#include <algo_types.h>

#include <iostream>
#include <type_traits>
#include <cstdint>

//
/**
 * @brief Compute connected components. 
 * The weak version was imported from cuML.
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @param graph input graph; assumed undirected for weakly CC [in]
 * @param labels gdf_column for the output labels [out]
 * @param connectivity_type 0=WEAK; 1=STRONG
 * @param stream the cuda stream
 */
template<typename IndexT,
         int TPB_X = 32>
std::enable_if_t<std::is_signed<IndexT>::value,gdf_error>
gdf_connected_components_impl(gdf_graph *graph,
                              gdf_column *labels,
                              cugraph_connect_t connectivity_type,
                              cudaStream_t stream)
{
  static auto row_offsets_ = [](const gdf_graph* G){
    return static_cast<const IndexT*>(G->adjList->offsets->data);
  };

  static auto col_indices_ = [](const gdf_graph* G){
    return static_cast<const IndexT*>(G->adjList->indices->data);
  };

  static auto nrows_ = [](const gdf_graph* G){
    return G->adjList->offsets->size - 1;
  };

  static auto nnz_ = [](const gdf_graph* G){
    return G->adjList->indices->size;
  };


  GDF_REQUIRE(graph != nullptr, GDF_INVALID_API_CALL);
  std::cout<<"Passed 1.\n";

  if( !graph->adjList )
    std::cerr<<"########### graph->adjlist nullptr.\n";
  
  GDF_REQUIRE(graph->adjList != nullptr, GDF_INVALID_API_CALL);
  std::cout<<"Passed 2.\n";
  
  GDF_REQUIRE(row_offsets_(graph) != nullptr, GDF_INVALID_API_CALL);
  std::cout<<"Passed 3.\n";

  GDF_REQUIRE(col_indices_(graph) != nullptr, GDF_INVALID_API_CALL);
  std::cout<<"Passed 4.\n";
  
  GDF_REQUIRE(labels != nullptr, GDF_INVALID_API_CALL);
  std::cout<<"Passed 5.\n";
  
  GDF_REQUIRE(labels->data != nullptr, GDF_INVALID_API_CALL);
  std::cout<<"Passed 6.\n";
  
  auto type_id = graph->adjList->offsets->dtype;
  GDF_REQUIRE( type_id == GDF_INT32 || type_id == GDF_INT64, GDF_UNSUPPORTED_DTYPE);

  std::cout<<"Passed 7.\n";
  
  GDF_REQUIRE( type_id == graph->adjList->indices->dtype, GDF_UNSUPPORTED_DTYPE);

  std::cout<<"Passed 8.\n";
  
  //TODO: relax this requirement:
  //
  GDF_REQUIRE( type_id == labels->dtype, GDF_UNSUPPORTED_DTYPE);

  std::cout<<"Passed 9.\n";

  //bool flag_dir = graph->prop->directed;//useless, for the time being...
  //TODO: direction_checker() to se this flag correctly; prop is not even allocated!

  std::cout<<"Passed 10.\n";
  
  if( connectivity_type == CUGRAPH_WEAK )
    {
      //check if graph is undirected; return w/ error, if not?
      //Yes, for now; in the future we may remove this constraint; 
      //
      //GDF_REQUIRE(flag_dir == false, GDF_INVALID_API_CALL);//useless check
      
      IndexT* p_d_labels = static_cast<IndexT*>(labels->data);
      const IndexT* p_d_row_offsets = row_offsets_(graph);
      const IndexT* p_d_col_ind = col_indices_(graph);

      IndexT nnz = nnz_(graph);
      IndexT nrows = nrows_(graph);

      std::cout<<"############## "
               <<"nrows = "<<nrows
               <<"; nnz = "<<nnz
               <<"p_d_labels valid: "<<(p_d_labels != nullptr)
               <<"p_d_row_offsets valid: "<<(p_d_row_offsets != nullptr)
               <<"p_d_col_ind valid: " << (p_d_col_ind != nullptr) <<"\n";
      MLCommon::Sparse::weak_cc_entry<IndexT, TPB_X>(p_d_labels,
                                                     p_d_row_offsets,
                                                     p_d_col_ind,
                                                     nnz,
                                                     nrows,
                                                     stream);

    }
  else
    {
      //dump error message and return unsupported, for now:
      //
      std::cerr<<"ERROR: Feature not supported, yet;"
               <<" at: " << __FILE__ << ":" << __LINE__ << std::endl;
      
      return GDF_INVALID_API_CALL;//for now...
    }
  return GDF_SUCCESS;
}

/**
 * @brief Compute connected components. 
 * The weak version was imported from cuML.
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 
 * @param graph input graph; assumed undirected for weakly CC [in]
 * @param labels gdf_column for the output labels [out]
 * @param connectivity_type 0=WEAK; 1=STRONG
 */
gdf_error gdf_connected_components(gdf_graph *graph,
                                   gdf_column *labels,
                                   cugraph_connect_t connectivity_type)
{
  cudaStream_t stream{nullptr};

  std::cout<<"############# reached here...; type is int32: "<<(labels->dtype==GDF_INT32) << "\n";
  
  switch( labels->dtype )//currently graph's row offsets, col_indices and labels are same type; that may change in the future
    {
    case GDF_INT32:
      return gdf_connected_components_impl<int32_t>(graph, labels, connectivity_type, stream);
      //    case GDF_INT64:
      //return gdf_connected_components_impl<int64_t>(graph, labels, connectivity_type, stream);
      // PROBLEM: relies on atomicMin(), which won't work w/ int64_t
      // should work with `unsigned long long` but using signed `Type`'s
      //(initialized to `-1`)
    default:
      break;//warning eater
    }
  return GDF_UNSUPPORTED_DTYPE;
}
