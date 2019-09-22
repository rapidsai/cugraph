///#include <gunrock/app/sm/sm_app.cuh>
#include <thrust/sequence.h>

#include "utilities/graph_utils.cuh"
#include "utilities/error_utils.h"
#include <cugraph.h>
#include <algo_types.h>

#include <iostream>
#include <type_traits>
#include <cstdint>

extern double sm(const int            num_nodes,
                 const int            num_edges,
                 const int           *row_offsets,
                 const int           *col_indices,
                 const unsigned long *edge_values,
                 const int            num_runs,
                 int           *subgraphs);



//#define _DEBUG_SM_

//
/**
 * @brief Subgraph matching. 
 * API for gunrock implementation.
 *
 * @tparam VertexT the indexing type for vertices
 * @tparam SizeT the type for sizes/dimensions
 * @tparam GValueT the type for edge weights
 * @param  graph input graph; assumed undirected [in]
 * @param  subgraphs   Return number of subgraphs [out]
 * @param  stream the cuda stream [in / optional]
 */
template <typename VertexT,
          typename SizeT,
          typename GValueT>
gdf_error gdf_subgraph_matching_impl(gdf_graph *graph,
                                     VertexT* subgraphs,
                                     cudaStream_t stream = nullptr)
{
  static auto row_offsets_ = [](const gdf_graph* G){
    return static_cast<const SizeT*>(G->adjList->offsets->data);
  };

  static auto col_indices_ = [](const gdf_graph* G){
    return static_cast<const VertexT*>(G->adjList->indices->data);
  };

  static auto values_ = [](const gdf_graph* G){
    return static_cast<const GValueT*>(G->adjList->edge_data->data);
  };


  static auto nrows_ = [](const gdf_graph* G){
    return static_cast<SizeT>(G->adjList->offsets->size - 1);
  };

  static auto nnz_ = [](const gdf_graph* G){
    return static_cast<SizeT>(G->adjList->indices->size);
  };
  
  GDF_REQUIRE(graph != nullptr, GDF_INVALID_API_CALL);
    
  GDF_REQUIRE(graph->adjList != nullptr, GDF_INVALID_API_CALL);
    
  GDF_REQUIRE(row_offsets_(graph) != nullptr, GDF_INVALID_API_CALL);

  GDF_REQUIRE(col_indices_(graph) != nullptr, GDF_INVALID_API_CALL);
    
  auto type_id = graph->adjList->offsets->dtype;
  GDF_REQUIRE( type_id == GDF_INT32 || type_id == GDF_INT64, GDF_UNSUPPORTED_DTYPE);
  
  GDF_REQUIRE( type_id == graph->adjList->indices->dtype, GDF_UNSUPPORTED_DTYPE);
  
  const SizeT* p_d_row_offsets = row_offsets_(graph);
  const VertexT* p_d_col_ind = col_indices_(graph);
  const GValueT* p_d_values = values_(graph);

  assert( p_d_values );

  SizeT nnz = nnz_(graph);
  SizeT nrows = nrows_(graph);

  //return unused (for now)...
  //
  auto t_elapsed = sm(nrows,
                      nnz,
                      p_d_row_offsets,
                      p_d_col_ind,
                      p_d_values,
                      1,
                      subgraphs);
    
  return GDF_SUCCESS;
}

/**
 * @brief Subgraph matching. 
 * API for gunrock implementation.
 *
 * @param  graph input graph; assumed undirected [in]
 * @param  subgraphs   Return number of matched subgraphs [out]
 */
gdf_error gdf_subgraph_matching(gdf_graph *graph,
                                gdf_column* subgraphs)

{
  static auto row_offsets_t_ = [](const gdf_graph* G){
    return G->adjList->offsets->dtype;
  };

  static auto col_indices_t_ = [](const gdf_graph* G){
    return G->adjList->indices->dtype;
  };

  static auto values_t_ = [](const gdf_graph* G){
    return G->adjList->edge_data->dtype;
  };

  
  auto subg_dtype = subgraphs->dtype;
  auto ro_dtype   = row_offsets_t_(graph);
  auto ci_dtype   = col_indices_t_(graph);
  auto v_dtype    = values_t_(graph);
  
  GDF_REQUIRE( subg_dtype == ci_dtype, GDF_INVALID_API_CALL);//currently Gunrock's API requires that graph's col indices and subgraphs must be same type;

  int* p_d_subg = static_cast<int*>(subgraphs->data);
  return gdf_subgraph_matching_impl<int, int, unsigned long>(graph, p_d_subg);
  
  /*
  switch( subg_dtype )
    {
    case GDF_INT32:
      {
        using VertexT = int32_t;
        VertexT* p_d_subg = static_cast<VertexT*>(subgraphs->data);
        switch( v_dtype )
          {
          case GDF_INT64:
            {
              using GValueT = unsigned long;
              return gdf_subgraph_matching_impl<VertexT, VertexT, GValueT>(graph, p_d_subg);
            }
          //more...
          default:
            break;//warning eater
          }
      }
    //more...
    default:
      break;//warning eater
    }
  */
  return GDF_UNSUPPORTED_DTYPE;
}
