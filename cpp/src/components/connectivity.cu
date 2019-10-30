#include "weak_cc.cuh"
#include "scc_matrix.cuh"

#include <thrust/sequence.h>

#include "utilities/graph_utils.cuh"
#include "utilities/error_utils.h"
#include <cugraph.h>
#include <algo_types.h>

#include <iostream>
#include <type_traits>
#include <cstdint>

#include "topology/topology.cuh"

//#define _DEBUG_WEAK_CC

//
/**
 * @brief Compute connected components. 
 * The weak version (for undirected graphs, only) was imported from cuML.
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 * 
 * The strong version (for directed or undirected graphs) is based on: 
 * [2] Gilbert, J. et al, 2011. "Graph Algorithms in the Language of Linear Algebra"
 *
 * C = I | A | A^2 |...| A^k
 * where matrix multiplication is via semi-ring: 
 * (combine, reduce) == (&, |) (bitwise ops)
 * Then: X = C & transpose(C); and finally, apply get_labels(X);
 *
 *
 * @tparam IndexT the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @param graph input graph; assumed undirected for weakly CC [in]
 * @param table of 2 gdf_columns: output labels and vertex indices [out]
 * @param connectivity_type CUGRAPH_WEAK or CUGRAPH_STRONG [in]
 * @param stream the cuda stream [in]
 */
template<typename IndexT,
         int TPB_X = 32>
std::enable_if_t<std::is_signed<IndexT>::value,gdf_error>
gdf_connected_components_impl(gdf_graph *graph,
                              cudf::table *table,
                              cugraph_cc_t connectivity_type,
                              cudaStream_t stream)
{
  using ByteT = unsigned char;//minimum addressable unit
  
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
  
  gdf_column* labels = table->get_column(0);
  gdf_column* verts = table->get_column(1);

  GDF_REQUIRE(graph != nullptr, GDF_INVALID_API_CALL);
    
  GDF_REQUIRE(graph->adjList != nullptr, GDF_INVALID_API_CALL);
    
  GDF_REQUIRE(row_offsets_(graph) != nullptr, GDF_INVALID_API_CALL);

  GDF_REQUIRE(col_indices_(graph) != nullptr, GDF_INVALID_API_CALL);
  
  GDF_REQUIRE(labels->data != nullptr, GDF_INVALID_API_CALL);

  GDF_REQUIRE(verts->data != nullptr, GDF_INVALID_API_CALL);
  
  auto type_id = graph->adjList->offsets->dtype;
  GDF_REQUIRE( type_id == GDF_INT32 || type_id == GDF_INT64, GDF_UNSUPPORTED_DTYPE);
  
  GDF_REQUIRE( type_id == graph->adjList->indices->dtype, GDF_UNSUPPORTED_DTYPE);
  
  //TODO: relax this requirement:
  //
  GDF_REQUIRE( type_id == labels->dtype, GDF_UNSUPPORTED_DTYPE);

  IndexT* p_d_labels = static_cast<IndexT*>(labels->data);
  IndexT* p_d_verts = static_cast<IndexT*>(verts->data);
  
  const IndexT* p_d_row_offsets = row_offsets_(graph);
  const IndexT* p_d_col_ind = col_indices_(graph);

  IndexT nnz = nnz_(graph);
  IndexT nrows = nrows_(graph);//static_cast<IndexT>(graph->adjList->offsets->size) - 1;
  
  if( connectivity_type == CUGRAPH_WEAK )
    {
      // using VectorT = thrust::device_vector<IndexT>;
      // VectorT d_ro(p_d_row_offsets, p_d_row_offsets + nrows + 1);
      // VectorT d_ci(p_d_col_ind, p_d_col_ind + nnz);

#ifdef _DEBUG_WEAK_CC
      IndexT last_elem{0};
      cudaMemcpy((void*)(&last_elem), p_d_row_offsets+nrows, sizeof(IndexT), cudaMemcpyDeviceToHost);
      std::cout<<"############## "
               <<"nrows = "<<nrows
               <<"; nnz = "<<nnz
               <<"; nnz_ro = "<<last_elem
               <<"; p_d_labels valid: "<<(p_d_labels != nullptr)
               <<"; p_d_row_offsets valid: "<<(p_d_row_offsets != nullptr)
               <<"; p_d_col_ind valid: " << (p_d_col_ind != nullptr)
               <<"\n";
      
      std::cout<<"############## d_ro:\n";
      print_v(d_ro, std::cout);

      std::cout<<"############## d_ci:\n";
      print_v(d_ci, std::cout);
#endif

      //check if graph is undirected; return w/ error, if not?
      //Yes, for now; in the future we may remove this constraint; 
      //
      bool is_symmetric = cugraph::detail::check_symmetry(nrows, p_d_row_offsets, nnz, p_d_col_ind);
#ifdef _DEBUG_WEAK_CC
      std::cout<<"############## "
               <<"; adj. matrix symmetric? " << is_symmetric
               <<"\n";
#endif
      
      GDF_REQUIRE( is_symmetric, GDF_INVALID_API_CALL);
      MLCommon::Sparse::weak_cc_entry<IndexT, TPB_X>(p_d_labels,
                                                     p_d_row_offsets,
                                                     p_d_col_ind,
                                                     nnz,
                                                     nrows,
                                                     stream);

    }
  else
    {
      
      //device memory requirements: 2n^2 + 2n x sizeof(IndexT) + 1 (for flag)
      //( n = |V|)
      //
      size_t n2 = 2*nrows;
      n2 = n2*(nrows*sizeof(ByteT) + sizeof(IndexT)) + 1;

      int device;
      cudaDeviceProp prop;
      
      cudaGetDevice(&device);
      cudaGetDeviceProperties(&prop, device);

      if( n2 > prop.totalGlobalMem )
        {

          //not enough memory, dump error message and return unsupported:
          //
          std::cerr<<"ERROR: Insufficient device memory for SCC;"
                   <<" at: " << __FILE__ << ":" << __LINE__ << std::endl;
      
          return GDF_MEMORYMANAGER_ERROR;
        }
      SCC_Data<ByteT, IndexT> sccd(nrows, p_d_row_offsets, p_d_col_ind);
      sccd.run_scc(p_d_labels);
      
    }

  //fill the vertex indices column:
  //
  thrust::sequence(thrust::device, p_d_verts, p_d_verts + nrows);
  
  return GDF_SUCCESS;
}

/**
 * @brief Compute connected components. 
 * The weak version (for undirected graphs, only) was imported from cuML.
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 * 
 * The strong version (for directed or undirected graphs) is based on: 
 * [2] Gilbert, J. et al, 2011. "Graph Algorithms in the Language of Linear Algebra"
 *
 * C = I | A | A^2 |...| A^k
 * where matrix multiplication is via semi-ring: 
 * (combine, reduce) == (&, |) (bitwise ops)
 * Then: X = C & transpose(C); and finally, apply get_labels(X);
 *
 *
 * @param graph input graph; assumed undirected for weakly CC [in]
 * @param connectivity_type CUGRAPH_WEAK or CUGRAPH_STRONG [in]
 * @param table of 2 gdf_columns: output labels and vertex indices [out]
 */
 gdf_error gdf_connected_components(gdf_graph *graph,
                                    cugraph_cc_t connectivity_type,
                                    cudf::table *table)  
{
  cudaStream_t stream{nullptr};

  GDF_REQUIRE(table != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(table->num_columns() > 1, GDF_INVALID_API_CALL);
  
  gdf_column* labels = table->get_column(0);
  gdf_column* verts = table->get_column(1);

  GDF_REQUIRE(labels != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(verts != nullptr, GDF_INVALID_API_CALL);

  auto dtype = labels->dtype;
  GDF_REQUIRE( dtype == verts->dtype, GDF_INVALID_API_CALL);
  
  switch( dtype )//currently graph's row offsets, col_indices and labels are same type; that may change in the future
    {
    case GDF_INT32:
      return gdf_connected_components_impl<int32_t>(graph, table, connectivity_type, stream);
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
