#include <cugraph.h>
#include <rmm_utils.h>
#include "converters/COOtoCSR.cuh"

template <typename IdxT>
struct permutation_functor{
  IdxT* permutation;
  permutation_functor(IdxT* p):permutation(p){}
  __host__ __device__
  IdxT operator()(IdxT in){
    return permutation[in];
  }
};

/**
 * This function takes a graph and a permutation vector and permutes the
 * graph according to the permutation vector. So each vertex id i becomes
 * vertex id permutation[i] in the permuted graph.
 * @param graph The graph to permute.
 * @param permutation The permutation vector to use, must be a valid permutation
 * i.e. contains all values 0-n exactly once.
 * @return The permuted graph.
 */
template<typename IdxT, typename ValT>
cugraph::Graph* permute_graph(cugraph::Graph* graph, IdxT* permutation) {
  CUGRAPH_EXPECTS(graph->adjList || graph->edgeList, "Graph requires connectivity information.");
  IdxT nnz;
  if (graph->edgeList) {
    nnz = graph->edgeList->src_indices->size;
  }
  else if (graph->adjList){
    nnz = graph->adjList->indices->size;
  }
  IdxT* src_indices;
  ALLOC_TRY(&src_indices, sizeof(IdxT) * nnz, nullptr);
  IdxT* dest_indices;
  ALLOC_TRY(&dest_indices, sizeof(IdxT) * nnz, nullptr);
  ValT* weights = nullptr;

  // Fill a copy of the data from either the edge list or adjacency list:
  if (graph->edgeList) {
    thrust::copy(rmm::exec_policy(nullptr)->on(nullptr),
                 (IdxT*)graph->edgeList->src_indices->data,
                 (IdxT*)graph->edgeList->src_indices->data + nnz,
                 src_indices);
    thrust::copy(rmm::exec_policy(nullptr)->on(nullptr),
                 (IdxT*)graph->edgeList->dest_indices->data,
                 (IdxT*)graph->edgeList->dest_indices->data + nnz,
                 dest_indices);
    weights = (ValT*) graph->edgeList->edge_data->data;
  }
  else if (graph->adjList) {
    cugraph::detail::offsets_to_indices((IdxT*) graph->adjList->offsets->data,
                                        (IdxT)graph->adjList->offsets->size - 1,
                                        src_indices);
    thrust::copy(rmm::exec_policy(nullptr)->on(nullptr),
                 (IdxT*) graph->adjList->indices->data,
                 (IdxT*) graph->adjList->indices->data + nnz,
                 dest_indices);
    weights = (ValT*)graph->adjList->edge_data->data;
  }

  // Permute the src_indices
  permutation_functor<IdxT>pf(permutation);
  thrust::transform(rmm::exec_policy(nullptr)->on(nullptr),
                    src_indices,
                    src_indices + nnz,
                    src_indices,
                    pf);

  // Permute the destination indices
  thrust::transform(rmm::exec_policy(nullptr)->on(nullptr),
                    dest_indices,
                    dest_indices + nnz,
                    dest_indices,
                    pf);

  // Call COO2CSR to get the new adjacency
  CSR_Result_Weighted<IdxT, ValT>new_csr;
  ConvertCOOtoCSR_weighted(src_indices,
                           dest_indices,
                           weights,
                           (int64_t) nnz,
                           new_csr);

  // Construct the result graph
  cugraph::Graph* result = new cugraph::Graph;
  result->adjList = new cugraph::gdf_adj_list;
  result->adjList->offsets = new gdf_column;
  result->adjList->indices = new gdf_column;
  result->adjList->edge_data = new gdf_column;
  result->adjList->ownership = 1;

  gdf_column_view(result->adjList->offsets,
                  new_csr.rowOffsets,
                  nullptr,
                  new_csr.size + 1,
                  graph->adjList->offsets->dtype);
  gdf_column_view(result->adjList->indices,
                  new_csr.colIndices,
                  nullptr,
                  nnz,
                  graph->adjList->offsets->dtype);
  gdf_column_view(result->adjList->edge_data,
                  new_csr.edgeWeights,
                  nullptr,
                  nnz,
                  graph->adjList->edge_data->dtype);

  ALLOC_FREE_TRY(src_indices, nullptr);
  ALLOC_FREE_TRY(dest_indices, nullptr);

  return result;
}
