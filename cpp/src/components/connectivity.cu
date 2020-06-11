#include "scc_matrix.cuh"
#include "weak_cc.cuh"

#include <thrust/sequence.h>

#include <algorithms.hpp>
#include <cstdint>
#include <graph.hpp>
#include <iostream>
#include <type_traits>
#include "utilities/error_utils.h"
#include "utilities/graph_utils.cuh"

#include "topology/topology.cuh"

namespace cugraph {
namespace detail {

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
 * @param connectivity_type CUGRAPH_WEAK or CUGRAPH_STRONG [in]
 * @param stream the cuda stream [in]
 */
template <typename VT, typename ET, typename WT, int TPB_X = 32>
std::enable_if_t<std::is_signed<VT>::value> connected_components_impl(
  experimental::GraphCSRView<VT, ET, WT> const &graph,
  cugraph_cc_t connectivity_type,
  VT *labels,
  cudaStream_t stream)
{
  using ByteT = unsigned char;  // minimum addressable unit

  CUGRAPH_EXPECTS(graph.offsets != nullptr, "Invalid API parameter: graph.offsets is nullptr");
  CUGRAPH_EXPECTS(graph.indices != nullptr, "Invalid API parameter: graph.indices is nullptr");

  VT nrows = graph.number_of_vertices;

  if (connectivity_type == cugraph_cc_t::CUGRAPH_WEAK) {
    MLCommon::Sparse::weak_cc_entry<VT, ET, TPB_X>(labels,
                                                   graph.offsets,
                                                   graph.indices,
                                                   graph.number_of_edges,
                                                   graph.number_of_vertices,
                                                   stream);
  } else {
    SCC_Data<ByteT, VT> sccd(nrows, graph.offsets, graph.indices);
    sccd.run_scc(labels);
  }
}
}  // namespace detail

template <typename VT, typename ET, typename WT>
void connected_components(experimental::GraphCSRView<VT, ET, WT> const &graph,
                          cugraph_cc_t connectivity_type,
                          VT *labels)
{
  cudaStream_t stream{nullptr};

  CUGRAPH_EXPECTS(labels != nullptr, "Invalid API parameter: labels parameter is NULL");

  return detail::connected_components_impl<VT, ET, WT>(graph, connectivity_type, labels, stream);
}

template void connected_components<int32_t, int32_t, float>(
  experimental::GraphCSRView<int32_t, int32_t, float> const &, cugraph_cc_t, int32_t *);
template void connected_components<int64_t, int64_t, float>(
  experimental::GraphCSRView<int64_t, int64_t, float> const &, cugraph_cc_t, int64_t *);

}  // namespace cugraph
