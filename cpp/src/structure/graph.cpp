/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// Graph analytics features

#include <memory>

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <utilities/error_utils.h>
#include <graph.hpp>


namespace cugraph {
namespace experimental {

std::unique_ptr<Graph> Graph:: from_edge_list(cudf::column_view const &src_indices,
                                              cudf::column_view const &dst_indices,
                                              cudf::column_view const &edge_data,
                                              size_t number_of_vertices,
                                              size_t number_of_edges) {

  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  CUGRAPH_EXPECTS( src_indices.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( dst_indices.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( src_indices.type() == dst_indices.type() , "Unsupported data type" );
  CUGRAPH_EXPECTS( src_indices.type().id() == cudf::experimental::type_to_id<int32_t>() , "Unsupported data type" );
  CUGRAPH_EXPECTS( src_indices.size() == dst_indices.size() , "Source and Destination must have same number of rows" );
  CUGRAPH_EXPECTS( src_indices.size() > 0 , "No edges");

  cudf::column_view empty;

  EdgeList coo(src_indices, dst_indices, edge_data);
  AdjList  csr(empty, empty, empty);
  AdjList  csc(empty, empty, empty);

  return std::unique_ptr<Graph>(new Graph(coo, csr, csc, number_of_vertices, number_of_edges));
}

std::unique_ptr<Graph> Graph:: from_edge_list(cudf::column_view const &src_indices,
                                              cudf::column_view const &dst_indices,
                                              size_t number_of_vertices,
                                              size_t number_of_edges) {

  cudf::column_view empty;
  return from_edge_list(src_indices, dst_indices, empty, number_of_vertices, number_of_edges);
}

std::unique_ptr<Graph> Graph:: from_adj_list(cudf::column_view const &offsets,
                                             cudf::column_view const &indices,
                                             cudf::column_view const &edge_data,
                                             size_t number_of_vertices,
                                             size_t number_of_edges) {

  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  CUGRAPH_EXPECTS( offsets.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( indices.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( offsets.type() == indices.type() , "Unsupported data type" );
  CUGRAPH_EXPECTS( offsets.type().id() == cudf::experimental::type_to_id<int32_t>() , "Unsupported data type" );
  CUGRAPH_EXPECTS( offsets.size() > 0 , "Offsets column is empty");

  cudf::column_view empty;
  EdgeList coo(empty, empty, empty);
  AdjList  csr(offsets, indices, edge_data);
  AdjList  csc(empty, empty, empty);

  return std::unique_ptr<Graph>(new Graph(coo, csr, csc, number_of_vertices, number_of_edges));
}

std::unique_ptr<Graph> Graph:: from_adj_list(cudf::column_view const &offsets,
                                             cudf::column_view const &indices,
                                             size_t number_of_vertices,
                                             size_t number_of_edges) {
  cudf::column_view empty;
  return from_adj_list(offsets, indices, empty, number_of_vertices, number_of_edges);
}

std::unique_ptr<Graph> Graph:: from_transposed_adj_list(cudf::column_view const &offsets,
                                                        cudf::column_view const &indices,
                                                        cudf::column_view const &edge_data,
                                                        size_t number_of_vertices,
                                                        size_t number_of_edges) {

  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  CUGRAPH_EXPECTS( offsets.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( indices.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( offsets.type() == indices.type() , "Unsupported data type" );
  CUGRAPH_EXPECTS( offsets.type().id() == cudf::experimental::type_to_id<int32_t>() , "Unsupported data type" );
  CUGRAPH_EXPECTS( offsets.size() > 0 , "Offsets column is empty");

  cudf::column_view empty;
  EdgeList coo(empty, empty, empty);
  AdjList  csr(empty, empty, empty);
  AdjList  csc(offsets, indices, edge_data);

  return std::unique_ptr<Graph>(new Graph(coo, csr, csc, number_of_vertices, number_of_edges));
}

std::unique_ptr<Graph> Graph:: from_transposed_adj_list(cudf::column_view const &offsets,
                                                        cudf::column_view const &indices,
                                                        size_t number_of_vertices,
                                                        size_t number_of_edges) {

  cudf::column_view empty;
  return from_transposed_adj_list(offsets, indices, empty, number_of_vertices, number_of_edges);
}

} //namespace experimental
} //namespace cugraph
