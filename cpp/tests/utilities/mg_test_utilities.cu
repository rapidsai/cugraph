/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <experimental/graph.hpp>
#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph_functions.hpp>
#include <utilities/shuffle_comm.cuh>

#include <utilities/mg_test_utilities.hpp>


namespace cugraph {
namespace test {

// Given a raft handle and an edgelist from reading a dataset (.mtx in this
// case), returns a tuple containing:
//  * graph_t instance for the partition accesible from the raft handle
//  * 4-tuple containing renumber info resulting from renumbering the
//    edgelist for the partition
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
std::tuple<
   std::unique_ptr<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, true>>, // multi_gpu=true
   rmm::device_uvector<vertex_t>
>
create_graph_for_gpu(raft::handle_t& handle,
                     edgelist_from_market_matrix_file_t<vertex_t, weight_t> edgelist_from_mm) {

   const auto &comm = handle.get_comms();
   auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
   auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

   int my_rank = comm.get_rank();

   //auto edgelist_from_mm = ::cugraph::test::read_edgelist_from_matrix_market_file<vertex_t, edge_t, weight_t>(graph_file_path);

   edge_t total_number_edges = static_cast<edge_t>(edgelist_from_mm.h_rows.size());

   //////////
   // Copy COO to device
   rmm::device_uvector<vertex_t> d_edgelist_rows(total_number_edges, handle.get_stream());
   rmm::device_uvector<vertex_t> d_edgelist_cols(total_number_edges, handle.get_stream());
   rmm::device_uvector<weight_t> d_edgelist_weights(total_number_edges, handle.get_stream());

   raft::update_device(
      d_edgelist_rows.data(), edgelist_from_mm.h_rows.data(), total_number_edges, handle.get_stream());
   raft::update_device(
      d_edgelist_cols.data(), edgelist_from_mm.h_cols.data(), total_number_edges, handle.get_stream());
   raft::update_device(
      d_edgelist_weights.data(), edgelist_from_mm.h_weights.data(), total_number_edges, handle.get_stream());

   std::cout<<"COPIED COO TO DEVICE FOR RANK: "<<my_rank<<" SIZE: "<<total_number_edges<<std::endl;

   //////////
   // Filter out edges that are not to be associated with this GPU.
   //
   // Create a edge_gpu_identifier, which will be used by the individual jobs to
   // identify if a edge belongs to a particular job's GPU.
   cugraph::experimental::detail::compute_gpu_id_from_edge_t<vertex_t>
      edge_gpu_identifier{false, comm.get_size(), row_comm.get_size(), col_comm.get_size()};
   //cugraph::experimental::detail::compute_gpu_id_from_vertex_t<vertex_t> vertex_gpu_identifier{comm.get_size()};

   auto edgelist_zip_it_begin =
      thrust::make_zip_iterator(thrust::make_tuple(d_edgelist_rows.begin(),
                                                   d_edgelist_cols.begin(),
                                                   d_edgelist_weights.begin()));
   bool is_transposed{store_transposed};

   auto new_end = thrust::remove_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    edgelist_zip_it_begin,
                                    edgelist_zip_it_begin + total_number_edges,
                                    [my_rank, is_transposed, edge_gpu_identifier] __device__(auto tup) {
                                       if(is_transposed) {
                                          return (edge_gpu_identifier(thrust::get<1>(tup), thrust::get<0>(tup)) != my_rank);
                                          //return (edge_gpu_identifier(thrust::get<0>(tup), thrust::get<1>(tup)) != my_rank);
                                       } else {
                                          return (edge_gpu_identifier(thrust::get<0>(tup), thrust::get<1>(tup)) != my_rank);
                                          //return (edge_gpu_identifier(thrust::get<1>(tup), thrust::get<0>(tup)) != my_rank);
                                       }
                                    });

   edge_t local_number_edges = thrust::distance(edgelist_zip_it_begin, new_end);
   // Free the memory used for the items remove_if "removed". This not only
   // frees memory, but keeps the actual vector sizes consistent with the data
   // being used from this point forward.
   d_edgelist_rows.resize(local_number_edges, handle.get_stream());
   d_edgelist_rows.shrink_to_fit(handle.get_stream());
   d_edgelist_cols.resize(local_number_edges, handle.get_stream());
   d_edgelist_cols.shrink_to_fit(handle.get_stream());
   d_edgelist_weights.resize(local_number_edges, handle.get_stream());
   d_edgelist_weights.shrink_to_fit(handle.get_stream());

   std::cout<<"FILTERED OUT EDGES NOT BELONGING TO RANK: "<<my_rank<<" NEW SIZE: "<<local_number_edges<<std::endl;

   //////////
   // renumber filtered edgelist_from_mm
   vertex_t* major_vertices{nullptr};
   vertex_t* minor_vertices{nullptr};
   if(is_transposed) {
      major_vertices = d_edgelist_cols.data();
      minor_vertices = d_edgelist_rows.data();
   } else {
      major_vertices = d_edgelist_rows.data();
      minor_vertices = d_edgelist_cols.data();
   }

   rmm::device_uvector<vertex_t> renumber_map_labels(0, handle.get_stream());
   cugraph::experimental::partition_t<vertex_t> partition(std::vector<vertex_t>(comm.get_size() + 1, 0),
                                   false, // is_hypergraph_partitioned(),
                                   row_comm.get_size(),
                                   col_comm.get_size(),
                                   row_comm.get_rank(),
                                   col_comm.get_rank());
   vertex_t number_of_vertices{};
   edge_t number_of_edges{};
   std::tie(renumber_map_labels, partition, number_of_vertices, number_of_edges) =
      ::cugraph::experimental::renumber_edgelist<vertex_t, edge_t, true> // multi_gpu=true
      (handle,
       major_vertices,  // edgelist_major_vertices, INOUT of vertex_t*
       minor_vertices,  // edgelist_minor_vertices, INOUT of vertex_t*
       local_number_edges,
       false, // is_hypergraph_partitioned
       true); // do_expensive_check
   std::cout<<"RENUMBERED IN RANK: "<<my_rank<<std::endl;

   cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
      d_edgelist_rows.data(),
      d_edgelist_cols.data(),
      d_edgelist_weights.data(),
      local_number_edges};

   //cugraph::experimental::partition_t<vertex_t> partition = std::get<1>(renumber_info);
   std::vector<cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t>> edgelist_vect;
   edgelist_vect.push_back(edgelist);
   cugraph::experimental::graph_properties_t properties;
   properties.is_symmetric = edgelist_from_mm.is_symmetric;
   properties.is_multigraph = false;

   // Finally, create instance of graph_t using filtered & renumbered edgelist
   return std::make_tuple(
      std::make_unique<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, true>>(
         handle,
         edgelist_vect,
         partition,
         edgelist_from_mm.number_of_vertices,
         total_number_edges,
         properties,
         false, // sorted_by_global_degree_within_vertex_partition
         true), // do_expensive_check
      std::move(renumber_map_labels));
}

// explicit instantiation
   //   std::tuple<rmm::device_uvector<int32_t>, cugraph::experimental::partition_t<int32_t>, int32_t, int32_t>&&
template
std::tuple<
   std::unique_ptr<cugraph::experimental::graph_t<int32_t, int32_t, float, true, true>>, // store_transposed=true multi_gpu=true
   rmm::device_uvector<int32_t>
>
create_graph_for_gpu(raft::handle_t& handle,
                     edgelist_from_market_matrix_file_t<int32_t, float> edgelist_from_mm);

} // namespace test
} // namespace cugraph
