/*
 * Copyright (c) 2021-, NVIDIA CORPORATION.
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

// MG test utility which returns a MG graph_t instance constructed from a
// edgelist (currently assumed to be generated from reading a .mtx file).  The
// data is partitioned, shuffled, and renumbered based on the current GPU rank
// before being passed to the graph_t ctor.
//
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, true> // multi_gpu=true
create_graph_for_gpu(raft::handle_t& handle,
                     edgelist_from_market_matrix_file_t<vertex_t, weight_t> edgelist_from_mm,
                     bool input_is_weighted) {

   const auto &comm = handle.get_comms();
   auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
   auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

   int my_rank = comm.get_rank();

   // Create a vertex_gpu_identifier, which will be used by the individual jobs
   // to identify if a vertex belongs to a particular GPU/job.
   cugraph::experimental::detail::compute_gpu_id_from_vertex_t<vertex_t> vertex_gpu_identifier{comm.get_size()};

   auto result = \
      cugraph::experimental::sort_and_shuffle_values
          <vertex_t*,
           cugraph::experimental::detail::compute_gpu_id_from_vertex_t<vertex_t>>
      (comm,
       edgelist_from_mm.h_cols.data(),
       edgelist_from_mm.h_rows.data(),
       vertex_gpu_identifier,
       handle.get_stream()
       );

   std::cout<<"SHUFFLED IN RANK: "<<my_rank<<std::endl;

    // renumber filtered edgelist_from_mm
    edge_t number_of_edges = static_cast<edge_t>(edgelist_from_mm.h_rows.size());
    auto renumber_info = ::cugraph::experimental::renumber_edgelist<vertex_t, edge_t, true> // multi_gpu=true
         (handle,
          edgelist_from_mm.h_cols.data(),  // edgelist_major_vertices, INOUT of vertex_t*
          edgelist_from_mm.h_rows.data(),  // edgelist_minor_vertices, INOUT of vertex_t*
          number_of_edges,
          false, // is_hypergraph_partitioned
          true); // do_expensive_check

    std::cout<<"RENUMBERED IN RANK: "<<my_rank<<std::endl;
    // create instance of graph_t using filtered & renumbered edgelist_from_mm

    rmm::device_uvector<vertex_t> d_edgelist_rows(number_of_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> d_edgelist_cols(number_of_edges, handle.get_stream());
    rmm::device_uvector<weight_t> d_edgelist_weights(input_is_weighted ? number_of_edges : 0,
                                                     handle.get_stream());

    raft::update_device(
        d_edgelist_rows.data(), edgelist_from_mm.h_rows.data(), number_of_edges, handle.get_stream());
    raft::update_device(
        d_edgelist_cols.data(), edgelist_from_mm.h_cols.data(), number_of_edges, handle.get_stream());
    if (input_is_weighted) {
       raft::update_device(
           d_edgelist_weights.data(), edgelist_from_mm.h_weights.data(), number_of_edges, handle.get_stream());
    }

    cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
        d_edgelist_rows.data(),
        d_edgelist_cols.data(),
        input_is_weighted ? d_edgelist_weights.data() : nullptr,
        number_of_edges};

    cugraph::experimental::partition_t<vertex_t> partition = std::get<1>(renumber_info);
    std::vector<cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t>> edgelist_vect;
    edgelist_vect.push_back(edgelist);
    cugraph::experimental::graph_properties_t properties;
    properties.is_symmetric = edgelist_from_mm.is_symmetric;
    properties.is_multigraph = false;

    return cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, true, true>( //store_transposed=true, multi_gpu=true
        handle,
        edgelist_vect,
        partition,
        edgelist_from_mm.number_of_vertices,
        number_of_edges,
        properties,
        false, // sorted_by_global_degree_within_vertex_partition
        false); // do_expensive_check
}

// explicit instantiation
template
cugraph::experimental::graph_t<int, int, float, true, true, void>
create_graph_for_gpu(raft::handle_t& handle,
                     edgelist_from_market_matrix_file_t<int, float> edgelist_from_mm,
                     bool input_is_weighted);

} // namespace test
} // namespace cugraph
