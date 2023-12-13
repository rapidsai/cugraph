/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#pragma once

// FIXME: remove all unused imports
#include <prims/extract_transform_e.cuh>
#include <prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh>
#include <prims/update_edge_src_dst_property.cuh>
#include <prims/reduce_op.cuh>
#include <prims/edge_bucket.cuh>
#include <prims/transform_e.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {

namespace {

template <typename vertex_t>
struct exclude_self_loop_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return src != dst
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : thrust::nullopt;
  }
};

template <typename edge_t>
struct is_two_or_greater_t {
  __device__ bool operator()(edge_t core_number) const { return core_number >= edge_t{2}; }
};

template <typename vertex_t>
struct extract_two_core_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src, vertex_t dst, bool src_in_two_core, bool dst_in_two_core, thrust::nullopt_t) const
  {
    return (src_in_two_core && dst_in_two_core)
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : thrust::nullopt;
  }
};



template <typename vertex_t, typename edge_t>
struct extract_low_to_high_degree_edges_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(vertex_t src,
                                                                            vertex_t dst,
                                                                            edge_t src_out_degree,
                                                                            edge_t dst_out_degree,
                                                                            thrust::nullopt_t) const
  {
    return (src_out_degree < dst_out_degree)
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : (((src_out_degree == dst_out_degree) &&
                 (src < dst) /* tie-breaking using vertex ID */)
                  ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src,
                                                                                           dst)}
                  : thrust::nullopt);
  }
};


// primary key: major_comm_rank secondary key: local edge partition index => primary key: local edge
// partition index secondary key: major_comm_rank
template <typename vertex_t, typename edge_t>
struct update_edge_id {
  //vertex_pairs_first
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  //raft::device_span<edge_t const> intersection_nbr_prop_0{};
  //raft::device_span<edge_t const> intersection_nbr_prop_1{};


  __device__ edge_t operator()(edge_t i)
  {
   //printf("\nin operator");
   return intersection_offsets[i+1] - intersection_offsets[i];
  }
};



template <typename vertex_t, typename edge_t, typename VertexPairIterator>
struct update_edge_id_ {
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  //thrust::zip_iterator<thrust::tuple<vertex_t*, vertex_t*>> vertex_pairs;
  //typedef rmm::device_uvector<vertex_t>::iterator  VtxItr
  //typedef thrust::tuple<VtxItr, VtxItr> IteratorTuple;
  //thrust::zip_iterator<thrust::tuple<rmm::device_uvector<vertex_t>::iterator, rmm::device_uvector<vertex_t>::iterator>> vertex_pairs_begin;
  //ZipIterator vertex_pairs_begin;
  //thust::tuple
  
  VertexPairIterator vertex_pairs_begin;
  //printf("\nvalue = %d", *vertex_pairs_begin);

  __device__ thrust::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    printf("\n i = %d", i);
    auto itr = thrust::upper_bound(thrust::seq, intersection_offsets.begin()+1, intersection_offsets.end(), i);

    auto idx = thrust::distance(intersection_offsets.begin()+1, itr);
    printf("\nthe idx = %d", idx);

    //printf("\n dereference = %d", *(vertex_pairs_begin + idx));
    printf("\n major = %d and minor = %d", thrust::get<0>(*(vertex_pairs_begin + idx)), thrust::get<1>(*(vertex_pairs_begin + idx)));
    printf("\nintersection = %d\n", intersection_offsets[i+1] - intersection_offsets[i]);
    /*
    auto local_edge_partition_id = static_cast<int>(i) / major_comm_size;
    auto major_comm_rank         = static_cast<int>(i) % major_comm_size;
    return group_counts[major_comm_rank * minor_comm_size + local_edge_partition_id];
    */
   //auto y = intersection_indices[0];
   
   thrust::tuple<vertex_t, vertex_t> pair = *(vertex_pairs_begin + idx);
   //thrust::tuple<vertex_t, vertex_t, vertex_t> pair_ = thrust::make_tuple(*(vertex_pairs_begin + idx), 1);
   
   //auto x = thrust::get<0>(*(vertex_pairs_begin + idx));
   //auto y = thrust::get<1>(*(vertex_pairs_begin + idx));
   //int z = pair;

   //auto pair_ = thrust::make_tuple(x, y, 1);
   //return 0;
   return pair;
   //return *(vertex_pairs_begin + idx);
  }
};


/*
template <typename vertex_t, typename edge_t>
struct update_num_triangles{

  raft::device_span<vertex_t const> num_triangles{};
  // should it be 'edge_t' instead of 'vertex_t'
  //__device__ vertex_t operator()(edge_t i) const
  //auto i = 0;
  __device__ vertex_t operator()(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return num_triangles[0];
  }

};
*/



template <typename vertex_t, typename edge_t>
struct intersection_op_t {
  __device__ thrust::tuple<edge_t, edge_t> operator()(
    vertex_t v0,
    vertex_t v1,
    edge_t v0_prop /* out degree */,
    edge_t v1_prop /* out degree */,
    raft::device_span<vertex_t const> intersection,
    std::byte, /* dummy */
    std::byte  /* dummy */
  ) const
  {
    return thrust::make_tuple(v0_prop + v1_prop, static_cast<edge_t>(intersection.size()));
  }
};


} // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
void ktruss(raft::handle_t const& handle,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
            vertex_t k,
            bool do_expensive_check)
{
  using weight_t = float;  // dummy

  std::cout << "k = " << k << std::endl;

  // 1. Check input arguments.

  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  CUGRAPH_EXPECTS(
    graph_view.is_symmetric(),
    "Invalid input arguments: triangle_count currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(
    !graph_view.is_multigraph(),
    "Invalid input arguments: triangle_count currently does not support multi-graphs.");
  

  // 2. Exclude self-loops (FIXME: better mask-out once we add masking support).

  std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
  std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>, edge_t>> edge_ids{std::nullopt};
  // FIXME: Maybe should not be optional
  std::optional<rmm::device_uvector<edge_t>> wgts{std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> wgts_{std::nullopt};
  //RAFT_CUDA_TRY(cudaDeviceSynchronize());
  //raft::print_device_vector("edge property before statement", (*wgts).data(), (*wgts).size(), std::cout); 

  if (graph_view.count_self_loops(handle) > edge_t{0}) {
    auto [srcs, dsts] = extract_transform_e(handle,
                                            graph_view,
                                            edge_src_dummy_property_t{}.view(),
                                            edge_dst_dummy_property_t{}.view(),
                                            edge_dummy_property_t{}.view(),
                                            exclude_self_loop_t<vertex_t>{});
    
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("src - self loop ", srcs.data(), srcs.size(), std::cout);
    raft::print_device_vector("dst - self loop ", dsts.data(), dsts.size(), std::cout);

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt, std::nullopt);
    }

    std::tie(*modified_graph, std::ignore, std::ignore, std::ignore, renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{true, graph_view.is_multigraph()},
        true);

    modified_graph_view = (*modified_graph).view();
  }

  // 3. Find (k+1)-core and exclude edges that do not belong to (k+1)-core (FIXME: better mask-out once we
  // add masking support).

  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    auto vertex_partition_range_lasts =
      renumber_map
        ? std::make_optional<std::vector<vertex_t>>(cur_graph_view.vertex_partition_range_lasts())
        : std::nullopt;

    rmm::device_uvector<edge_t> core_numbers(cur_graph_view.number_of_vertices(),
                                             handle.get_stream());
    core_number(
      handle, cur_graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{k+1}, size_t{k+1});

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("core_number after ", core_numbers.data(), core_numbers.size(), std::cout);

    edge_src_property_t<decltype(cur_graph_view), bool> edge_src_in_two_cores(handle,
                                                                              cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), bool> edge_dst_in_two_cores(handle,
                                                                              cur_graph_view);
    auto in_two_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), is_two_or_greater_t<edge_t>{});
    rmm::device_uvector<bool> in_two_core_flags(core_numbers.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 in_two_core_first,
                 in_two_core_first + core_numbers.size(),
                 in_two_core_flags.begin());
    update_edge_src_property(
      handle, cur_graph_view, in_two_core_flags.begin(), edge_src_in_two_cores);
    update_edge_dst_property(
      handle, cur_graph_view, in_two_core_flags.begin(), edge_dst_in_two_cores);
    auto [srcs, dsts] = extract_transform_e(handle,
                                            cur_graph_view,
                                            edge_src_in_two_cores.view(),
                                            edge_dst_in_two_cores.view(),
                                            edge_dummy_property_t{}.view(),
                                            extract_two_core_t<vertex_t>{});

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt, std::nullopt);
    }

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};

    std::tie(*modified_graph, std::ignore, std::ignore, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{true, graph_view.is_multigraph()},
        true);

    modified_graph_view = (*modified_graph).view();

    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                   (*tmp_renumber_map).data(),
                                                   (*tmp_renumber_map).size(),
                                                   (*renumber_map).data(),
                                                   *vertex_partition_range_lasts);
    }
    renumber_map = std::move(tmp_renumber_map);
    //RAFT_CUDA_TRY(cudaDeviceSynchronize());
    //raft::print_device_vector("rneumber map - low - to - high", (*renumber_map).data(), (*renumber_map).size(), std::cout);
  }

  // 4. Keep only the edges from a low-degree vertex to a high-degree vertex.
  {
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    auto vertex_partition_range_lasts =
      renumber_map
        ? std::make_optional<std::vector<vertex_t>>(cur_graph_view.vertex_partition_range_lasts())
        : std::nullopt;

    auto out_degrees = cur_graph_view.compute_out_degrees(handle);

    edge_src_property_t<decltype(cur_graph_view), edge_t> edge_src_out_degrees(handle,
                                                                               cur_graph_view);
    edge_dst_property_t<decltype(cur_graph_view), edge_t> edge_dst_out_degrees(handle,
                                                                               cur_graph_view);
    update_edge_src_property(handle, cur_graph_view, out_degrees.begin(), edge_src_out_degrees);
    update_edge_dst_property(handle, cur_graph_view, out_degrees.begin(), edge_dst_out_degrees);
    auto [srcs, dsts] = extract_transform_e(handle,
                                            cur_graph_view,
                                            edge_src_out_degrees.view(),
                                            edge_dst_out_degrees.view(),
                                            edge_dummy_property_t{}.view(),
                                            extract_low_to_high_degree_edges_t<vertex_t, edge_t>{});
    
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("src - low - to - high", srcs.data(), srcs.size(), std::cout);
    raft::print_device_vector("dst - low - to - high", dsts.data(), dsts.size(), std::cout);

    if constexpr (multi_gpu) {
      std::tie(srcs, dsts, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t>(
          handle, std::move(srcs), std::move(dsts), std::nullopt, std::nullopt, std::nullopt);
    }

    // This should be renamed as it is an edge property
    // FIXME: Do this only if there is at least one pair
    wgts = std::make_optional<rmm::device_uvector<edge_t>>(srcs.size(), handle.get_stream());
    wgts_ = std::make_optional<rmm::device_uvector<edge_t>>(srcs.size(), handle.get_stream());


    //rmm::device_uvector<edge_t> wgts(srcs.size(), handle.get_stream());

    thrust::uninitialized_fill(handle.get_thrust_policy(),
                               (*wgts).begin(),
                               (*wgts).end(),
                               edge_t{0});
    
    //auto wgts_ = 

    // std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
    


    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("edge property", (*wgts_).data(), (*wgts_).size(), std::cout);   

    //std::optional<edge_property_t<edge_t, edge_t const*>> edge_ids{std::nullopt};
    
    thrust::copy(handle.get_thrust_policy(),
                 (*wgts).begin(),
                 (*wgts).end(),
                 (*wgts_).begin());
    
    std::cout<<"done copying"<<std::endl;

    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
    std::tie(*modified_graph, std::ignore, edge_ids, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_t, int32_t, false, multi_gpu>(
        handle,
        std::nullopt,
        std::move(srcs),
        std::move(dsts),
        std::nullopt,
        std::move(wgts),
        //std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{false /* now asymmetric */, cur_graph_view.is_multigraph()},
        true);
    
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("edge property before statement", (*wgts).data(), (*wgts).size(), std::cout); 

    //rmm::device_uvector<vertex_t>&& edgelist_srcs,
    //rmm::device_uvector<int32_t>,
    

    //std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids
    //std::optional<rmm::device_uvector<int64_t>>



    modified_graph_view = (*modified_graph).view();
    
    if (renumber_map) {  // collapse renumber_map
      unrenumber_int_vertices<vertex_t, multi_gpu>(handle,
                                                   (*tmp_renumber_map).data(),
                                                   (*tmp_renumber_map).size(),
                                                   (*renumber_map).data(),
                                                   *vertex_partition_range_lasts);
    }
    renumber_map = std::move(tmp_renumber_map);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("renumber map - low - to - high", (*renumber_map).data(), (*renumber_map).size(), std::cout);
  }
  
  // 5. Decompress the resulting graph to an edge list

  rmm::device_uvector<edge_t> cur_graph_counts(size_t{0}, handle.get_stream());
  {
    //RAFT_CUDA_TRY(cudaDeviceSynchronize());
    //raft::print_device_vector("edge property after statement", (*wgts).data(), (*wgts).size(), std::cout); 
    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
    cur_graph_counts.resize(cur_graph_view.local_vertex_partition_range_size(),
                            handle.get_stream());
    std::cout<<"local_vertex_partition_range_size = " << cur_graph_view.local_vertex_partition_range_size() << std::endl;

    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_t>> edgelist_prop {std::nullopt};

    /*
    std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore) = decompress_to_edgelist(
      handle,
      cur_graph_view,
      std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
      std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
      std::make_optional<raft::device_span<vertex_t const>>((*renumber_map).data(),
                                                            (*renumber_map).size()));
    */
   
   
   //auto edge_id = std::optional<edge_property_view_t<edge_t, edge_t const*>>{(*wgts)};

   std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view = (*edge_ids).view();

   std::tie(edgelist_srcs, edgelist_dsts, std::ignore, wgts) = decompress_to_edgelist(
      handle,
      cur_graph_view,
      std::optional<edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},
     //std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
      edge_id_view,
      //wgts,
      std::optional<raft::device_span<vertex_t const>>(std::nullopt));

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("edge property after decompressing", (*wgts).data(), (*wgts).size(), std::cout); 
    if (edgelist_prop){
      std::cout<<"there are edge properties"<<std::endl;
    }
    else{
      std::cout<<"there are no edge properties"<<std::endl;
    }

    auto vertex_pairs_begin = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

    //vertex_t* v_begin = thrust::make_zip_iterator(edgelist_srcs.begin(), edgelist_dsts.begin());

    size_t num_vertex_pairs = edgelist_srcs.size();
    auto out_degrees = cur_graph_view.compute_out_degrees(handle);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout<< "number of vertex pairs = " << num_vertex_pairs  << std::endl;
    raft::print_device_vector("src ", edgelist_srcs.data(), edgelist_srcs.size(), std::cout);
    raft::print_device_vector("dst ", edgelist_dsts.data(), edgelist_dsts.size(), std::cout);
    raft::print_device_vector("out_degrees ", out_degrees.data(), out_degrees.size(), std::cout);
    //std::cout<< "outdegree = " << out_degrees << std::endl;


    // sort and reduce

    //auto vertex_pairs_end = thrust::make_zip_iterator(edgelist_srcs.end(), edgelist_dsts.end());
    //auto vertex_pairs_begin = thrust::make_zip_iterator(std::get<0>(vertex_pairs).data(), std::get<1>(vertex_pairs).data());


    // 6. Find intersection of edge endppints for each partition using detail::nbr_intersection
    // FIXME: Move this in its own {}
    rmm::device_uvector<size_t> intersection_offsets(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> intersection_indices(size_t{0}, handle.get_stream());
    rmm::device_uvector<edge_t> r_nbr_intersection_property_values0(size_t{0}, handle.get_stream());
    rmm::device_uvector<edge_t> r_nbr_intersection_property_values1(size_t{0}, handle.get_stream());

    /*
    [[maybe_unused]] rmm::device_uvector<edge_property_value_t>
        r_nbr_intersection_property_values0(size_t{0}, handle.get_stream());
      [[maybe_unused]] rmm::device_uvector<edge_property_value_t>
        r_nbr_intersection_property_values1(size_t{0}, handle.get_stream());
    */

    // do_expensive_check = true;
    // std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt}
    // auto view_ = (*edge_ids).view();
    //std::tie(intersection_offsets, intersection_indices) =

    //create src_dst_intersection_size
    //call adjacency diff on offsets. 
    /*
    (p, q, 'intersection_size')
    (p, r, 1)
    (q, r, 1)
    */
    // tabulate with the size of intersection_indices, and call binary search on intersection_offsets
    // sort and reduce 
    /*
    std::tie(intersection_offsets, intersection_indices, r_nbr_intersection_property_values0, r_nbr_intersection_property_values1) = 
      detail::nbr_intersection(handle,
                               cur_graph_view,
                               //cugraph::edge_dummy_property_t{}.view(),
                               (*edge_ids).view(),
                               vertex_pairs_begin,
                               vertex_pairs_begin + num_vertex_pairs,
                               std::array<bool, 2>{true, true},
                               do_expensive_check);
    */

   // FIXME: Initially each eadge should have an edge property of 0


    std::tie(intersection_offsets, intersection_indices) = 
      detail::nbr_intersection(handle,
                               cur_graph_view,
                               cugraph::edge_dummy_property_t{}.view(),
                               //(*edge_ids).view(),
                               vertex_pairs_begin,
                               vertex_pairs_begin + num_vertex_pairs,
                               std::array<bool, 2>{true, true},
                               do_expensive_check);


    
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("intersection_offsets ", intersection_offsets.data(), intersection_offsets.size(), std::cout);
    raft::print_device_vector("intersection_indices ", intersection_indices.data(), intersection_indices.size(), std::cout);
    //raft::print_device_vector("r_nbr_intersection_property_values0 ", r_nbr_intersection_property_values0.data(), r_nbr_intersection_property_values0.size(), std::cout);
    //raft::print_device_vector("r_nbr_intersection_property_values1 ", r_nbr_intersection_property_values1.data(), r_nbr_intersection_property_values1.size(), std::cout);



    //auto x = intersection_offsets;
    /*
     thrust::tabulate(handle.get_thrust_policy(),
                         intersection_offsets.begin(),
                         intersection_offsets.end(),
                         update_edge_id_{
                          raft::device_span<size_t const>(
                            intersection_offsets.data(), intersection_offsets.size()),
                          raft::device_span<vertex_t const>(
                            intersection_indices.data(), intersection_indices.size())
                          });
    */

    std::cout<<"the weights size = " << (*wgts).size() << std::endl;
    /*
    rmm::device_uvector<edge_t> wgts_(6, handle.get_stream());
    thrust::uninitialized_fill(handle.get_thrust_policy(),
                               (wgts_).begin(),
                               (wgts_).end(),
                               edge_t{0});

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    raft::print_device_vector("new edge property --- ", (wgts_).data(), (wgts_).size(), std::cout);
    */

    /*
    thrust::tabulate(handle.get_thrust_policy(),
                         (*wgts).begin(),
                         (*wgts).end(),
                         update_edge_id<vertex_t, edge_t>{
                          raft::device_span<size_t const>(
                            intersection_offsets.data(), intersection_offsets.size()),
                          raft::device_span<vertex_t const>(
                            intersection_indices.data(), intersection_indices.size()),
                          raft::device_span<edge_t const>(
                            r_nbr_intersection_property_values0.data(), r_nbr_intersection_property_values0.size())
                          raft::device_span<edge_t const>(
                            r_nbr_intersection_property_values1.data(), r_nbr_intersection_property_values1.size())
                          });
    */

    //create src_dst_intersection_size
    //call adjacency diff on offsets. 
    /*
    (p, q, 'intersection_size')
    (p, r, 1)
    (q, r, 1)
    */
    // tabulate with the size of intersection_indices, and call binary search on intersection_offsets
    // sort and reduce 

    //thrust::tabulate(vertex_pairs_begin,)
    //typedef rmm::device_uvector<vertex_t>::iterator  VtxItr
    //typedef thrust::tuple<VtxItr, VtxItr> IteratorTuple;
    //typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    //assert(std::is_same_v<typename ZipIterator::value_type, decltype(your_pair)::value_type>)

    //allocate_dataframe_buffer<decltype(vertex_pair)> (4, hand.get_stream)

    /*
    auto vertex_pair_buffer = allocate_dataframe_buffer<decltype(*vertex_pairs_begin)>(
          4, handle.get_stream());
    */

    auto vertex_pair_buffer__ = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          3, handle.get_stream());

    auto vertex_pair_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          4, handle.get_stream());

    //thust::tuple<vertex_t, vertex_t> pair = thrust::make_tuple(10, 12);
    //thrust::tuple<int, int> pair(10, 12);
    //thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");
    printf("\ntabulate\n");
    thrust::tabulate(handle.get_thrust_policy(),
                         get_dataframe_buffer_begin(vertex_pair_buffer),
                         get_dataframe_buffer_begin(vertex_pair_buffer) + 4,
                         update_edge_id_<vertex_t, edge_t, decltype(vertex_pairs_begin)>{
                          // vertex_pairs
                          raft::device_span<size_t const>(
                            intersection_offsets.data(), intersection_offsets.size()),
                          raft::device_span<vertex_t const>(
                            intersection_indices.data(), intersection_indices.size()),
                          vertex_pairs_begin
                          });
    

    thrust::sort(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(vertex_pair_buffer),
                 get_dataframe_buffer_end(vertex_pair_buffer));
    
    
    rmm::device_uvector<vertex_t> num_triangles(4, handle.get_stream());

    thrust::fill(handle.get_thrust_policy(), num_triangles.begin(), num_triangles.end(), size_t{1});

    rmm::device_uvector<vertex_t> num_triangles_(3, handle.get_stream());
    //thrust::fill(num_triangles.begin(), num_triangles.end(), size_t{1});

    //int tie_ = std::tie(vertex_pair_buffer);
    //auto v = std::get<0>(tie_);

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          get_dataframe_buffer_begin(vertex_pair_buffer),
                          get_dataframe_buffer_end(vertex_pair_buffer),
                          num_triangles.begin(),
                          get_dataframe_buffer_begin(vertex_pair_buffer__),
                          num_triangles_.begin(),
                          thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>{}
                          
    );

    printf("\nDone reducing\n");



    //cugraph::edge_bucket_t<vertex_t, void, !store_transposed /* src_major */, true, true> edge_list_(handle);
    //auto store_transposed = false;
    //auto multi_gpu = false;
    // Note: ensure 'edge_list_' and 'cur_graph_view' have the same transpose flag
    cugraph::edge_bucket_t<vertex_t, void, true, multi_gpu, true> edge_list_(handle);

    edge_list_.insert(std::get<0>(vertex_pair_buffer).begin(),
                      std::get<0>(vertex_pair_buffer).end(),
                      std::get<1>(vertex_pair_buffer).begin());

    
    cugraph::edge_property_t<decltype(cur_graph_view), vertex_t> edge_value_output(handle,
                                                                                 cur_graph_view);
    
    /*
    cugraph::transform_e(
        handle,
        cur_graph_view,
        edge_list_,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        update_num_triangles<vertex_t, edge_t>{
          // num_triangles
          raft::device_span<vertex_t const>(
                      num_triangles_.data(), num_triangles_.size()),
          
        },
        edge_value_output.mutable_view(),
        false);
    */

   auto x = num_triangles_.data();

    cugraph::transform_e(
        handle,
        cur_graph_view,
        edge_list_,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto src, auto dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) {
          return 2;
        },
        edge_value_output.mutable_view(),
        false);


  
    raft::print_device_vector("new num triangles", num_triangles_.data(), num_triangles_.size(), std::cout);


    /*
    thrust::reduce_by_key(handle_.get_thrust_policy(),
                          std::tie(std::get<0>(vertex_pair_buffer).begin(), std::get<1>(vertex_pair_buffer).begin()),
                          std::tie(std::get<0>(vertex_pair_buffer).end(), std::get<1>(vertex_pair_buffer).end()),
                          num_triangles.begin(),
                          get_dataframe_buffer_begin(vertex_pair_buffer__),
                          num_triangles_.begin()
    );
    
    */

    //auto vertex_pairs_begin_ = thrust::make_zip_iterator(get_dataframe_buffer_begin(vertex_pair_buffer), get_dataframe_buffer_begin(vertex_pair_buffer)+4);
    //auto vertex_pairs_begin_ = thrust::make_zip_iterator(vertex_pair_buffer, vertex_pair_buffer+4);
    //auto x_ =  get_dataframe_buffer_begin(vertex_pair_buffer);
    //int y = thrust::get<0>(*(x_ + 1));
    //int a = vertex_pair_buffer;
    //int z = std::get<0>(vertex_pair_buffer);
    //auto vertex_pairs_begin__ = std::tie(std::get<0>(vertex_pair_buffer), std::get<1>(vertex_pair_buffer))


    vertex_pairs_begin = thrust::make_zip_iterator(std::get<0>(vertex_pair_buffer).begin(), std::get<1>(vertex_pair_buffer).begin());
    raft::print_device_vector("edge ids", std::get<1>(vertex_pair_buffer).data(), std::get<1>(vertex_pair_buffer).size(), std::cout);



    //auto y = thrust::get<2>(*get_dataframe_buffer_begin(vertex_pair_buffer));
    //printf("x_ = %d\n", thrust::get<0>(*(x_)));
    //printf("y = %d = \n", y+1);

    //printf("\n major = %d and minor = %d\n", thrust::get<0>(*(vertex_pairs_begin_)), thrust::get<1>(*(vertex_pairs_begin_)));
  




    //RAFT_CUDA_TRY(cudaDeviceSynchronize());
    //raft::print_device_vector("new edge property", (*wgts).data(), (*wgts).size(), std::cout);






    // intersection_offsets.size()
    //for (size_t i = 0; i < 1; i++) {
      /*
      auto intersesection = raft::device_span<typename GraphViewType::vertex_type const>(
        intersection_indices + intersection_offsets[i], intersection_indices + intersection_offsets[i + 1]);
      */
     //std::cout<< "The intersection size =  " << intersection_offsets.size() << std::endl;
     //raft::print_device_vector("intersection_indices[0] ", intersection_indices.data()[0], 1, std::cout);
     //std::cout<< "intersection offsets = " << (intersection_offsets) << std::endl;

      //auto intersection = raft::device_span<vertex_t const>(
      //  intersection_indices.data() + intersection_offsets.data()[i], intersection_indices.data() + intersection_offsets.data()[i + 1]);

      //  raft::print_device_vector("intersection = ", intersection.data(), intersection.size(), std::cout);
    //}
    
    

  }

  
  
  
  /*
    cugraph::per_v_pair_transform_dst_nbr_intersection(
      *handle,
      cur_graph_view,
      cugraph::edge_dummy_property_t{}.view(),
      vertex_pairs_begin,
      vertex_pairs_begin + num_vertex_pairs,
      out_degrees.begin(),
      intersection_op_t<vertex_t, edge_t>{},
      cugraph::get_dataframe_buffer_begin(mg_result_buffer));
    */


}


} // namespace cugraph



/*
/home/nfs/jnke/debug_jaccard/cugraph/cpp/src/community/ktruss_impl.cuh(148): error: no suitable conversion function from "thrust::tuple<int, int, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>" to "int32_t" exists
*/