#pragma once

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>
//#include <prims/extract_transform_e.cuh>
//#include <prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>


namespace cugraph {
template <typename vertex_t>
struct extract_one_core_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(vertex_t src,
                                                                            vertex_t dst,
                                                                            uint8_t src_one_core,
                                                                            uint8_t dst_one_core,
                                                                            thrust::nullopt_t) const
  {
    return (src_one_core == uint8_t{1}) && (dst_one_core == uint8_t{1})
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : thrust::nullopt;
  }
};

template <typename edge_t>
struct is_one_or_greater_t {
  __device__ uint8_t operator()(edge_t core_number) const
  {
    return core_number > edge_t{0} ? uint8_t{1} : uint8_t{0};
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<vertex_t> 
trim(raft::handle_t const& handle,
       graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
       std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
       bool do_expensive_check)
{
    //std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
    //std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};
    //std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
    //auto vertex_partition_range_lasts =
    //     std::make_optional<std::vector<vertex_t>>(graph_view.vertex_partition_range_lasts());

    rmm::device_uvector<edge_t> core_numbers(graph_view.number_of_vertices(), handle.get_stream());

    core_number(
      handle, graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{1}, size_t{1});

    edge_src_property_t<decltype(graph_view), uint8_t> edge_src_out_one_cores(handle,
                                                                                 graph_view);
    edge_dst_property_t<decltype(graph_view), uint8_t> edge_dst_out_one_cores(handle,
                                                                                 graph_view);
    auto out_one_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), is_one_or_greater_t<edge_t>{});

    rmm::device_uvector<uint8_t> out_one_core_flags(core_numbers.size(), handle.get_stream());

    thrust::copy(handle.get_thrust_policy(),
                 out_one_core_first,
                 out_one_core_first + core_numbers.size(),
                 out_one_core_flags.begin());

    update_edge_src_property(
      handle, graph_view, out_one_core_flags.begin(), edge_src_out_one_cores);
    update_edge_dst_property(
      handle, graph_view, out_one_core_flags.begin(), edge_dst_out_one_cores);

    rmm::device_uvector<size_t> subgraph_offsets_out(2, handle.get_stream());
    std::vector<size_t> h_subgraph_offsets_out{{0, out_one_core_flags.size()}};
    raft::update_device(subgraph_offsets_out.data(),
                      h_subgraph_offsets_out.data(),
                      h_subgraph_offsets_out.size(),
                      handle.get_stream());
    handle.sync_stream();

    auto [src, dst, wgt, offsets] = extract_induced_subgraphs(
    handle,
    graph_view,
    edge_weight_view,
    raft::device_span<size_t const>{subgraph_offsets_out.data(), subgraph_offsets_out.size()},
    raft::device_span<vertex_t const>{out_one_core_flags.data(), out_one_core_flags.size()},
    do_expensive_check);

    // in degree 1 core
    core_number(
      handle, graph_view, core_numbers.data(), k_core_degree_type_t::IN, size_t{1}, size_t{1});

    edge_src_property_t<decltype(graph_view), uint8_t> edge_src_in_one_cores(handle,
                                                                                 graph_view);
    edge_dst_property_t<decltype(graph_view), uint8_t> edge_dst_in_one_cores(handle,
                                                                                 graph_view);
    auto in_one_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), is_one_or_greater_t<edge_t>{});
    rmm::device_uvector<uint8_t> in_one_core_flags(core_numbers.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 in_one_core_first,
                 in_one_core_first + core_numbers.size(),
                 in_one_core_flags.begin());
    update_edge_src_property(
      handle, graph_view, in_one_core_flags.begin(), edge_src_in_one_cores);
    update_edge_dst_property(
      handle, graph_view, in_one_core_flags.begin(), edge_dst_in_one_cores);
    //auto [srcs, dsts] = extract_transform_e(handle,
    //                                        graph_view,
    //                                        edge_src_in_one_cores.view(),
    //                                        edge_dst_in_one_cores.view(),
    //                                        edge_dummy_property_t{}.view(),
    //                                        extract_one_core_t<vertex_t>{});
    rmm::device_uvector<size_t> subgraph_offsets_in(2, handle.get_stream());
    std::vector<size_t> h_subgraph_offsets_in{{0, in_one_core_flags.size()}};
    raft::update_device(subgraph_offsets_in.data(),
                      h_subgraph_offsets_in.data(),
                      h_subgraph_offsets_in.size(),
                      handle.get_stream());
    handle.sync_stream();

   
    auto [src_, dst_, wgt_, offsets_] = extract_induced_subgraphs(
    handle,
    graph_view,
    edge_weight_view,
    raft::device_span<size_t const>{subgraph_offsets_in.data(), subgraph_offsets_in.size()},
    raft::device_span<vertex_t const>{in_one_core_flags.data(), in_one_core_flags.size()},
    do_expensive_check);
/*
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
*/
}
}
