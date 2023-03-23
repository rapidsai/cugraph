#pragma once

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>
//#include <prims/extract_transform_e.cuh>
#include <prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh>
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
struct extract_zero_core_t {
  __device__ thrust::optional<thrust::tuple<vertex_t, vertex_t>> operator()(vertex_t src,
                                                                            vertex_t dst,
                                                                            uint8_t src_in_zero_core,
                                                                            uint8_t dst_in_zero_core,
                                                                            thrust::nullopt_t) const
  {
    return (src_in_zero_core == uint8_t{0}) && (dst_in_zero_core == uint8_t{0})
             ? thrust::optional<thrust::tuple<vertex_t, vertex_t>>{thrust::make_tuple(src, dst)}
             : thrust::nullopt;
  }
};

template <typename edge_t>
struct is_zero_or_greater_t {
  __device__ uint8_t operator()(edge_t core_number) const
  {
    return core_number >= edge_t{0} ? uint8_t{1} : uint8_t{0};
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<vertex_t> 
trim(raft::handle_t const& handle,
       graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view)
{
//    std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
//    auto cur_graph_view = modified_graph_view ? *modified_graph_view : graph_view;
//    auto vertex_partition_range_lasts =
//      renumber_map?
//         std::make_optional<std::vector<vertex_t>>(graph_view.vertex_partition_range_lasts())
//        : std::nullopt;
    std::optional<graph_t<vertex_t, edge_t, false, multi_gpu>> modified_graph{std::nullopt};
    std::optional<graph_view_t<vertex_t, edge_t, false, multi_gpu>> modified_graph_view{std::nullopt};
    std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
    auto vertex_partition_range_lasts =
         std::make_optional<std::vector<vertex_t>>(graph_view.vertex_partition_range_lasts());

    rmm::device_uvector<edge_t> core_numbers(graph_view.number_of_vertices(), handle.get_stream());

    core_number(
      handle, graph_view, core_numbers.data(), k_core_degree_type_t::OUT, size_t{0}, size_t{0});

    edge_src_property_t<decltype(graph_view), uint8_t> edge_src_in_zero_cores(handle,
                                                                                 graph_view);
    edge_dst_property_t<decltype(graph_view), uint8_t> edge_dst_in_zero_cores(handle,
                                                                                 graph_view);
    auto in_zero_core_first =
      thrust::make_transform_iterator(core_numbers.begin(), is_zero_or_greater_t<edge_t>{});
    rmm::device_uvector<uint8_t> in_zero_core_flags(core_numbers.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 in_zero_core_first,
                 in_zero_core_first + core_numbers.size(),
                 in_zero_core_flags.begin());
    update_edge_src_property(
      handle, graph_view, in_zero_core_flags.begin(), edge_src_in_zero_cores);
    update_edge_dst_property(
      handle, graph_view, in_zero_core_flags.begin(), edge_dst_in_zero_cores);
    auto [srcs, dsts] = extract_transform_e(handle,
                                            graph_view,
                                            edge_src_in_zero_cores.view(),
                                            edge_dst_in_zero_cores.view(),
                                            edge_dummy_property_t{}.view(),
                                            extract_zero_core_t<vertex_t>{});

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

}
}
