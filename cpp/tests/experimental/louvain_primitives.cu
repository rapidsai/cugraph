#include <experimental/graph.hpp>

#include <utilities/shuffle_comm.cuh>

#include <rmm/device_uvector.hpp>

// TODO:
//    The following 3 include files are not referenced in my file,
//    however if they are not included prior to the two pattern
//    includes below then the compilation fails.  I suspect these
//    should be included somehow in the patterns includes below
//
#include <matrix_partition_device.cuh>
#include <patterns/edge_op_utils.cuh>
#include <utilities/host_scalar_comm.cuh>
// END EXTRA INCLUDES

#include <patterns/copy_v_transform_reduce_key_aggregated_out_nbr.cuh>
#include <patterns/transform_reduce_by_adj_matrix_row_col_key_e.cuh>

template <typename graph_t>
void update_by_delta_modularity(raft::handle_t const& handle,
                                graph_t const& graph_view,
                                std::vector<typename graph_t::vertex_type> cluster_h,
                                typename graph_t::weight_type total_edge_weight,
                                typename graph_t::weight_type resolution)
{
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  vertex_t num_verts = graph_view.get_number_of_vertices();

  rmm::device_uvector<vertex_t> cluster_v(num_verts, handle.get_stream());

  raft::update_device(cluster_v.data(), cluster_h.data(), num_verts, handle.get_stream());

  auto tmp = cugraph::experimental::transform_reduce_by_adj_matrix_col_key_e(
    handle,
    graph_view,
    cluster_v.begin(),
    cluster_v.begin(),
    cluster_v.begin(),
    [] __device__(auto r, auto c, auto w, auto rv, auto cv) {
      printf("transform reduce (%d,%d,%g,%d,%d)\n", (int)r, (int)c, (float)w, (int)rv, (int)cv);
      return w;
    },
    weight_t{0});

  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(1),
                   [b_key   = std::get<0>(tmp).begin(),
                    e_key   = std::get<0>(tmp).end(),
                    b_value = std::get<1>(tmp).begin(),
                    e_value = std::get<1>(tmp).end()] __device__(auto) {
                     char comma[2] = {0, 0};
                     printf("keys = ");
                     for (auto p = b_key; p != e_key; ++p) {
                       printf("%s%d", comma, *p);
                       comma[0] = ',';
                     }
                     printf("\n");
                     printf("values = ");
                     comma[0] = (char)0;
                     for (auto p = b_value; p != e_value; ++p) {
                       printf("%s%g", comma, *p);
                       comma[0] = ',';
                     }
                     printf("\n");
                   });

  if (graph_t::is_multi_gpu) {
    cugraph::experimental::sort_and_shuffle_kv_pairs(
      handle.get_comms(),
      std::get<0>(tmp).begin(),
      std::get<0>(tmp).end(),
      std::get<1>(tmp).begin(),
      [] __device__(auto) { return 0; },
      handle.get_stream());
  }

  rmm::device_vector<thrust::tuple<vertex_t, weight_t>> output_v(num_verts);

  // NOTE: these are populated and shuffled before call to update_by_delta_modularity
  rmm::device_uvector<weight_t> old_cluster_sum_v(num_verts, handle.get_stream());
  rmm::device_uvector<weight_t> src_vertex_weight_v(num_verts, handle.get_stream());
  rmm::device_uvector<weight_t> src_cluster_weight_v(num_verts, handle.get_stream());

  copy_v_transform_reduce_key_aggregated_out_nbr(
    handle,
    graph_view,
    thrust::make_zip_iterator(thrust::make_tuple(
      old_cluster_sum_v.begin(), src_vertex_weight_v.begin(), src_cluster_weight_v.begin())),
    cluster_v.begin(),
    std::get<0>(tmp).begin(),
    std::get<0>(tmp).end(),
    std::get<1>(tmp).begin(),
    [total_edge_weight, resolution] __device__(
      auto src, auto neighbor_cluster, auto new_cluster_sum, auto src_info, auto a_new) {
      auto old_cluster_sum = thrust::get<0>(src_info);
      auto k_k             = thrust::get<1>(src_info);
      auto a_old           = thrust::get<2>(src_info);

      weight_t delta_modularity = 2 * (((new_cluster_sum - old_cluster_sum) / total_edge_weight) -
                                       resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                                         (total_edge_weight * total_edge_weight));

      printf("compute delta_modularity (%d, %d, %g, %g, %g, %g, %g)\n",
             src,
             neighbor_cluster,
             new_cluster_sum,
             old_cluster_sum,
             k_k,
             a_old,
             a_new);

      return thrust::make_tuple(neighbor_cluster, delta_modularity);
    },
    [] __device__(auto p1, auto p2) {
      if (thrust::get<1>(p1) < thrust::get<1>(p2))
        return p1;
      else
        return p2;
    },
    thrust::make_tuple(vertex_t{-1}, weight_t{std::numeric_limits<weight_t>::max()}),
    output_v.begin());

  //
  //  output_v contains (for each vertex) the pair (neighbor_cluster, delta_modularity)
  //  for all neighboring clusters with the highest delta modularity
  //
  //  This can be used locally on each GPU to complete this portion of the algorithm
  //
}

int main(int argc, char** argv)
{
  using vertex_t = int32_t;
  using edge_t   = int32_t;
  using weight_t = float;

  weight_t resolution{1};

  raft::handle_t handle{};

  std::vector<vertex_t> src_h = {0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4};
  std::vector<vertex_t> dst_h = {1, 2, 3, 4, 0, 2, 0, 1, 0, 4, 0, 3};
  std::vector<weight_t> w_h   = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  vertex_t num_verts = 5;
  edge_t num_edges   = dst_h.size();
  weight_t total_edge_weight{static_cast<weight_t>(num_edges)};

  rmm::device_uvector<vertex_t> src_v(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(num_edges, handle.get_stream());
  rmm::device_uvector<weight_t> weight_v(num_edges, handle.get_stream());

  raft::update_device(src_v.data(), src_h.data(), num_edges, handle.get_stream());
  raft::update_device(dst_v.data(), dst_h.data(), num_edges, handle.get_stream());
  raft::update_device(weight_v.data(), w_h.data(), num_edges, handle.get_stream());

  cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    src_v.data(), dst_v.data(), weight_v.data(), num_edges};

  auto graph = cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false>(
    handle, edgelist, num_verts, cugraph::experimental::graph_properties_t{}, false, true);

  auto graph_view = graph.view();

  std::vector<vertex_t> cluster_h = {0, 1, 1, 3, 3};
  rmm::device_uvector<vertex_t> cluster_v(num_verts, handle.get_stream());

  raft::update_device(cluster_v.data(), cluster_h.data(), num_verts, handle.get_stream());

  CUDA_TRY(cudaDeviceSynchronize());

  update_by_delta_modularity(handle, graph_view, cluster_h, total_edge_weight, resolution);
};
