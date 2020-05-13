/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <graph.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <nvgraph/include/high_res_clock.h>
#include <nvgraph/include/util.cuh>
#include <utilities/cuda_utils.cuh>
#include <utilities/graph_utils.cuh>

#include <converters/COOtoCSR.cuh>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
__global__  // __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
  void
  compute_vertex_sums(vertex_t n_vertex,
                      edge_t const *offsets,
                      weight_t const *weights,
                      weight_t *output)
{
  // FIXME:  Do this at a WARP level and do an inline reduce
  //         to better handle high degree vertices?
  //
  int src = blockDim.x * blockIdx.x + threadIdx.x;

  if ((src < n_vertex)) {
    weight_t sum{0.0};

    for (int i = offsets[src]; i < offsets[src + 1]; ++i) { sum += weights[i]; }

    output[src] = sum;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void kernel_modularity_no_matrix(vertex_t n_vertex,
                                            vertex_t n_clusters,
                                            weight_t m2,
                                            edge_t const *offsets,
                                            vertex_t const *indices,
                                            weight_t const *weights,
                                            vertex_t const *cluster,
                                            weight_t const *vertex_sums,
                                            weight_t const *cluster_sums,
                                            weight_t *Q_arr)
{
  int src = blockIdx.x * blockDim.x + threadIdx.x;

  if (src < n_vertex) {
    weight_t Ai{0.0};

    vertex_t c_i = cluster[src];
    weight_t ki  = vertex_sums[src];

    for (int j = offsets[src]; j < offsets[src + 1]; ++j) {
      if (c_i != cluster[indices[j]]) Ai += weights[j];
    }

    weight_t sum_k = m2 - cluster_sums[c_i];
    Q_arr[src]     = (Ai - ((ki * sum_k) / m2)) / m2;
  }
}

//
//  NEW APPROACH!!!
//
//  Parallelizing Louvain is hard.  There are a bunch of attempts
//  at identifying which work can be done in parallel before doing
//  the parallel work.
//
//  For now, start with the simplest parallel model which will
//  be serial over the vertices in the worst case.  This should get
//  the correct answer while we work an a more optimal implementation.
//
//  TODO:  This might need to be just 1 warp for now...
//
template <typename vertex_t, typename edge_t, typename weight_t>
__global__  // __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
  void
  update_each_assignment_by_delta_modularity(weight_t m2,
                                             vertex_t n_vertex,
                                             edge_t n_edges,
                                             edge_t const *offsets,
                                             vertex_t const *indices,
                                             weight_t const *weights,
                                             weight_t const *vertex_weights,
                                             weight_t volatile *cluster_weights,
                                             vertex_t volatile *cluster,
                                             vertex_t volatile *vertex_ready)
{
  unsigned int tid = threadIdx.x;  // 0 ~ 31

  if (blockIdx.x > 0) return;

  __shared__ edge_t start_idx;
  __shared__ edge_t end_idx;
  __shared__ vertex_t local_max_loc[WARP_SIZE];
  __shared__ weight_t local_max[WARP_SIZE];

  for (vertex_t src = 0; src < n_vertex; ++src) {
    if (tid == 0) {
      start_idx = offsets[src];
      end_idx   = offsets[src + 1];
    }

    __syncwarp();

    local_max[tid]     = -1.0;
    local_max_loc[tid] = -1;

    //
    //  So, first we're going to compute the delta modularity
    //  for each edge associated with this vertex and store
    //  the maximum
    //
    vertex_t old_cluster = cluster[src];
    weight_t degc_totw   = vertex_weights[src] / m2;

    for (int loc = start_idx + tid; loc < end_idx; loc += WARP_SIZE) {
      vertex_t dst         = indices[loc];
      vertex_t new_cluster = cluster[dst];

      if (old_cluster != new_cluster) {
        weight_t delta_mod{-1.0};
        weight_t old_cluster_sum{0.0};
        weight_t new_cluster_sum{0.0};

        //
        // TODO:  This could be computed by the warp once into
        //        a temp array.  TOO MUCH MEMORY.  If we stay
        //        with 1 warp doing all the work could use
        //        rmm::device_vector<weight_t>(num_verts)...
        //        but with multiple warps, each warp would need its
        //        own array of that size.
        //
        for (edge_t i = offsets[src]; i < offsets[src + 1]; ++i) {
          vertex_t j = indices[i];

          if (j != src) {
            vertex_t cluster_j = cluster[j];

#ifdef DEBUG
            if (src == 3)
              printf(
                "---- src = %d, j = %d, old_cluster = %d, new_cluster = %d, cluster_j = %d, "
                "weights = %g\n",
                src,
                j,
                old_cluster,
                new_cluster,
                cluster_j,
                weights[i]);
#endif

            if (cluster_j == new_cluster) {
              new_cluster_sum += weights[i];
            } else if (cluster_j == old_cluster) {
              old_cluster_sum += weights[i];
            }
          }
        }

        delta_mod =
          new_cluster_sum - degc_totw * cluster_weights[new_cluster] -
          (old_cluster_sum - (degc_totw * (cluster_weights[old_cluster] - vertex_weights[src])));

#ifdef DEBUG
        if (src == 3)
          printf(
            "*** src = %d, dst = %d, loc = %d, (%g - (%g * %g)) - (%g - (%g * (%g - %g))) = %g\n",
            src,
            dst,
            loc,
            new_cluster_sum,
            degc_totw,
            cluster_weights[new_cluster],
            old_cluster_sum,
            degc_totw,
            cluster_weights[old_cluster],
            vertex_weights[src],
            delta_mod);
#endif

        if (delta_mod > local_max[tid]) {
          local_max[tid]     = delta_mod;
          local_max_loc[tid] = loc;
        }
      } else {
        if (local_max[tid] < 0.0) {
          local_max[tid]     = 0.0;
          local_max_loc[tid] = loc;
        }
      }
    }

    __syncwarp();

    // Now we'll do a reduction
    unsigned stride = WARP_SIZE / 2;

    while ((tid < stride) && (stride > 0)) {
      if (((tid + stride) < WARP_SIZE) && ((local_max[tid + stride] > local_max[tid]))) {
        local_max[tid]     = local_max[tid + stride];
        local_max_loc[tid] = local_max_loc[tid + stride];
      }

      stride /= 2;
    }

    __syncwarp();

    //
    //  Now we've identified the best new cluster for this
    //  vertex, update it.
    //
    if (tid == 0) {
      if (local_max[0] > weight_t{0.0}) {
        cluster_weights[cluster[src]] -= vertex_weights[src];
        cluster[src] = cluster[indices[local_max_loc[0]]];
        cluster_weights[cluster[src]] += vertex_weights[src];

#ifdef DEBUG
        printf("assigning vertex %d to cluster %d, score = %g, cluster_weights[%d] = %g\n",
               src,
               cluster[src],
               local_max[0],
               cluster[src],
               cluster_weights[cluster[src]]);
#endif
      }

      vertex_ready[src] = 1;
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void generate_superverticies_graph(
  cugraph::experimental::GraphCSRView<vertex_t, edge_t, weight_t> &current_graph,
  vertex_t new_number_of_vertices,
  rmm::device_vector<vertex_t> &cluster_v)
{
  cudaStream_t stream{0};

  rmm::device_vector<vertex_t> tmp_src_v(current_graph.number_of_edges);
  rmm::device_vector<vertex_t> new_src_v(current_graph.number_of_edges);
  rmm::device_vector<vertex_t> new_dst_v(current_graph.number_of_edges);
  rmm::device_vector<weight_t> new_weight_v(current_graph.number_of_edges);

  vertex_t *d_old_src    = tmp_src_v.data().get();
  vertex_t *d_old_dst    = current_graph.indices;
  weight_t *d_old_weight = current_graph.edge_data;
  vertex_t *d_new_src    = new_src_v.data().get();
  vertex_t *d_new_dst    = new_dst_v.data().get();
  vertex_t *d_clusters   = cluster_v.data().get();
  weight_t *d_new_weight = new_weight_v.data().get();

  //
  //  First, let's expand the CSR sources into a COO
  //
  current_graph.get_source_indices(d_old_src);

  //
  //  Now we'll renumber the COO
  //
  thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<edge_t>(0),
    thrust::make_counting_iterator<edge_t>(current_graph.number_of_edges),
    [d_old_src, d_old_dst, d_new_src, d_new_dst, d_clusters, d_new_weight, d_old_weight] __device__(
      edge_t e) {
      d_new_src[e]    = d_clusters[d_old_src[e]];
      d_new_dst[e]    = d_clusters[d_old_dst[e]];
      d_new_weight[e] = d_old_weight[e];
    });

  thrust::stable_sort_by_key(
    rmm::exec_policy(stream)->on(stream),
    d_new_dst,
    d_new_dst + current_graph.number_of_edges,
    thrust::make_zip_iterator(thrust::make_tuple(d_new_src, d_new_weight)));
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream)->on(stream),
    d_new_src,
    d_new_src + current_graph.number_of_edges,
    thrust::make_zip_iterator(thrust::make_tuple(d_new_dst, d_new_weight)));

  //
  //  Now we reduce by key to combine the weights of duplicate
  //  edges.
  //
  auto start     = thrust::make_zip_iterator(thrust::make_tuple(d_new_src, d_new_dst));
  auto new_start = thrust::make_zip_iterator(thrust::make_tuple(d_old_src, d_old_dst));
  auto new_end   = thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                                       start,
                                       start + current_graph.number_of_edges,
                                       d_new_weight,
                                       new_start,
                                       d_old_weight,
                                       thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>(),
                                       thrust::plus<weight_t>());

  current_graph.number_of_edges    = thrust::distance(new_start, new_end.first);
  current_graph.number_of_vertices = new_number_of_vertices;

  detail::fill_offset(d_old_src,
                      current_graph.offsets,
                      new_number_of_vertices,
                      current_graph.number_of_edges,
                      stream);
  CUDA_CHECK_LAST();
}

template <typename vertex_t, typename edge_t, typename weight_t>
void compute_vertex_sums(experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                         rmm::device_vector<weight_t> &sums)
{
  dim3 block_size_1d =
    dim3((graph.number_of_vertices + BLOCK_SIZE_1D * 4 - 1) / BLOCK_SIZE_1D * 4, 1, 1);
  dim3 grid_size_1d = dim3(BLOCK_SIZE_1D * 4, 1, 1);

  compute_vertex_sums<vertex_t, edge_t, weight_t><<<block_size_1d, grid_size_1d>>>(
    graph.number_of_vertices, graph.offsets, graph.edge_data, sums.data().get());
}

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t modularity(weight_t m2,
                    experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                    rmm::device_vector<vertex_t> const &cluster,
                    rmm::device_vector<weight_t> const &vertex_sums,
                    rmm::device_vector<weight_t> const &cluster_sums,
                    rmm::device_vector<weight_t> &Q_arr)
{
  int nthreads = min(graph.number_of_vertices, CUDA_MAX_KERNEL_THREADS);
  int nblocks  = min((graph.number_of_vertices + nthreads - 1) / nthreads, CUDA_MAX_BLOCKS);
  kernel_modularity_no_matrix<vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(graph.number_of_vertices,
                            cluster.size(),
                            m2,
                            graph.offsets,
                            graph.indices,
                            graph.edge_data,
                            cluster.data().get(),
                            vertex_sums.data().get(),
                            cluster_sums.data().get(),
                            Q_arr.data().get());

  CUDA_CALL(cudaDeviceSynchronize());

  weight_t Q = thrust::reduce(
    thrust::cuda::par, Q_arr.begin(), Q_arr.begin() + graph.number_of_vertices, weight_t{0.0});

  return -Q;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void update_each_assignment_by_delta_modularity(
  weight_t m2,
  experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  rmm::device_vector<weight_t> const &vertex_weights,
  rmm::device_vector<weight_t> &cluster_weights,
  rmm::device_vector<vertex_t> &cluster,
  rmm::device_vector<vertex_t> &vertex_ready)
{
  thrust::fill(vertex_ready.begin(), vertex_ready.end(), vertex_t{0});

  // dim3 block_size_1d = dim3((graph.number_of_vertices + WARP_SIZE - 1) / WARP_SIZE, 1, 1);
  dim3 block_size_1d = dim3(graph.number_of_vertices, 1, 1);
  dim3 grid_size_1d  = dim3(WARP_SIZE, 1, 1);

  update_each_assignment_by_delta_modularity<vertex_t, edge_t, weight_t>
    <<<block_size_1d, grid_size_1d>>>(m2,
                                      graph.number_of_vertices,
                                      graph.number_of_edges,
                                      graph.offsets,
                                      graph.indices,
                                      graph.edge_data,
                                      vertex_weights.data().get(),
                                      cluster_weights.data().get(),
                                      cluster.data().get(),
                                      vertex_ready.data().get());

  CUDA_CALL(cudaDeviceSynchronize());
}

template <typename vertex_t, typename edge_t, typename weight_t>
void louvain(experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
             weight_t *final_modularity,
             int *num_level,
             vertex_t *cluster_vec,
             int max_iter)
{
  *num_level = 0;

  vertex_t n_vertex = graph.number_of_vertices;
  vertex_t n_edges  = graph.number_of_edges;

  //
  //  We create a copy of the graph.  We're going to reuse
  //  this memory to edit the graph during the algorithm.
  //
  rmm::device_vector<edge_t> offsets_v(graph.offsets, graph.offsets + n_vertex + 1);
  rmm::device_vector<vertex_t> indices_v(graph.indices, graph.indices + n_edges);
  rmm::device_vector<weight_t> weights_v(graph.edge_data, graph.edge_data + n_edges);

  //
  //  The cluster graph will have n_vertex edges and at most n_vertex
  //  vertices
  //
  rmm::device_vector<vertex_t> cluster_offsets_v(n_vertex + 1);
  rmm::device_vector<vertex_t> cluster_indices_v(n_vertex);

  //
  //  We need to store to sum the weights for each vertex and cluster
  //
  rmm::device_vector<weight_t> vertex_weights_v(n_vertex);
  rmm::device_vector<weight_t> cluster_weights_v(n_vertex);

  //
  //  TODO:  Can I use cluster_vec directly
  //
  rmm::device_vector<vertex_t> cluster_v(n_vertex);
  rmm::device_vector<weight_t> Q_arr_v(n_vertex);  // really a temp space
  rmm::device_vector<vertex_t> tmp_arr_v(n_vertex);
  rmm::device_vector<vertex_t> cluster_inverse_v(n_vertex);  // really a temp space

  //  TODO:  not needed in current version... for one of the parallel variations
  rmm::device_vector<vertex_t> vertex_ready_v(n_vertex);

  HighResClock hr_clock;
  double timed, diff_time;

  weight_t new_Q;
  weight_t cur_Q;

  // TODO:  Would like this to be a cpp file... need to move some things like this into the kernels
  // file...
  weight_t m2              = thrust::reduce(thrust::cuda::par, weights_v.begin(), weights_v.end());
  weight_t best_modularity = -1;

  //
  //  Initialize every cluster to reference each vertex to itself
  //
  thrust::sequence(thrust::cuda::par, cluster_v.begin(), cluster_v.end());
  thrust::sequence(thrust::cuda::par, cluster_offsets_v.begin(), cluster_offsets_v.end());
  thrust::sequence(thrust::cuda::par, cluster_indices_v.begin(), cluster_indices_v.end());

  //
  //  Start with the base clustering (every vertex in its own cluster)
  //
  thrust::copy(cluster_v.begin(), cluster_v.end(), cluster_vec);

  cugraph::experimental::GraphCSRView<vertex_t, edge_t, weight_t> current_graph(
    offsets_v.data().get(), indices_v.data().get(), weights_v.data().get(), n_vertex, n_edges);

  cugraph::experimental::GraphCSRView<vertex_t, edge_t, weight_t> cluster_graph(
    cluster_offsets_v.data().get(), cluster_indices_v.data().get(), nullptr, n_vertex, n_vertex);

  //
  //  Sum the weights of all edges departing a vertex.  This is
  //  loop invariant, so we'll compute it here.
  //
  cugraph::detail::compute_vertex_sums(graph, vertex_weights_v);

  //
  //  Cluster weights are equivalent to vertex weights with this initial
  //  graph
  //
  thrust::copy(vertex_weights_v.begin(), vertex_weights_v.end(), cluster_weights_v.begin());

  hr_clock.start();

  //
  //  Compute modularity of initial clustering
  //
  new_Q = modularity<vertex_t, edge_t, weight_t>(
    m2, current_graph, cluster_v, vertex_weights_v, cluster_weights_v, Q_arr_v);

  hr_clock.stop(&timed);
  diff_time = timed;

  std::cout << "Initial modularity value: " << new_Q << " runtime: " << (diff_time / 1000)
            << std::endl;

  weight_t delta_Q = 1;

  do {
    cur_Q = new_Q;
    weight_t inner_cur_Q;

    while (delta_Q > 0.0001) {
      inner_cur_Q = new_Q;

      hr_clock.start();

      // Compute delta modularity for each edges
      update_each_assignment_by_delta_modularity(
        m2, current_graph, vertex_weights_v, cluster_weights_v, cluster_v, vertex_ready_v);

      hr_clock.stop(&timed);
      diff_time = timed;
      std::cout << "Complete update_cluster_by_delta_modularity  runtime: " << (diff_time / 1000)
                << std::endl;

#ifdef DEBUG
      std::cout << "clustering = (";
      for (int i = 0; i < current_graph.number_of_vertices; ++i) {
        std::cout << (i == 0 ? "" : ", ") << " " << i << ": " << cluster_v[i];
      }
      std::cout << ")" << std::endl;
#endif

      // FIXME:  make a cluster graph object and pass it around
      new_Q = modularity<vertex_t, edge_t, weight_t>(
        m2, current_graph, cluster_v, vertex_weights_v, cluster_weights_v, Q_arr_v);

      delta_Q = new_Q - inner_cur_Q;

      std::cout << "new_Q = " << new_Q << ", delta_Q = " << delta_Q << std::endl;
    }

    if (new_Q > cur_Q) {
      //
      //  Now we're going to renumber the clusters from 0 to (k-1), where k is the number of
      //  clusters in this level of the dendogram.
      //

      // TODO:  move this into a function for better scoping
      //
      thrust::copy(cluster_v.begin(), cluster_v.end(), tmp_arr_v.begin());
      thrust::sort(tmp_arr_v.begin(), tmp_arr_v.end());
      auto tmp_end = thrust::unique(tmp_arr_v.begin(), tmp_arr_v.end());

      vertex_t num_clusters = thrust::distance(tmp_arr_v.begin(), tmp_end);

      cluster_v.resize(num_clusters);
      tmp_arr_v.resize(num_clusters);
      cluster_weights_v.resize(num_clusters);

      vertex_t *d_tmp_arr         = tmp_arr_v.data().get();
      vertex_t *d_cluster_inverse = cluster_inverse_v.data().get();
      vertex_t *d_cluster         = cluster_v.data().get();

      cudaStream_t stream{0};

      thrust::fill(cluster_inverse_v.begin(), cluster_inverse_v.end(), vertex_t{-1});

      thrust::for_each(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<vertex_t>(0),
                       thrust::make_counting_iterator<vertex_t>(num_clusters),
                       [d_tmp_arr, d_cluster_inverse] __device__(vertex_t i) {
                         d_cluster_inverse[d_tmp_arr[i]] = i;
                       });

      thrust::for_each(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<vertex_t>(0),
                       thrust::make_counting_iterator<vertex_t>(current_graph.number_of_vertices),
                       [d_cluster, d_cluster_inverse] __device__(vertex_t i) {
                         d_cluster[i] = d_cluster_inverse[d_cluster[i]];
                       });

      thrust::for_each(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<vertex_t>(0),
                       thrust::make_counting_iterator<vertex_t>(current_graph.number_of_vertices),
                       [cluster_vec, d_cluster] __device__(vertex_t i) {
                         cluster_vec[i] = d_cluster[cluster_vec[i]];
                       });

#ifdef DEBUG
      std::cout << "clustering = (";
      for (int i = 0; i < current_graph.number_of_vertices; ++i) {
        std::cout << (i == 0 ? "" : ", ") << " " << i << ": " << cluster_v[i];
      }
      std::cout << ")" << std::endl;
#endif

      // generate super vertices graph
      generate_superverticies_graph(current_graph, num_clusters, cluster_v);

      //
      //  Sum the weights of all edges departing a vertex.  This is
      //  loop invariant, so we'll compute it here.
      //
      cugraph::detail::compute_vertex_sums(current_graph, vertex_weights_v);

      //
      //  Cluster weights are equivalent to vertex weights with this initial
      //  graph
      //
      thrust::copy(vertex_weights_v.begin(), vertex_weights_v.end(), cluster_weights_v.begin());

      thrust::sequence(
        thrust::cuda::par, cluster_v.begin(), cluster_v.begin() + current_graph.number_of_vertices);

      //
      //  Check... modularity should not have changed
      //
      new_Q = modularity<vertex_t, edge_t, weight_t>(
        m2, current_graph, cluster_v, vertex_weights_v, cluster_weights_v, Q_arr_v);

      delta_Q = new_Q - cur_Q;

#ifdef DEBUG
      std::cout << "after creating super vertices, new_Q = " << new_Q << ", delta_Q = " << delta_Q
                << std::endl;

      std::cout << "new offsets = (";
      for (int i = 0; i < current_graph.number_of_vertices + 1; ++i) {
        std::cout << (i == 0 ? "" : ", ") << " " << offsets_v[i];
      }
      std::cout << ")" << std::endl;

      std::cout << "new indices = (";
      for (int i = 0; i < current_graph.number_of_edges; ++i) {
        std::cout << (i == 0 ? "" : ", ") << " " << indices_v[i];
      }
      std::cout << ")" << std::endl;

      std::cout << "new weights = (";
      for (int i = 0; i < current_graph.number_of_edges; ++i) {
        std::cout << (i == 0 ? "" : ", ") << " " << weights_v[i];
      }
      std::cout << ")" << std::endl;

      std::cout << "vertex weights = (";
      for (int i = 0; i < current_graph.number_of_vertices; ++i) {
        std::cout << (i == 0 ? "" : ", ") << " " << vertex_weights_v[i];
      }
      std::cout << ")" << std::endl;

      std::cout << "cluster map = (";
      for (int i = 0; i < current_graph.number_of_vertices; ++i) {
        std::cout << (i == 0 ? "" : ", ") << " " << cluster_v[i];
      }
      std::cout << ")" << std::endl;
#endif

      best_modularity = new_Q;
    }
  } while (delta_Q > 0.001);

  *final_modularity = best_modularity;
}

template void louvain(
  experimental::GraphCSRView<int32_t, int32_t, float> const &, float *, int *, int32_t *, int);
template void louvain(
  experimental::GraphCSRView<int32_t, int32_t, double> const &, double *, int *, int32_t *, int);

}  // namespace detail
}  // namespace cugraph
