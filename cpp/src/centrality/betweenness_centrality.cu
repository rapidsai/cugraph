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

#include <vector>

#include <thrust/transform.h>

#include <raft/cudart_utils.h>

#include <algorithms.hpp>
#include <graph.hpp>
#include <utilities/error.hpp>

#include <raft/handle.hpp>
#include "betweenness_centrality.cuh"
#include "betweenness_centrality_kernels.cuh"

namespace cugraph {
namespace detail {
namespace {
template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality_impl(const raft::handle_t &handle,
                                 experimental::GraphCSRView<VT, ET, WT> const &graph,
                                 result_t *result,
                                 bool normalize,
                                 bool endpoints,
                                 WT const *weight,
                                 VT number_of_sources,
                                 VT const *sources,
                                 VT total_number_of_sources)
{
  // Current Implementation relies on BFS
  // FIXME: For SSSP version
  // Brandes Algorithm expects non negative weights for the accumulation
  bool is_edge_betweenness = false;
  verify_betweenness_centrality_input<VT, ET, WT, result_t>(
    result, is_edge_betweenness, normalize, endpoints, weight, number_of_sources, sources);
  cugraph::detail::BC<VT, ET, WT, result_t> bc(graph);
  bc.configure(
    result, is_edge_betweenness, normalize, endpoints, weight, sources, number_of_sources);
  bc.compute();
  bc.rescale_by_total_sources_used(total_number_of_sources);
}

template <typename VT, typename ET, typename WT, typename result_t>
void edge_betweenness_centrality_impl(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                      result_t *result,
                                      bool normalize,
                                      WT const *weight,
                                      VT const number_of_sources,
                                      VT const *sources)
{
  // Current Implementation relies on BFS
  // FIXME: For SSSP version
  // Brandes Algorithm expects non negative weights for the accumulation
  bool is_edge_betweenness = true;
  bool endpoints           = false;
  verify_betweenness_centrality_input<VT, ET, WT, result_t>(
    result, is_edge_betweenness, normalize, endpoints, weight, number_of_sources, sources);
  cugraph::detail::BC<VT, ET, WT, result_t> bc(graph);
  bc.configure(
    result, is_edge_betweenness, normalize, endpoints, weight, sources, number_of_sources);
  bc.compute();
}
}  // namespace

template <typename VT, typename ET, typename WT, typename result_t>
void verify_betweenness_centrality_input(result_t *result,
                                         bool is_edge_betweenness,
                                         bool normalize,
                                         bool endpoints,
                                         WT const *weights,
                                         VT const number_of_sources,
                                         VT const *sources)
{
  static_assert(std::is_same<VT, int>::value, "VT should be int");
  static_assert(std::is_same<ET, int>::value, "ET should be int");
  static_assert(std::is_same<WT, float>::value || std::is_same<WT, double>::value,
                "WT should be float or double");
  static_assert(std::is_same<result_t, float>::value || std::is_same<result_t, double>::value,
                "result_t should be float or double");

  CUGRAPH_EXPECTS(result != nullptr, "Invalid API parameter: betwenness pointer is NULL");
  CUGRAPH_EXPECTS(number_of_sources >= 0, "Number of sources must be positive or equal to 0.");
  if (number_of_sources != 0) {
    CUGRAPH_EXPECTS(sources != nullptr,
                    "Sources cannot be NULL if number_of_source is different from 0.");
  }
  if (is_edge_betweenness) {
    CUGRAPH_EXPECTS(!endpoints, "Endpoints is not supported for edge betweenness centrality.");
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::setup()
{
  number_of_vertices_ = graph_.number_of_vertices;
  number_of_edges_    = graph_.number_of_edges;
  offsets_ptr_        = graph_.offsets;
  indices_ptr_        = graph_.indices;
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::configure(result_t *betweenness,
                                         bool is_edge_betweenness,
                                         bool normalized,
                                         bool endpoints,
                                         WT const *weights,
                                         VT const *sources,
                                         VT number_of_sources)
{
  // --- Bind betweenness output vector to internal ---
  betweenness_         = betweenness;
  normalized_          = normalized;
  endpoints_           = endpoints;
  sources_             = sources;
  number_of_sources_   = number_of_sources;
  edge_weights_ptr_    = weights;
  is_edge_betweenness_ = is_edge_betweenness;

  // --- Working data allocation ---
  initialize_work_vectors();
  initialize_pointers_to_vectors();

  // --- Get Device Information ---
  initialize_device_information();

  // --- Confirm that configuration went through ---
  configured_ = true;
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_work_vectors()
{
  distances_vec_.resize(number_of_vertices_);
  predecessors_vec_.resize(number_of_vertices_);
  sp_counters_vec_.resize(number_of_vertices_);
  deltas_vec_.resize(number_of_vertices_);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_pointers_to_vectors()
{
  distances_    = distances_vec_.data().get();
  predecessors_ = predecessors_vec_.data().get();
  sp_counters_  = sp_counters_vec_.data().get();
  deltas_       = deltas_vec_.data().get();
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_device_information()
{
  // TODO(xcadet) Update to use raft::handle
  CUDA_TRY(cudaGetDevice(&device_id_));
  CUDA_TRY(cudaDeviceGetAttribute(&max_grid_dim_1D_, cudaDevAttrMaxGridDimX, device_id_));
  CUDA_TRY(cudaDeviceGetAttribute(&max_block_dim_1D_, cudaDevAttrMaxBlockDimX, device_id_));
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute()
{
  CUGRAPH_EXPECTS(configured_, "BC must be configured before computation");
  if (sources_) {
    for (VT source_idx = 0; source_idx < number_of_sources_; ++source_idx) {
      VT source_vertex = sources_[source_idx];
      compute_single_source(source_vertex);
    }
  } else {
    for (VT source_vertex = 0; source_vertex < number_of_vertices_; ++source_vertex) {
      compute_single_source(source_vertex);
    }
  }
  rescale();
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute_single_source(VT source_vertex)
{
  // Step 1) Singe-source shortest-path problem
  cugraph::bfs(
    graph_, distances_, predecessors_, sp_counters_, source_vertex, graph_.prop.directed);

  // FIXME: Remove that with a BC specific class to gather
  //        information during traversal

  // Numeric max value is replaced by -1 as we look for the maximal depth of
  // the traversal, this value is avalaible within the bfs implementation and
  // there could be a way to access it directly and avoid both replace and the
  // max
  thrust::replace(rmm::exec_policy(stream_)->on(stream_),
                  distances_,
                  distances_ + number_of_vertices_,
                  std::numeric_limits<VT>::max(),
                  static_cast<VT>(-1));
  auto current_max_depth = thrust::max_element(
    rmm::exec_policy(stream_)->on(stream_), distances_, distances_ + number_of_vertices_);
  VT max_depth = 0;
  CUDA_TRY(cudaMemcpy(&max_depth, current_max_depth, sizeof(VT), cudaMemcpyDeviceToHost));
  // Step 2) Dependency accumulation
  accumulate(source_vertex, max_depth);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate(VT source_vertex, VT max_depth)
{
  dim3 grid_configuration, block_configuration;
  block_configuration.x = max_block_dim_1D_;
  grid_configuration.x  = min(max_grid_dim_1D_, (number_of_edges_ / block_configuration.x + 1));

  initialize_dependencies();

  if (is_edge_betweenness_) {
    accumulate_edges(max_depth, grid_configuration, block_configuration);
  } else if (endpoints_) {
    accumulate_vertices_with_endpoints(
      source_vertex, max_depth, grid_configuration, block_configuration);
  } else {
    accumulate_vertices(max_depth, grid_configuration, block_configuration);
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_dependencies()
{
  thrust::fill(rmm::exec_policy(stream_)->on(stream_),
               deltas_,
               deltas_ + number_of_vertices_,
               static_cast<result_t>(0));
}
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate_edges(VT max_depth,
                                                dim3 grid_configuration,
                                                dim3 block_configuration)
{
  for (VT depth = max_depth; depth >= 0; --depth) {
    edges_accumulation_kernel<VT, ET, WT, result_t>
      <<<grid_configuration, block_configuration, 0, stream_>>>(betweenness_,
                                                                number_of_vertices_,
                                                                graph_.indices,
                                                                graph_.offsets,
                                                                distances_,
                                                                sp_counters_,
                                                                deltas_,
                                                                depth);
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate_vertices_with_endpoints(VT source_vertex,
                                                                  VT max_depth,
                                                                  dim3 grid_configuration,
                                                                  dim3 block_configuration)
{
  for (VT depth = max_depth; depth > 0; --depth) {
    endpoints_accumulation_kernel<VT, ET, WT, result_t>
      <<<grid_configuration, block_configuration, 0, stream_>>>(betweenness_,
                                                                number_of_vertices_,
                                                                graph_.indices,
                                                                graph_.offsets,
                                                                distances_,
                                                                sp_counters_,
                                                                deltas_,
                                                                depth);
  }
  add_reached_endpoints_to_source_betweenness(source_vertex);
  add_vertices_dependencies_to_betweenness();
}

// Distances should contain -1 for unreached nodes,

// FIXME: There might be a cleaner way to add a value to a single
//        score in the betweenness vector
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::add_reached_endpoints_to_source_betweenness(VT source_vertex)
{
  VT number_of_unvisited_vertices = thrust::count(
    rmm::exec_policy(stream_)->on(stream_), distances_, distances_ + number_of_vertices_, -1);
  VT number_of_visited_vertices_except_source =
    number_of_vertices_ - number_of_unvisited_vertices - 1;
  rmm::device_vector<VT> buffer(1);
  buffer[0] = number_of_visited_vertices_except_source;
  thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                    buffer.begin(),
                    buffer.end(),
                    betweenness_ + source_vertex,
                    betweenness_ + source_vertex,
                    thrust::plus<result_t>());
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::add_vertices_dependencies_to_betweenness()
{
  thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                    deltas_,
                    deltas_ + number_of_vertices_,
                    betweenness_,
                    betweenness_,
                    thrust::plus<result_t>());
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate_vertices(VT max_depth,
                                                   dim3 grid_configuration,
                                                   dim3 block_configuration)
{
  for (VT depth = max_depth; depth > 0; --depth) {
    accumulation_kernel<VT, ET, WT, result_t>
      <<<grid_configuration, block_configuration, 0, stream_>>>(betweenness_,
                                                                number_of_vertices_,
                                                                graph_.indices,
                                                                graph_.offsets,
                                                                distances_,
                                                                sp_counters_,
                                                                deltas_,
                                                                depth);
  }
  add_vertices_dependencies_to_betweenness();
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale()
{
  bool modified           = false;
  result_t rescale_factor = static_cast<result_t>(1);
  if (normalized_) {
    if (is_edge_betweenness_) {
      rescale_edges_betweenness_centrality(rescale_factor, modified);
    } else {
      rescale_vertices_betweenness_centrality(rescale_factor, modified);
    }
  } else {
    if (!graph_.prop.directed) {
      rescale_factor /= static_cast<result_t>(2);
      modified = true;
    }
  }
  apply_rescale_factor_to_betweenness(rescale_factor);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale_edges_betweenness_centrality(result_t &rescale_factor,
                                                                    bool &modified)
{
  result_t casted_number_of_vertices_ = static_cast<result_t>(number_of_vertices_);
  if (number_of_vertices_ > 1) {
    rescale_factor /= ((casted_number_of_vertices_) * (casted_number_of_vertices_ - 1));
    modified = true;
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale_vertices_betweenness_centrality(result_t &rescale_factor,
                                                                       bool &modified)
{
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices_);
  if (number_of_vertices_ > 2) {
    if (endpoints_) {
      rescale_factor /= (casted_number_of_vertices * (casted_number_of_vertices - 1));
    } else {
      rescale_factor /= ((casted_number_of_vertices - 1) * (casted_number_of_vertices - 2));
    }
    modified = true;
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::apply_rescale_factor_to_betweenness(result_t rescale_factor)
{
  size_t result_size = number_of_vertices_;
  if (is_edge_betweenness_) result_size = number_of_edges_;
  thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                    betweenness_,
                    betweenness_ + result_size,
                    thrust::make_constant_iterator(rescale_factor),
                    betweenness_,
                    thrust::multiplies<result_t>());
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale_by_total_sources_used(VT total_number_of_sources_used)
{
  result_t rescale_factor = static_cast<result_t>(1);
  result_t casted_total_number_of_sources_used =
    static_cast<result_t>(total_number_of_sources_used);
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices_);

  if (normalized_) {
    if (number_of_vertices_ > 2 && total_number_of_sources_used > 0) {
      rescale_factor *= (casted_number_of_vertices / casted_total_number_of_sources_used);
    }
  } else if (!graph_.prop.directed) {
    if (number_of_vertices_ > 2 && total_number_of_sources_used > 0) {
      rescale_factor *= (casted_number_of_vertices / casted_total_number_of_sources_used);
    }
  }
  apply_rescale_factor_to_betweenness(rescale_factor);
}
}  // namespace detail

namespace opg {
// TODO(xcadet) Move it to own file
template <typename VT, typename ET, typename WT>
class OPGBatchGraphCSRDistributor {
 public:
  OPGBatchGraphCSRDistributor(const raft::handle_t &handle,
                              experimental::GraphCSRView<VT, ET, WT> &graph)
    : handle_(handle), graph_(graph)
  {
    rank_ = handle.get_comms().get_rank();
  }

  ~OPGBatchGraphCSRDistributor() {}
  void distribute(rmm::device_vector<ET> *d_offsets,
                  rmm::device_vector<VT> *d_indices,
                  rmm::device_vector<WT> *d_edge_data)
  {
    distribute_graph_info();
    if (rank_ != 0) {
      CUGRAPH_EXPECTS(d_offsets != nullptr,
                      "A pointer to a resizable device vector must be provided");
      CUGRAPH_EXPECTS(graph_.offsets == nullptr, "graph's offsets is already assigned");
      CUGRAPH_EXPECTS(d_indices != nullptr,
                      "A pointer to a resizable device vector must be provided");
      CUGRAPH_EXPECTS(graph_.indices == nullptr, "graph's indices is already assigned");

      d_offsets->resize(graph_.number_of_vertices + 1);
      graph_.offsets = d_offsets->data().get();

      d_indices->resize(graph_.number_of_edges);
      graph_.indices = d_indices->data().get();

      if (d_edge_data) {
        CUGRAPH_EXPECTS(graph_.edge_data == nullptr, "graph's edge_data is already assigned");
        d_edge_data->resize(graph_.number_of_edges);
        graph_.edge_data = d_edge_data->data().get();
      }
    }
    distribute_graph_offsets();
    distribute_graph_indices();
    if (d_edge_data) { distribute_graph_edge_data(); }
    CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
  }

  void distribute_graph_info()
  {
    initalize_graph_data_storage();
    if (rank_ == 0) { fill_graph_data_storage(); }
    handle_.get_comms().bcast(
      d_graph_data_storage_.data().get(), d_graph_data_storage_.size(), 0, handle_.get_stream());
    if (rank_ != 0) { read_from_graph_data_storage(); }
  }

  void distribute_graph_indices()
  {
    size_t indices_size = graph_.number_of_edges;
    handle_.get_comms().bcast(graph_.indices, indices_size, 0, handle_.get_stream());
  }

  void distribute_graph_offsets()
  {
    size_t offsets_size = graph_.number_of_vertices + 1;
    handle_.get_comms().bcast(graph_.offsets, offsets_size, 0, handle_.get_stream());
  }

  void distribute_graph_edge_data()
  {
    size_t edge_data_size = graph_.number_of_edges;
    handle_.get_comms().bcast(graph_.edge_data, edge_data_size, 0, handle_.get_stream());
  }

  size_t get_required_size_for_graph_data()
  {
    size_t required_size_for_number_of_vertices = sizeof(int);
    size_t required_size_for_number_of_edges    = sizeof(int);
    size_t required_size_for_graph_properties   = sizeof(experimental::GraphProperties);
    size_t total_required_size                  = required_size_for_number_of_vertices +
                                 required_size_for_number_of_edges +
                                 required_size_for_graph_properties;
    return total_required_size;
  }

 private:
  void initalize_graph_data_storage()
  {
    size_t required_size_for_graph_data = get_required_size_for_graph_data();
    h_graph_data_storage_.resize(required_size_for_graph_data);
    d_graph_data_storage_.resize(required_size_for_graph_data);
  }

  void fill_graph_data_storage()
  {
    CUGRAPH_EXPECTS(rank_ == 0, "Only the Node with rank == 0 should fill the graph_data_storage");
    size_t position = 0;
    memcpy(h_graph_data_storage_.data() + position, &graph_.number_of_vertices, sizeof(VT));
    position += sizeof(VT);
    memcpy(h_graph_data_storage_.data() + position, &graph_.number_of_edges, sizeof(ET));
    position += sizeof(ET);
    memcpy(
      h_graph_data_storage_.data() + position, &graph_.prop, sizeof(experimental::GraphProperties));
    thrust::copy(
      h_graph_data_storage_.begin(), h_graph_data_storage_.end(), d_graph_data_storage_.begin());
  }

  void read_from_graph_data_storage()
  {
    CUGRAPH_EXPECTS(rank_ != 0, "Only the Node with rank != 0 should read the graph_data_storage");
    thrust::copy(
      d_graph_data_storage_.begin(), d_graph_data_storage_.end(), h_graph_data_storage_.begin());
    size_t position           = 0;
    char *storage_start       = h_graph_data_storage_.data();
    graph_.number_of_vertices = *reinterpret_cast<VT *>(storage_start + position);
    position += sizeof(VT);
    graph_.number_of_edges = *reinterpret_cast<ET *>(storage_start + position);
    position += sizeof(ET);
    graph_.prop = *reinterpret_cast<experimental::GraphProperties *>(storage_start + position);
  }

  experimental::GraphCSRView<VT, ET, WT> &graph_;
  int rank_;
  // TODO(xcadet) Look into a way to make is more flexible
  const raft::handle_t &handle_;
  thrust::host_vector<char> h_graph_data_storage_;
  rmm::device_vector<char> d_graph_data_storage_;
};

template <typename VT, typename ET, typename WT, typename result_t>
void setup(const raft::handle_t &handle,
           experimental::GraphCSRView<VT, ET, WT> const *original_graph,
           const result_t *result_ptr)
{
  // printf("[DBG][OPG] Setup\n");
  int rank      = handle.get_comms().get_rank();
  int device_id = handle.get_device();
  printf("[DBG][OPG] Rank(%d)\n", rank);
  printf("[DBG][OPG] Graph is Null(%d)\n", original_graph == nullptr);
}
void receive_output_destination() {}

void get_batch() {}  // printf("[DBG][OPG] Get Batch\n"); }

void process() {}  // printf("[DBG][OPG] Process\n"); }

template <typename VT, typename result_t>
void combine(const raft::handle_t &handle, result_t *src_result, result_t *dst_result, VT size)
{
  handle.get_comms().reduce(src_result, dst_result, size, raft::comms::op_t::SUM, 0, 0);
}
}  // namespace opg
#include <chrono>
#include <ctime>
#include <ratio>

template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(const raft::handle_t &handle,
                            experimental::GraphCSRView<VT, ET, WT> const *graph,
                            result_t *result,
                            bool normalize,
                            bool endpoints,
                            WT const *weight,
                            VT k,
                            VT const *vertices,
                            VT total_number_of_sources_used)
{
  if (handle.comms_initialized()) {
    printf("[DBG][OPG] Started BATCH-OPG-BC\n");
    int rank = handle.get_comms().get_rank();
    experimental::GraphCSRView<VT, ET, WT> local_graph;
    if (graph) { local_graph = *graph; }
    opg::OPGBatchGraphCSRDistributor<VT, ET, WT> distributor(handle, local_graph);
    rmm::device_vector<ET> d_local_offsets;
    rmm::device_vector<VT> d_local_indices;
    rmm::device_vector<WT> d_local_edge_data;
    if (graph) {
      distributor.distribute(nullptr, nullptr, nullptr);
    } else {
      // TODO(xcadet) Enable edge_data transfer
      distributor.distribute(&d_local_offsets, &d_local_indices, nullptr /*&d_local_edge_data*/);
    }
    // TODO(xcadet) Through this approach we need an extra |V| device memory,
    //              should probaly directly use the allocated data for rank 0
    // auto t1                                 = std::chrono::high_resolution_clock::now();
    // auto t2                                 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> time_span = t2 - t1;
    // printf("[PROF][DBG] time_span %lf\n", time_span.count());
    rmm::device_vector<result_t> betweenness(local_graph.number_of_vertices, 0);
    // opg::get_batch();
    detail::betweenness_centrality_impl(handle,
                                        local_graph,
                                        betweenness.data().get(),
                                        normalize,
                                        endpoints,
                                        weight,
                                        k,
                                        vertices,
                                        total_number_of_sources_used);
    opg::combine<VT, result_t>(
      handle, betweenness.data().get(), result, local_graph.number_of_vertices);
    printf("[DBG][OPG] Rank(%d)\n", rank);
    printf("[DBG][OPG] End of computation\n");
  } else {
    printf("[DBG][OPG] Started Regular-BC\n");
    detail::betweenness_centrality_impl(handle,
                                        *graph,
                                        result,
                                        normalize,
                                        endpoints,
                                        weight,
                                        k,
                                        vertices,
                                        total_number_of_sources_used);
  }
}  // namespace cugraph

template void betweenness_centrality<int, int, float, float>(
  const raft::handle_t &,
  experimental::GraphCSRView<int, int, float> const *,
  float *,
  bool,
  bool,
  float const *,
  int,
  int const *,
  int);
template void betweenness_centrality<int, int, double, double>(
  const raft::handle_t &,
  experimental::GraphCSRView<int, int, double> const *,
  double *,
  bool,
  bool,
  double const *,
  int,
  int const *,
  int);

template <typename VT, typename ET, typename WT, typename result_t>
void edge_betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                 result_t *result,
                                 bool normalize,
                                 WT const *weight,
                                 VT k,
                                 VT const *vertices)
{
  detail::edge_betweenness_centrality_impl(graph, result, normalize, weight, k, vertices);
}

template void edge_betweenness_centrality<int, int, float, float>(
  experimental::GraphCSRView<int, int, float> const &,
  float *,
  bool,
  float const *,
  int,
  int const *);
template void edge_betweenness_centrality<int, int, double, double>(
  experimental::GraphCSRView<int, int, double> const &,
  double *,
  bool,
  double const *,
  int,
  int const *);
}  // namespace cugraph
