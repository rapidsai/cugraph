/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cugraph/graph.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/lap/lap.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/random.h>

#include <iostream>
#include <limits>

//#define TIMING

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

namespace cugraph {
namespace detail {

template <typename weight_t>
weight_t default_precision()
{
  return 0;
}

template <>
float default_precision()
{
  return float{1e-6};
}

template <>
double default_precision()
{
  return double{1e-6};
}

template <typename index_t, typename weight_t>
weight_t hungarian(raft::handle_t const &handle,
                   index_t num_rows,
                   index_t num_cols,
                   weight_t const *d_original_cost,
                   index_t *d_assignment,
                   weight_t precision,
                   rmm::cuda_stream_view stream_view)
{
  // FIXME: if num_cols != num_rows we can copy it and fill with zeros to make it square //
  //  TODO:  Can Date/Nagi implementation in raft handle rectangular matrices?
  //
  CUGRAPH_EXPECTS(num_rows == num_cols, "Current implementation only supports square matrices");

  rmm::device_uvector<index_t> col_assignments_v(num_rows, stream_view);

  // Create an instance of LinearAssignmentProblem using problem size, number of subproblems
  raft::lap::LinearAssignmentProblem<index_t, weight_t> lpx(handle, num_rows, 1, precision);

  // Solve LAP(s) for given cost matrix
  lpx.solve(d_original_cost, d_assignment, col_assignments_v.data());

  return lpx.getPrimalObjectiveValue(0);
}

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian_sparse(raft::handle_t const &handle,
                          GraphCOOView<vertex_t, edge_t, weight_t> const &graph,
                          vertex_t num_workers,
                          vertex_t const *workers,
                          vertex_t *assignment,
                          weight_t precision,
                          rmm::cuda_stream_view stream_view)
{
  CUGRAPH_EXPECTS(assignment != nullptr, "Invalid input argument: assignment pointer is NULL");
  CUGRAPH_EXPECTS(graph.edge_data != nullptr,
                  "Invalid input argument: graph must have edge data (costs)");

#ifdef TIMING
  HighResTimer hr_timer;

  hr_timer.start("prep");
#endif

  //
  //  Translate sparse matrix into dense bipartite matrix.
  //    rows are the workers, columns are the tasks
  //
  vertex_t num_rows = num_workers;
  vertex_t num_cols = graph.number_of_vertices - num_rows;

  vertex_t matrix_dimension = std::max(num_rows, num_cols);

  rmm::device_uvector<weight_t> cost_v(matrix_dimension * matrix_dimension, stream_view);
  rmm::device_uvector<vertex_t> tasks_v(num_cols, stream_view);
  rmm::device_uvector<vertex_t> temp_tasks_v(graph.number_of_vertices, stream_view);
  rmm::device_uvector<vertex_t> temp_workers_v(graph.number_of_vertices, stream_view);

  weight_t *d_cost         = cost_v.data();
  vertex_t *d_tasks        = tasks_v.data();
  vertex_t *d_temp_tasks   = temp_tasks_v.data();
  vertex_t *d_temp_workers = temp_workers_v.data();
  vertex_t *d_src_indices  = graph.src_indices;
  vertex_t *d_dst_indices  = graph.dst_indices;
  weight_t *d_edge_data    = graph.edge_data;

  //
  //  Renumber vertices internally.  Workers will become
  //  rows, tasks will become columns
  //
  thrust::sequence(rmm::exec_policy(stream_view), temp_tasks_v.begin(), temp_tasks_v.end());

  thrust::for_each(rmm::exec_policy(stream_view),
                   workers,
                   workers + num_workers,
                   [d_temp_tasks] __device__(vertex_t v) { d_temp_tasks[v] = -1; });

  auto temp_end = thrust::copy_if(rmm::exec_policy(stream_view),
                                  temp_tasks_v.begin(),
                                  temp_tasks_v.end(),
                                  d_tasks,
                                  [] __device__(vertex_t v) { return v >= 0; });

  vertex_t size = thrust::distance(d_tasks, temp_end);
  tasks_v.resize(size, stream_view);

  //
  // Now we'll assign costs into the dense array
  //
  thrust::fill(
    rmm::exec_policy(stream_view), temp_workers_v.begin(), temp_workers_v.end(), vertex_t{-1});
  thrust::fill(
    rmm::exec_policy(stream_view), temp_tasks_v.begin(), temp_tasks_v.end(), vertex_t{-1});
  thrust::fill(rmm::exec_policy(stream_view), cost_v.begin(), cost_v.end(), weight_t{0});

  thrust::for_each(
    rmm::exec_policy(stream_view),
    thrust::make_counting_iterator<vertex_t>(0),
    thrust::make_counting_iterator<vertex_t>(num_rows),
    [d_temp_workers, workers] __device__(vertex_t v) { d_temp_workers[workers[v]] = v; });

  thrust::for_each(
    rmm::exec_policy(stream_view),
    thrust::make_counting_iterator<vertex_t>(0),
    thrust::make_counting_iterator<vertex_t>(num_cols),
    [d_temp_tasks, d_tasks] __device__(vertex_t v) { d_temp_tasks[d_tasks[v]] = v; });

  thrust::for_each(rmm::exec_policy(stream_view),
                   thrust::make_counting_iterator<edge_t>(0),
                   thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
                   [d_temp_workers,
                    d_temp_tasks,
                    d_cost,
                    matrix_dimension,
                    d_src_indices,
                    d_dst_indices,
                    d_edge_data] __device__(edge_t loc) {
                     vertex_t src = d_temp_workers[d_src_indices[loc]];
                     vertex_t dst = d_temp_tasks[d_dst_indices[loc]];

                     if ((src >= 0) && (dst >= 0)) {
                       d_cost[src * matrix_dimension + dst] = d_edge_data[loc];
                     }
                   });

#ifdef TIMING
  hr_timer.stop();

  hr_timer.start("hungarian");
#endif

  //
  //  temp_assignment_v will hold the assignment in the dense
  //  bipartite matrix numbering
  //
  rmm::device_uvector<vertex_t> temp_assignment_v(matrix_dimension, stream_view);
  vertex_t *d_temp_assignment = temp_assignment_v.data();

  weight_t min_cost = detail::hungarian(
    handle, matrix_dimension, matrix_dimension, d_cost, d_temp_assignment, precision, stream_view);

#ifdef TIMING
  hr_timer.stop();

  hr_timer.start("translate");
#endif

  //
  //  Translate the assignment back to the original vertex ids
  //
  thrust::for_each(rmm::exec_policy(stream_view),
                   thrust::make_counting_iterator<vertex_t>(0),
                   thrust::make_counting_iterator<vertex_t>(num_rows),
                   [d_tasks, d_temp_assignment, assignment] __device__(vertex_t id) {
                     assignment[id] = d_tasks[d_temp_assignment[id]];
                   });

#ifdef TIMING
  hr_timer.stop();

  hr_timer.display(std::cout);
#endif

  return min_cost;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian(raft::handle_t const &handle,
                   GraphCOOView<vertex_t, edge_t, weight_t> const &graph,
                   vertex_t num_workers,
                   vertex_t const *workers,
                   vertex_t *assignment)
{
  rmm::cuda_stream_view stream_view{};

  return detail::hungarian_sparse(handle,
                                  graph,
                                  num_workers,
                                  workers,
                                  assignment,
                                  detail::default_precision<weight_t>(),
                                  stream_view);
}

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian(raft::handle_t const &handle,
                   GraphCOOView<vertex_t, edge_t, weight_t> const &graph,
                   vertex_t num_workers,
                   vertex_t const *workers,
                   vertex_t *assignment,
                   weight_t precision)
{
  rmm::cuda_stream_view stream_view{};

  return detail::hungarian_sparse(
    handle, graph, num_workers, workers, assignment, precision, stream_view);
}

template int32_t hungarian<int32_t, int32_t, int32_t>(
  raft::handle_t const &,
  GraphCOOView<int32_t, int32_t, int32_t> const &,
  int32_t,
  int32_t const *,
  int32_t *,
  int32_t);

template float hungarian<int32_t, int32_t, float>(raft::handle_t const &,
                                                  GraphCOOView<int32_t, int32_t, float> const &,
                                                  int32_t,
                                                  int32_t const *,
                                                  int32_t *,
                                                  float);
template double hungarian<int32_t, int32_t, double>(raft::handle_t const &,
                                                    GraphCOOView<int32_t, int32_t, double> const &,
                                                    int32_t,
                                                    int32_t const *,
                                                    int32_t *,
                                                    double);

template int32_t hungarian<int32_t, int32_t, int32_t>(
  raft::handle_t const &,
  GraphCOOView<int32_t, int32_t, int32_t> const &,
  int32_t,
  int32_t const *,
  int32_t *);

template float hungarian<int32_t, int32_t, float>(raft::handle_t const &,
                                                  GraphCOOView<int32_t, int32_t, float> const &,
                                                  int32_t,
                                                  int32_t const *,
                                                  int32_t *);
template double hungarian<int32_t, int32_t, double>(raft::handle_t const &,
                                                    GraphCOOView<int32_t, int32_t, double> const &,
                                                    int32_t,
                                                    int32_t const *,
                                                    int32_t *);

namespace dense {

template <typename index_t, typename weight_t>
weight_t hungarian(raft::handle_t const &handle,
                   weight_t const *costs,
                   index_t num_rows,
                   index_t num_cols,
                   index_t *assignment)
{
  rmm::cuda_stream_view stream_view{};

  return detail::hungarian(handle,
                           num_rows,
                           num_cols,
                           costs,
                           assignment,
                           detail::default_precision<weight_t>(),
                           stream_view);
}

template <typename index_t, typename weight_t>
weight_t hungarian(raft::handle_t const &handle,
                   weight_t const *costs,
                   index_t num_rows,
                   index_t num_cols,
                   index_t *assignment,
                   weight_t precision)
{
  rmm::cuda_stream_view stream_view{};

  return detail::hungarian(handle, num_rows, num_cols, costs, assignment, precision, stream_view);
}

template int32_t hungarian<int32_t, int32_t>(
  raft::handle_t const &, int32_t const *, int32_t, int32_t, int32_t *);
template float hungarian<int32_t, float>(
  raft::handle_t const &, float const *, int32_t, int32_t, int32_t *);
template double hungarian<int32_t, double>(
  raft::handle_t const &, double const *, int32_t, int32_t, int32_t *);
template int32_t hungarian<int32_t, int32_t>(
  raft::handle_t const &, int32_t const *, int32_t, int32_t, int32_t *, int32_t);
template float hungarian<int32_t, float>(
  raft::handle_t const &, float const *, int32_t, int32_t, int32_t *, float);
template double hungarian<int32_t, double>(
  raft::handle_t const &, double const *, int32_t, int32_t, int32_t *, double);

}  // namespace dense

}  // namespace cugraph
