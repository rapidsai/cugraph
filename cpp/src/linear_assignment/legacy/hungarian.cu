/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
//#define TIMING

#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>
#ifdef TIMING
#include <cugraph/utilities/high_res_timer.hpp>
#endif

#include <raft/solver/linear_assignment.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <iostream>
#include <limits>

namespace cugraph {
namespace detail {

template <typename weight_t>
weight_t default_epsilon()
{
  return 0;
}

template <>
float default_epsilon()
{
  return float{1e-6};
}

template <>
double default_epsilon()
{
  return double{1e-6};
}

template <typename index_t, typename weight_t>
weight_t hungarian(raft::handle_t const& handle,
                   index_t num_rows,
                   index_t num_cols,
                   weight_t const* d_original_cost,
                   index_t* d_assignment,
                   weight_t epsilon)
{
  if (num_rows == num_cols) {
    rmm::device_uvector<index_t> col_assignments_v(num_rows, handle.get_stream());

    // Create an instance of LinearAssignmentProblem using problem size, number of subproblems
    raft::solver::LinearAssignmentProblem<index_t, weight_t> lpx(handle, num_rows, 1, epsilon);

    // Solve LAP(s) for given cost matrix
    lpx.solve(d_original_cost, d_assignment, col_assignments_v.data());

    return lpx.getPrimalObjectiveValue(0);
  } else {
    //
    //  Create a square matrix, copy d_original_cost into it.
    //  Fill the extra rows/columns with max(d_original_cost)
    //
    index_t n         = std::max(num_rows, num_cols);
    weight_t max_cost = thrust::reduce(handle.get_thrust_policy(),
                                       d_original_cost,
                                       d_original_cost + (num_rows * num_cols),
                                       weight_t{0},
                                       thrust::maximum<weight_t>());

    rmm::device_uvector<weight_t> tmp_cost_v(n * n, handle.get_stream());
    rmm::device_uvector<index_t> tmp_row_assignment_v(n, handle.get_stream());
    rmm::device_uvector<index_t> tmp_col_assignment_v(n, handle.get_stream());

    thrust::transform(handle.get_thrust_policy(),
                      thrust::make_counting_iterator<index_t>(0),
                      thrust::make_counting_iterator<index_t>(n * n),
                      tmp_cost_v.begin(),
                      [max_cost, d_original_cost, n, num_rows, num_cols] __device__(index_t i) {
                        index_t row = i / n;
                        index_t col = i % n;

                        return ((row < num_rows) && (col < num_cols))
                                 ? d_original_cost[row * num_cols + col]
                                 : max_cost;
                      });

    raft::solver::LinearAssignmentProblem<index_t, weight_t> lpx(handle, n, 1, epsilon);

    // Solve LAP(s) for given cost matrix
    lpx.solve(tmp_cost_v.begin(), tmp_row_assignment_v.begin(), tmp_col_assignment_v.begin());

    weight_t tmp_objective_value = lpx.getPrimalObjectiveValue(0);

    raft::copy(d_assignment, tmp_row_assignment_v.begin(), num_rows, handle.get_stream());

    return tmp_objective_value - max_cost * std::abs(num_rows - num_cols);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian_sparse(raft::handle_t const& handle,
                          legacy::GraphCOOView<vertex_t, edge_t, weight_t> const& graph,
                          vertex_t num_workers,
                          vertex_t const* workers,
                          vertex_t* assignment,
                          weight_t epsilon)
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

  rmm::device_uvector<weight_t> cost_v(matrix_dimension * matrix_dimension, handle.get_stream());
  rmm::device_uvector<vertex_t> tasks_v(num_cols, handle.get_stream());
  rmm::device_uvector<vertex_t> temp_tasks_v(graph.number_of_vertices, handle.get_stream());
  rmm::device_uvector<vertex_t> temp_workers_v(graph.number_of_vertices, handle.get_stream());

  weight_t* d_cost         = cost_v.data();
  vertex_t* d_tasks        = tasks_v.data();
  vertex_t* d_temp_tasks   = temp_tasks_v.data();
  vertex_t* d_temp_workers = temp_workers_v.data();
  vertex_t* d_src_indices  = graph.src_indices;
  vertex_t* d_dst_indices  = graph.dst_indices;
  weight_t* d_edge_data    = graph.edge_data;

  //
  //  Renumber vertices internally.  Workers will become
  //  rows, tasks will become columns
  //
  thrust::sequence(handle.get_thrust_policy(), temp_tasks_v.begin(), temp_tasks_v.end());

  thrust::for_each(handle.get_thrust_policy(),
                   workers,
                   workers + num_workers,
                   [d_temp_tasks] __device__(vertex_t v) { d_temp_tasks[v] = -1; });

  auto temp_end = thrust::copy_if(handle.get_thrust_policy(),
                                  temp_tasks_v.begin(),
                                  temp_tasks_v.end(),
                                  d_tasks,
                                  [] __device__(vertex_t v) { return v >= 0; });

  vertex_t size = thrust::distance(d_tasks, temp_end);
  tasks_v.resize(size, handle.get_stream());

  //
  // Now we'll assign costs into the dense array
  //
  thrust::fill(
    handle.get_thrust_policy(), temp_workers_v.begin(), temp_workers_v.end(), vertex_t{-1});
  thrust::fill(handle.get_thrust_policy(), temp_tasks_v.begin(), temp_tasks_v.end(), vertex_t{-1});
  thrust::fill(handle.get_thrust_policy(), cost_v.begin(), cost_v.end(), weight_t{0});

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<vertex_t>(0),
    thrust::make_counting_iterator<vertex_t>(num_rows),
    [d_temp_workers, workers] __device__(vertex_t v) { d_temp_workers[workers[v]] = v; });

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<vertex_t>(0),
    thrust::make_counting_iterator<vertex_t>(num_cols),
    [d_temp_tasks, d_tasks] __device__(vertex_t v) { d_temp_tasks[d_tasks[v]] = v; });

  thrust::for_each(handle.get_thrust_policy(),
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
  rmm::device_uvector<vertex_t> temp_assignment_v(matrix_dimension, handle.get_stream());
  vertex_t* d_temp_assignment = temp_assignment_v.data();

  weight_t min_cost = detail::hungarian(
    handle, matrix_dimension, matrix_dimension, d_cost, d_temp_assignment, epsilon);

#ifdef TIMING
  hr_timer.stop();

  hr_timer.start("translate");
#endif

  //
  //  Translate the assignment back to the original vertex ids
  //
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator<vertex_t>(0),
                   thrust::make_counting_iterator<vertex_t>(num_rows),
                   [d_tasks, d_temp_assignment, assignment] __device__(vertex_t id) {
                     assignment[id] = d_tasks[d_temp_assignment[id]];
                   });

#ifdef TIMING
  hr_timer.stop();

  hr_timer.display_and_clear(std::cout);
#endif

  return min_cost;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian(raft::handle_t const& handle,
                   legacy::GraphCOOView<vertex_t, edge_t, weight_t> const& graph,
                   vertex_t num_workers,
                   vertex_t const* workers,
                   vertex_t* assignment)
{
  return detail::hungarian_sparse(
    handle, graph, num_workers, workers, assignment, detail::default_epsilon<weight_t>());
}

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian(raft::handle_t const& handle,
                   legacy::GraphCOOView<vertex_t, edge_t, weight_t> const& graph,
                   vertex_t num_workers,
                   vertex_t const* workers,
                   vertex_t* assignment,
                   weight_t epsilon)
{
  return detail::hungarian_sparse(handle, graph, num_workers, workers, assignment, epsilon);
}

template int32_t hungarian<int32_t, int32_t, int32_t>(
  raft::handle_t const&,
  legacy::GraphCOOView<int32_t, int32_t, int32_t> const&,
  int32_t,
  int32_t const*,
  int32_t*,
  int32_t);

template float hungarian<int32_t, int32_t, float>(
  raft::handle_t const&,
  legacy::GraphCOOView<int32_t, int32_t, float> const&,
  int32_t,
  int32_t const*,
  int32_t*,
  float);
template double hungarian<int32_t, int32_t, double>(
  raft::handle_t const&,
  legacy::GraphCOOView<int32_t, int32_t, double> const&,
  int32_t,
  int32_t const*,
  int32_t*,
  double);

template int32_t hungarian<int32_t, int32_t, int32_t>(
  raft::handle_t const&,
  legacy::GraphCOOView<int32_t, int32_t, int32_t> const&,
  int32_t,
  int32_t const*,
  int32_t*);

template float hungarian<int32_t, int32_t, float>(
  raft::handle_t const&,
  legacy::GraphCOOView<int32_t, int32_t, float> const&,
  int32_t,
  int32_t const*,
  int32_t*);
template double hungarian<int32_t, int32_t, double>(
  raft::handle_t const&,
  legacy::GraphCOOView<int32_t, int32_t, double> const&,
  int32_t,
  int32_t const*,
  int32_t*);

namespace dense {

template <typename index_t, typename weight_t>
weight_t hungarian(raft::handle_t const& handle,
                   weight_t const* costs,
                   index_t num_rows,
                   index_t num_cols,
                   index_t* assignment)
{
  return detail::hungarian(
    handle, num_rows, num_cols, costs, assignment, detail::default_epsilon<weight_t>());
}

template <typename index_t, typename weight_t>
weight_t hungarian(raft::handle_t const& handle,
                   weight_t const* costs,
                   index_t num_rows,
                   index_t num_cols,
                   index_t* assignment,
                   weight_t epsilon)
{
  return detail::hungarian(handle, num_rows, num_cols, costs, assignment, epsilon);
}

template int32_t hungarian<int32_t, int32_t>(
  raft::handle_t const&, int32_t const*, int32_t, int32_t, int32_t*);
template float hungarian<int32_t, float>(
  raft::handle_t const&, float const*, int32_t, int32_t, int32_t*);
template double hungarian<int32_t, double>(
  raft::handle_t const&, double const*, int32_t, int32_t, int32_t*);
template int32_t hungarian<int32_t, int32_t>(
  raft::handle_t const&, int32_t const*, int32_t, int32_t, int32_t*, int32_t);
template float hungarian<int32_t, float>(
  raft::handle_t const&, float const*, int32_t, int32_t, int32_t*, float);
template double hungarian<int32_t, double>(
  raft::handle_t const&, double const*, int32_t, int32_t, int32_t*, double);

}  // namespace dense

}  // namespace cugraph
