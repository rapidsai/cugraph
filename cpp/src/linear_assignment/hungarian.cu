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
#include <iostream>
#include <limits>

#include <graph.hpp>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/for_each.h>
#include <thrust/random.h>

#include <utilities/error.hpp>

#include "utilities/cuda_utils.cuh"
#include <utilities/high_res_timer.hpp>

//#define DETAILED_TIMING

#ifdef DETAILED_TIMING
HighResTimer hr_timer;

#define TIMER_START(LABEL)  { hr_timer.start(LABEL); }
#define TIMER_STOP()        { cudaDeviceSynchronize(); hr_timer.stop(); }

#else

#define TIMER_START(LABEL)
#define TIMER_STOP()

#endif

static int64_t hungarian_seed_s{time(nullptr)};

//#define DEBUG

namespace cugraph { 
namespace detail {

template <typename index_t, typename weight_t>
void display_dense(index_t num_rows, index_t num_cols, weight_t const *d_cost, cudaStream_t stream) {
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<int>(0),
                   thrust::make_counting_iterator<int>(1),
                   [num_rows, num_cols, d_cost] __device__ (int) {
                     printf("Dense Matrix:\n--------\n");
                     for (int i = 0 ; i < num_rows ; ++i) {
                       for (int j = 0 ; j < num_cols ; ++j) {
                         //printf(" %6.2g", d_cost[i * num_cols + j]);
                         printf(" %6g", d_cost[i * num_cols + j]);
                       }
                       printf("\n");
                     }
                     printf("------\n");
                   });
}

template <typename index_t>
void display_vector(index_t size, bool const *v, cudaStream_t stream) {
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<int>(0),
                   thrust::make_counting_iterator<int>(1),
                   [size, v] __device__ (int) {
                     printf("Vector:\n--------\n");
                     for (int i = 0 ; i < size ; ++i) {
                       printf(" %s", v[i] ? "true" : "false");
                     }
                     printf("\n------\n");
                   });
}

template <typename index_t>
void display_vector(index_t size, float const *v, cudaStream_t stream) {
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<int>(0),
                   thrust::make_counting_iterator<int>(1),
                   [size, v] __device__ (int) {
                     printf("Vector:\n--------\n");
                     for (int i = 0 ; i < size ; ++i) {
                       printf(" %g", v[i]);
                     }
                     printf("\n------\n");
                   });
}

template <typename index_t>
void display_vector(index_t size, double const *v, cudaStream_t stream) {
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<int>(0),
                   thrust::make_counting_iterator<int>(1),
                   [size, v] __device__ (int) {
                     printf("Vector:\n--------\n");
                     for (int i = 0 ; i < size ; ++i) {
                       printf(" %lg", v[i]);
                     }
                     printf("\n------\n");
                   });
}

template <typename index_t>
void display_vector(index_t size, int const *v, cudaStream_t stream) {
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<int>(0),
                   thrust::make_counting_iterator<int>(1),
                   [size, v] __device__ (int) {
                     printf("Vector:\n--------\n");
                     for (int i = 0 ; i < size ; ++i) {
                       printf(" %d", v[i]);
                     }
                     printf("\n------\n");
                   });
}

template <typename index_t, typename weight_t>
weight_t hungarian(index_t num_rows, index_t num_cols, weight_t const *d_original_cost, index_t *d_assignment, cudaStream_t stream) {

  // FIXME:  Call raft implementation
  //  TODO:  Can Date/Nagi implementation in raft handle rectangular matrices?

  //
  // FIXME: Current implementation only supports square matrices
  //
  CUGRAPH_EXPECTS(num_rows == num_cols, "Current implementation only supports square matrices");

  index_t size = num_rows * num_cols;
  rmm::device_vector<weight_t> cost_v(size);

  weight_t *d_cost = cost_v.data().get();

  thrust::copy(rmm::exec_policy(stream)->on(stream), d_original_cost, d_original_cost + size, d_cost);

#ifdef DEBUG
  display_dense(num_rows, num_cols, d_cost, stream);
#endif

  bool done = false;

  while (!done) {
    //
    // TODO:  I think it might be better to add noise *FIRST*
    //        That should eliminate the need to recompute, probably ever.
    //
    //        Based on how the noise is added, this should not affect
    //        the results.
    //

    //
    //  Now we will compute the assignment
    //
    done = detail::compute_assignment(num_rows, num_cols, d_cost, d_assignment, stream);
    if (!done) {
      add_noise(num_rows, num_cols, d_cost);
    }
  }

#ifdef DETAILED_TIMING
  hr_timer.display_and_clear(std::cout);
#endif
                   
  return thrust::transform_reduce(rmm::exec_policy(stream)->on(stream),
                                  thrust::make_counting_iterator<index_t>(0),
                                  thrust::make_counting_iterator<index_t>(num_rows),
                                  [d_assignment, d_original_cost, num_cols] __device__ (index_t row) {
                                    if (d_assignment[row] == num_cols)
                                      return weight_t{0};
                                    else {
                                      return d_original_cost[row * num_cols + d_assignment[row]];
                                    }
                                  },
                                  weight_t{0},
                                  thrust::plus<weight_t>());
}
  
template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian_sparse(GraphCOOView<vertex_t, edge_t, weight_t> const &graph,
                          vertex_t num_workers,
                          vertex_t const *workers,
                          vertex_t *assignment,
                          cudaStream_t stream)
{
  CUGRAPH_EXPECTS(assignment != nullptr, "Invalid API parameter: assignment pointer is NULL");
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "Invalid API parameter: graph must have edge data (costs)");

  //
  //  Translate sparse matrix into dense bipartite matrix.
  //    rows are the workers, columns are the tasks
  //
  vertex_t num_rows = num_workers;
  vertex_t num_cols = graph.number_of_vertices - num_rows;

  vertex_t matrix_dimension = std::max(num_rows, num_cols);

  rmm::device_vector<weight_t> cost_v(matrix_dimension * matrix_dimension);
  rmm::device_vector<vertex_t> tasks_v(num_cols);
  rmm::device_vector<vertex_t> temp_tasks_v(graph.number_of_vertices);
  rmm::device_vector<vertex_t> temp_workers_v(graph.number_of_vertices);

  weight_t *d_cost = cost_v.data().get();
  vertex_t *d_tasks = tasks_v.data().get();
  vertex_t *d_temp_tasks = temp_tasks_v.data().get();
  vertex_t *d_temp_workers = temp_workers_v.data().get();
  vertex_t *d_src_indices = graph.src_indices;
  vertex_t *d_dst_indices = graph.dst_indices;
  weight_t *d_edge_data = graph.edge_data;

  //
  //  Renumber vertices internally.  Workers will become
  //  rows, tasks will become columns
  //
  thrust::sequence(rmm::exec_policy(stream)->on(stream), temp_tasks_v.begin(), temp_tasks_v.end());

  thrust::for_each(rmm::exec_policy(stream)->on(stream), workers, workers + num_workers,
                   [d_temp_tasks] __device__ (vertex_t v) {
                     d_temp_tasks[v] = -1;
                   });
                     
  auto temp_end = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                                  temp_tasks_v.begin(),
                                  temp_tasks_v.end(),
                                  d_tasks,
                                  [] __device__ (vertex_t v) {
                                    return v >= 0;
                                  });

  vertex_t size = thrust::distance(d_tasks, temp_end);
  tasks_v.resize(size);

  //
  // Now we'll assign costs into the dense array
  //
  thrust::fill(rmm::exec_policy(stream)->on(stream), temp_workers_v.begin(), temp_workers_v.end(), vertex_t{-1});
  thrust::fill(rmm::exec_policy(stream)->on(stream), temp_tasks_v.begin(), temp_tasks_v.end(), vertex_t{-1});
  thrust::fill(rmm::exec_policy(stream)->on(stream), cost_v.begin(), cost_v.end(), weight_t{0});

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<vertex_t>(0),
                   thrust::make_counting_iterator<vertex_t>(num_rows),
                   [d_temp_workers, workers] __device__ (vertex_t v) {
                     d_temp_workers[workers[v]] = v;
                   });

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<vertex_t>(0),
                   thrust::make_counting_iterator<vertex_t>(num_cols),
                   [d_temp_tasks, d_tasks] __device__ (vertex_t v) {
                     d_temp_tasks[d_tasks[v]] = v;
                   });

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<edge_t>(0),
                   thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
                   [d_temp_workers, d_temp_tasks, d_cost, matrix_dimension,
                    d_src_indices, d_dst_indices, d_edge_data] __device__ (edge_t loc) {
                     vertex_t src = d_temp_workers[d_src_indices[loc]];
                     vertex_t dst = d_temp_tasks[d_dst_indices[loc]];

                     if ((src >= 0) && (dst >= 0)) {
                       d_cost[src * matrix_dimension + dst] = d_edge_data[loc];
                     }
                   });

  //
  //  temp_assignment_v will hold the assignment in the dense
  //  bipartite matrix numbering
  //
  rmm::device_vector<vertex_t> temp_assignment_v(matrix_dimension);
  vertex_t *d_temp_assignment = temp_assignment_v.data().get();

  weight_t min_cost = detail::hungarian(matrix_dimension, matrix_dimension, d_cost, d_temp_assignment, stream); 

  //
  //  Translate the assignment back to the original vertex ids
  //
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<vertex_t>(0),
                   thrust::make_counting_iterator<vertex_t>(num_rows),
                   [d_tasks, d_temp_assignment, assignment] __device__ (vertex_t id) {
                     assignment[id] = d_tasks[d_temp_assignment[id]];
                   });

  return min_cost;
}

} //namespace detail

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian_sparse(GraphCOOView<vertex_t, edge_t, weight_t> const &graph,
                          vertex_t num_workers,
                          vertex_t const *workers,
                          vertex_t *assignment)
{
  cudaStream_t stream{0};

  return detail::hungarian_sparse(graph, num_workers, workers, assignment, stream);
}

template <typename index_t, typename weight_t>
weight_t hungarian_dense(weight_t const *costs,
                         index_t num_rows,
                         index_t num_cols,
                         index_t *assignment) {

  cudaStream_t stream{0};

  return detail::hungarian(num_rows, num_cols, costs, assignment, stream);
}
                   

template int32_t hungarian_sparse<int32_t, int32_t, int32_t>(GraphCOOView<int32_t,int32_t,int32_t> const &, int32_t, int32_t const *, int32_t *);
template float hungarian_sparse<int32_t, int32_t, float>(GraphCOOView<int32_t,int32_t,float> const &, int32_t, int32_t const *, int32_t *);
template double hungarian_sparse<int32_t, int32_t, double>(GraphCOOView<int32_t,int32_t,double> const &, int32_t, int32_t const *, int32_t *);

template int32_t hungarian_dense<int32_t, int32_t>(int32_t const *, int32_t, int32_t, int32_t *);
template float hungarian_dense<int32_t, float>(float const *, int32_t, int32_t, int32_t *);
template double hungarian_dense<int32_t, double>(double const *, int32_t, int32_t, int32_t *);

} //namespace cugraph 
