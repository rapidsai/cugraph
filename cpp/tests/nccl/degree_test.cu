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

#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <comms_mpi.hpp>
#include <graph.hpp>
#include "gtest/gtest.h"

#include "utilities/test_utilities.hpp"

// ref Degree on the host
template <typename idx_t>
void ref_degree_h(std::vector<idx_t> &ind_h, std::vector<idx_t> &degree)
{
  for (size_t i = 0; i < degree.size(); i++) degree[i] = 0;
  for (size_t i = 0; i < ind_h.size(); i++) degree[ind_h[i]] += 1;
}

// global to local offsets by shifting all offsets by the first offset value
template <typename T>
void shift_by_front(std::vector<T> &v)
{
  auto start = v.front();
  for (auto i = size_t{0}; i < v.size(); ++i) v[i] -= start;
}

// 1D partitioning such as each GPU has about the same number of edges
template <typename T>
void opg_edge_partioning(
  int r, int p, std::vector<T> &ind_h, std::vector<size_t> &part_offset, size_t &e_loc)
{
  // set first and last partition offsets
  part_offset[0] = 0;
  part_offset[p] = ind_h.size();
  // part_offset[p] = *(std::max_element(ind_h.begin(), ind_h.end()));
  auto loc_nnz = ind_h.size() / p;
  for (int i = 1; i < p; i++) {
    // get the first vertex ID of each partition
    auto start_nnz = i * loc_nnz;
    auto start_v   = 0;
    for (auto j = size_t{0}; j < ind_h.size(); ++j) {
      if (j >= start_nnz) {
        start_v = j;
        break;
      }
    }
    part_offset[i] = start_v;
  }
  e_loc = part_offset[r + 1] - part_offset[r];
}
TEST(degree, success)
{
  int v = 6;

  // host
  std::vector<int> src_h  = {0, 0, 2, 2, 2, 3, 3, 4, 4, 5, 5},
                   dest_h = {1, 2, 0, 1, 4, 4, 5, 3, 5, 3, 1};
  std::vector<int> degree_h(v, 0.0), degree_ref(v, 0.0);

  // MG
  int p;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));
  cugraph::experimental::Comm comm(p);
  std::vector<size_t> part_offset(p + 1);
  auto i = comm.get_rank();
  size_t e_loc;

  opg_edge_partioning(i, p, src_h, part_offset, e_loc);
#ifdef OPG_VERBOSE
  sleep(i);
  for (auto j = part_offset.begin(); j != part_offset.end(); ++j) std::cout << *j << ' ';
  std::cout << std::endl;
  std::cout << "eloc: " << e_loc << std::endl;
#endif
  std::vector<int> src_loc_h(src_h.begin() + part_offset[i],
                             src_h.begin() + part_offset[i] + e_loc),
    dest_loc_h(dest_h.begin() + part_offset[i], dest_h.begin() + part_offset[i] + e_loc);
  shift_by_front(src_loc_h);

  // print mg info
  printf("#   Rank %2d - Pid %6d - device %2d\n", comm.get_rank(), getpid(), comm.get_dev());

  // local device
  thrust::device_vector<int> src_d(src_loc_h.begin(), src_loc_h.end());
  thrust::device_vector<int> dest_d(dest_loc_h.begin(), dest_loc_h.end());
  thrust::device_vector<int> degree_d(v);

  // load local chunck to cugraph
  cugraph::experimental::GraphCOO<int, int, float> G(thrust::raw_pointer_cast(src_d.data()),
                                                     thrust::raw_pointer_cast(dest_d.data()),
                                                     nullptr,
                                                     degree_h.size(),
                                                     e_loc);
  G.set_communicator(comm);

  // OUT degree
  G.degree(thrust::raw_pointer_cast(degree_d.data()), cugraph::experimental::DegreeDirection::IN);
  thrust::copy(degree_d.begin(), degree_d.end(), degree_h.begin());
  ref_degree_h(dest_h, degree_ref);
  // sleep(i);
  for (size_t j = 0; j < degree_h.size(); ++j) EXPECT_EQ(degree_ref[j], degree_h[j]);
  std::cout << "Rank " << i << " done checking." << std::endl;
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  {
    auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
    rmm::mr::set_default_resource(resource.get());
    int rc = RUN_ALL_TESTS();
  }
  MPI_Finalize();
  return rc;
}
