#include "gtest/gtest.h"
#include <cugraph.h>
#include "test_utils.h"
#include <string.h>
#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <graph.hpp>
#include "comms/mpi/comms_mpi.hpp"

// ref Degree on the host
template<typename idx_t>
void ref_degree_h(std::vector<idx_t> & ind_h,
                  std::vector<idx_t> & degree) {
  for (size_t i = 0; i < degree.size(); i++)
    degree[i] = 0;
  for (size_t i = 0; i < ind_h.size(); i++)
    degree[ind_h[i]] += 1;
}

TEST(degree, success)
{
  int v = 6;

  //host
  std::vector<int> src_h= {0, 0, 2, 2, 2, 3, 3, 4, 4, 5}, 
                   dest_h={1, 2, 0, 1, 4, 4, 5, 3, 5, 3};
  std::vector<int> degree_h(v, 0.0), degree_ref(v, 0.0);

  //device
  thrust::device_vector<int> src_d(src_h.begin(), src_h.begin()+src_h.size());
  thrust::device_vector<int> dest_d(dest_h.begin(), dest_h.begin()+dest_h.size());
  thrust::device_vector<int> degree_d(v);

  //MG
  int p;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));
  cugraph::experimental::Comm comm(p);

  // print mg info
  printf("#   Rank %2d - Pid %6d - device %2d\n", comm.get_rank(), getpid(), comm.get_dev());

  // load cugraph (fix me : split per process)
  cugraph::experimental::GraphCOO<int,int,float> G(thrust::raw_pointer_cast(src_d.data()), 
                                                   thrust::raw_pointer_cast(dest_d.data()), 
                                                   nullptr, degree_h.size(), dest_h.size());
  G.set_communicator(comm);

  // IN degree
  G.degree(thrust::raw_pointer_cast(degree_d.data()), cugraph::experimental::DegreeDirection::IN);
  thrust::copy(degree_d.begin(), degree_d.end(), degree_h.begin());
  ref_degree_h(dest_h, degree_ref); 
  for (size_t j = 0; j < degree_h.size(); ++j)
    EXPECT_EQ(degree_ref[j], degree_h[j]);

  // OUT degree
  G.degree(thrust::raw_pointer_cast(degree_d.data()), cugraph::experimental::DegreeDirection::OUT);
  thrust::copy(degree_d.begin(), degree_d.end(), degree_h.begin());
  ref_degree_h(src_h, degree_ref);
  for (size_t j = 0; j < degree_h.size(); ++j)
    EXPECT_EQ(degree_ref[j], degree_h[j]);
}

int main( int argc, char** argv )
{
    testing::InitGoogleTest(&argc,argv);
    MPI_Init(&argc, &argv);
    rmmInitialize(nullptr);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    MPI_Finalize();
    return rc;
}