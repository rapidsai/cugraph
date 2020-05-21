#include <mpi.h>
#include <nccl.h>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include "gtest/gtest.h"
#include "test_utils.h"

TEST(allgather, success)
{
  int p = 1, r = 0, dev = 0, dev_count = 0;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &r));
  CUDA_RT_CALL(cudaGetDeviceCount(&dev_count));

  // shortcut for device ID here
  // may need something smarter later
  dev = r % dev_count;
  // cudaSetDevice must happen before ncclCommInitRank
  CUDA_RT_CALL(cudaSetDevice(dev));

  // print info
  printf("#   Rank %2d - Pid %6d - device %2d\n", r, getpid(), dev);

  // NCCL init
  ncclUniqueId id;
  ncclComm_t comm;
  if (r == 0) NCCLCHECK(ncclGetUniqueId(&id));
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NCCLCHECK(ncclCommInitRank(&comm, p, id, r));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // allocate device buffers
  int size = 3;
  float *sendbuff, *recvbuff;
  CUDA_RT_CALL(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDA_RT_CALL(cudaMalloc(&recvbuff, size * p * sizeof(float)));

  // init values
  thrust::fill(
    thrust::device_pointer_cast(sendbuff), thrust::device_pointer_cast(sendbuff + size), (float)r);
  thrust::fill(
    thrust::device_pointer_cast(recvbuff), thrust::device_pointer_cast(recvbuff + size * p), -1.0f);

  // ncclAllGather
  NCCLCHECK(ncclAllGather(
    (const void *)sendbuff, (void *)recvbuff, size, ncclFloat, comm, cudaStreamDefault));

  // expect each rankid printed size times in ascending order
  if (r == 0) {
    thrust::device_ptr<float> dev_ptr(recvbuff);
    std::cout.precision(15);
    thrust::copy(dev_ptr, dev_ptr + size * p, std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
  }

  // free device buffers
  CUDA_RT_CALL(cudaFree(sendbuff));
  CUDA_RT_CALL(cudaFree(recvbuff));

  // finalizing NCCL
  NCCLCHECK(ncclCommDestroy(comm));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  rmmInitialize(nullptr);
  int rc = RUN_ALL_TESTS();
  rmmFinalize();
  MPI_Finalize();
  return rc;
}
