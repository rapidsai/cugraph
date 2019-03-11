/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <mpi.h>
#include <algorithm>
#include "gtest/gtest.h"
#include <cugraph.h>
#include "cuda_profiler_api.h"
#include "test_utils.h"
#include "global.h"
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

static std::string zero_file;

std::string get_file_name(std::string zeroFile, int file_id, int total_file_count) {
  if (file_id == 0) { return zeroFile; }
  int digits = 1 + std::floor(std::log10(file_id));
  std::string file_prefix = zeroFile.substr(0, zeroFile.size() - digits);
  return file_prefix + std::to_string(file_id);
}

int read_single_file(std::string fileName,
        std::vector<LOCINT>& s,
        std::vector<LOCINT>& d) {
    s.clear();
    d.clear();
    std::ifstream f(fileName);
    if (!f) { return 1; }
    LOCINT src, dst;
    while (f>>src>>dst) {
        s.push_back(src);
        d.push_back(dst);
    }
    f.close();
    return 0;
}

void read_file(std::string zeroFile, std::vector<LOCINT>& s, std::vector<LOCINT>& d) {
    int rank, ntask;
    MPI_Comm_size(MPI_COMM_WORLD, &ntask);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto input_file       = get_file_name(zeroFile, rank, ntask);
    int file_read_err = read_single_file(input_file, s, d);
    int all_files_err = 0;
    MPI_Allreduce(&file_read_err, &all_files_err, 1, LOCINT_MPI, MPI_SUM, MPI_COMM_WORLD);
    if (0 != all_files_err) {
  MPI_Barrier(MPI_COMM_WORLD);
        if (0 != file_read_err) {
            std::stringstream s;
            s<<"Unable to open "<<input_file<<" by process "<<rank<<"\n";
            std::cerr<<s.str();
            std::cerr<<std::flush;
        }
  MPI_Barrier(MPI_COMM_WORLD);
        exit(EXIT_FAILURE);
    }
  MPI_Barrier(MPI_COMM_WORLD);
}

LOCINT get_global_vertex_max(std::vector<LOCINT> &s, std::vector<LOCINT> &d) {
    LOCINT src_max = *(std::max_element(s.begin(), s.end()));
    LOCINT dst_max = *(std::max_element(d.begin(), d.end()));
    LOCINT ver_max = std::max(src_max, dst_max);
    LOCINT global_ver_max = -1;
    MPI_Allreduce(&ver_max, &global_ver_max, 1, LOCINT_MPI, MPI_MAX, MPI_COMM_WORLD);
    return global_ver_max;
}

TEST(MultiPagerank, Generic)
{
  std::vector<LOCINT> s;
  std::vector<LOCINT> d;
  read_file(zero_file, s, d);
  LOCINT global_v = get_global_vertex_max(s, d);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  CUDA_RT_CALL(cudaSetDevice(rank));
  gdf_column *col_src = new gdf_column, 
             *col_dest = new gdf_column, 
             *col_pagerank = new gdf_column, 
             *col_vidx = new gdf_column;

  create_gdf_column(s, col_src);
  create_gdf_column(d, col_dest);

  float damping_factor=0.85;
  int max_iter=3;

  gdf_error err = gdf_multi_pagerank (global_v, col_src, col_dest, col_vidx, col_pagerank, damping_factor, max_iter);
  int err_count = (err != GDF_SUCCESS);
  int gl_err    = 0;
  MPI_Allreduce(&err_count, &gl_err, 1, LOCINT_MPI, MPI_SUM, MPI_COMM_WORLD);
  ASSERT_EQ(gl_err, 0);

  gdf_col_delete(col_src);
  gdf_col_delete(col_dest);
  gdf_col_delete(col_pagerank);
  gdf_col_delete(col_vidx);
}

//USAGE
//NTASKS - {1..8}
//TYPE - {small, large, huge, gigantic, bigdata}
//mpirun -np <NTASKS> ./gtests/MULTI_PAGERANK_FILE_TEST /datasets/pagerank_demo/<NTASKS>/Input-<TYPE>/edges/part-00000
int main(int argc, char **argv)  {
  srand(42);
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  zero_file = std::string(argv[1]);
  int r = RUN_ALL_TESTS();
  MPI_Finalize();
  return r;
}
