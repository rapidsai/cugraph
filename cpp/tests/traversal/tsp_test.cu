/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// TSP solver tests
// Author: Hugo Linsenmaier hlinsenmaier@nvidia.com

#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <utilities/high_res_clock.h>
#include <algorithms.hpp>
#include <cmath>
#include <graph.hpp>
#include <raft/error.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

typedef struct Route {
  uint num_packages;
  float *x;
  float *y;
  float *vol;
  uint *order;
} route;

static int readInput(char *fname,
                     route *input)  // ATT and CEIL_2D edge weight types are not supported
{
  int ch, cnt, in1, nodes;
  float in2, in3;
  FILE *f;
  char str[256];  // potential for buffer overrun

  f = fopen(fname, "rt");
  if (f == NULL) {
    fprintf(stderr, "could not open file %s\n", fname);
    exit(-1);
  }

  ch = getc(f);
  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);
  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);
  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

  ch = getc(f);
  while ((ch != EOF) && (ch != ':')) ch = getc(f);
  fscanf(f, "%s\n", str);
  nodes = atoi(str);
  printf("nodes: %i\n", nodes);
  input->num_packages = (uint)nodes;
  if (nodes <= 2) {
    fprintf(stderr, "only %d nodes\n", nodes);
    exit(-1);
  }

  input->x = (float *)malloc(sizeof(float) * nodes);
  if (input->x == NULL) {
    fprintf(stderr, "cannot allocate %d xcoords\n", nodes);
    exit(-1);
  }
  input->y = (float *)malloc(sizeof(float) * nodes);
  if (input->y == NULL) {
    fprintf(stderr, "cannot allocate %d ycoords\n", nodes);
    exit(-1);
  }

  ch = getc(f);
  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  fscanf(f, "%s\n", str);
  if (strcmp(str, "NODE_COORD_SECTION") != 0) {
    fprintf(stderr, "wrong file format\n");
    exit(-1);
  }

  cnt = 0;
  while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) {
    input->x[cnt] = in2;
    input->y[cnt] = in3;
    printf("idx: %i: %f, y: %f\n", in1, in2, in3);
    cnt++;
    if (cnt > nodes) {
      fprintf(stderr, "inconsistent data: input too long\n");
      exit(-1);
    }
    if (cnt != in1) {
      fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);
      exit(-1);
    }
  }
  if (cnt != nodes) {
    fprintf(stderr, "inconsistent data: read %d instead of %d nodes\n", cnt, nodes);
    exit(-1);
  }

  fscanf(f, "%s", str);
  if (strcmp(str, "EOF") != 0) {
    fprintf(stderr, "didn't see 'EOF' at end of file\n");
    exit(-1);
  }

  fclose(f);
  return nodes;
}

int main(int argc, char *argv[])
{
  route input;
  int nodes = readInput(argv[1], &input);
  printf("Read %d locations from %s \n", nodes, argv[1]);
  int restarts = atoi(argv[2]);
  if (restarts < 1) {
    fprintf(stderr, "Zero or negative restarts: %d\n", restarts);
    exit(-1);
  }

  raft::handle_t handle;
  rmm::device_uvector<int> route(static_cast<size_t>(nodes), nullptr);
  // device alloc
  rmm::device_uvector<float> x_pos(static_cast<size_t>(nodes), nullptr);
  rmm::device_uvector<float> y_pos(static_cast<size_t>(nodes), nullptr);
  int *d_route   = route.data();
  float *d_x_pos = x_pos.data();
  float *d_y_pos = y_pos.data();
  int k          = 4;
  bool verbose   = true;

  CUDA_TRY(cudaMemcpy(d_x_pos, input.x, sizeof(float) * nodes, cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(d_y_pos, input.y, sizeof(float) * nodes, cudaMemcpyHostToDevice));

  float final_cost =
    cugraph::traveling_salesman(handle, d_route, d_x_pos, d_y_pos, nodes, restarts, k, verbose);
  cudaDeviceSynchronize();
  std::cout << "Final cost is: " << final_cost << "\n";
  return 0;
}
