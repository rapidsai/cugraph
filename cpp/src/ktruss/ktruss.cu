/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

/**
 * ---------------------------------------------------------------------------*
 * @brief KTruss implementation
 *
 * @file ktruss.cu
 * --------------------------------------------------------------------------*/


#include <cugraph.h>
#include "utilities/error_utils.h"
#include <Hornet.hpp>
#include "Static/KTruss/KTruss.cuh"
#include <StandardAPI.hpp>
#include <rmm_utils.h>
#include <nvgraph_gdf.h>

using namespace hornets_nest;

#define CPU_INIT 0


namespace cugraph {
namespace detail {

void ktruss_max_impl(Graph *graph,
                          int *k_max) {
  // gdf_error err = gdf_add_adj_list(graph);
  // if (err != GDF_SUCCESS)
  //   return err;
  int * src = static_cast<int*>(graph->edgeList->src_indices->data);
  int * dst = static_cast<int*>(graph->edgeList->dest_indices->data);



  #if CPU_INIT==1
    using HornetGraph = hornet::gpu::Hornet<int>;
    using UpdatePtr   = ::hornet::BatchUpdatePtr<int, hornet::EMPTY, hornet::DeviceType::DEVICE>;
    using Update      = ::hornet::gpu::BatchUpdate<int>;

    UpdatePtr ptr(graph->edgeList->src_indices->size, src, dst);
    Update batch(ptr);
    number_of_vertices(graph);

    HornetGraph hnt(graph->numberOfVertices+1);
    hnt.insert(batch);

    printf("GPU Init : CSIDE %d \n",graph->numberOfVertices,graph->edgeList->src_indices->size);

  #else
    int *offs, *adjs;
    offs = (int*)malloc(sizeof(int) * (graph->numberOfVertices + 1));
    adjs = (int*)malloc(sizeof(int) * (graph->adjList->indices->size));

    cudaMemcpy(offs,static_cast<int*>(graph->adjList->offsets->data), sizeof(int) * (graph->numberOfVertices + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(adjs,static_cast<int*>(graph->adjList->indices->data), sizeof(int) * (graph->adjList->indices->size), cudaMemcpyDeviceToHost);

    // for(int i=0; i<(graph->numberOfVertices); i++){
    //   int o = offs[i];
    //   int len = offs[i+1]-offs[i];

    //   printf("%d : ",i);
    //   for(int j=0; j<len; j++){
    //     printf("%d, ",adjs[o+j]);
    //   }
    //   printf("\n");
    // }



    // using HornetInit  = hornet::HornetInit<int>;
    HornetInit init(graph->numberOfVertices, graph->adjList->indices->size, offs,adjs);

    HornetGraph hnt(init);

    // printf("%d %d\n",graph->numberOfVertices, graph->adjList->indices->size);fflush(stdout);

    printf("CPU Init : CSIDE %d %d\n",graph->numberOfVertices,graph->adjList->indices->size);

  #endif

  KTruss kt(hnt);

  kt.init();
  kt.reset();

  #if CPU_INIT==0
    kt.createOffSetArray();
  #else
    kt.copyOffsetArrayHost(offs);
  #endif

  kt.setInitParameters(4, 8, 2, 64000, 32);
  kt.reset(); 
  kt.sortHornet();


  kt.run();

  *k_max = kt.getMaxK();

  kt.release();


  #if CPU_INIT==1
    free(offs);
    free(adjs);
  #endif


}

} // detail namespace

void k_truss_max(Graph *graph,
                          int *k_max) {
  // CUGRAPH_EXPECTS(graph->edgeList->src_indices != nullptr || graph->edgeList->dest_indices != nullptr, "Invalid API parameter");
  // CUGRAPH_EXPECTS(graph->edgeList->src_indices->data != nullptr || graph->edgeList->dest_indices->data != nullptr, "Invalid API parameter");

  // CUGRAPH_EXPECTS(graph->edgeList->src_indices->dtype == GDF_INT32, "Unsupported data type");
  // CUGRAPH_EXPECTS(graph->edgeList->dest_indices->dtype == GDF_INT32, "Unsupported data type");

  detail::ktruss_max_impl(graph, k_max);
}

namespace detail {

void ktruss_subgraph_impl(Graph *graph,
                          int k,
                          Graph *ktrussgraph) {
  int * src = static_cast<int*>(graph->edgeList->src_indices->data);
  int * dst = static_cast<int*>(graph->edgeList->dest_indices->data);
  using HornetGraph = hornet::gpu::Hornet<int>;
  using UpdatePtr   = ::hornet::BatchUpdatePtr<int, hornet::EMPTY, hornet::DeviceType::DEVICE>;
  using Update      = ::hornet::gpu::BatchUpdate<int>;


  UpdatePtr ptr(graph->edgeList->src_indices->size, src, dst);
  Update batch(ptr);
  number_of_vertices(graph);
  HornetGraph hnt(graph->numberOfVertices);
  hnt.insert(batch);

  
  // KTruss kt(hnt);

  // kt.init();
  // kt.reset();

  // kt.createOffSetArray();
  // kt.setInitParameters(4, 8, 2, 64000, 32);
  // kt.reset(); 
  // kt.sortHornet();

  // printf("Number of edges before %d : ",hnt.nE());

  // // kt.runForK(k);

  // printf("Number of edges after  %d : ",hnt.nE());


  // kt.release();
  
}
} // detail namespace

void k_truss_subgraph(Graph *graph,
                          int k,
                          Graph *ktrussgraph) {

  CUGRAPH_EXPECTS(graph->edgeList->src_indices != nullptr || graph->edgeList->dest_indices != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->edgeList->src_indices->data != nullptr || graph->edgeList->dest_indices->data != nullptr, "Invalid API parameter");

  CUGRAPH_EXPECTS(graph->edgeList->src_indices->dtype == GDF_INT32, "Unsupported data type");
  CUGRAPH_EXPECTS(graph->edgeList->dest_indices->dtype == GDF_INT32, "Unsupported data type");

  detail::ktruss_subgraph_impl(graph, k,NULL);
}


}//namespace cugraph