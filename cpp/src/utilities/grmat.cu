/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// Graph generation
// Author: Ramakrishna Prabhu ramakrishnap@nvidia.com

#include <stdio.h>
#include <string>
#include <omp.h>

// Utilities and correctness-checking
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/sort_omp.cuh>
#include <gunrock/csr.cuh>
#include <gunrock/graphio/grmat.cuh>
#include <gunrock/coo.cuh>


#include <moderngpu.cuh>

// boost includes
#include <boost/config.hpp>
#include <boost/utility.hpp>

#include <gunrock/util/shared_utils.cuh>

#include <cudf/cudf.h>
#include <thrust/extrema.h>
#include "utilities/error_utils.h"
#include "graph_utils.cuh"

#include <rmm_utils.h>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::graphio;
using namespace gunrock::graphio::grmat;

template <typename VertexId, typename Value, typename SizeT>
__global__ void Remove_Self_Loops (VertexId* row, VertexId* col, Value* val, SizeT edges)
{
   SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

   if (i < edges)
   {
       if (row[i] == col[i])
       {
           col[i] = 0;
       }
   }
}

//    rmat (default: rmat_scale = 10, a = 0.57, b = c = 0.19)
//                Generate R-MAT graph as input
//                --rmat_scale=<vertex-scale>
//                --rmat_nodes=<number-nodes>
//                --rmat_edgefactor=<edge-factor>
//                --rmat_edges=<number-edges>
//                --rmat_a=<factor> --rmat_b=<factor> --rmat_c=<factor>
//                --rmat_self_loops If this option is supplied, then self loops will be retained
//                --rmat_undirected If this option is not mentioned, then the graps will be undirected
//        Optional arguments:
//        [--device=<device_index>] Set GPU(s) for testing (Default: 0).
//        [--quiet]                 No output (unless --json is specified).
//        [--random_seed]           This will enable usage of random seed, else it will use same seed
//        [--normalized]\n

template<
    typename VertexId,
    typename SizeT,
    typename Value>
gdf_error main_(gdf_column *src,  gdf_column *dest, gdf_column *val, CommandLineArgs *args, size_t &vertices, size_t &edges)
{
    CpuTimer cpu_timer, cpu_timer2;
    SizeT rmat_nodes = 1 << 10;
    SizeT rmat_edges = 1 << 10;
    SizeT rmat_scale = 10;
    SizeT rmat_edgefactor = 48;
    double rmat_a = 0.57;
    double rmat_b = 0.19;
    double rmat_c = 0.19;
    double rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
    double rmat_vmin = 1;
    double rmat_vmultipiler = 64;
    int rmat_seed = 888;
    bool undirected = false;
    bool self_loops = false;
    SizeT rmat_all_edges = rmat_edges;
    std::string file_name;
    bool quiet = false;

    typedef Coo_nv<VertexId, Value> EdgeTupleType;

    cpu_timer.Start();

    if (args->CheckCmdLineFlag ("rmat_scale") && args->CheckCmdLineFlag ("rmat_nodes"))
    {
        printf ("Please mention scale or nodes, not both \n");
        return GDF_UNSUPPORTED_METHOD;
    }
    else if (args->CheckCmdLineFlag ("rmat_edgefactor") && args->CheckCmdLineFlag ("rmat_edges"))
    {
        printf ("Please mention edgefactor or edge, not both \n");
        return GDF_UNSUPPORTED_METHOD;
    }

    self_loops = args->CheckCmdLineFlag ("rmat_self_loops");
    // graph construction or generation related parameters
    if (args -> CheckCmdLineFlag("normalized"))
        undirected = args -> CheckCmdLineFlag("rmat_undirected");
    else undirected = true;   // require undirected input graph when unnormalized
    quiet = args->CheckCmdLineFlag("quiet");

    args->GetCmdLineArgument("rmat_scale", rmat_scale);
    rmat_nodes = 1 << rmat_scale;
    args->GetCmdLineArgument("rmat_nodes", rmat_nodes);
    args->GetCmdLineArgument("rmat_edgefactor", rmat_edgefactor);
    rmat_edges = rmat_nodes * rmat_edgefactor;
    args->GetCmdLineArgument("rmat_edges", rmat_edges);
    args->GetCmdLineArgument("rmat_a", rmat_a);
    args->GetCmdLineArgument("rmat_b", rmat_b);
    args->GetCmdLineArgument("rmat_c", rmat_c);
    rmat_d = 1 - (rmat_a + rmat_b + rmat_c);
    args->GetCmdLineArgument("rmat_d", rmat_d);
    args->GetCmdLineArgument("rmat_vmin", rmat_vmin);
    args->GetCmdLineArgument("rmat_vmultipiler", rmat_vmultipiler);
    args->GetCmdLineArgument("file_name", file_name);
    if (args->CheckCmdLineFlag("random_seed"))
    {
        rmat_seed = -1;
    }
    EdgeTupleType coo;

    if (undirected == true)
    {
        rmat_all_edges = 2 * rmat_edges;
    }
    else
    {
        rmat_all_edges = rmat_edges;
    }

    std::vector<int> temp_devices;
    if (args->CheckCmdLineFlag("device"))  // parse device list
    {
      args->GetCmdLineArguments<int>("device", temp_devices);
    }
    else  // use single device with index 0
    {
      int gpu_idx;
      util::GRError(cudaGetDevice(&gpu_idx),
          "cudaGetDevice failed", __FILE__, __LINE__);
      temp_devices.push_back(gpu_idx);
    }
    int *gpu_idx = new int[temp_devices.size()];
    for (unsigned int i=0; i<temp_devices.size(); i++)
        gpu_idx[i] = temp_devices[i];

    if (!quiet)
    {
        printf ("---------Graph properties-------\n"
                "      Undirected : %s\n"
                "      Nodes : %lld\n"
                "      Edges : %lld\n"
                "      a = %f, b = %f, c = %f, d = %f\n\n\n", ((undirected == true)? "True": "False"), (long long)rmat_nodes,
                              (long long)(rmat_edges * ((undirected == true)? 2: 1)), rmat_a, rmat_b, rmat_c, rmat_d);
    }

    if (util::SetDevice(gpu_idx[0]))
        return GDF_CUDA_ERROR;

    cudaStream_t stream {nullptr};
    ALLOC_TRY((void**)&coo.row, sizeof(VertexId) * rmat_all_edges, stream);
    ALLOC_TRY((void**)&coo.col, sizeof(VertexId) * rmat_all_edges, stream);
    if (val != nullptr)
    {
        ALLOC_TRY((void**)&coo.val, sizeof(Value) * rmat_all_edges, stream);
    }
    if ((coo.row == NULL) ||(coo.col == NULL))
    {
        if (!quiet)
            printf ("Error: Cuda malloc failed \n");
        if (coo.row != nullptr)
                ALLOC_FREE_TRY(coo.row, stream);
        if (coo.col != nullptr)
                ALLOC_FREE_TRY(coo.col, stream);
        return GDF_CUDA_ERROR;
    }
    cpu_timer2.Start();
    cudaError_t status = cudaSuccess;
    if(val == nullptr)
        status = BuildRmatGraph_coo_nv<false, VertexId, SizeT, Value, EdgeTupleType>(rmat_nodes, rmat_edges, coo, undirected,
                                               rmat_a, rmat_b, rmat_c, rmat_d, rmat_vmultipiler, rmat_vmin, rmat_seed,
                                               quiet, temp_devices.size(), gpu_idx);
    else
        status = BuildRmatGraph_coo_nv<true, VertexId, SizeT, Value, EdgeTupleType>(rmat_nodes, rmat_edges, coo, undirected,
                                               rmat_a, rmat_b, rmat_c, rmat_d, rmat_vmultipiler, rmat_vmin, rmat_seed,
                                               quiet, temp_devices.size(), gpu_idx);

    cpu_timer2.Stop();
    if (status == cudaSuccess)
    {
        if (!quiet)
            printf ("Graph has been generated \n");
    }
    else
    {
        if (coo.row != nullptr)
                ALLOC_FREE_TRY(coo.row, stream);
        if (coo.col != nullptr)
                ALLOC_FREE_TRY(coo.col, stream);
        if (coo.val != nullptr)
                ALLOC_FREE_TRY(coo.val, stream);

        return GDF_CUDA_ERROR;
    }

    int block_size = (sizeof(VertexId) == 4) ? 1024 : 512;
    int grid_size = rmat_all_edges / block_size + 1;

    if (util::SetDevice(gpu_idx[0]))
        return GDF_CUDA_ERROR;
    if ((self_loops != false) && (val != nullptr))
    {
        Remove_Self_Loops
              <VertexId, Value, SizeT>
              <<<grid_size, block_size, 0>>>
              (coo.row, coo.col, coo.val, rmat_all_edges);
    }

    cugraph::detail::remove_duplicate (coo.row, coo.col, coo.val, rmat_all_edges);

    thrust::device_ptr<VertexId> tmp;

    VertexId nodes_row = 0;
    VertexId nodes_col = 0;

    cudaMemcpy((void*)&nodes_row, (void*)&(coo.row[rmat_all_edges-1]), sizeof(VertexId), cudaMemcpyDeviceToHost);

    tmp = thrust::max_element(rmm::exec_policy(stream)->on(stream),
                                thrust::device_pointer_cast((VertexId*)(coo.col)),
                                thrust::device_pointer_cast((VertexId*)(coo.col + rmat_all_edges)));
    nodes_col = tmp[0];

    VertexId max_nodes = (nodes_row > nodes_col)? nodes_row: nodes_col;

    cpu_timer.Stop();

    if ((src != nullptr) && (dest != nullptr))
    {
        src->data = coo.row;
        src->size = rmat_all_edges;
        src->valid = nullptr;

        dest->data = coo.col;
        dest->size = rmat_all_edges;
        dest->valid = nullptr;
    }
    else
    {
        if (coo.row != nullptr)
            ALLOC_FREE_TRY(coo.row, stream);
        if (coo.col != nullptr)
            ALLOC_FREE_TRY(coo.col, stream);
        if (coo.val != nullptr)
            ALLOC_FREE_TRY(coo.val, stream);
        if (!quiet)
            printf ("Error : Pointers for gdf column are null, releasing allocated memory for graph\n");

        return GDF_CUDA_ERROR;
    }

    if (val != nullptr)
    {
        val->data = coo.val;
        val->size = rmat_all_edges;
        val->valid = nullptr;
    }

    vertices = max_nodes+1;
    edges = rmat_all_edges;

    if (!quiet)
        printf ("Time to generate the graph %f ms\n"
                "Total time %f ms\n", cpu_timer2.ElapsedMillis(), cpu_timer.ElapsedMillis());

    
}

void free_args (char argc, char** args)
{
    for (int i = 0; i < argc; i++)
        free(args[i]);
}

gdf_error gdf_grmat_gen (const char* argv, size_t& vertices, size_t& edges, gdf_column *src,  gdf_column *dest, gdf_column *val)
{
    int argc = 0;
    char* arg[32] = {0};
    char* tmp = nullptr;
    char tmp_argv [1024] = {0};

    strcpy(tmp_argv, argv);

    tmp = strtok (tmp_argv, " ");
    for (int i = 0; tmp != nullptr; i++)
    {
        arg[i] = (char*) malloc (sizeof(char)*(strlen(tmp)+1));
        strcpy(arg[i], tmp);
        argc += 1;
        tmp = strtok(NULL, " ");
    }

    CommandLineArgs args(argc, arg);

    int graph_args = argc - args.ParsedArgc() - 1;
    gdf_error status = GDF_CUDA_ERROR;

    if (src == nullptr || dest == nullptr)
    {
        free_args(argc, arg);
        return GDF_DATASET_EMPTY;
    }

    CUGRAPH_EXPECTS ((src->dtype == dest->dtype), GDF_DTYPE_MISMATCH);
    CUGRAPH_EXPECTS (src->null_count == 0, "Column must be valid");

    if (argc < 2 || args.CheckCmdLineFlag("help"))
    {
        free_args(argc, arg);
        return GDF_UNSUPPORTED_METHOD;
    }


    if (src->dtype == GDF_INT64)
    {
        if ((val != nullptr) && (val->dtype == GDF_FLOAT64))
        {
            status = main_<long long, long long, double> (src, dest, val, &args, vertices, edges);
        }
        else
        {
            status = main_<long long, long long, float> (src, dest, val, &args, vertices, edges);
        }
    }
    else
    {
        if ((val != nullptr) && (val->dtype == GDF_FLOAT64))
        {
            status = main_ <int, int, double> (src, dest, val, &args, vertices, edges);
        }
        else
        {
            status = main_ <int, int, float> (src, dest, val, &args, vertices, edges);
        }
    }

    free_args(argc, arg);

    CUGRAPH_EXPECTS((src->size == dest->size), "Column size mismatch");
    CUGRAPH_EXPECTS ((src->dtype == dest->dtype), GDF_DTYPE_MISMATCH);
    CUGRAPH_EXPECTS (src->null_count == 0, "Column must be valid");

    return status;
}
