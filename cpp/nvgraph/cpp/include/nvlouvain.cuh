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
#pragma once
#include <string>
#include <cstring>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>

#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cusparse.h>

#include "graph_utils.cuh"
#include "modularity.cuh"
#include "delta_modularity.cuh"
#include "high_res_clock.h"
#include "size2_selector.cuh"
#include "thrust_coarse_generator.cuh"

namespace nvlouvain{

//#define VERBOSE true

#define LOG() (log<<COLOR_GRN<<"[ "<< time_now() <<" ] "<<COLOR_WHT)


/*
The main program of louvain
*/
template<typename IdxType=int, typename ValType>
NVLOUVAIN_STATUS louvain(IdxType* csr_ptr, IdxType* csr_ind, ValType* csr_val,
                  const size_t num_vertex, const size_t num_edges, 
                  bool& weighted, bool has_init_cluster,
                  IdxType* init_cluster, // size = n_vertex
                  ValType& final_modularity,
                  IdxType* cluster_vec, // size = n_vertex
                  IdxType& num_level,
                  std::ostream& log = std::cout){
#ifndef ENABLE_LOG
  log.setstate(std::ios_base::failbit);
#endif
  num_level = 0;
  cusparseHandle_t cusp_handle;
  cusparseCreate(&cusp_handle);

  int n_edges = num_edges;
  int n_vertex = num_vertex;

  thrust::device_vector<IdxType> csr_ptr_d(csr_ptr, csr_ptr + n_vertex + 1);
  thrust::device_vector<IdxType> csr_ind_d(csr_ind, csr_ind + n_edges);
  thrust::device_vector<ValType> csr_val_d(csr_val, csr_val + n_edges);

  //std::vector<IdxType> clustering(n_vertex);
  thrust::device_vector<IdxType> clustering(n_vertex);
  int upper_bound = 100;

  HighResClock hr_clock;
  double timed, diff_time;
  //size_t mem_tot= 0;
  //size_t mem_free = 0;

  int c_size(n_vertex);
  unsigned int best_c_size = (unsigned) n_vertex;
  unsigned current_n_vertex(n_vertex);
  int num_aggregates(n_edges);
  ValType m2 = thrust::reduce(thrust::cuda::par, csr_val_d.begin(), csr_val_d.begin() + n_edges);

  ValType best_modularity = -1;

  thrust::device_vector<IdxType> new_csr_ptr(n_vertex, 0);
  thrust::device_vector<IdxType> new_csr_ind(n_edges, 0);
  thrust::device_vector<ValType> new_csr_val(n_edges, 0);

  thrust::device_vector<IdxType> cluster_d(n_vertex);
  thrust::device_vector<IdxType> aggregates_tmp_d(n_vertex, 0);
  thrust::device_vector<IdxType> cluster_inv_ptr(c_size + 1, 0);
  thrust::device_vector<IdxType> cluster_inv_ind(n_vertex, 0);
  thrust::device_vector<ValType> k_vec(n_vertex, 0);
  thrust::device_vector<ValType> Q_arr(n_vertex, 0);
  thrust::device_vector<ValType> delta_Q_arr(n_edges, 0);
  thrust::device_vector<ValType> cluster_sum_vec(c_size, 0);
  thrust::host_vector<IdxType> best_cluster_h(n_vertex, 0);
  Vector<IdxType> aggregates((int) current_n_vertex, 0);

  IdxType* cluster_inv_ptr_ptr = thrust::raw_pointer_cast(cluster_inv_ptr.data());
  IdxType* cluster_inv_ind_ptr = thrust::raw_pointer_cast(cluster_inv_ind.data());
  IdxType* csr_ptr_ptr = thrust::raw_pointer_cast(csr_ptr_d.data());
  IdxType* csr_ind_ptr = thrust::raw_pointer_cast(csr_ind_d.data());
  ValType* csr_val_ptr = thrust::raw_pointer_cast(csr_val_d.data());
  IdxType* cluster_ptr = thrust::raw_pointer_cast(cluster_d.data());

  if(!has_init_cluster){
    // if there is no initialized cluster
    // the cluster as assigned as a sequence (a cluster for each vertex)
    // inv_clusters will also be 2 sequence
    thrust::sequence(thrust::cuda::par, cluster_d.begin(), cluster_d.end());
    thrust::sequence(thrust::cuda::par, cluster_inv_ptr.begin(), cluster_inv_ptr.end());
    thrust::sequence(thrust::cuda::par, cluster_inv_ind.begin(), cluster_inv_ind.end());
  }
  else{
    // assign initialized cluster to cluster_d device vector
    // generate inverse cluster in CSR formate
    if(init_cluster == nullptr){
      final_modularity = -1;
      return NVLOUVAIN_ERR_BAD_PARAMETERS;
    }

    thrust::copy(init_cluster, init_cluster + n_vertex , cluster_d.begin());
    generate_cluster_inv(current_n_vertex, c_size, cluster_d.begin(), cluster_inv_ptr, cluster_inv_ind);
  }
 
  dim3 block_size_1d((n_vertex + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size_1d(BLOCK_SIZE_1D, 1, 1); 
  dim3 block_size_2d((n_vertex + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, (n_vertex + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, 1);
  dim3 grid_size_2d(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);

  ValType* k_vec_ptr = thrust::raw_pointer_cast(k_vec.data());
  ValType* Q_arr_ptr = thrust::raw_pointer_cast(Q_arr.data());
  ValType* cluster_sum_vec_ptr = thrust::raw_pointer_cast(cluster_sum_vec.data());
  ValType* delta_Q_arr_ptr =  thrust::raw_pointer_cast(delta_Q_arr.data());

  ValType new_Q, cur_Q, delta_Q, delta_Q_final;
  unsigned old_c_size(c_size); 
  bool updated = true;

  hr_clock.start();
  // Get the initialized modularity
  new_Q = modularity( n_vertex, n_edges, c_size, m2,
              csr_ptr_ptr, csr_ind_ptr, csr_val_ptr,
              cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr, 
              weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr); // delta_Q_arr_ptr is temp_i


  hr_clock.stop(&timed);
  diff_time = timed;

  LOG()<<"Initial modularity value: "<<COLOR_MGT<<new_Q<<COLOR_WHT<<" runtime: "<<diff_time/1000<<"\n";  

  bool contin(true);
  int bound = 0;
  int except = 3;

  
  do{ 
    bound = 0;  
    

    block_size_1d = dim3((current_n_vertex + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
    grid_size_1d  = dim3(BLOCK_SIZE_1D, 1, 1); 
    cur_Q = new_Q;
    old_c_size = c_size;

#ifdef VERBOSE
      LOG()<<"Current cluster inv: \n";
      nvlouvain::display_vec(cluster_inv_ptr, log);
      nvlouvain::display_vec(cluster_inv_ind, log);
#endif


    hr_clock.start();
    // Compute delta modularity for each edges
    build_delta_modularity_vector(cusp_handle, current_n_vertex, c_size, m2, updated, 
                                  csr_ptr_d, csr_ind_d, csr_val_d, 
                                  cluster_d, 
                                  cluster_inv_ptr_ptr, cluster_inv_ind_ptr, 
                                  k_vec_ptr, cluster_sum_vec_ptr, delta_Q_arr_ptr);
     
    //display_vec(delta_Q_arr);
    hr_clock.stop(&timed);
    diff_time = timed;
    LOG()<<"Complete build_delta_modularity_vector  runtime: "<<diff_time/1000<<"\n";
    //LOG()<<"Initial modularity value: "<<COLOR_MGT<<new_Q<<COLOR_WHT<<" runtime: "<<diff_time/1000<<"\n";  


    //  Start aggregates 
    Matching_t config = nvlouvain::USER_PROVIDED;
    //Size2Selector<IdxType, ValType> size2_sector(config, 0, 50, 0.6, true, false, 0);
    int agg_deterministic = 1;
    int agg_max_iterations = 25;
    ValType agg_numUnassigned_tol = 0.85;
    bool agg_two_phase =  false;
    bool agg_merge_singletons = true;
    

    if (current_n_vertex<8)
    {
      agg_merge_singletons = false;
      //agg_max_iterations = 4;
    }


    Size2Selector<IdxType, ValType> size2_sector(config, agg_deterministic, agg_max_iterations, agg_numUnassigned_tol, agg_two_phase, agg_merge_singletons, 0); 

    //hollywood-2009 0.5


#ifdef DEBUG
    if((unsigned)cluster_d.size()!= current_n_vertex)
      //LOG()<<"Error cluster_d.size()!= current_n_verte:qx"<< cluster_d.size() <<" != "<< current_n_vertex <<"\n";
#endif 

#ifdef VERBOSE
    //LOG()<<"n_vertex: "<< csr_ptr_d.size()<<" "<<csr_ind_d.size()<< " " << csr_val_d.size()<<" a_size: "<<aggregates.size()<<std::endl;
#endif

    hr_clock.start();
    size2_sector.setAggregates(cusp_handle, current_n_vertex, n_edges, csr_ptr_ptr, csr_ind_ptr, csr_val_ptr , aggregates, num_aggregates);
    CUDA_CALL(cudaDeviceSynchronize());
    hr_clock.stop(&timed);
    diff_time = timed;

    LOG()<<"Complete aggregation size: "<< num_aggregates<<" runtime: "<<diff_time/1000<<std::endl;

    // Done aggregates 
    c_size = num_aggregates;
    thrust::copy(thrust::device, aggregates.begin(), aggregates.begin() + current_n_vertex, cluster_d.begin());
    weighted = true;

    // start update modularty 
    hr_clock.start();
    CUDA_CALL(cudaDeviceSynchronize());

    generate_cluster_inv(current_n_vertex, c_size, cluster_d.begin(), cluster_inv_ptr, cluster_inv_ind);
    CUDA_CALL(cudaDeviceSynchronize());

    hr_clock.stop(&timed);
    diff_time = timed;

    LOG()<<"Complete generate_cluster_inv runtime: "<<diff_time/1000<<std::endl;


#ifdef VERBOSE     
      display_vec(cluster_inv_ptr, log);
      display_vec(cluster_inv_ind, log);
#endif


    hr_clock.start();
    new_Q = modularity(current_n_vertex, n_edges, c_size, m2,
                       csr_ptr_ptr, csr_ind_ptr, csr_val_ptr,
                       cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr,
                       weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr); //delta_Q_arr_ptr is temp_i and Q_arr is also temp store


    hr_clock.stop(&timed);
    diff_time = timed;
    // Done update modularity
   
    delta_Q = new_Q - cur_Q; 
    

    if(best_modularity < new_Q){
      best_c_size = c_size;
    }

    LOG()<<"modularity: "<<COLOR_MGT<<new_Q<<COLOR_WHT
       <<" delta modularity: " <<delta_Q
       <<" best_modularity:"<< min(best_modularity, new_Q)
       <<" moved: "<< (old_c_size - best_c_size)
       <<" runtime: "<<diff_time/1000<<std::endl;  


    // start shinking graph
    if(best_modularity < new_Q ){

      LOG()<< "Start Update best cluster\n";
      updated = true;
      num_level ++;

      thrust::copy(thrust::device, cluster_d.begin(), cluster_d.begin() + current_n_vertex, aggregates_tmp_d.begin());

      // if we would like to record the best cluster assignment for each level 
      // we push back current cluster assignment to cluster_vec
      //TODO

      best_modularity = new_Q;
      best_c_size = c_size;

      hr_clock.start(); 
      // generate super vertices graph 
      generate_superverticies_graph(current_n_vertex, best_c_size, 
                                    csr_ptr_d, csr_ind_d, csr_val_d, 
                                    new_csr_ptr, new_csr_ind, new_csr_val, 
                                    aggregates_tmp_d);

      CUDA_CALL(cudaDeviceSynchronize());
      if(current_n_vertex == num_vertex){
        // copy inital aggregates assignments as initial clustering
        thrust::copy(thrust::device, aggregates_tmp_d.begin(), aggregates_tmp_d.begin() +  current_n_vertex, clustering.begin());
      } 
      else{
        // update, clustering[i] = aggregates[clustering[i]];
        update_clustering((int)num_vertex, thrust::raw_pointer_cast(clustering.data()), thrust::raw_pointer_cast(aggregates_tmp_d.data()));
      }
      hr_clock.stop(&timed);
      diff_time = timed;
      LOG() <<"Complete generate_superverticies_graph size of graph: "<<current_n_vertex<<" -> "<<best_c_size<<" runtime: "<<diff_time/1000<<std::endl;
  
      // update cluster_d as a sequence
      thrust::sequence(thrust::cuda::par, cluster_d.begin(), cluster_d.begin() + current_n_vertex);  
      cudaCheckError(); 
    
      // generate cluster inv in CSR form as sequence
      thrust::sequence(thrust::cuda::par, cluster_inv_ptr.begin(), cluster_inv_ptr.begin() + best_c_size+1);
      thrust::sequence(thrust::cuda::par, cluster_inv_ind.begin(), cluster_inv_ind.begin() + best_c_size);

      cluster_inv_ptr_ptr = thrust::raw_pointer_cast(cluster_inv_ptr.data());
      cluster_inv_ind_ptr = thrust::raw_pointer_cast(cluster_inv_ind.data());
      
      //display_vec(cluster_inv_ind, log);
      hr_clock.start(); 
      // get new modularity after we generate super vertices. 
      IdxType* new_csr_ptr_ptr = thrust::raw_pointer_cast(new_csr_ptr.data());
      IdxType* new_csr_ind_ptr = thrust::raw_pointer_cast(new_csr_ind.data());
      ValType* new_csr_val_ptr = thrust::raw_pointer_cast(new_csr_val.data());


      new_Q = modularity( best_c_size, n_edges, best_c_size, m2,
                          new_csr_ptr_ptr, new_csr_ind_ptr, new_csr_val_ptr,
                          cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr,
                          weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr);
      
      hr_clock.stop(&timed);

      diff_time = timed;
 
      // modularity keeps the same after we generate super vertices 
      // shouldn't happen
      if(std::fabs(new_Q - best_modularity) > 0.0001){
        
        printf("Warning new_Q != best_Q %f != %f \n", new_Q, best_modularity);
#if 0
        printf("best_c_size = %d\n", best_c_size);

        std::ofstream ouf("./log/Error_"+time_now()+".log");
        display_vec(aggregates_tmp_d, ouf);
        ouf<<"Error new_Q != best_Q "<< new_Q<<" != "<< best_modularity<<"\n";
        ouf<<"old graph with size =  "<<current_n_vertex<< "\n";
        display_vec(csr_ptr_d, ouf);  
        display_vec(csr_ind_d, ouf);  
        display_vec(csr_val_d, ouf);  
 
        ouf<<"new graph \n";
        display_vec(new_csr_ptr, ouf);
        display_vec(new_csr_ind, ouf);
        display_vec(new_csr_val, ouf);

        generate_cluster_inv(current_n_vertex, c_size, aggregates_tmp_d.begin(), cluster_inv_ptr, cluster_inv_ind);

        ValType Q = modularity( current_n_vertex, n_edges, c_size, m2,
                        csr_ptr_d, csr_ind_d, csr_val_d, 
                        cluster_d, cluster_inv_ptr, cluster_inv_ind, 
                        weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr); // delta_Q_arr_ptr is temp_i
        CUDA_CALL(cudaDeviceSynchronize());

        LOG()<<Q<<std::endl;

  
        ouf<<"non block Q recompute "<< Q<<std::endl;
                   
        display_vec(Q_arr, ouf);
        display_vec(delta_Q_arr, ouf);

        ouf.close();

#endif
      } 

      LOG()<<"Update vectors and variables\n";
      
      
      if(cur_Q - new_Q && (bound < upper_bound)){
        current_n_vertex = best_c_size;
        n_edges = new_csr_ptr[ best_c_size ];
        thrust::copy(thrust::device, new_csr_ptr.begin(), new_csr_ptr.begin() + current_n_vertex + 1, csr_ptr_d.begin()); 
        thrust::copy(thrust::device, new_csr_ind.begin(), new_csr_ind.begin() + n_edges, csr_ind_d.begin());
        thrust::copy(thrust::device, new_csr_val.begin(), new_csr_val.begin() + n_edges, csr_val_d.begin());
      }

      //cudaMemGetInfo(&mem_free, &mem_tot);
      //std::cout<<"Mem usage : "<< (float)(mem_tot-mem_free)/(1<<30) <<std::endl;
    }else {
      LOG()<<"Didn't increase in modularity\n";
      updated = false;
      except --;
    }
    // end better   

    
    delta_Q_final = cur_Q - new_Q;

    contin = ((delta_Q_final > 0.0001 || except >0) && (bound < upper_bound));

    LOG()<<"======================= modularity: "<<COLOR_MGT<<new_Q<<COLOR_WHT<<" delta modularity: "<<delta_Q_final
         << " runtime: "<<diff_time/1000<<" best_c_size: "<<best_c_size <<std::endl;  

 
    ++bound;

  } while(contin);

#ifdef VERBOSE
    display_vec(cluster_d);   
    display_vec(csr_ptr_d);
    display_vec(csr_ind_d);
    display_vec(csr_val_d);

#endif

  //LOG()<<"Final modularity: "<<COLOR_MGT<<best_modularity<<COLOR_WHT<<std::endl;
  log.clear();  
  final_modularity = best_modularity;
  cudaMemcpy ( cluster_vec, thrust::raw_pointer_cast(clustering.data()), n_vertex*sizeof(int), cudaMemcpyDefault );
  return NVLOUVAIN_OK;
}

template<typename IdxType=int, typename ValType>
NVLOUVAIN_STATUS louvain(IdxType* csr_ptr, IdxType* csr_ind, ValType* csr_val,
                  const size_t num_vertex, const size_t num_edges, 
                  bool& weighted, bool has_init_cluster,
                  IdxType* init_cluster, // size = n_vertex
                  ValType& final_modularity,
                  std::vector< std::vector<int> >& cluster_vec,
//                  std::vector< IdxType* >& cluster_vec,
                  IdxType& num_level,
                  std::ostream& log = std::cout){
#ifndef ENABLE_LOG
  log.setstate(std::ios_base::failbit);
#endif
  num_level = 0;
  cusparseHandle_t cusp_handle;
  cusparseCreate(&cusp_handle);

  int n_edges = num_edges;
  int n_vertex = num_vertex;

  thrust::device_vector<IdxType> csr_ptr_d(csr_ptr, csr_ptr + n_vertex + 1);
  thrust::device_vector<IdxType> csr_ind_d(csr_ind, csr_ind + n_edges);
  thrust::device_vector<ValType> csr_val_d(csr_val, csr_val + n_edges);


  int upper_bound = 100;

  HighResClock hr_clock;
  double timed, diff_time;

  int c_size(n_vertex);
  unsigned int best_c_size = (unsigned) n_vertex;
  int current_n_vertex(n_vertex);
  int num_aggregates(n_edges);
  ValType m2 = thrust::reduce(thrust::cuda::par, csr_val_d.begin(), csr_val_d.begin() + n_edges);

  ValType best_modularity = -1;

  thrust::device_vector<IdxType> new_csr_ptr(n_vertex, 0);
  thrust::device_vector<IdxType> new_csr_ind(n_edges, 0);
  thrust::device_vector<ValType> new_csr_val(n_edges, 0);

  thrust::device_vector<IdxType> cluster_d(n_vertex);
  thrust::device_vector<IdxType> aggregates_tmp_d(n_vertex, 0);
  thrust::device_vector<IdxType> cluster_inv_ptr(c_size + 1, 0);
  thrust::device_vector<IdxType> cluster_inv_ind(n_vertex, 0);
  thrust::device_vector<ValType> k_vec(n_vertex, 0);
  thrust::device_vector<ValType> Q_arr(n_vertex, 0);
  thrust::device_vector<ValType> delta_Q_arr(n_edges, 0);
  thrust::device_vector<ValType> cluster_sum_vec(c_size, 0);
  std::vector<IdxType> best_cluster_h(n_vertex, 0);
  Vector<IdxType> aggregates(current_n_vertex, 0);

  IdxType* cluster_inv_ptr_ptr = thrust::raw_pointer_cast(cluster_inv_ptr.data());
  IdxType* cluster_inv_ind_ptr = thrust::raw_pointer_cast(cluster_inv_ind.data());
  IdxType* csr_ptr_ptr = thrust::raw_pointer_cast(csr_ptr_d.data());
  IdxType* csr_ind_ptr = thrust::raw_pointer_cast(csr_ind_d.data());
  ValType* csr_val_ptr = thrust::raw_pointer_cast(csr_val_d.data());
  IdxType* cluster_ptr = thrust::raw_pointer_cast(cluster_d.data());

 
 

  if(!has_init_cluster){
    // if there is no initialized cluster
    // the cluster as assigned as a sequence (a cluster for each vertex)
    // inv_clusters will also be 2 sequence
    thrust::sequence(thrust::cuda::par, cluster_d.begin(), cluster_d.end());
    thrust::sequence(thrust::cuda::par, cluster_inv_ptr.begin(), cluster_inv_ptr.end());
    thrust::sequence(thrust::cuda::par, cluster_inv_ind.begin(), cluster_inv_ind.end());
  }
  else{
    // assign initialized cluster to cluster_d device vector
    // generate inverse cluster in CSR formate
    if(init_cluster == nullptr){
      final_modularity = -1;
      return NVLOUVAIN_ERR_BAD_PARAMETERS;
    }

    thrust::copy(init_cluster, init_cluster + n_vertex , cluster_d.begin());
    generate_cluster_inv(current_n_vertex, c_size, cluster_d.begin(), cluster_inv_ptr, cluster_inv_ind);
  }
 
  dim3 block_size_1d((n_vertex + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
  dim3 grid_size_1d(BLOCK_SIZE_1D, 1, 1); 
  dim3 block_size_2d((n_vertex + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, (n_vertex + BLOCK_SIZE_2D -1)/ BLOCK_SIZE_2D, 1);
  dim3 grid_size_2d(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);

  ValType* k_vec_ptr = thrust::raw_pointer_cast(k_vec.data());
  ValType* Q_arr_ptr = thrust::raw_pointer_cast(Q_arr.data());
  ValType* cluster_sum_vec_ptr = thrust::raw_pointer_cast(cluster_sum_vec.data());
  ValType* delta_Q_arr_ptr =  thrust::raw_pointer_cast(delta_Q_arr.data());

  ValType new_Q, cur_Q, delta_Q, delta_Q_final;
  unsigned old_c_size(c_size); 
  bool updated = true;

  hr_clock.start();
  // Get the initialized modularity
  new_Q = modularity( n_vertex, n_edges, c_size, m2,
              csr_ptr_ptr, csr_ind_ptr, csr_val_ptr,
              cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr, 
              weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr); // delta_Q_arr_ptr is temp_i

  hr_clock.stop(&timed);
  diff_time = timed;

  LOG()<<"Initial modularity value: "<<COLOR_MGT<<new_Q<<COLOR_WHT<<" runtime: "<<diff_time/1000<<"\n";  

  bool contin(true);
  int bound = 0;
  int except = 3;

  
  do{ 
    bound = 0;  
    block_size_1d = dim3((current_n_vertex + BLOCK_SIZE_1D -1)/ BLOCK_SIZE_1D, 1, 1);
    grid_size_1d  = dim3(BLOCK_SIZE_1D, 1, 1); 
    cur_Q = new_Q;
    old_c_size = c_size;

#ifdef VERBOSE  
      LOG()<<"Current cluster inv: \n";
      nvlouvain::display_vec(cluster_inv_ptr, log);
      nvlouvain::display_vec(cluster_inv_ind, log);
#endif


    hr_clock.start();
    // Compute delta modularity for each edges
    build_delta_modularity_vector(cusp_handle, current_n_vertex, c_size, m2, updated, 
                                  csr_ptr_d, csr_ind_d, csr_val_d, 
                                  cluster_d, 
                                  cluster_inv_ptr_ptr, cluster_inv_ind_ptr, 
                                  k_vec_ptr, cluster_sum_vec_ptr, delta_Q_arr_ptr);
     
    //display_vec(delta_Q_arr);
    hr_clock.stop(&timed);
    diff_time = timed;
    LOG()<<"Complete build_delta_modularity_vector  runtime: "<<diff_time/1000<<"\n";

    //  Start aggregates 
    Matching_t config = nvlouvain::USER_PROVIDED;
//    Size2Selector<IdxType, ValType> size2_sector(config, 0, 50, 0.6, true, false, 0);
    Size2Selector<IdxType, ValType> size2_sector(config, 1, 25, 0.85, false, true, 0); 
    //hollywood-2009 0.5


#ifdef DEBUG
    if((unsigned)cluster_d.size()!= current_n_vertex)
      //LOG()<<"Error cluster_d.size()!= current_n_verte:qx"<< cluster_d.size() <<" != "<< current_n_vertex <<"\n";
#endif 

#ifdef VERBOSE
    //LOG()<<"n_vertex: "<< csr_ptr_d.size()<<" "<<csr_ind_d.size()<< " " << csr_val_d.size()<<" a_size: "<<aggregates.size()<<std::endl;
#endif

    hr_clock.start();
    size2_sector.setAggregates(cusp_handle, current_n_vertex, n_edges, csr_ptr_ptr, csr_ind_ptr, csr_val_ptr , aggregates, num_aggregates);
    CUDA_CALL(cudaDeviceSynchronize());
    hr_clock.stop(&timed);
    diff_time = timed;

    LOG()<<"Complete aggregation size: "<< num_aggregates<<" runtime: "<<diff_time/1000<<std::endl;

    // Done aggregates 
    c_size = num_aggregates;
    thrust::copy(thrust::device, aggregates.begin(), aggregates.begin() + current_n_vertex, cluster_d.begin());
    weighted = true;

    // start update modularty 
    hr_clock.start();
    CUDA_CALL(cudaDeviceSynchronize());

    generate_cluster_inv(current_n_vertex, c_size, cluster_d.begin(), cluster_inv_ptr, cluster_inv_ind);
    CUDA_CALL(cudaDeviceSynchronize());

    hr_clock.stop(&timed);
    diff_time = timed;

    LOG()<<"Complete generate_cluster_inv runtime: "<<diff_time/1000<<std::endl;


#ifdef VERBOSE   
      display_vec(cluster_inv_ptr, log);
      display_vec(cluster_inv_ind, log);
#endif


    hr_clock.start();
    new_Q = modularity(current_n_vertex, n_edges, c_size, m2,
                       csr_ptr_ptr, csr_ind_ptr, csr_val_ptr,
                       cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr,
                       weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr); //delta_Q_arr_ptr is temp_i and Q_arr is also temp store


    hr_clock.stop(&timed);
    diff_time = timed;
    // Done update modularity
   
    delta_Q = new_Q - cur_Q; 
    
    if(best_modularity < new_Q){
      best_c_size = c_size;
    }

    LOG()<<"modularity: "<<COLOR_MGT<<new_Q<<COLOR_WHT
       <<" delta modularity: " <<delta_Q
       <<" best_modularity:"<< min(best_modularity, new_Q)
       <<" moved: "<< (old_c_size - best_c_size)
       <<" runtime: "<<diff_time/1000<<std::endl;  


    // start shinking graph
    if(best_modularity < new_Q ){

      LOG()<< "Start Update best cluster\n";
      updated = true;
      num_level ++;

      thrust::copy(thrust::device, cluster_d.begin(), cluster_d.begin() + current_n_vertex, aggregates_tmp_d.begin());

      // if we would like to record the best cluster assignment for each level 
      // we push back current cluster assignment to cluster_vec
    

      best_cluster_h.resize(current_n_vertex);
      thrust::copy( cluster_d.begin(), cluster_d.begin() + current_n_vertex, best_cluster_h.begin());
      cudaCheckError();
      cluster_vec.push_back(best_cluster_h);

      best_modularity = new_Q;
      best_c_size = c_size;

      hr_clock.start(); 
      // generate super vertices graph 
      generate_superverticies_graph(current_n_vertex, best_c_size, 
                                    csr_ptr_d, csr_ind_d, csr_val_d, 
                                    new_csr_ptr, new_csr_ind, new_csr_val, 
                                    aggregates_tmp_d);

      CUDA_CALL(cudaDeviceSynchronize());
      hr_clock.stop(&timed);
      diff_time = timed;
      LOG() <<"Complete generate_superverticies_graph size of graph: "<<current_n_vertex<<" -> "<<best_c_size<<" runtime: "<<diff_time/1000<<std::endl;
  
      // update cluster_d as a sequence
      thrust::sequence(thrust::cuda::par, cluster_d.begin(), cluster_d.begin() + current_n_vertex);  
      cudaCheckError(); 
    
      // generate cluster inv in CSR form as sequence
      thrust::sequence(thrust::cuda::par, cluster_inv_ptr.begin(), cluster_inv_ptr.begin() + best_c_size+1);
      thrust::sequence(thrust::cuda::par, cluster_inv_ind.begin(), cluster_inv_ind.begin() + best_c_size);

      cluster_inv_ptr_ptr = thrust::raw_pointer_cast(cluster_inv_ptr.data());
      cluster_inv_ind_ptr = thrust::raw_pointer_cast(cluster_inv_ind.data());

      hr_clock.start(); 
      // get new modularity after we generate super vertices. 
      IdxType* new_csr_ptr_ptr = thrust::raw_pointer_cast(new_csr_ptr.data());
      IdxType* new_csr_ind_ptr = thrust::raw_pointer_cast(new_csr_ind.data());
      ValType* new_csr_val_ptr = thrust::raw_pointer_cast(new_csr_val.data());


      new_Q = modularity( best_c_size, n_edges, best_c_size, m2,
                          new_csr_ptr_ptr, new_csr_ind_ptr, new_csr_val_ptr,
                          cluster_ptr, cluster_inv_ptr_ptr, cluster_inv_ind_ptr,
                          weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr);
      
      hr_clock.stop(&timed);

      diff_time = timed;
 
      // modularity keeps the same after we generate super vertices 
      // shouldn't happen
      if(std::fabs(new_Q - best_modularity) > 0.0001){
        
        printf("Warning new_Q != best_Q %f != %f \n", new_Q, best_modularity);
#if 0
        printf("best_c_size = %d\n", best_c_size);

        std::ofstream ouf("./log/Error_"+time_now()+".log");
        display_vec(aggregates_tmp_d, ouf);
        ouf<<"Error new_Q != best_Q "<< new_Q<<" != "<< best_modularity<<"\n";
        ouf<<"old graph with size =  "<<current_n_vertex<< "\n";
        display_vec(csr_ptr_d, ouf);  
        display_vec(csr_ind_d, ouf);  
        display_vec(csr_val_d, ouf);  
 
        ouf<<"new graph \n";
        display_vec(new_csr_ptr, ouf);
        display_vec(new_csr_ind, ouf);
        display_vec(new_csr_val, ouf);

        generate_cluster_inv(current_n_vertex, c_size, aggregates_tmp_d.begin(), cluster_inv_ptr, cluster_inv_ind);

        ValType Q = modularity( current_n_vertex, n_edges, c_size, m2,
                        csr_ptr_d, csr_ind_d, csr_val_d, 
                        cluster_d, cluster_inv_ptr, cluster_inv_ind, 
                        weighted, k_vec_ptr, Q_arr_ptr, delta_Q_arr_ptr); // delta_Q_arr_ptr is temp_i
        CUDA_CALL(cudaDeviceSynchronize());

        LOG()<<Q<<std::endl;

  
        ouf<<"non block Q recompute "<< Q<<std::endl;
                   
        display_vec(Q_arr, ouf);
        display_vec(delta_Q_arr, ouf);

        ouf.close();

#endif
      } 

      LOG()<<"Update vectors and variables\n";
      
      
      if(cur_Q - new_Q && (bound < upper_bound)){
        current_n_vertex = best_c_size;
        n_edges = new_csr_ptr[ best_c_size ];
        thrust::copy(thrust::device, new_csr_ptr.begin(), new_csr_ptr.begin() + current_n_vertex + 1, csr_ptr_d.begin()); 
        thrust::copy(thrust::device, new_csr_ind.begin(), new_csr_ind.begin() + n_edges, csr_ind_d.begin());
        thrust::copy(thrust::device, new_csr_val.begin(), new_csr_val.begin() + n_edges, csr_val_d.begin());
      }

    }else {
      LOG()<<"Didn't increase in modularity\n";
      updated = false;
      except --;
    }
    // end better   

    
    delta_Q_final = cur_Q - new_Q;

    contin = ((delta_Q_final > 0.0001 || except >0) && (bound < upper_bound));

    LOG()<<"======================= modularity: "<<COLOR_MGT<<new_Q<<COLOR_WHT<<" delta modularity: "<<delta_Q_final
         << " runtime: "<<diff_time/1000<<" best_c_size: "<<best_c_size <<std::endl;  

 
    ++bound;

  }while(contin);

#ifdef VERBOSE
    display_vec(cluster_d);   
    display_vec(csr_ptr_d);
    display_vec(csr_ind_d);
    display_vec(csr_val_d);
#endif

  //LOG()<<"Final modularity: "<<COLOR_MGT<<best_modularity<<COLOR_WHT<<std::endl;
  log.clear();  
  final_modularity = best_modularity;
  return NVLOUVAIN_OK; 
}


}
