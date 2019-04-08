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

#ifndef GRAPH_CONTRACTING_VISITOR_HXX
#define GRAPH_CONTRACTING_VISITOR_HXX
//
//

#include <multi_valued_csr_graph.hxx> //which includes all other headers... 
#include <range_view.hxx> // TODO: to be changed to thrust/range_view.h, when toolkit gets in sync with Thrust
#include <thrust_traits.hxx>
///#include <graph_contracting_structs.hxx>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>//
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/distance.h>//
#include <thrust/unique.h>//

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/transpose.h>//

//debugging only:
#include <cstdio>

#define __CUDA_ARCH_THRESHOLD__ 300
///#define __CUDA_ARCH_THRESHOLD__ 350
//
namespace nvgraph
{
  


  //SpMv + SpMM + SpMM:
  //  cntrctd_vertex_data = S*v(g_vertex_data);
  //  cntrctd_edge_data   = (S*G(g_edge_data)*St).values
  //
  //see GraphContractionFunctor::computeRestrictionOperator() for S matrix CSR data
  //
  template<typename VectorI,            //vector type for indices
           typename VectorV,            //vector type for values 
           typename VWrapper,           //wrapper type around raw pointer or other type of array wrapper
           typename VertexCombineFctr,  //vertex "multiplication" functor type
           typename VertexReduceFctr,   //vertex "addition" functor type
           typename EdgeCombineFctr,    //edge "multiplication" functor type
           typename EdgeReduceFctr>     //edge "addition" functor type
  struct SemiringContractionUtilities
  {
    typedef typename VectorI::value_type IndexT;
    typedef typename VectorV::value_type ValT;

  typedef typename VectorPtrT<typename VectorI::value_type,VectorI>::PtrT PtrI;
  typedef typename VectorPtrT<typename VectorV::value_type,VectorV>::PtrT PtrV;

    SemiringContractionUtilities(const VectorI& g_row_offsets, //original graph CSR 
                                 const VectorI& g_col_indices,
                                 const VectorI& S_row_offsets,
                                 const VectorI& S_col_indices,
                                 VertexCombineFctr& v_combine,
                                 VertexReduceFctr&  v_reduce,
                                 EdgeCombineFctr&   e_combine,
                                 EdgeReduceFctr&    e_reduce):
      m_g_row_offsets(g_row_offsets),
      m_g_col_indices(g_col_indices),
      m_v_combine(v_combine),
      m_v_reduce(v_reduce),
      m_e_combine(e_combine),
      m_e_reduce(e_reduce),
      m_n_agg(S_row_offsets.size()-1),
      m_g_nr(g_row_offsets.size()-1), // == S_nc
      m_g_nnz(g_row_offsets.back()),
      m_s_nnz(S_row_offsets.back())
    { 
      VectorV S_vals(m_s_nnz, 1);

      PtrV p_S_vals(S_vals.data().get());
      VWrapper S_vals_w(p_S_vals, p_S_vals+S_vals.size());

      //NOT necessarily square!
      m_S = make_csr_matrix(m_g_nr, S_row_offsets, S_col_indices, S_vals_w);

      m_St = cusp::csr_matrix<IndexT, ValT, cusp::device_memory>(m_g_nr, m_n_agg, m_s_nnz);
      cusp::transpose(m_S, m_St);
      cudaCheckError();
    }

    virtual ~SemiringContractionUtilities(void)
    {
    }

    const VectorI& get_row_ptr(void) const
    {
      return m_cntrctd_row_offsets;
    }
    
    const VectorI& get_col_ind(void) const
    {
      return m_cntrctd_col_indices;
    }

    IndexT get_subg_nnz(void) const
    {
      return m_cntrctd_row_offsets.back();
    }

    virtual void update_vertex_data(/*In: */const VWrapper& g_vertex_data,//multivalue vertex entry of original graph, size==g_nr
                                    /*Out:*/VWrapper& cntrctd_vertex_data)//multivalue vertex entry of contracted graph, size==n_agg==S_nr (assumed allocated!)
    {
      //SpMv:
      //
      assert( m_g_nr == g_vertex_data.size() );
      cusp::array1d<ValT, cusp::device_memory> x(g_vertex_data.cbegin(), g_vertex_data.cend());
      cusp::array1d<ValT, cusp::device_memory> y(m_n_agg,0);

      cusp::constant_functor<ValT> initialize;
      cusp::multiply(m_S, x, y, initialize, m_v_combine, m_v_reduce);
      cudaCheckError();

      thrust::copy(y.begin(), y.end(), cntrctd_vertex_data.begin());
      cudaCheckError();
    }

    virtual void update_topology_only(void)
    {
      cudaCheckError();
      //SpMM+SpMM: S*G*St
      //
      VectorV empty(m_g_nnz, 1);//0 => empty G matrix, use 1's as values

      PtrV ptr_e(&empty[0]);
      VWrapper g_edge_data(ptr_e, ptr_e+m_g_nnz);
      cudaCheckError();
      
      cusp::csr_matrix<IndexT, ValT, cusp::device_memory> G =
        make_square_csr_matrix(m_g_row_offsets, m_g_col_indices, g_edge_data);
      cudaCheckError();

      cusp::constant_functor<ValT> initialize;

      //L=S*G
      cusp::csr_matrix<IndexT, ValT, cusp::device_memory> L;//no need to allocate!
      cusp::multiply(m_S, G, L, initialize, m_e_combine, m_e_reduce);
      cudaCheckError();

      //R = L*St
      cusp::csr_matrix<IndexT, ValT, cusp::device_memory> R;//no need to allocate!
      cusp::multiply(L, m_St, R, initialize, m_e_combine, m_e_reduce);
      cudaCheckError();

      //##### debug:
      //std::cout<<"S:\n";cusp::print(m_S);
      //std::cout<<"R:\n";cusp::print(R);

      size_t r_sz = R.row_offsets.size();
      assert( r_sz > 0 );
  
      size_t cntrctd_nnz = R.row_offsets.back();
      ///size_t cntrctd_nr = r_sz-1;

      //allocate cntrctd_csr_data:
      m_cntrctd_row_offsets = VectorI(r_sz, 0);
      m_cntrctd_col_indices = VectorI(cntrctd_nnz, 0);

      thrust::copy(R.row_offsets.begin(), R.row_offsets.end(), m_cntrctd_row_offsets.begin());
      cudaCheckError();
      thrust::copy(R.column_indices.begin(), R.column_indices.end(), m_cntrctd_col_indices.begin());
      cudaCheckError();
    }

    virtual void update_edge_data(/*In: */const VWrapper& g_edge_data,  //multivalue edge entry of original graph, size==g_nnz
                                  /*Out:*/VWrapper& cntrctd_edge_data)  //multivalue edge entry of contracted graph, size==nnz(S*G*St) (assumed allocated!)
    {
      //SpMM+SpMM: S*G*St
      //
      assert( m_g_nnz == g_edge_data.size() );
      cusp::csr_matrix<IndexT, ValT, cusp::device_memory> G =
        make_square_csr_matrix(m_g_row_offsets, m_g_col_indices, g_edge_data);
      cudaCheckError();

      cusp::constant_functor<ValT> initialize;
      cudaCheckError();

      //L=S*G
      cusp::csr_matrix<IndexT, ValT, cusp::device_memory> L;//no need to allocate!
      cusp::multiply(m_S, G, L, initialize, m_e_combine, m_e_reduce);
      cudaCheckError();

      //R = L*St //##### crash here:
      cusp::csr_matrix<IndexT, ValT, cusp::device_memory> R;//no need to allocate!
      cusp::multiply(L, m_St, R, initialize, m_e_combine, m_e_reduce);
      cudaCheckError();

      size_t r_sz = R.row_offsets.size();
      assert( r_sz > 0 );
  
      size_t cntrctd_nnz = R.row_offsets.back();
      ///size_t cntrctd_nr = r_sz-1;

      //allocate cntrctd_csr_data:
      m_cntrctd_row_offsets = VectorI(r_sz, 0);
      m_cntrctd_col_indices = VectorI(cntrctd_nnz, 0);

      thrust::copy(R.row_offsets.begin(), R.row_offsets.end(), m_cntrctd_row_offsets.begin());
      cudaCheckError();
      
      thrust::copy(R.column_indices.begin(), R.column_indices.end(), m_cntrctd_col_indices.begin());
      cudaCheckError();
      
      thrust::copy(R.values.begin(), R.values.end(), cntrctd_edge_data.begin());
      cudaCheckError();
    }

    virtual void update_all(/*In: */const VWrapper& g_vertex_data,//multivalue vertex entry of original graph, size==g_nr
                            /*Out:*/VWrapper& cntrctd_vertex_data,//multivalue vertex entry of contracted graph, size==n_agg==S_nr (assumed allocated!)
                            /*In: */const VWrapper& g_edge_data,  //multivalue edge entry of original graph, size==g_nnz
                            /*Out:*/VWrapper& cntrctd_edge_data)  //multivalue edge entry of contracted graph, size==nnz(S*G*St) (assumed allocated!)
    {
      update_vertex_data(g_vertex_data, cntrctd_vertex_data);
      update_edge_data(g_edge_data, cntrctd_edge_data);
    }

  protected:
    static cusp::csr_matrix<IndexT,
                            ValT,
                            cusp::device_memory>
    make_csr_matrix(size_t nc,
                    const VectorI& row_offsets,
                    const VectorI& col_indices,
                    const VWrapper& vals)
    {
      size_t nr  = row_offsets.size()-1;
      size_t nz = row_offsets.back();

      cusp::csr_matrix<IndexT, ValT, cusp::device_memory> A(nr, nc, nz);

      //copy:
      //
      A.row_offsets    = row_offsets;
      A.column_indices = col_indices;

      thrust::copy(vals.cbegin(), vals.cend(), A.values.begin());
      cudaCheckError();

      return A;
    }

    static cusp::csr_matrix<IndexT,
                            ValT,
                            cusp::device_memory>
    make_square_csr_matrix(const VectorI& row_offsets,
                           const VectorI& col_indices,
                           const VWrapper& vals)
    {
      size_t nc  = row_offsets.size()-1;

      return make_csr_matrix(nc, row_offsets, col_indices, vals);
    }
    
  private:
    //Input:
    //
    const VectorI& m_g_row_offsets; //original graph CSR data:
    const VectorI& m_g_col_indices;
    cusp::csr_matrix<IndexT, ValT, cusp::device_memory> m_S; //aggreagate matrix
    cusp::csr_matrix<IndexT, ValT, cusp::device_memory> m_St; //aggreagate matrix transpose

    //Output:
    //
    VectorI m_cntrctd_row_offsets;  //contracted graph CSR data:
    VectorI m_cntrctd_col_indices;

    //I/O:
    //
    VertexCombineFctr& m_v_combine; //vertex "multiplication" functor
    VertexReduceFctr&  m_v_reduce;  //vertex "addition" functor
    EdgeCombineFctr& m_e_combine;   //edge "multiplication" functor
    EdgeReduceFctr& m_e_reduce;     //edge "addition" functor

    const size_t m_n_agg;
    const size_t m_g_nr; // == S_nc
    const size_t m_g_nnz;
    const size_t m_s_nnz;
    
  };

  //generic value updater
  //
  template<typename VectorV,            //Vector of values
       typename VectorI,            //Vector of indices
           typename VertexCombineFctr,  //vertex "multiplication" functor type
           typename VertexReduceFctr,   //vertex "addition" functor type
           typename EdgeCombineFctr,    //edge "multiplication" functor type
           typename EdgeReduceFctr,     //edge "addition" functor type
       size_t CTA_SIZE>             //only used by the specialized template
  struct ContractionValueUpdater
  {
    typedef typename VectorI::value_type IndexT;
    //typedef typename VectorPtrT<typename VectorI::value_type,VectorV>::PtrT PtrI;

    typedef typename VectorV::value_type ValueT;
    typedef typename VectorPtrT<typename VectorV::value_type,VectorV>::PtrT PtrV;

  //TODO: make template argument:
    typedef range_view<PtrV> VWrapper;

    //v_src, v_dest assumed pre-allocated!
    //
    ContractionValueUpdater(/*const */VectorV& v_src,
              VectorV& v_dest,
                            VertexCombineFctr& v_combine,
                            VertexReduceFctr&  v_reduce,
                            EdgeCombineFctr&   e_combine,
                            EdgeReduceFctr&    e_reduce):
      v_s_(v_src),
      v_d_(v_dest),
      m_v_combine(v_combine),
      m_v_reduce(v_reduce),
      m_e_combine(e_combine),
      m_e_reduce(e_reduce)
    {
    }

    //TODO: more efficient solution with VWrapper, to avoid device memory traffic
    //
    void update_from(///Hash_Workspace<IndexT,ValueT>& hash_wk,//only used by the specialized template
           ///size_t num_aggregates,//only used by the specialized template
           const VectorI& R_row_offsets,
           const VectorI& R_column_indices,
           const VectorI& g_row_offsets,
           const VectorI& g_col_indices)
           ///const VectorI& aggregates,//only used by the specialized template
           ///const VectorI& cg_row_offsets,//only used by the specialized template
           ///const VectorI& cg_col_indices,//only used by the specialized template
           ///const VectorI& Ac_pos)//only used by the specialized template
    {
      // PtrI ptr(&seq[0]);  
      // int* raw_ptr = ptr.get();
      // PtrI ptr0(raw_ptr);
      // range_view<PtrI> rv0(ptr0, ptr0+n);

      size_t n_s = v_s_.size();
      PtrV ptr_src(&v_s_[0]);
      //ValueT* p_s = v_s_.data().get();
      VWrapper g_edge_data(ptr_src, ptr_src+n_s);
      ///VWrapper g_edge_data(v_s_.cbegin(), v_s_.cend());//nope...

      size_t n_d = v_d_.size();
      PtrV ptr_dst(&v_d_[0]);
      //ValueT* p_d = v_d_.data().get();
      VWrapper cg_edge_data(ptr_dst, ptr_dst+n_d);
      //R == S
      //
      SemiringContractionUtilities<VectorI, VectorV, VWrapper,VertexCombineFctr,VertexReduceFctr,EdgeCombineFctr,EdgeReduceFctr>
        sr(g_row_offsets,
           g_col_indices,
           R_row_offsets,
           R_column_indices,
           m_v_combine,
           m_v_reduce,
           m_e_combine,
           m_e_reduce);

      sr.update_edge_data(g_edge_data, cg_edge_data);
    }
      
    const VectorV& get_cg_vals(void) const
    {
      return v_d_;
    }
  private:
    /*const */VectorV& v_s_;
    VectorV& v_d_;

    VertexCombineFctr& m_v_combine;
    VertexReduceFctr&  m_v_reduce;
    EdgeCombineFctr&   m_e_combine;
    EdgeReduceFctr&    m_e_reduce;
  };

  //partial specialization for (Combine, Reduce) == (*,+)
  //
  // template<typename VectorV, 
  //        typename VectorI,
  //        size_t CTA_SIZE>
  // struct ContractionValueUpdater<VectorV,
  //                                VectorI,
  //                                thrust::multiplies<typename VectorV::value_type>,
  //                                thrust::plus<typename VectorV::value_type>,
  //                                thrust::multiplies<typename VectorV::value_type>,
  //                                thrust::plus<typename VectorV::value_type>,
  //                                CTA_SIZE>
  // {
  //   typedef typename VectorI::value_type IndexT;
  //   //typedef typename VectorPtrT<typename VectorI::value_type,VectorV>::PtrT PtrI;

  //   typedef typename VectorV::value_type ValueT;
  //   typedef typename VectorPtrT<typename VectorV::value_type,VectorV>::PtrT PtrV;

  //   //v_src, v_dest assumed pre-allocated!
  //   //
  //   ContractionValueUpdater(/*const */VectorV& v_src,
  //               VectorV& v_dest,
  //                           thrust::multiplies<ValueT>& ,
  //                           thrust::plus<ValueT>&  ,
  //                           thrust::multiplies<ValueT>& ,
  //                           thrust::plus<ValueT>& ):
  //     v_s_(v_src),
  //     v_d_(v_dest)
  //   {
  //   }

  //   void update_from(Hash_Workspace<IndexT,ValueT>& hash_wk,
  //            size_t num_aggregates,
  //            const VectorI& R_row_offsets,
  //            const VectorI& R_column_indices,
  //            const VectorI& g_row_offsets,
  //            const VectorI& g_col_indices,
  //            const VectorI& aggregates,
  //            const VectorI& cg_row_offsets,
  //            const VectorI& cg_col_indices,
  //            const VectorI& Ac_pos)
  //   {
  //     fill_A_dispatch<CTA_SIZE>(hash_wk,
  //                 num_aggregates, 
  //                 R_row_offsets.data().get(), 
  //                 R_column_indices.data().get(), 
  //                 g_row_offsets.data().get(),
  //                 g_col_indices.data().get(),
  //                 v_s_.data().get(),
  //                 aggregates.data().get(), 
  //                 cg_row_offsets.data().get(), 
  //                 cg_col_indices.data().get(), 
  //                 thrust::raw_pointer_cast( &Ac_pos.front() ),
  //                 v_d_.data().get());
  //     cudaCheckError();
  //   }
      
  //   const VectorV& get_cg_vals(void) const
  //   {
  //     return v_d_;
  //   }
  // private:
  //   /*const */VectorV& v_s_;
  //   VectorV& v_d_;
  // };

  


  template<typename VectorI,
       typename VectorV,
           typename VertexCombineFctr,  //vertex "multiplication" functor type
           typename VertexReduceFctr,   //vertex "addition" functor type
           typename EdgeCombineFctr,    //edge "multiplication" functor type
           typename EdgeReduceFctr,     //edge "addition" functor type
       typename VectorB = VectorI,
       size_t CTA_SIZE = 128>
  struct GraphContractionFunctor
  {
    typedef typename VectorI::value_type IndexT;
    typedef typename VectorV::value_type ValueT;
    typedef typename VectorB::value_type ValueB;

    typedef typename VectorPtrT<typename VectorB::value_type,VectorB>::PtrT PtrB;
    typedef typename VectorPtrT<typename VectorI::value_type,VectorI>::PtrT PtrI;
    typedef typename VectorPtrT<typename VectorV::value_type,VectorV>::PtrT PtrV;
    //       num_aggregates != m_aggregates.size()!!!
    //       Need m_num_aggregates const member
    //
    GraphContractionFunctor(size_t g_n_vertices,
                            const VectorI& aggregates, /*const */
                            size_t num_aggregates,
                            VertexCombineFctr& v_combine,
                            VertexReduceFctr&  v_reduce,
                            EdgeCombineFctr&   e_combine,
                            EdgeReduceFctr&    e_reduce):
      m_num_rows(g_n_vertices), 
      m_aggregates(aggregates),
      m_num_aggregates(num_aggregates),
      m_v_combine(v_combine),
      m_v_reduce(v_reduce),
      m_e_combine(e_combine),
      m_e_reduce(e_reduce)
    {
      computeRestrictionOperator();
      cudaCheckError();
    }

    virtual ~GraphContractionFunctor(void)
    {
    }

    const VectorI& get_aggregates(void) const
    {
      return m_aggregates;
    }

    size_t get_num_aggregates(void) const
    {
      return m_num_aggregates;
    }

    const VectorI& get_R_row_offsets(void) const
    {
      return m_R_row_offsets;
    }
    
    const VectorI& get_R_column_indices(void) const
    {
       return m_R_column_indices;
    }

  VertexCombineFctr& get_v_combine(void)
  {
    return m_v_combine;
  }

    VertexReduceFctr&  get_v_reduce(void)
  {
    return m_v_reduce;
  }

    EdgeCombineFctr&   get_e_combine(void)
  {
    return m_e_combine;
  }

    EdgeReduceFctr&    get_e_reduce(void)
  {
    return m_e_reduce;
  }
    
  protected:
    void computeRestrictionOperator(void)
    {
      size_t n_aggregates = m_num_aggregates;//nope: m_aggregates.size();
      m_R_row_offsets.resize(n_aggregates+1);//create one more row for the pseudo aggregate (?)
      VectorI R_row_indices(m_aggregates);

      m_R_column_indices.resize(m_num_rows);
      thrust::sequence(m_R_column_indices.begin(),m_R_column_indices.end());
      cudaCheckError();

      thrust::sort_by_key(R_row_indices.begin(),R_row_indices.end(),m_R_column_indices.begin());
      cudaCheckError();

      thrust::lower_bound(R_row_indices.begin(),
        R_row_indices.end(),
        thrust::counting_iterator<ValueT>(0),
        thrust::counting_iterator<ValueT>(m_R_row_offsets.size()),
        m_R_row_offsets.begin());
      cudaCheckError();
    }

    //code "parked" for the time being;
    //it uses the AMGX approach which has a bug
    //un-debuggable due to nvcc failure with -g -G pair
    //(bug: https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=1813290&cmtNo)
    //
    struct NoValueUpdater
  {
    void update_from(///Hash_Workspace<IndexT,ValueT>& hash_wk,
                       ///size_t num_aggregates,
                       const VectorI& R_row_offsets,
                       const VectorI& R_column_indices,
                       const VectorI& g_row_offsets,
                       const VectorI& g_col_indices)
           ///const VectorI& aggregates,
           ///const VectorI& cg_row_offsets,
           ///const VectorI& cg_col_indices,
           ///const VectorI& Ac_pos)
    {
    //no-op...
    }
  };

    virtual void operator() (VectorI& g_row_ptr_,
               VectorI& g_col_ind_)
    {
    NoValueUpdater updater;//dummy object...

    contract(g_row_ptr_, g_col_ind_, updater);
    }

    virtual void operator () (VectorV& g_vals_,
                VectorI& g_row_ptr_,
                VectorI& g_col_ind_)
    {
    ContractionValueUpdater<VectorV,
                              VectorI,
                              VertexCombineFctr,
                              VertexReduceFctr,
                              EdgeCombineFctr,
                              EdgeReduceFctr,
                              CTA_SIZE>
        updater(g_vals_,
                m_cg_values,
                m_v_combine,
                m_v_reduce,
                m_e_combine,
                m_e_reduce);
      
    contract(g_row_ptr_, g_col_ind_, updater); 
    }
    
    const VectorI& get_row_ptr(void) const
    {
      return m_cg_row_offsets;
    }
    
    const VectorI& get_col_ind(void) const
    {
      return m_cg_col_indices;
    }

    IndexT get_subg_nnz(void) const
    {
      return m_cg_row_offsets.back();
    }

    template<typename ValUpdaterFctr>
    void contract(VectorI& g_row_offsets, //contracted
      VectorI& g_col_indices, //contracted
      ValUpdaterFctr fctrv)
    {
      //notation mapping from AMGX->nvGRAPH:
      //
      //S (Restriction) matrix data:
      //R_row_offsets          -> m_R_row_offsets
      //R_column_indices       -> m_R_column_indices
      //
      //Graph matrix data:
      //A.row_offsets          -> g_row_offsets
      //A.col_indices          -> g_col_indices
      //
      //Contracted matrix data:
      //Ac.row_offsets         -> m_cg_row_offsets
      //Ac.col_indices         -> m_cg_col_indices
      //
      //num_aggregates != m_aggregates.size()!!!
      //
      ///size_t num_aggregates = m_aggregates.size(); //nope...
      //size_t sz_aggregates = m_aggregates.size();
      // TODO: check why no size() for amgx::IVector

      m_cg_row_offsets.resize( m_num_aggregates+1 );

      //##### update topology:
      //{
      // Hash_Workspace<IndexT,ValueT> hash_wk;

      // compute_sparsity_dispatch<CTA_SIZE, false, true>(hash_wk, 
      //                                                  m_num_aggregates,//????? 
      //                                                  m_R_row_offsets.data().get(), 
      //                                                  m_R_column_indices.data().get(), 
      //                                                  g_row_offsets.data().get(), 
      //                                                  g_col_indices.data().get(), 
      //                                                  m_aggregates.data().get(), 
      //                                                  m_cg_row_offsets.data().get(),
      //                                                  static_cast<IndexT*>(0), //ok
      //                                                  static_cast<IndexT*>(0));//ok
      // cudaCheckError();

      // // Compute the number of non-zeroes.
      // thrust::exclusive_scan( m_cg_row_offsets.begin(), m_cg_row_offsets.end(), m_cg_row_offsets.begin() );
      // cudaCheckError();

      ///IndexT nonzero_blocks = m_cg_row_offsets[m_num_aggregates];

      // // Vector to store the positions in the hash table.
      ///VectorI Ac_pos(nonzero_blocks);

      // compute_sparsity_dispatch<CTA_SIZE, false, false>(hash_wk, 
      //                                                   m_num_aggregates,///????? 
      //                                                   m_R_row_offsets.data().get(), 
      //                                                   m_R_column_indices.data().get(), 
      //                                                   g_row_offsets.data().get(), 
      //                                                   g_col_indices.data().get(), 
      //                                                   m_aggregates.data().get(), 
      //                                                   m_cg_row_offsets.data().get(), 
      //                                                   m_cg_col_indices.data().get(),
      //                                                   thrust::raw_pointer_cast( &Ac_pos.front() ));
      // cudaCheckError();
      //} end update topology

      //##### update values:
      //{
      //act (or not) on values:
      //
      fctrv.update_from(///hash_wk,
            ///m_num_aggregates,///????? 
            m_R_row_offsets, 
            m_R_column_indices,
            g_row_offsets, 
            g_col_indices);
            ///m_aggregates, 
            ///m_cg_row_offsets, 
            ///m_cg_col_indices, 
            ///Ac_pos);
      //}end update values
      
    }

  private:
    size_t m_num_rows;    // number of vertices in the original graph
    VectorI m_aggregates; // labels of vertices to be collapsed (vertices with same label will be collapsed into one)
    const size_t m_num_aggregates; // != m_aggregates.size() !!!

    //Restrictor CSR info
    //Restrictor = S "matrix" in algorithm 4.5 in "Graph Algorithms in the language of Linear Algebra")
    VectorI m_R_row_offsets;
    VectorI m_R_column_indices;

    //Contracted graph data:
    VectorI m_cg_row_offsets;
    VectorI m_cg_col_indices;
    VectorV m_cg_values;

    //Contraction functors:
    //
    VertexCombineFctr& m_v_combine;
    VertexReduceFctr&  m_v_reduce;
    EdgeCombineFctr&   m_e_combine;
    EdgeReduceFctr&    m_e_reduce;
  };

namespace{ //unnamed..
  template<typename VectorI>
  size_t validate_contractor_input(const VectorI& v, size_t g_nrows)
  {
    typedef typename VectorI::value_type IndexT;
    typedef typename VectorI::iterator Iterator;

    size_t n = v.size();

    if( n == 0 )
      FatalError("0-sized array input in graph contraction.",NVGRAPH_ERR_BAD_PARAMETERS);

     if( n != g_nrows )
      FatalError("Aggregate array size must match number of vertices of original graph",NVGRAPH_ERR_BAD_PARAMETERS);

     //find min/max values in aggregates...
     //and check if min==0 and max <= g_nrows-1...
     VectorI res(v);//copy
     cudaCheckError();
     thrust::pair<Iterator, Iterator> result = thrust::minmax_element(res.begin(), res.end());
     if( *result.first != 0 )
       FatalError("Aggregate array values must start from 0.",NVGRAPH_ERR_BAD_PARAMETERS);
     cudaCheckError();

     if( static_cast<size_t>(*result.second) > g_nrows-1 )
       FatalError("Aggregate array values must be less than number of vertices of original graph.",NVGRAPH_ERR_BAD_PARAMETERS);

     //then make sure all values in between are covered...
     //use count_distinct() and see if there are max-min+1
     size_t n_expected = *result.second - *result.first + 1;
     
     thrust::sort(res.begin(), res.end());
     cudaCheckError();
     size_t counts = thrust::distance(res.begin(), thrust::unique(res.begin(), res.end()));
     cudaCheckError();

     if( counts != n_expected )
       FatalError("Aggregate array intermediate values (between 0 and max(aggregates)) are missing.",NVGRAPH_ERR_BAD_PARAMETERS);

     //return # aggregates (not to be confused with aggregates.size()!)
     return n_expected;
  }
}//end unnamed namespace


  //(the C header will have something similar)
  //add more enums for additional Functor Types;
  //
  //CAVEAT: NrFctrTypes MUST be last in enum!
  //additions can be made anywhere between enum...=0 and NrFctrTypes!
  //
  typedef enum{Multiply=0, Sum, Min, Max, NrFctrTypes} SemiRingFunctorTypes;

  //Partial specialization to select proper
  //functor through an integer, at compile time (?)
  //
  template<SemiRingFunctorTypes, typename ValueT> 
  struct SemiRingFctrSelector;

  template<typename ValueT>
  struct SemiRingFctrSelector<Multiply, ValueT>
  {
    typedef typename thrust::multiplies<ValueT> FctrType;
  };

  template<typename ValueT>
  struct SemiRingFctrSelector<Sum, ValueT>
  {
    typedef typename thrust::plus<ValueT> FctrType;
  };

  template<typename ValueT>
  struct SemiRingFctrSelector<Min, ValueT>
  {
    typedef typename thrust::minimum<ValueT> FctrType;
  };

  template<typename ValueT>
  struct SemiRingFctrSelector<Max, ValueT>
  {
    typedef typename thrust::maximum<ValueT> FctrType;
  };

  //...add more specializations for additional Functor Types

  //Acyclic Visitor
  //         (A. Alexandrescu, "Modern C++ Design", Section 10.4), 
  //         where *concrete* Visitors must be parameterized by all 
  //         the possibile template args of the Visited classes (visitees);
  //

  //Visitor for SubGraph extraction:
  //
  template<typename VectorI, 
           typename VectorV,
           typename VertexCombineFctr,  //vertex "multiplication" functor type
           typename VertexReduceFctr,   //vertex "addition" functor type
           typename EdgeCombineFctr,    //edge "multiplication" functor type
           typename EdgeReduceFctr>     //edge "addition" functor type>
  struct GraphContractionVisitor: 
    VisitorBase,
    Visitor<Graph<typename VectorI::value_type> >,
    Visitor<CsrGraph<typename VectorI::value_type> >,
    Visitor<ValuedCsrGraph<typename VectorI::value_type, typename VectorV::value_type> >,
    Visitor<MultiValuedCsrGraph<typename VectorI::value_type, typename VectorV::value_type> >
  {
    typedef typename VectorI::value_type IndexType_;
    typedef typename VectorV::value_type ValueType_;
    typedef typename VectorPtrT<typename VectorI::value_type,VectorI>::PtrT PtrI;
  typedef typename VectorPtrT<typename VectorV::value_type,VectorV>::PtrT PtrV;
  typedef range_view<PtrV> VWrapper;

    typedef GraphContractionFunctor<VectorI,
                                    VectorV,
                                    VertexCombineFctr,
                                    VertexReduceFctr,
                                    EdgeCombineFctr,
                                    EdgeReduceFctr > CFunctor;

    //TODO: avoid copy from raw pointer
    //
    GraphContractionVisitor(CsrGraph<IndexType_>& graph,    
                            const VectorI& aggregates, /*const */
                            cudaStream_t stream,
                            VertexCombineFctr& v_combine,
                            VertexReduceFctr&  v_reduce,
                            EdgeCombineFctr&   e_combine,
                            EdgeReduceFctr&    e_reduce):
      m_g_row_ptr_(graph.get_raw_row_offsets(),
               graph.get_raw_row_offsets()+graph.get_num_vertices()+1),
      m_g_col_ind_(graph.get_raw_column_indices(),
               graph.get_raw_column_indices()+graph.get_num_edges()),
      //       num_aggregates != m_aggregates.size()!!!
      //       need to calculate num_aggregates (validate_..() does it)
      //       and pass it to contractor:
      //
      contractor_(graph.get_num_vertices(),
                  aggregates,
                  validate_contractor_input(aggregates, graph.get_num_vertices()),
                  v_combine,
                  v_reduce,
                  e_combine,
                  e_reduce),
      stream_(stream),
      contracted_graph_(0)
    {
      cudaCheckError();
      //empty...
    }

    void Visit(Graph<IndexType_>& graph)
    {
      //no-op...
    }

    void Visit(CsrGraph<IndexType_>& graph_src)
    {
      //(non-AMGX version):
      //SemiRing::update_topology(contractor_.get_row_ptr(), contractor_.get_col_ind());
      typedef typename SemiRingFctrSelector<Multiply, ValueType_>::FctrType MultiplyFctr;
      typedef typename SemiRingFctrSelector<Sum, ValueType_>::FctrType SumFctr;

      MultiplyFctr mult;
      SumFctr sum;

    SemiringContractionUtilities<VectorI, VectorV, VWrapper,MultiplyFctr,SumFctr,MultiplyFctr,SumFctr>
        sr(m_g_row_ptr_,
           m_g_col_ind_,
           contractor_.get_R_row_offsets(),
           contractor_.get_R_column_indices(),
           mult,
           sum,
           mult,
           sum);
      
      sr.update_topology_only();
      
      ///contractor_(m_g_row_ptr_, m_g_col_ind_);//just drop it, no-op, here, all work done by sr

      size_t rowptr_sz = sr.get_row_ptr().size();
      assert( rowptr_sz >= 1 );

      size_t contrctd_nrows = rowptr_sz-1;
      size_t contrctd_nnz = sr.get_subg_nnz();

      if( contracted_graph_ )
        delete contracted_graph_;
      
      contracted_graph_ = new CsrGraph<IndexType_>(contrctd_nrows, contrctd_nnz, stream_);

      //TODO: more efficient solution: investigate if/how copy can be avoided
      //
      thrust::copy(sr.get_row_ptr().begin(), sr.get_row_ptr().end(), contracted_graph_->get_raw_row_offsets());
      cudaCheckError();
      thrust::copy(sr.get_col_ind().begin(), sr.get_col_ind().end(), contracted_graph_->get_raw_column_indices());
      cudaCheckError();
    }

    void Visit(ValuedCsrGraph<IndexType_,ValueType_>& graph_src)
    {
      size_t g_nrows = graph_src.get_num_vertices();
      size_t g_nnz = graph_src.get_num_edges();

      VectorV vals(graph_src.get_raw_values(), graph_src.get_raw_values()+g_nnz);

      //(non-AMGX version):
      //SemiRing::update_topology(contractor_.get_row_ptr(), contractor_.get_col_ind());
      typedef typename SemiRingFctrSelector<Multiply, ValueType_>::FctrType MultiplyFctr;
      typedef typename SemiRingFctrSelector<Sum, ValueType_>::FctrType SumFctr;

      MultiplyFctr mult;
      SumFctr sum;

    SemiringContractionUtilities<VectorI, VectorV, VWrapper,MultiplyFctr,SumFctr,MultiplyFctr,SumFctr>
        sr(m_g_row_ptr_,
           m_g_col_ind_,
           contractor_.get_R_row_offsets(),
           contractor_.get_R_column_indices(),
           mult,
           sum,
           mult,
           sum);
      
      sr.update_topology_only();
      
      ///contractor_(vals, m_g_row_ptr_, m_g_col_ind_);//just drop it, no-op, here, all work done by sr and updater, below
      
      size_t rowptr_sz = sr.get_row_ptr().size();
      assert( rowptr_sz >= 1 );

      size_t contrctd_nrows = rowptr_sz-1;
      size_t contrctd_nnz = sr.get_subg_nnz();

      ValuedCsrGraph<IndexType_,ValueType_>* subg = new ValuedCsrGraph<IndexType_,ValueType_>(contrctd_nrows, contrctd_nnz, stream_);

      //TODO: more efficient solution: investigate if/how copy can be avoided
      //
      thrust::copy(sr.get_row_ptr().begin(), sr.get_row_ptr().end(), subg->get_raw_row_offsets());
      cudaCheckError();
      thrust::copy(sr.get_col_ind().begin(), sr.get_col_ind().end(), subg->get_raw_column_indices());
      cudaCheckError();

      //handling the values:
      //
      VertexCombineFctr v_combine;
      VertexReduceFctr  v_reduce;
      EdgeCombineFctr   e_combine;
      EdgeReduceFctr    e_reduce;

      //TODO: more efficient solution with VWrapper, to avoid device memory traffic
      //
      VectorV cg_values(subg->get_raw_values(), subg->get_raw_values()+contrctd_nnz);
         
      ContractionValueUpdater<VectorV,//VWrapper?
                              VectorI,
                              VertexCombineFctr,
                              VertexReduceFctr,
                              EdgeCombineFctr,
                              EdgeReduceFctr,
                              128>//useless...; only used with AMGX version
        updater(vals,
                cg_values,
                v_combine,
                v_reduce,
                e_combine,
                e_reduce);

      updater.update_from(contractor_.get_R_row_offsets(),
                          contractor_.get_R_column_indices(),
                          m_g_row_ptr_,
                          m_g_col_ind_);
                          

      //TODO: more efficient solution with VWrapper, to avoid device memory traffic
      //
      thrust::copy(cg_values.begin(), cg_values.end(), subg->get_raw_values());
      cudaCheckError();
      

      if( contracted_graph_ )
        delete contracted_graph_;
      
      contracted_graph_ = subg;
    }

    void Visit(MultiValuedCsrGraph<IndexType_,ValueType_>& graph_src)
    {
       //(non-AMGX version):
      //SemiRing::update_topology(contractor_.get_row_ptr(), contractor_.get_col_ind());
      typedef typename SemiRingFctrSelector<Multiply, ValueType_>::FctrType MultiplyFctr;
      typedef typename SemiRingFctrSelector<Sum, ValueType_>::FctrType SumFctr;

      MultiplyFctr mult;
      SumFctr sum;

    SemiringContractionUtilities<VectorI, VectorV, VWrapper,MultiplyFctr,SumFctr,MultiplyFctr,SumFctr>
        sr(m_g_row_ptr_,
           m_g_col_ind_,
           contractor_.get_R_row_offsets(),
           contractor_.get_R_column_indices(),
           mult,
           sum,
           mult,
           sum);
      cudaCheckError();
      sr.update_topology_only();
      cudaCheckError();
            
      ///contractor_(m_g_row_ptr_, m_g_col_ind_);//just drop it, no-op, here, all work done by sr and reduce_*_data(), below
      
      //construct the contracted graph out of contractor_ newly acquired data
    size_t rowptr_sz = sr.get_row_ptr().size();
      assert( rowptr_sz >= 1 );

      size_t contrctd_nrows = rowptr_sz-1;
      size_t contrctd_nnz = sr.get_subg_nnz();
      cudaCheckError();

      if( contracted_graph_ )
        delete contracted_graph_;
      cudaCheckError();
      
      MultiValuedCsrGraph<IndexType_,ValueType_>* mv_cntrctd_graph = 
        new MultiValuedCsrGraph<IndexType_,ValueType_>(contrctd_nrows, contrctd_nnz, stream_);

      cudaCheckError();

      //TODO: more efficient solution: investigate if/how copy can be avoided
      //
      thrust::copy(sr.get_row_ptr().begin(), sr.get_row_ptr().end(), mv_cntrctd_graph->get_raw_row_offsets());
      cudaCheckError();
      thrust::copy(sr.get_col_ind().begin(), sr.get_col_ind().end(), mv_cntrctd_graph->get_raw_column_indices());
      cudaCheckError();


      //reduce vertex and edge data for the contracted graph
      reduce_vertex_data(graph_src, *mv_cntrctd_graph);
      reduce_edge_data(graph_src, *mv_cntrctd_graph);

      contracted_graph_ = mv_cntrctd_graph;
    }

    const CFunctor& get_contractor(void) const
    {
      return contractor_;
    }

    CsrGraph<IndexType_>* get_contracted_graph(void) // TODO: change to unique_ptr, when moving to C++1*
    {
      return contracted_graph_;
    }

    const VectorI& get_aggregates(void) const
    {
      return contractor_.get_aggregates();
    }

  protected:
    //virtual reductors for contracted vertices and edges:
    //
    virtual void reduce_vertex_data(MultiValuedCsrGraph<IndexType_,ValueType_>& graph_src,
                                    MultiValuedCsrGraph<IndexType_,ValueType_>& graph_dest)
    {
    SemiringContractionUtilities<VectorI, VectorV, VWrapper,VertexCombineFctr,VertexReduceFctr,EdgeCombineFctr,EdgeReduceFctr>
        sr(m_g_row_ptr_,
           m_g_col_ind_,
           contractor_.get_R_row_offsets(),
           contractor_.get_R_column_indices(),
           contractor_.get_v_combine(),
           contractor_.get_v_reduce(),
           contractor_.get_e_combine(),
           contractor_.get_e_reduce());
      cudaCheckError();

      if ( graph_dest.get_num_vertices() == 0 )
        FatalError("Empty contracted graph (no vertices).",NVGRAPH_ERR_BAD_PARAMETERS);

      //allocate graph_dest vertex data and fill it:
    //
    size_t ng = graph_src.get_num_vertex_dim();
    graph_dest.allocateVertexData(ng, stream_);
    cudaCheckError();

    for(unsigned int i=0;i<ng;++i)
    {
      Vector<ValueType_>& v_src = graph_src.get_vertex_dim(i);
      Vector<ValueType_>& v_dest = graph_dest.get_vertex_dim(i);

      size_t n_src = v_src.get_size();
      PtrV ptr_src(v_src.raw());
      VWrapper rv_src(ptr_src, ptr_src+n_src);

      size_t n_dest = v_dest.get_size();
      assert( graph_dest.get_num_vertices() == n_dest );

      PtrV ptr_dest(v_dest.raw());
      VWrapper rv_dest(ptr_dest, ptr_dest+n_dest);

      sr.update_vertex_data(rv_src, rv_dest);
      cudaCheckError();
    }
    }

    virtual void reduce_edge_data(MultiValuedCsrGraph<IndexType_,ValueType_>& graph_src,
                                  MultiValuedCsrGraph<IndexType_,ValueType_>& graph_dest)
    {
    SemiringContractionUtilities<VectorI, VectorV, VWrapper,VertexCombineFctr,VertexReduceFctr,EdgeCombineFctr,EdgeReduceFctr>
        sr(m_g_row_ptr_,
           m_g_col_ind_,
           contractor_.get_R_row_offsets(),
           contractor_.get_R_column_indices(),
           contractor_.get_v_combine(),
           contractor_.get_v_reduce(),
           contractor_.get_e_combine(),
           contractor_.get_e_reduce());
      cudaCheckError();

      //There can be a contracted graph with no edges,
      //but such a case warrants a warning:
      //
      if ( graph_dest.get_num_edges() == 0 )
        WARNING("Contracted graph is disjointed (no edges)");
      
      //allocate graph_dest edge data and fill it:
    //
    size_t ng = graph_src.get_num_edge_dim();
    graph_dest.allocateEdgeData(ng, stream_);
    cudaCheckError();

    for(unsigned int i=0;i<ng;++i)
    {
      Vector<ValueType_>& v_src = graph_src.get_edge_dim(i);
      Vector<ValueType_>& v_dest = graph_dest.get_edge_dim(i);

      size_t n_src = v_src.get_size();
      PtrV ptr_src(v_src.raw());
      VWrapper rv_src(ptr_src, ptr_src+n_src);

      size_t n_dest = v_dest.get_size();
      assert( graph_dest.get_num_edges() == n_dest );

      PtrV ptr_dest(v_dest.raw());
      VWrapper rv_dest(ptr_dest, ptr_dest+n_dest);

      sr.update_edge_data(rv_src, rv_dest);
      cudaCheckError();
    }
    }

  private:
    VectorI m_g_row_ptr_;
    VectorI m_g_col_ind_;
    CFunctor contractor_;
    cudaStream_t stream_;
    CsrGraph<IndexType_>* contracted_graph_; // to be constructed
  };


  


  //###################################################### Nested-if-then-else solution: 
  //
  //easier on number of recursive template instantiations
  //i.e., less-likely to run into compilation problems like:
  //'error: excessive recursion at instantiation of function ...';
  //or the newly(as of cuda8.0) available flag: -ftemplate-depth <depth>
  //
  //generic empty template:
  //
  template<typename VectorI,
       typename VectorV,
       typename T1,
       typename T2,
       typename T3,
       size_t Level, size_t n, size_t N>
  struct NestedTypedIfThenElser;    

  //Level 3 (ceiling of recursion):
  //
  template<typename VectorI,
       typename VectorV,
       typename T1,
       typename T2,
       typename T3,
       size_t n, size_t N>
  struct NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 3, n, N>
  {
  typedef typename VectorI::value_type IndexT;
    typedef typename VectorV::value_type ValueT;

  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    if( i4 == n )//reached both ceiling of Level recursion and bottom of n value recursion
    {
      ///std::cout<<"OK: tuple("<<i1<<","<<i2<<","<<i3<<","<<i4<<") hit!\n";//stop, everything hit...
      typedef typename SemiRingFctrSelector<(SemiRingFunctorTypes)n, ValueT>::FctrType T4;

      typedef T1 VertexCombineFctr;
      typedef T2 VertexReduceFctr;
      typedef T3 EdgeCombineFctr;
      typedef T4 EdgeReduceFctr;

      VertexCombineFctr v_combine;
      VertexReduceFctr  v_reduce;
      EdgeCombineFctr   e_combine;
      EdgeReduceFctr    e_reduce;

      GraphContractionVisitor<VectorI,
                              VectorV,
                              VertexCombineFctr,
                              VertexReduceFctr,
                              EdgeCombineFctr,
                              EdgeReduceFctr>
      visitor(graph,
          aggregates,
          stream,
          v_combine,
          v_reduce,
          e_combine,
          e_reduce);
      cudaCheckError();

      graph.Accept(visitor);
      cudaCheckError();
      return visitor.get_contracted_graph();
    }
    else //continue with same level (3), but next decreasing n value
    return NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 3, n-1, N>::iffer(i1, i2, i3, i4, 
                                            graph, 
                                            aggregates,
                                            stream); 
  }
  };

  //Level 3 bottom:
  //
  template<typename VectorI,
       typename VectorV,
       typename T1,
       typename T2,
       typename T3,
       size_t N>
  struct NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 3, 0, N>
  {
  typedef typename VectorI::value_type IndexT;
    typedef typename VectorV::value_type ValueT;

  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    if( i4 == 0 )
    {
      ///std::cout<<"OK: tuple("<<i1<<","<<i2<<","<<i3<<","<<i4<<") hit!\n";//stop, everything hit...
      typedef typename SemiRingFctrSelector<(SemiRingFunctorTypes)0, ValueT>::FctrType T4;

      typedef T1 VertexCombineFctr;
      typedef T2 VertexReduceFctr;
      typedef T3 EdgeCombineFctr;
      typedef T4 EdgeReduceFctr;

      VertexCombineFctr v_combine;
      VertexReduceFctr  v_reduce;
      EdgeCombineFctr   e_combine;
      EdgeReduceFctr    e_reduce;

      GraphContractionVisitor<VectorI,
                              VectorV,
                              VertexCombineFctr,
                              VertexReduceFctr,
                              EdgeCombineFctr,
                              EdgeReduceFctr>
      visitor(graph,
          aggregates,
          stream,
          v_combine,
          v_reduce,
          e_combine,
          e_reduce);

      graph.Accept(visitor);
      return visitor.get_contracted_graph();
    }
    else
    {
      std:: stringstream ss;
      ss<<"ERROR: tuple("<<i1<<","<<i2<<","<<i3<<","<<i4<<") not hit on Level 3.";
      FatalError(ss.str().c_str(),NVGRAPH_ERR_BAD_PARAMETERS);
      //return 0;
    }
  }
  };

  //Level 2 generic:
  //
  template<typename VectorI,
       typename VectorV,
       typename T1,
       typename T2,
       typename T3,
       size_t n, size_t N>
  struct NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 2, n, N>
  {
  typedef typename VectorI::value_type IndexT;
    typedef typename VectorV::value_type ValueT;

  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    if( i3 == n )
    {
      typedef typename SemiRingFctrSelector<(SemiRingFunctorTypes)n, ValueT>::FctrType RT;//replace T3!
      return NestedTypedIfThenElser<VectorI, VectorV, T1, T2, RT, 3, N-1, N>::iffer(i1, i2, i3, i4, 
                                         graph, 
                                         aggregates,
                                         stream);//continue with next increasing level (3)
      //with 1st possible value (N-1)
    }
    else
    return NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 2, n-1, N>::iffer(i1, i2, i3, i4, 
                                         graph, 
                                         aggregates,
                                         stream);//continue with same level (2), but next decreasing n value 
  }
  };

  //Level 2 bottom:
  //
  template<typename VectorI,
       typename VectorV,
       typename T1,
       typename T2,
       typename T3,
       size_t N>
  struct NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 2, 0, N>
  {
  typedef typename VectorI::value_type IndexT;
    typedef typename VectorV::value_type ValueT;

  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    if( i3 == 0 )
    {
      typedef typename SemiRingFctrSelector<(SemiRingFunctorTypes)0, ValueT>::FctrType RT;//replace T3!
      return NestedTypedIfThenElser<VectorI, VectorV, T1, T2, RT, 3, N-1, N>::iffer(i1, i2, i3, i4, 
                                         graph, 
                                         aggregates,
                                         stream);//continue with next increasing level (3)
      //with 1st possible value (N-1)
    }
    else
    {
      std:: stringstream ss;
      ss<<"ERROR: tuple("<<i1<<","<<i2<<","<<i3<<","<<i4<<") not hit on Level 2.";
      FatalError(ss.str().c_str(),NVGRAPH_ERR_BAD_PARAMETERS);
      //return 0;
    }
  }
  };

  //Level 1 generic:
  //
  template<typename VectorI,
       typename VectorV,
       typename T1,
       typename T2,
       typename T3,
       size_t n, size_t N>
  struct NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 1, n, N>
  {
  typedef typename VectorI::value_type IndexT;
  typedef typename VectorV::value_type ValueT;

  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    if( i2 == n )
    {
      typedef typename SemiRingFctrSelector<(SemiRingFunctorTypes)n, ValueT>::FctrType RT;//replace T2!
      return NestedTypedIfThenElser<VectorI, VectorV, T1, RT, T3, 2, N-1, N>::iffer(i1, i2, i3, i4, 
                                         graph, 
                                         aggregates,
                                         stream);//continue with next increasing level (2)
      //with 1st possible value (N-1)
    }
    else
    return NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 1, n-1, N>::iffer(i1, i2, i3, i4, 
                                         graph, 
                                         aggregates,
                                         stream);//continue with same level (1), but next decreasing n value 
  }
  };

  //Level 1 bottom:
  //
  template<typename VectorI,
       typename VectorV,
       typename T1,
       typename T2,
       typename T3,
       size_t N>
  struct NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 1, 0, N>
  {
  typedef typename VectorI::value_type IndexT;
  typedef typename VectorV::value_type ValueT;

  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    if( i2 == 0 )
    {
      typedef typename SemiRingFctrSelector<(SemiRingFunctorTypes)0, ValueT>::FctrType RT;//replace T2!
      return NestedTypedIfThenElser<VectorI, VectorV, T1, RT, T3, 2, N-1, N>::iffer(i1, i2, i3, i4, 
                                         graph, 
                                         aggregates,
                                         stream);//continue with next increasing level (2)
      //with 1st possible value (N-1)
    }
    else
    {
      std:: stringstream ss;
      ss<<"ERROR: tuple("<<i1<<","<<i2<<","<<i3<<","<<i4<<") not hit on Level 1.";
      FatalError(ss.str().c_str(),NVGRAPH_ERR_BAD_PARAMETERS);
      //return 0;
    }
  }
  };

  //Level 0 generic:
  //
  template<typename VectorI,
       typename VectorV,
       typename T1,
       typename T2,
       typename T3,
       size_t n, size_t N>
  struct NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 0, n, N>
  {
  typedef typename VectorI::value_type IndexT;
  typedef typename VectorV::value_type ValueT;

  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    if( i1 == n )
    {
      typedef typename SemiRingFctrSelector<(SemiRingFunctorTypes)n, ValueT>::FctrType RT;//replace T1!
      return NestedTypedIfThenElser<VectorI, VectorV, RT, T2, T3, 1, N-1, N>::iffer(i1, i2, i3, i4, 
                                     graph, 
                                     aggregates,
                                     stream);//continue with next increasing level (1)
      //with 1st possible value (N-1)
    }
    else
    return NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 0, n-1, N>::iffer(i1, i2, i3, i4, 
                                   graph, 
                                   aggregates,
                                   stream);//continue with same level (0), but next decreasing n value 
  }
  };

  //Level 0 bottom:
  //
  template<typename VectorI,
       typename VectorV,
       typename T1,
       typename T2,
       typename T3,
       size_t N>
  struct NestedTypedIfThenElser<VectorI, VectorV, T1, T2, T3, 0, 0, N>
  {
  typedef typename VectorI::value_type IndexT;
  typedef typename VectorV::value_type ValueT;

  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    if( i1 == 0 )
    {
      typedef typename SemiRingFctrSelector<(SemiRingFunctorTypes)0, ValueT>::FctrType RT;//replace T1!
      return NestedTypedIfThenElser<VectorI, VectorV, RT, T2, T3, 1, N-1, N>::iffer(i1, i2, i3, i4, 
                                         graph, 
                                         aggregates,
                                         stream);//continue with next increasing level (1)
      //with 1st possible value (N-1)
    }
    else
    {
      std:: stringstream ss;
      ss<<"ERROR: tuple("<<i1<<","<<i2<<","<<i3<<","<<i4<<") not hit on Level 0.";
      FatalError(ss.str().c_str(),NVGRAPH_ERR_BAD_PARAMETERS);
      //return 0;
    }
  }
  };

  //Wrapper:
  //
  //N = # possible (consecutive 0-based) values
  //that each tuple element can take
  //
  template<typename VectorI,
       typename VectorV, 
       size_t N>
  struct NestedTypedIfThenElseWrapper
  {
  typedef typename VectorI::value_type IndexT;
    typedef typename VectorV::value_type ValueT;

  struct Unused{};//placeholder to be replaced by actual types
  
  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    return NestedTypedIfThenElser<VectorI, VectorV, Unused, Unused, Unused, 0, N-1, N>::iffer(i1, i2, i3, i4, 
                                                graph, 
                                                aggregates,
                                                stream);
  }
  };


  template<typename VectorI,
       typename VectorV, 
       typename T1,
       size_t N>
  struct NestedTypedIfThenElseWrapperT
  {
    typedef typename VectorI::value_type IndexT;
    typedef typename VectorV::value_type ValueT;

  struct Unused{};//placeholder to be replaced by actual types
  
  static CsrGraph<IndexT>* iffer(size_t i1, size_t i2, size_t i3, size_t i4, 
                   CsrGraph<IndexT>& graph,
                   VectorI& aggregates,
                   cudaStream_t stream)
  {
    return NestedTypedIfThenElser<VectorI, VectorV, T1, Unused, Unused, 1, N-1, N>::iffer(i1, i2, i3, i4, 
                                                graph, 
                                                aggregates,
                                                stream);
  }
  };


  

  template<typename IndexT, typename ValueT>
  CsrGraph<IndexT>* contract_from_aggregates(CsrGraph<IndexT>& graph, 
                                             IndexT* p_aggregates,
                                             size_t n,
                                             cudaStream_t stream,
                                             const SemiRingFunctorTypes& vCombine,
                                             const SemiRingFunctorTypes& vReduce,
                                             const SemiRingFunctorTypes& eCombine,
                                             const SemiRingFunctorTypes& eReduce)
  {
    typedef thrust::device_vector<IndexT> VectorI;
    typedef thrust::device_vector<ValueT> VectorV;

    VectorI aggregates(p_aggregates, p_aggregates+n);

  //Nested if-then-else solution:
  //
  //(no need for constness, they're NOT template args)
  //
  return NestedTypedIfThenElseWrapper<VectorI, VectorV, NrFctrTypes>::iffer((size_t)vCombine, 
                                      (size_t)vReduce, 
                                      (size_t)eCombine, 
                                      (size_t)eReduce, 
                                      graph, aggregates, stream);

  //Flatened if-then-else solution:
  //
     //const size_t M = NrFctrTypes;
     //const size_t M2 = M*M;
     //const size_t M3 = M2*M;
  
     //size_t i
     //  = (size_t)vCombine * M3
     //  + (size_t)vReduce *  M2
     //  + (size_t)eCombine * M
     //  + (size_t)eReduce;
    
    //return Selector<NComboTypes-1, NrFctrTypes, VectorI, VectorV>::iffer(i, graph, aggregates, stream);
  }

    template<typename IndexT, typename ValueT, typename T>
  CsrGraph<IndexT>* contract_from_aggregates_t(CsrGraph<IndexT>& graph, 
                                             IndexT* p_aggregates,
                                             size_t n,
                                             cudaStream_t stream,
                                             const SemiRingFunctorTypes& vCombine,
                                             const SemiRingFunctorTypes& vReduce,
                                             const SemiRingFunctorTypes& eCombine,
                                             const SemiRingFunctorTypes& eReduce)
  {
    typedef thrust::device_vector<IndexT> VectorI;
    typedef thrust::device_vector<ValueT> VectorV;

    VectorI aggregates(p_aggregates, p_aggregates+n);

  //Nested if-then-else solution:
  //
  //(no need for constness, they're NOT template args)
  //
  return NestedTypedIfThenElseWrapperT<VectorI, VectorV, T, NrFctrTypes>::iffer((size_t)vCombine, 
                                      (size_t)vReduce, 
                                      (size_t)eCombine, 
                                      (size_t)eReduce, 
                                      graph, aggregates, stream);

  //Flatened if-then-else solution:
  //
     //const size_t M = NrFctrTypes;
     //const size_t M2 = M*M;
     //const size_t M3 = M2*M;
  
     //size_t i
     //  = (size_t)vCombine * M3
     //  + (size_t)vReduce *  M2
     //  + (size_t)eCombine * M
     //  + (size_t)eReduce;
    
    //return Selector<NComboTypes-1, NrFctrTypes, VectorI, VectorV>::iffer(i, graph, aggregates, stream);
  }

}

#endif
