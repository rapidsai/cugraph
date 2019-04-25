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

#ifndef GRAPH_CONCRETE_VISITORS_HXX
#define GRAPH_CONCRETE_VISITORS_HXX

#include <multi_valued_csr_graph.hxx> //which includes all other headers... 
#include <range_view.hxx> // TODO: to be changed to thrust/range_view.h, when toolkit gets in sync with Thrust
#include <thrust_traits.hxx>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/distance.h>//
#include <thrust/unique.h>//
#include <thrust/merge.h>//
#include <thrust/sort.h>//
#include <thrust/find.h>//
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>

namespace nvgraph
{
  //get unique elements and return their count:
  //
  template<typename Container>
  size_t count_get_distinct(const Container& v, //in
			    Container& res)     //out
  {
    res.assign(v.begin(), v.end());//copy
  
    size_t counts = thrust::distance(res.begin(), thrust::unique(res.begin(), res.end()));
    res.resize(counts);
    return  counts;
  }

  //Adapted from: https://github.com/thrust/thrust/blob/master/examples/expand.cu
  //
  //Note:
  //C++03 doesn’t allow default template arguments on function templates. 
  //This was considered a “defect” by Bjarne Stroustrup, subsequently fixed in C++11. 
  //See, for example: http://stackoverflow.com/questions/2447458/default-template-arguments-for-function-templates 
  //
  template<typename T, 
	   template<typename> class Allocator, 
	   template<typename, typename> class Vector>
  typename Vector<T, Allocator<T> >::iterator expand(Vector<T, Allocator<T> >& counts,
													 Vector<T, Allocator<T> >& values,
													 Vector<T, Allocator<T> >& out)
  {
    typedef typename Vector<T, Allocator<T> >::iterator Iterator;

    Iterator first1 = counts.begin();
    Iterator last1 =  counts.end();
  
    Iterator first2 = values.begin();
    Iterator output = out.begin();

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  
    difference_type input_size  = thrust::distance(first1, last1);
    difference_type output_size = thrust::reduce(first1, last1);

    // scan the counts to obtain output offsets for each input element
    Vector<difference_type, Allocator<difference_type> > output_offsets(input_size, 0);
    thrust::exclusive_scan(first1, last1, output_offsets.begin()); 

    // scatter the nonzero counts into their corresponding output positions
    Vector<difference_type, Allocator<difference_type> > output_indices(output_size, 0);
    thrust::scatter_if
      (thrust::counting_iterator<difference_type>(0),
       thrust::counting_iterator<difference_type>(input_size),
       output_offsets.begin(),
       first1,
       output_indices.begin());

    // compute max-scan over the output indices, filling in the holes
    thrust::inclusive_scan
      (output_indices.begin(),
       output_indices.end(),
       output_indices.begin(),
       thrust::maximum<difference_type>());

    // gather input values according to index array (output = first2[output_indices])
    Iterator output_end = output; thrust::advance(output_end, output_size);
    thrust::gather(output_indices.begin(),
		   output_indices.end(),
		   first2,
		   output);

    // return output + output_size
    thrust::advance(output, output_size);
    return output;
  }



  //
  //


  
  //##### Change 1: reverse hash was wrong: hash[val_i] = index of first occurence of val_i #####
  //
  template<typename Container>
  struct MinLeftRightPlusValue
  {
    typedef typename VectorPtrT<typename Container::value_type,Container>::PtrT PtrT;
    typedef typename Container::value_type ValT;

    explicit MinLeftRightPlusValue(ValT delta):
      delta_(delta)
    {
    }
    
    __host__ __device__
    ValT operator() (ValT left, ValT right)
    {
      ValT rs = right + delta_;
      return (left < rs? left : rs);
    }

  private:
    ValT delta_;
  };

  //given vector v[i] = val_i, 
  //return reverse hash vector:
  //hash[val_i] = i (index of first occurence of val_i, if val_i exists in v[]; 
  //                 else, last occurence of closest value less than val_i):
  //
  //advantage:    works trully like a hash, no need for search
  //
  // 
  //pre-conditions: (1.) v sorted in ascending order
  //                (2.) value_type is integer type 
  //
  //Ex:
  //v:    0,1,3,6,7,8,8;
  //hash: 0,1,1,2,2,2,3,4,5;
  //
  template<typename Container>
  void reverse_hash(Container& v,    //in
		    Container& hash) //out
  {
    typedef typename Container::value_type ValT;

    if( v.empty() )
      return;

    size_t sz = v.size();
    size_t seq_sz = v.back()-v.front()+1;

    thrust::counting_iterator<ValT> seq_first(v.front());
    thrust::counting_iterator<ValT> seq_last(v.back()+1);

    Container hash1(seq_sz, ValT(-1));
    Container hash2(seq_sz, ValT(-1));
    hash.assign(seq_sz, ValT(-1));

    thrust::upper_bound(v.begin(), v.end(),
			seq_first, seq_last, //seq.begin(), seq.end(),//ok 
			hash1.begin(),
			thrust::less<ValT>());

    //
    thrust::lower_bound(v.begin(), v.end(),
			seq_first, seq_last, //seq.begin(), seq.end(), //ok
			hash2.begin(),
			thrust::less<ValT>());

    thrust::transform(hash2.begin(), hash2.end(),
		      hash1.begin(),
		      hash.begin(),
		      MinLeftRightPlusValue<Container>(-1));
    
  }

  //better use thrust::gather(...)
  //see /home/aschaffer/Development/Sources/Cuda_Thrust/filter_via_gather.cu
  template<typename VectorR, typename IndexT>
  struct Filter
  {
    typedef typename VectorR::value_type RetT;

    explicit Filter(VectorR& src):
      m_src(&src[0])
    {
    }
    __host__ __device__
    RetT operator()(const IndexT& k)
    {
      return m_src[k];
    }
  private:
    typename VectorPtrT<typename VectorR::value_type,VectorR>::PtrT m_src;
  };

  template<typename Container, typename IndexT>
  struct CleanFctr
  {
    explicit CleanFctr(Container& used):
      m_used(&used[0])
    {
    }
    __host__ __device__
    bool operator()(const IndexT& k)
    {
      return (m_used[k] == 0);
    }
  private:
    typename VectorPtrT<typename Container::value_type,Container>::PtrT m_used;
  };

  //
  //
  template<typename VectorV, 
	   typename VectorI>
  struct ValueUpdater
  {
    typedef typename VectorI::value_type IndexT;
    //typedef typename VectorPtrT<typename VectorI::value_type,VectorV>::PtrT PtrI;

    typedef typename VectorV::value_type ValueT;
    typedef typename VectorPtrT<typename VectorV::value_type,VectorV>::PtrT PtrV;

    explicit ValueUpdater(VectorV& v_src,
			  VectorV& v_dest):
      v_s_(v_src),
      v_d_(v_dest)
    {
    }
  
    ///__host__ __device__
    ValueT at(IndexT j) const
    {
      return v_s_[j];
    }

    struct ValFiller
    {
      explicit ValFiller(VectorV& v_src):
	m_s(&v_src[0])
      {
      }

      __host__ __device__
      ValueT operator() (IndexT k)
      {
	return m_s[k];
      }
    private:
      PtrV m_s;
    };

    //#####  Change 5: const K #####
    //
    void update_from(const VectorI& K)
    {
      size_t actual_nnz = K.size();

      v_d_.assign(actual_nnz, ValueT(0));

      ValFiller valfill(v_s_);
      thrust::transform(K.begin(), K.end(),
			v_d_.begin(),
			valfill);
    }
      
    const VectorV& get_subg_vals(void) const
    {
      return v_d_;
    }
  private:
    VectorV& v_s_;
    VectorV& v_d_;
  };

  template<typename VectorI,
	   typename VectorB = VectorI>
  struct Offsets2RowIndex
  {

    typedef typename VectorI::value_type IndexT;
    typedef typename VectorPtrT<typename VectorB::value_type,VectorB>::PtrT PtrB;
    typedef typename VectorPtrT<typename VectorI::value_type,VectorI>::PtrT PtrI;


    Offsets2RowIndex(VectorB& hash_rows,
		     VectorI& offsets,
		     VectorI& I0,
		     VectorI& vSub,
		     VectorI& row_ptr,
		     VectorI& col_ind,
		     VectorI& I,
		     VectorI& J,
		     VectorI& K,
		     VectorB& U):
      m_hash_sz(hash_rows.size()),
      m_off_sz(offsets.size()),
      m_hash_rows(&hash_rows[0]),
      m_offsets(&offsets[0]),
      m_i0(&I0[0]),
      m_row_subset(&vSub[0]),
      m_row_ptr(&row_ptr[0]),
      m_col_ind(&col_ind[0]),
      m_i(&I[0]),
      m_j(&J[0]),
      m_k(&K[0]),
      m_used(&U[0])
    {	  
    }

	
            
    //k = element in range[]:{0,1,...,nnz-1}
    //
    __host__ __device__
    IndexT operator() (IndexT k)
    {
      IndexT subg_row_index = m_i0[k];
	
      IndexT g_row_index = m_row_subset[subg_row_index];

      //j = col_ind[ row_ptr[g_row_index] + k - offsets[subg_row_index]]
      //
      IndexT row_ptr_i = m_row_ptr[g_row_index]+
	k-
	m_offsets[subg_row_index];

      IndexT col_index = m_col_ind[row_ptr_i];

      //is col_index in row_subset?
      //
      if( (col_index < m_hash_sz) && (m_hash_rows[col_index] == 1) )
	//col_index in subset, too=>it's a hit!
	{
	  m_i[k] = g_row_index;
	  m_j[k] = col_index;

	  ///m_v[k] = m_fctr.at(row_ptr_i);//ok, but couples it with vals...
	  m_k[k] = row_ptr_i;

	  m_used[k] = 1;
	}
      //else ...nothing
	  
      return g_row_index;
    }
  private:
    const size_t m_hash_sz;
    const size_t m_off_sz;

    PtrB m_hash_rows;

    PtrI m_offsets;

    PtrI m_offset_indices;

    PtrI m_row_subset;

    PtrI m_row_ptr;

    PtrI m_col_ind;

    PtrI m_i0;

    PtrI m_i;

    PtrI m_j;

    PtrI m_k;

    PtrB m_used;
  };

  template<typename VectorI,
	   typename VectorB>
  size_t fill_hash_nz2ijv(VectorB& hash_rows,
			  VectorI& range,         //in/out
			  VectorI& nzs,
			  VectorI& offsets,
			  VectorI& vSub,
			  VectorI& row_ptr,
			  VectorI& col_ind,
			  VectorI& I,
			  VectorI& J,
			  VectorI& K,
			  VectorB& U)
  {
    typedef typename VectorI::value_type IndexT;

    size_t nnz = range.size();
    size_t nrows_subg = nzs.size();

    VectorI I0(nnz, IndexT(0));
    VectorI dummy(nnz, IndexT(0));

    //make m_offset_indices increasing sequence
    //from 0,...,offsets.size()-1
    //
    VectorI offset_indices(nrows_subg, IndexT(0));
    thrust::sequence(offset_indices.begin(),
		     offset_indices.end(),
		     IndexT(0));

    expand(nzs, offset_indices, I0);

    Offsets2RowIndex<VectorI, /*VectorV, ValueUpdater, VectorSz,*/ VectorB > 
      off_fctr(hash_rows, 
	       offsets, 
	       I0,
	       vSub, 
	       row_ptr, 
	       col_ind, 
	       I,J,K,U);

    //why unused dummy? 
    //because functor must return something  
    //and must store result of functor somewhere!
    //
    thrust::transform(range.begin(), range.end(), 
		      dummy.begin(), //unused...
		      off_fctr);

    CleanFctr<VectorB, IndexT> cleaner(U);
    range.erase(thrust::remove_if(range.begin(), range.end(), cleaner), range.end());

    size_t actual_nnz = range.size();

    VectorI truncated_i(actual_nnz, IndexT(0));
    VectorI truncated_j(actual_nnz, IndexT(0));
    ///VectorV truncated_v(actual_nnz, IndexT(0));
    VectorI truncated_k(actual_nnz, IndexT(0));

    Filter<VectorI, IndexT> filter_i(I);
    thrust::transform(range.begin(), range.end(), 
		      truncated_i.begin(),
		      filter_i);
    I = truncated_i; // vector copy!

    Filter<VectorI, IndexT> filter_j(J);
    thrust::transform(range.begin(), range.end(), 
		      truncated_j.begin(),
		      filter_j);
    J = truncated_j; // vector copy!

    Filter<VectorI, IndexT> filter_k(K);
    thrust::transform(range.begin(), range.end(), 
		      truncated_k.begin(),
		      filter_k);
    K = truncated_k; // vector copy!

    // Filter<VectorV, IndexT> filter_v(V);
    // thrust::transform(range.begin(), range.end(), 
    // 					truncated_v.begin(),
    // 					filter_v);
    // V = truncated_v; // vector copy!
      
    //scoo.m_v[] == subg.vals !
    ///fctr.update_vals(scoo.get_v());

    U.assign(actual_nnz,1);//just for consistency, 
    //                       not really necessary

    return actual_nnz;
  }


  template<typename Container>
  struct NzCounter
  {
    typedef typename Container::value_type IndexT;
    typedef typename VectorPtrT<typename Container::value_type,Container>::PtrT PtrT;

    explicit NzCounter(Container& row_ptr):
      m_row_ptr(&row_ptr[0])
    {
    }
            
    __host__ __device__
    IndexT operator() (const IndexT& i)
    {
      return m_row_ptr[i+1]-m_row_ptr[i];
    }
  private:
    PtrT m_row_ptr;
  };

  template<typename Container>
  struct HashFctr
  {
    typedef typename Container::value_type IndexT;

    explicit HashFctr(Container& hash_src):
      m_hash(&hash_src[0])
    {
    }
    __host__ __device__
    IndexT operator() (const IndexT& src_elem)
    {
      IndexT hit(1);
      m_hash[src_elem] = hit;
      return hit;
    }
  private:
    typename VectorPtrT<typename Container::value_type,Container>::PtrT m_hash;
  };

  template<typename VectorI,
	   typename VectorB>
  size_t make_hash(VectorI& src,
		   VectorB& hash_src,
		   bool is_sorted = false)
  {
    typedef typename VectorI::value_type IndexT;
    typedef typename VectorB::value_type ValueB;

    assert( !src.empty() );
        
    IndexT max_entry(0);
    if( is_sorted )
      max_entry = src.back();
    else
      max_entry = thrust::reduce(src.begin(), src.end(), 
				 0, 
				 thrust::maximum<IndexT>());
        
    hash_src.assign(max_entry+1, 0);
    VectorB dummy(hash_src);	
        
    HashFctr<VectorB> hctr(hash_src);

    //why unused dummy? 
    //because functor must return something  
    //and must store result of functor somewhere!
    //
    thrust::transform(src.begin(), src.end(), 
		      dummy.begin(), //unused...
		      hctr);

    return hash_src.size();
  }


  //#####  Change 2: subg row_ptr extraction failed on missing indices #####

  /**
   * @brief Compute the CSR row indices of the extracted graph.
   *
   *    Note that source is an array of row indices that are
   *    part of the subgraph.  If a vertex appears a source multiple
   *    times in the subgraph it appears multiple times in the source
   *    vector.
   *
   *  @param[in] actual_nnz      Number of non-zeros in the subgraph matrix
   *                              (aka the number of edges)
   *  @param[in] nrows           Number of vertices in the subgraph
   *  @param[in] source          Array of row indices that the source of an edge
   *                              (NOTE: this array is assumed to be sorted)
   *  @param[out] subg_row_ptr   The computed subgraph row pointer
   */
  template<typename VectorI>
  void make_subg_row_ptr(size_t actual_nnz,     //in: # non-zeros in subgraph matrix
			 size_t nrows,          //in: |vSub|
			 VectorI& source,    //in: array of row indices where there 
			 //    are non-zeros (assumed sorted)
			 VectorI& subg_row_ptr) //out:subgraph row_ptr
  {
    typedef typename VectorI::value_type IndexT;

    //
    //  Nothing to do here.
    //
    if( actual_nnz == 0 )
      return;
  
    VectorI counts(nrows, 0);

    //
    //  We want to count how many times the element occurs.  We
    //  do this (based on the assumption that the list is sorted)
    //  by computing the upper bound of the range for each row id,
    //  and the lower bound for the range of each row id and
    //  computing the difference.
    //
    VectorI ub(nrows), lb(nrows);
    thrust::upper_bound(source.begin(), source.end(),
			thrust::make_counting_iterator(size_t{0}),
			thrust::make_counting_iterator(nrows),
			ub.begin());

    //
    //  At this point ub[i] is the offset of the end of the string
    //  of occurrences for row id i.
    //

    thrust::lower_bound(source.begin(), source.end(),
			thrust::make_counting_iterator(size_t{0}),
			thrust::make_counting_iterator(nrows),
			lb.begin());

    //
    //  At this point lb[i] is the offset of the beginning of the string
    //  of occurrences for row id i.
    //

    thrust::transform(ub.begin(), ub.end(), lb.begin(), counts.begin(), thrust::minus<int>());

    //
    //  Counts is now the number of times each index occurs in the data.  So we
    //  can compute prefix sums to create our new row index array.
    //
    thrust::exclusive_scan(counts.begin(), counts.end(),
			   subg_row_ptr.begin());

    subg_row_ptr.back() = actual_nnz;
  }

  //used by renumber_indices(...)
  //
  template<typename Container>
  struct Hasher
  {
    typedef typename Container::value_type IndexT;
    typedef typename VectorPtrT<typename Container::value_type,Container>::PtrT PtrT;

    explicit Hasher(Container& hash_src):
      m_hash(&hash_src[0])
    {
    }
    __host__ __device__
    IndexT operator() (IndexT i, IndexT v)
    {
      m_hash[v] = i;
      return v;
    }

    __host__ __device__
    IndexT operator() (IndexT u)
    {
      return m_hash[u];
    }
  private:
    PtrT m_hash;
  };

  //#####  Change 3: index renumbering must be split into hash construction and hash usage #####
  //constructs hash table
  //from set of indices into reduced set of indices:
  //row_idx{5,7,10,12}->{0,1,2,3};
  // so that given u{12,7} you get: w{3,1}
  //w[i]=hash[u[i]]; 
  //
  //Pre-conditions:
  //(1.) row_idx is sorted (increasing order);
  //(2.) row_idx has no duplicates;
  //
  template<typename VectorI>
  void renumber_indices(VectorI& row_idx, //in: subset of row indices; 
			//    pre-conditions=
			//    {sorted (increasingly), no duplicates} 
			VectorI& hash_t)  //out: renumbering hash table
  {
    typedef typename VectorI::value_type IndexT;
    size_t n = row_idx.size();
    VectorI dummy(n,IndexT(0));

    IndexT max_entry = row_idx.back();//...since row_idx is sorted increasingly 
    hash_t.assign(max_entry+1, -1);

    Hasher<VectorI> hasher(hash_t);

    thrust::counting_iterator<IndexT> first(0);

    thrust::transform(first, first+n, 
		      row_idx.begin(), 
		      dummy.begin(),
		      hasher);
  }

  template<typename VectorI>
  void get_renumbered_indices(VectorI& u,       //in: in=subset of row_idx; 
			      VectorI& hash_t,  //in: renumbering hash table
			      VectorI& w)       //out:renumbered: hash[u[i]]
  {
    typedef typename VectorI::value_type IndexT;

    Hasher<VectorI> hasher(hash_t);

    thrust::transform(u.begin(), u.end(),
		      w.begin(),
		      hasher);
  }

  template<typename VectorI,
	   typename VectorV,
	   typename VectorB = VectorI>
  struct SubGraphExtractorFunctor
  {
    typedef typename VectorI::value_type IndexT;
    typedef typename VectorV::value_type ValueT;
    typedef typename VectorB::value_type ValueB;

    typedef typename VectorPtrT<typename VectorB::value_type,VectorB>::PtrT PtrB;
    typedef typename VectorPtrT<typename VectorI::value_type,VectorI>::PtrT PtrI;
    typedef typename VectorPtrT<typename VectorV::value_type,VectorV>::PtrT PtrV;
  
    //constructor for edge subset:
    //requires additional info: col_ind, row_ptr
    //
    //pre-conditions: (1.) eSub sorted in ascending order;
    //                (2.) eSub has no duplicates;
    //
    SubGraphExtractorFunctor(const VectorI& eSub, bool /*unused*/):
      edgeSubset(eSub),
      is_vertex_extraction(false)
    {
    }

    explicit SubGraphExtractorFunctor(const VectorI& vSubset):
      vertexSubset(vSubset),
      is_vertex_extraction(true)
    {
      //make sure vertexSubset_ is sorted increasingly:
      ///sort_ifnot(vertexSubset);

      row_ptr_subg.assign(vSubset.size()+1, IndexT(0)); // can be pre-allocated
    }
   
    
    virtual ~SubGraphExtractorFunctor(void)
    {
    }
    
    const VectorV& get_vals(void) const
    {
      return vals_subg;
    }

    VectorV& get_vals(void)
    {
      return vals_subg;
    }
    
    const VectorI& get_row_ptr(void) const
    {
      return row_ptr_subg;
    }
    
    const VectorI& get_col_ind(void) const
    {
      return col_ind_subg;
    }
    
    struct NoValueUpdater
    {
      //#####  Change 5: const K #####
      //
      void update_from(const VectorI& K)
      {
	//no-op....
      }
    };

    virtual void operator () (VectorI& row_ptr_,
			      VectorI& col_ind_)
    {
      NoValueUpdater no_op;
      if( is_vertex_extraction )
	extract_subgraph_by_vertex(row_ptr_, col_ind_, no_op);
      else
	extract_subgraph_by_edge(row_ptr_, col_ind_, no_op);
    }

    
    virtual void operator () (VectorV& vals_,
			      VectorI& row_ptr_,
			      VectorI& col_ind_)
    {
      ValueUpdater<VectorV, VectorI> fctrv(vals_, vals_subg);
      if( is_vertex_extraction )
	extract_subgraph_by_vertex(row_ptr_, col_ind_, fctrv);
      else
	extract_subgraph_by_edge(row_ptr_, col_ind_, fctrv);
    }

    IndexT get_subg_nnz(void) const
    {
      return row_ptr_subg.back();
    }

    const VectorI& get_I(void) const
    {
      return I;
    }

    const VectorI& get_J(void) const
    {
      return J;
    }

    const VectorI& get_K(void) const
    {
      return K;
    }


    const VectorI& get_hash_table(void) const
    {
      return hash_t;
    }

    const VectorI& get_vertex_subset(void) const
    {
      return vertexSubset;
    }


  protected:
    
    template<typename ValUpdaterFctr>
    void extract_subgraph_by_vertex(VectorI& row_ptr_,
				    VectorI& col_ind_,
				    ValUpdaterFctr fctrv)
    {
      typedef typename VectorI::value_type IndexT;
      //typedef typename VectorV::value_type ValueT;
      typedef typename VectorB::value_type ValueB;

      if( vertexSubset.empty() )
	return; //nothing to do

      //Pre-condition (new): vertexSubset sorted!
      size_t nrows_subg = vertexSubset.size();
        
      //step 1: subgraph *upper-bound* 
      //of #non-zeros per row:
      VectorI nzs(nrows_subg, 0);
      //count_nz_per_row(row_ptr_, vertexSubset, nzs);
      NzCounter<VectorI> count_nzs(row_ptr_);
      thrust::transform(vertexSubset.begin(), vertexSubset.end(), 
			nzs.begin(), 
			count_nzs);
        
      //step 2: offsets of where each
      //subgraph row *could* have entries;
      //
      //TODO: change to an exclusive prefix scan!
      //
      VectorI offsets(nrows_subg, 0);
      thrust::exclusive_scan(nzs.begin(), nzs.end(),
			     offsets.begin());
        
      //step 3: total # non-zero entries; this is used as upper bound
      //for # non-zero entries of subgraph;
      //
      size_t nnz = offsets.back()+nzs.back();

      VectorI range(nnz, IndexT(0));//increasing sequence
      thrust::sequence(range.begin(), range.end(),IndexT(0));//or, counting_iterator
	
      VectorB hash_rows;
      size_t hash_sz = make_hash(vertexSubset, hash_rows, true);
        
      //step 4: create hash map between nz entry and corresponding 
      // I[], J[], V[], Used[] SoA; update vals_
      //
      I.assign(nnz, IndexT(0));
      J.assign(nnz, IndexT(0));
      K.assign(nnz, IndexT(0));

      VectorB U(nnz, ValueB(0));

      size_t actual_nnz = fill_hash_nz2ijv(hash_rows, 
					   range, 
					   nzs, 
					   offsets, 
					   vertexSubset, 
					   row_ptr_, 
					   col_ind_, 
					   I, J, K, U); 

      //#####  Change 4: subg row_ptr extraction requires renumbering first #####
      renumber_indices(vertexSubset, hash_t);

      VectorI I_sg(actual_nnz, IndexT(0));
      get_renumbered_indices(I,      //in: in=sources; 
			     hash_t, //in: renumbering hash table
			     I_sg);  //out:renumbered: sources[]

#ifdef DEBUG_NEW
      std::cout<<"I_sg: ";
      print_v(I_sg, std::cout);

      std::cout<<"nnz="<<actual_nnz<<std::endl;
      std::cout<<"I.size()="<<I.size()<<std::endl;
#endif
	
      //####################################  Change 2:
      //step 5: extract subgraph CSR data:
      //
      make_subg_row_ptr(actual_nnz,
			nrows_subg,
			I_sg,
			row_ptr_subg);
	       
      //step 6: update col_ind and re-number:
      //
      col_ind_subg.assign(actual_nnz, IndexT(0));

      //####################################  Change 3:
      get_renumbered_indices(J,            //in: in=sinks; 
			     hash_t,       //in: renumbering hash table
			     col_ind_subg);//out:renumbered: col_ind[]

      //#####  Change 7: get edge subset from original graph #####
      edgeSubset = K; // copy !!!

      //act (or not) on values:
      //
      fctrv.update_from(K);
    }

    //#####  Change 6: separate logic for extraction by edges #####
    //
    template<typename ValUpdaterFctr>
    void extract_subgraph_by_edge(VectorI& row_ptr,
				  VectorI& col_ind,
				  ValUpdaterFctr fctrv)
    {
      if( edgeSubset.empty() )
	return; //nothing to do

      size_t nedges = edgeSubset.size();

      K = edgeSubset; // copy!!!

      VectorI sinks0(nedges);

      //get edge sinks:
      //just extract the col_ind 
      //values at indices specified by eSub:
      //
      //
      //old solution...
      // Filter<Container, ValT> filter(col_ind);
      // thrust::transform(eSub.begin(), eSub.end(), 
      // 		sinks0.begin(),
      // 		filter);
      //
      //...replace with gather:
      //
      thrust::gather(edgeSubset.begin(), edgeSubset.end(), //range of indexes...
		     col_ind.begin(),          //...into source
		     sinks0.begin());          //destination (result)

      //subg_col_ind[] = sink entries corresponding 
      //to *sorted* source entries
      //at this point both sources and sinks are sorted,
      //but that doesn't mean that sinks[i] and sources[i] form edges...
      //(use multi_sort_SoA?)
      //
      //Actually: since sources[] should come out sorted regardless of sinks[]
      //the corresponding sinks[] are just sinks0[] before sorting it!
      //
      //J[] is just the unsorted sinks:
      //
      J = sinks0; // copy!!!
  
#ifdef DEBUG_EDGES
      std::cout<<"sinks:";
      print_v(J, std::cout);
#endif

      //sort sinks to later do a merge with them:
      //
      thrust::sort(sinks0.begin(), sinks0.end()); 

      //hash[val_i] = i (index of first occurence of val_i, if val_i exists in v[]; 
      //                 else, last occurence of closest value less than val_i):
      //
      //(not ot be confused with renumbering hash, hash_t)
      //
      VectorI hash;
      reverse_hash(row_ptr, hash);

#ifdef DEBUG_EDGES
      std::cout<<"hash:";
      print_v(hash, std::cout);
#endif
  
      //now get sources:
      //apply hash on eSub,
      //i.e., extract the hash 
      //values at indices specified by eSub:
      //(the result should be sorted, 
      // because eSub is assumed sorted
      // and hash has indices of a sorted array: row_ptr)
      //
      I.assign(nedges, IndexT(0)); //I[] = sources !!!
      //
      //old solution...
      // Filter<Container, ValT> hash_app(hash);
      // thrust::transform(eSub.begin(), eSub.end(), 
      // 		sources.begin(),
      // 		hash_app);
      //
      //replaced by gather...
      //
      thrust::gather(edgeSubset.begin(), edgeSubset.end(), //range of indexes...
		     hash.begin(),             //...into source
		     I.begin());         //destination (result)

      assert( sinks0.size() == I.size() );

#ifdef DEBUG_EDGES
      std::cout<<"sources:";
      print_v(I, std::cout);
#endif

      //now merge sinks with sources
      //
      VectorI v(nedges<<1);//twice as many edges...
      thrust::merge(sinks0.begin(), sinks0.end(),
		    I.begin(), I.end(),
		    v.begin());

      size_t nrows_subg = count_get_distinct(v, vertexSubset);

      //renumber row (vertex) indices:
      //
      renumber_indices(vertexSubset, hash_t);

      get_renumbered_indices(I, //in: in=sources; 
			     hash_t,  //in: renumbering hash table
			     sinks0); //out:renumbered: sources[]

      //create subgraph row_ptr,
      //operating on sources:
      //
      row_ptr_subg.resize(nrows_subg+1);
      make_subg_row_ptr(nedges,     //==actual_nnz
			nrows_subg,
			sinks0,
			row_ptr_subg);
 
      //renumber subg_col_ind:
      //
      col_ind_subg.resize(nedges);
      get_renumbered_indices(J,             //in: in=sinks; 
			     hash_t,        //in: renumbering hash table
			     col_ind_subg); //out:renumbered: subg_col_ind[]

      //act (or not) on values:
      //
      fctrv.update_from(K);
    }

  private:
    VectorI vertexSubset;  //original graph vertex indices used in subgraph

    //####################################  Change 7:
    //
    VectorI edgeSubset;    //original graph edge indices used in subgraph

    
    VectorV vals_subg;     //not used for non-valued graphs
    VectorI row_ptr_subg;
    VectorI col_ind_subg;

    //useful for mapping graph <--> subgraph:
    //
    VectorI I;      //subgraph's set of (original graph) row indices
    VectorI J;      //subgraph's set of (original graph) col indices
                    //hence, (I[k], J[k]) is an edge in subgraph

    VectorI K;      //subgraph's set of (original graph) edge indices 
    
    VectorI hash_t;

    const bool is_vertex_extraction;
  };






  //Acyclic Visitor
  //         (A. Alexandrescu, "Modern C++ Design", Section 10.4), 
  //         where *concrete* Visitors must be parameterized by all 
  //         the possibile template args of the Visited classes (visitees);
  //

  //Visitor for SubGraph extraction:
  //
  template<typename VectorI, 
		   typename VectorV>
  struct SubGraphExtractorVisitor: 
    VisitorBase,
    Visitor<Graph<typename VectorI::value_type> >,
    Visitor<CsrGraph<typename VectorI::value_type> >,
    Visitor<ValuedCsrGraph<typename VectorI::value_type, typename VectorV::value_type> >,
    Visitor<MultiValuedCsrGraph<typename VectorI::value_type, typename VectorV::value_type> >
  {
    typedef typename VectorI::value_type IndexType_;
    typedef typename VectorV::value_type ValueType_;
    typedef typename VectorPtrT<typename VectorI::value_type,VectorI>::PtrT PtrI;

    //TODO: avoid copy from raw pointer
    //
    SubGraphExtractorVisitor(CsrGraph<IndexType_>& graph,    
			     const VectorI& vSub, 
			     cudaStream_t stream):
      row_ptr_(graph.get_raw_row_offsets(), graph.get_raw_row_offsets()+graph.get_num_vertices()+1),
      col_ind_(graph.get_raw_column_indices(), graph.get_raw_column_indices()+graph.get_num_edges()),	  
      extractor_(vSub),
      stream_(stream)
    {
    }

    //TODO: avoid copy from raw pointer
    //
    SubGraphExtractorVisitor(CsrGraph<IndexType_>& graph,
			     const VectorI& eSub,       
			     cudaStream_t stream,
			     bool use_edges):     //just to differentiate vertex vs. edge semantics; value not used
      row_ptr_(graph.get_raw_row_offsets(), graph.get_raw_row_offsets()+graph.get_num_vertices()+1),
      col_ind_(graph.get_raw_column_indices(), graph.get_raw_column_indices()+graph.get_num_edges()),
      extractor_(eSub, false),       //different semantics!
      stream_(stream)
    {
    }  

    void Visit(Graph<IndexType_>& graph)
    {
      //no-op...
    }

    void Visit(CsrGraph<IndexType_>& graph)
    {
      // size_t g_nrows = graph.get_num_vertices();
      // size_t g_nnz = graph.get_num_edges();

      // VectorI row_ptr(graph.get_raw_row_offsets(), graph.get_raw_row_offsets()+g_nrows+1);
      // VectorI col_ind(graph.get_raw_column_indices(), graph.get_raw_column_indices()+g_nnz);

      extractor_(row_ptr_, col_ind_);//TODO: modify operator to work directly with PtrI

      size_t rowptr_sz = extractor_.get_row_ptr().size();
      assert( rowptr_sz >= 1 );

      size_t subg_nrows = rowptr_sz-1;
      size_t subg_nnz = extractor_.get_subg_nnz();

      subgraph_ = new CsrGraph<IndexType_>(subg_nrows, subg_nnz, stream_);

      //TODO: more efficient solution: investigate if/how copy can be avoided
      //
      thrust::copy(extractor_.get_row_ptr().begin(), extractor_.get_row_ptr().end(), subgraph_->get_raw_row_offsets());
      thrust::copy(extractor_.get_col_ind().begin(), extractor_.get_col_ind().end(), subgraph_->get_raw_column_indices());
    }

    //might not need to implement following Visit methods,
    //the one above for CsrGraph might work for derived
    //classes...
    void Visit(ValuedCsrGraph<IndexType_,ValueType_>& graph)
    {
      size_t g_nrows = graph.get_num_vertices();
      size_t g_nnz = graph.get_num_edges();

      // VectorI row_ptr(graph.get_raw_row_offsets(), graph.get_raw_row_offsets()+g_nrows+1);
      // VectorI col_ind(graph.get_raw_column_indices(), graph.get_raw_column_indices()+g_nnz);
      VectorV vals(graph.get_raw_values(), graph.get_raw_values()+g_nnz);

      extractor_(vals, row_ptr_, col_ind_);//TODO: modify operator to work directly with PtrI

      size_t rowptr_sz = extractor_.get_row_ptr().size();
      assert( rowptr_sz >= 1 );

      size_t subg_nrows = rowptr_sz-1;
      size_t subg_nnz = extractor_.get_subg_nnz();

      ValuedCsrGraph<IndexType_,ValueType_>* subg = new ValuedCsrGraph<IndexType_,ValueType_>(subg_nrows, subg_nnz, stream_);

      //TODO: more efficient solution: investigate if/how copy can be avoided
      //
      thrust::copy(extractor_.get_row_ptr().begin(), extractor_.get_row_ptr().end(), subg->get_raw_row_offsets());
      thrust::copy(extractor_.get_col_ind().begin(), extractor_.get_col_ind().end(), subg->get_raw_column_indices());
      thrust::copy(extractor_.get_vals().begin(), extractor_.get_vals().end(), subg->get_raw_values());

      subgraph_ = subg;
    }

    void Visit(MultiValuedCsrGraph<IndexType_,ValueType_>& graph)
    {
      size_t g_nrows = graph.get_num_vertices();
      size_t g_nnz = graph.get_num_edges();

      // VectorI row_ptr(graph.get_raw_row_offsets(), graph.get_raw_row_offsets()+g_nrows+1);
      // VectorI col_ind(graph.get_raw_column_indices(), graph.get_raw_column_indices()+g_nnz);
      /// VectorV vals(graph.get_raw_values(), graph.get_raw_values()+g_nnz);

	  ///extractor_(vals, row_ptr_, col_ind_);
      extractor_(row_ptr_, col_ind_);//TODO: modify operator to work directly with PtrI

      size_t rowptr_sz = extractor_.get_row_ptr().size();
      assert( rowptr_sz >= 1 );

      size_t subg_nrows = rowptr_sz-1;
      size_t subg_nnz = extractor_.get_subg_nnz();
      
      MultiValuedCsrGraph<IndexType_,ValueType_>* subg = new MultiValuedCsrGraph<IndexType_,ValueType_>(subg_nrows, subg_nnz, stream_);

      //TODO: more efficient solution: investigate if/how copy can be avoided
      //
      thrust::copy(extractor_.get_row_ptr().begin(), extractor_.get_row_ptr().end(), subg->get_raw_row_offsets());
      thrust::copy(extractor_.get_col_ind().begin(), extractor_.get_col_ind().end(), subg->get_raw_column_indices());
      ///thrust::copy(extractor_.get_vals().begin(), extractor_.get_vals().end(), subg->get_raw_values());

      //additional data extraction:
      //
      get_vertex_data(graph, extractor_.get_vertex_subset(), *subg);
      get_edge_data(graph, extractor_.get_K(), *subg);

      subgraph_ = subg;
    }

    const SubGraphExtractorFunctor<VectorI, VectorV>& get_extractor(void) const
    {
      return extractor_;
    }

    CsrGraph<IndexType_>* get_subgraph(void) // TODO: change to unique_ptr, when moving to C++1*
    {
      return subgraph_;
    }
  protected:
    void get_edge_data(MultiValuedCsrGraph<IndexType_,ValueType_>& graph_src,
		       const VectorI& K, //subset of graph edge set
		       MultiValuedCsrGraph<IndexType_,ValueType_>& graph_dest)
    {
      typedef thrust::device_ptr<ValueType_> PtrV;

      size_t ng = graph_src.get_num_edge_dim();
      size_t nedges = K.size();

      assert( nedges == graph_dest.get_num_edges() );

      graph_dest.allocateEdgeData(ng, stream_);
      
      for(unsigned int i=0;i<ng;++i)
		{
		  Vector<ValueType_>& v_src = graph_src.get_edge_dim(i);
		  Vector<ValueType_>& v_dest = graph_dest.get_edge_dim(i);

		  size_t n_src = v_src.get_size();
		  PtrV ptr_src(v_src.raw());
		  range_view<PtrV> rv_src(ptr_src, ptr_src+n_src);

		  size_t n_dest = v_dest.get_size();
		  assert( nedges == n_dest );

		  PtrV ptr_dest(v_dest.raw());
		  range_view<PtrV> rv_dest(ptr_dest, ptr_dest+n_dest);

		  thrust::gather(K.begin(), K.end(), //map of indices
						 rv_src.begin(),     //source
						 rv_dest.begin());   //source[map]
		}
    }

    void get_vertex_data(MultiValuedCsrGraph<IndexType_,ValueType_>& graph_src,
			 const VectorI& K,// subset of graph vertex set == vSub
			 MultiValuedCsrGraph<IndexType_,ValueType_>& graph_dest)
    {
      typedef thrust::device_ptr<ValueType_> PtrV;

      size_t ng = graph_src.get_num_vertex_dim();
      size_t nrows = K.size();//remember, K==vSub, here!

      assert( nrows == graph_dest.get_num_vertices() );

      graph_dest.allocateVertexData(ng, stream_);
      
      for(unsigned int i=0;i<ng;++i)
		{
		  Vector<ValueType_>& v_src = graph_src.get_vertex_dim(i);
		  Vector<ValueType_>& v_dest = graph_dest.get_vertex_dim(i);

		  size_t n_src = v_src.get_size();
		  PtrV ptr_src(v_src.raw());
		  range_view<PtrV> rv_src(ptr_src, ptr_src+n_src);

		  size_t n_dest = v_dest.get_size();
		  assert( nrows == n_dest );

		  PtrV ptr_dest(v_dest.raw());
		  range_view<PtrV> rv_dest(ptr_dest, ptr_dest+n_dest);

		  thrust::gather(K.begin(), K.end(), //map of indices
						 rv_src.begin(),     //source
						 rv_dest.begin());   //source[map]
		}
    }
  private:
    VectorI row_ptr_;
    VectorI col_ind_;
    SubGraphExtractorFunctor<VectorI, VectorV> extractor_;
    cudaStream_t stream_;
    CsrGraph<IndexType_>* subgraph_; // to be constructed
  };

  template<typename T>
  struct BoundValidator
  {
    BoundValidator(const T& lower_bound,
		   const T& upper_bound):
      lbound_(lower_bound),
      ubound_(upper_bound)
    {
    }

    __host__ __device__
    bool operator() (T k)
    {
      return ( k < lbound_ || k > ubound_ );
    }

  private:
    T lbound_;
    T ubound_;
  };

  template<typename Container>
  struct NotSortedAscendingly
  {
    typedef typename Container::value_type VType;
    typedef typename VectorPtrT<VType,Container>::PtrT PtrT;

    NotSortedAscendingly(Container& rv, const size_t& sz):
      ptr_(&rv[0]),
      sz_(sz)
    {
        
    }
    
    __host__ __device__
    bool operator() (VType k)
    {
      if( k+1 < sz_ )
	return ptr_[k+1] < ptr_[k];
      else
	return false;
    }
  private:
    PtrT ptr_;//no reference! must be copy constructed
    size_t sz_;
  };

  template<typename VectorI>
  void validate_input(VectorI& v, typename VectorI::value_type sz)
  {
    typedef typename VectorI::value_type IndexT;

    size_t n = v.size();

    if( n == 0 )
      FatalError("0-sized array input in subgraph extraction.",NVGRAPH_ERR_BAD_PARAMETERS);

    IndexT lb = 0;
    IndexT ub = sz-1;
    BoundValidator<IndexT> bvld(lb, ub);//closed interval!
    typename VectorI::iterator pos = thrust::find_if(v.begin(), v.end(), bvld);
    if( pos != v.end() )
      FatalError("Input is not a valid subset of the graph's corresponding set.",NVGRAPH_ERR_BAD_PARAMETERS);

    VectorI seq(n,0);
    thrust::sequence(seq.begin(), seq.end());
    NotSortedAscendingly<VectorI> nsa_f(v, n);
    pos = thrust::find_if(seq.begin(), seq.end(), nsa_f);
    if( pos != seq.end() )
      FatalError("Input array not sorted in ascending order.",NVGRAPH_ERR_BAD_PARAMETERS);

    pos = thrust::unique(v.begin(), v.end());
    if( pos != v.end() )
      FatalError("Input array has duplicates.",NVGRAPH_ERR_BAD_PARAMETERS);
	
  }

  template<typename IndexT, typename ValueT>
  CsrGraph<IndexT>* extract_from_vertex_subset(CsrGraph<IndexT>& graph, 
					       IndexT* pV, size_t n, cudaStream_t stream)
  {
    typedef thrust::device_vector<IndexT> VectorI;
    typedef thrust::device_vector<ValueT> VectorV;
    VectorI vSub(pV, pV+n);

    validate_input(vSub, graph.get_num_vertices());

    SubGraphExtractorVisitor<VectorI, VectorV> visitor(graph, vSub, stream);
    graph.Accept(visitor);
    return visitor.get_subgraph();
  }

  template<typename IndexT, typename ValueT>
  CsrGraph<IndexT>* extract_from_edge_subset(CsrGraph<IndexT>& graph, 
					     IndexT* pV, size_t n, cudaStream_t stream)
  {
    typedef thrust::device_vector<IndexT> VectorI;
    typedef thrust::device_vector<ValueT> VectorV;
    VectorI vSub(pV, pV+n);

    validate_input(vSub, graph.get_num_edges());

    SubGraphExtractorVisitor<VectorI, VectorV> visitor(graph, vSub, stream, true);
    graph.Accept(visitor);
    return visitor.get_subgraph();
  }
  
}//end namespace

#endif
