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

#include <vector>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <cassert>

namespace nvgraph{
namespace debug{

//Sequential CSR graph extractor
//for DEBUGGING purposes, only
//
template<typename VectorI,
	 typename VectorV,
	 typename VectorB = VectorI>
struct SeqSubGraphExtractorFunctor
{
  typedef typename VectorI::value_type IndexT;
  typedef typename VectorV::value_type ValueT;
  typedef typename VectorB::value_type ValueB;

  explicit SeqSubGraphExtractorFunctor(const VectorI& vSubset):
    vertexSubset(vSubset)
  {
	//make sure vertexSubset_ is sorted increasingly:
	///sort_ifnot(vertexSubset);
  }
    
  virtual ~SeqSubGraphExtractorFunctor(void)
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

  struct ValueUpdater
  {
	ValueUpdater(const VectorV& v_src,
				 VectorV& v_dest):
	  v_s_(v_src),
	  v_d_(v_dest)
	{
	}

	//debug: (sequential version only)
	void operator() (const IndexT& j)
	{
	  v_d_.push_back(v_s_[j]);
	}
            
	ValueT at(IndexT j) const
	{
	  return v_s_[j];
	}
      
	void update_vals(const VectorV& vals)
	{
	  v_d_ = vals;
	}
  private:
	const VectorV& v_s_;
	VectorV& v_d_;
  };

  struct NoValueUpdater
  {
	void operator() (const IndexT& j)
	{
	  //no-op...
	}

	ValueT at(IndexT j) const
	{
	  return ValueT(0); //nothing meaningful...
	}

	void update_vals(const VectorV& vals)
	{
	  //no-op...
	}
  };
    
  virtual void operator () (VectorI& row_ptr_,
							VectorI& col_ind_)
  {
	NoValueUpdater fctr;
	sequential_extract_subgraph(row_ptr_, col_ind_, fctr);
  }
    
  virtual void operator () (VectorV& vals_,
							VectorI& row_ptr_,
							VectorI& col_ind_)
  {
	ValueUpdater fctr(vals_, vals_subg);
	sequential_extract_subgraph(row_ptr_, col_ind_, fctr);
  }
    
protected:
    
  //for debugging purposes, only:
  //
  template<typename ValUpdaterFctr>
  void sequential_extract_subgraph(const VectorI& row_ptr_,
								   const VectorI& col_ind_,
								   ValUpdaterFctr& fctr)
  {
	VectorI all_zeros;
        
	IndexT last_updated_pos(0);
	//
	size_t nrows_subg = vertexSubset.size();
        
	VectorB hash_rows;
	size_t hash_sz = make_hash(vertexSubset, hash_rows);//assume *NOT* sorted
        
	row_ptr_subg.assign(nrows_subg+1, IndexT(0));
	all_zeros.reserve(nrows_subg);
        
	IndexT nz_subg(0);
        
	//this loop assumes sorted vertexSubset
	//
	for(IndexT i=IndexT(0);i<IndexT(nrows_subg);++i)
	  {
		IndexT row_index = vertexSubset[i];
		bool first_nz_inrow = true;
            
		for(IndexT j=row_ptr_[row_index]; j<row_ptr_[row_index+1];++j)
		  {
			IndexT k = col_ind_[j];
			if( (k<hash_sz) && (hash_rows[k] == 1) )//in vertex subset!
			  ///if( std::binary_search(vertexSubset.begin(), vertexSubset.end(), k) )//in vertex subset!
			  {
				///vals_subg.push_back(vals_[j]);//functor! (no-op vs push_back())
				fctr(j);//synch issues for parallel!
                    
				col_ind_subg.push_back(k);//synch issues for parallel!
                    
				++nz_subg;
                    
				//synch issues for parallel:
				//
				if( first_nz_inrow )//update row_ptr_subg
				  {
					row_ptr_subg[i] = last_updated_pos;
					first_nz_inrow = false;
				  }
                    
				++last_updated_pos;//synch issues for parallel!
			  }
		  }//end for(j;..)
            
		//special cases of a row with all zeros: mark it!
		if (first_nz_inrow)
		  {
			all_zeros.push_back(i);
		  }
	  }//end for(i;...)
        
	assert( nz_subg == col_ind_subg.size() );
        
	//last entry in row_ptr_subg:
	row_ptr_subg.back() = nz_subg;
        
	//handle all zero row cases:
	fix_zero_rows(all_zeros, row_ptr_subg);
        
	//assume *NOT* sorted
	remap_indices(vertexSubset, col_ind_subg);
  }

  struct UpdateRowPtr
  {
	explicit UpdateRowPtr(VectorI& row_p): row_p_(row_p)
	{
	}
            
	void operator() (const IndexT& i)
	{
	  row_p_[i] = row_p_[i+1];
	}
  private:
	VectorI& row_p_;
  };
    
  //correct row_ptr: iterate all_zeros from end towards beginning
  //and correct row_ptr_ at corresponding index
  //
  static void fix_zero_rows(const VectorI& all_zeros,
							VectorI& row_ptr)
  {    
	UpdateRowPtr correcter(row_ptr);
        
	//reverse traversal!
	//
	std::for_each(all_zeros.rbegin(), all_zeros.rend(), correcter);
  }
   
  template<typename Container>
  struct HashFctr
  {
	explicit HashFctr(Container& hash_src):
	  m_hash(hash_src)
	{
	}
	IndexT operator() (const IndexT& src_elem)
	{
	  IndexT hit(1);
	  m_hash[src_elem] = hit;
	  return hit;
	}
  private:
	Container& m_hash;
  };

  static size_t make_hash(const VectorI& src,
						  VectorB& hash_src,
						  bool is_sorted = false)
  {
	assert( !src.empty() );
        
	IndexT max_entry(0);
	if( is_sorted )
	  max_entry = src.back();
	else
	  max_entry = *std::max_element(src.begin(), src.end());
        
	hash_src.assign(max_entry+1, 0);
	VectorB dummy(hash_src);	
        
	HashFctr<VectorB> hctr(hash_src);

	//why unused dummy? 
	//because functor must return something  
	//and must store result of functor somewhere!
	//
	std::transform(src.begin(), src.end(), 
				   dummy.begin(), //unused...
				   hctr);

	return hash_src.size();
  }
    
  //re-number vertices:
  //
  static void remap_indices(const VectorI& src,
							VectorI& index_set,
							bool is_sorted = false)
  {
	IndexT max_entry(0);
	if( is_sorted )
	  max_entry = src.back();
	else
	  max_entry = *std::max_element(src.begin(), src.end());
        
	//use hash_src vector as hash-table:
	//
	VectorI hash_src(max_entry+1, IndexT(0));
        
	IndexT counter(0);
	for(typename VectorI::const_iterator pos = src.begin();
		pos != src.end();
		++pos)
	  {
		hash_src[*pos]=counter++;//SEQUENTIALITY!!!
	  }
        
	IndexT set_sz(index_set.size());
	VectorI old_index_set(index_set);
        
	for(IndexT k = IndexT(0);k<set_sz;++k)
	  {
		index_set[k] = hash_src[old_index_set[k]];
	  }
  }

private:
  VectorI vertexSubset;
    
  VectorV vals_subg;     //not used for non-valued graphs
  VectorI row_ptr_subg;
  VectorI col_ind_subg;
};

}//end namespace debug
}//end namespace nvgraph
