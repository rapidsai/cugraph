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

#ifndef incidence_graph_hxx
#define incidence_graph_hxx

#include <iostream>
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <cassert>



#define DEBUG_
//

namespace nvgraph{
namespace debug{

typedef std::vector<std::vector<int> > MatrixI;

//IndexT = index type to store in the incidence Matrix
//VertexT = value type to store for each vertex
//EdgetT = value type to store for each edge
//
//Graph stored by inidence matrix
//for DEBUGGING purposes, only
//(of small graphs)
//
template<typename IndexT, typename VertexT, typename EdgeT>
struct Graph
{
  typedef IndexT TypeI;
  typedef VertexT TypeV;
  typedef EdgeT TypeE;
    
  Graph(void): nrows_(0), ncols_(0)
  {
  }
    
  explicit Graph(const MatrixI& incidence):
    nrows_(incidence.size()),
    ncols_(incidence[0].size()),//throws on empty incidence!
    incidence_(incidence)
  {
	//construct the other members?
  }
    
  virtual ~Graph(){}

  void add_vertex(const VertexT& value)
  {
	//add row and column:
	++nrows_;
	++ncols_;

	for(typename MatrixI::iterator row=incidence_.begin();row!=incidence_.end();++row)
	  {
		(*row).push_back(IndexT(0));
	  }
        
	// for(auto& row:incidence_)
	//   {
	// 	row.push_back(IndexT(0));
	//   }
	incidence_.push_back(std::vector<IndexT>(ncols_,IndexT(0)));
        
	vertex_values_.push_back(value);
  }
    
  void add_edge(const EdgeT& value,
				const std::pair<IndexT,IndexT>& endpoints /*first = source, second=sink*/)
  {
	IndexT i = endpoints.first;
	IndexT j = endpoints.second;
        
	incidence_[i][j] = IndexT(1);
	edge_values_.insert(std::make_pair(endpoints,value));
  }
    
  friend std::ostream& operator<<(std::ostream& os, const Graph& g)
  {
	g.print(os);
        
	return os;
  }
    
  const MatrixI& get_incidence(void) const
  {
    return incidence_;
  }

  MatrixI& get_incidence(void)
  {
    return incidence_;
  }
    
  size_t get_nrows(void) const
  {
    return nrows_;
  }

  size_t& get_nrows(void)
  {
    return nrows_;
  }
    
  size_t get_ncols(void) const
  {
    return ncols_;
  }

  size_t& get_ncols(void)
  {
    return ncols_;
  }
    
  size_t get_nnz(void) const
  {
    return edge_values_.size();
  }
    
  const std::map<std::pair<IndexT, IndexT>, EdgeT>& get_edges(void) const
  {
    return edge_values_;
  }
    
  //must be public (for CsrGraph(Graph&))...why?
  std::map<std::pair<IndexT, IndexT>, EdgeT>& get_edges(void)
  {
    return edge_values_;
  }
    
  std::vector<VertexT>& get_vertices(void)
  {
    return vertex_values_;
  }
    
protected:
  struct RowPrinter
  {
	explicit RowPrinter(std::ostream& o):
	  m_os(o)
	{
	}

	void operator()(const std::vector<IndexT>& row)
	{
	  std::copy(row.begin(), row.end(), std::ostream_iterator<IndexT>(m_os, ","));
	  m_os<<"\n";
	}
  private:
	std::ostream& m_os;
  };

  void print_incidence(std::ostream& os) const
  {
    os<<"(nr,nc):("<<nrows_<<","<<ncols_<<")\n";
	
	RowPrinter rprint(os);
	std::for_each(incidence_.begin(), incidence_.end(), rprint);

    // std::for_each(incidence_.begin(), incidence_.end(), [&os](const std::vector<IndexT>& row){
	// 	std::copy(row.begin(), row.end(), std::ostream_iterator<IndexT>(os, ","));
	// 	os<<"\n";
    //   });
  }
    
  void print_vertices(std::ostream& os) const
  {
    int i=0;
	for(typename std::vector<VertexT>::const_iterator it=vertex_values_.begin();
		it!=vertex_values_.end();
		++it)
	  {
		os<<"v["<<i<<"]="<<*it<<",";
		++i;
	  }

	
    // for(auto entry:vertex_values_)
    //   {
	// 	os<<"v["<<i<<"]="<<entry<<",";
	// 	++i;
    //   }

    os<<"\n";
  }
    
  void print_edges(std::ostream& os) const
  {        

	for(typename std::map<std::pair<IndexT, IndexT>, EdgeT>::const_iterator it=edge_values_.begin();
		it!=edge_values_.end();
		++it)
	  {
		os<<"("<<it->first.first<<","<<it->first.second<<")="<<it->second<<",";
	  }

	  // for(auto entry:edge_values_)
	  // 	{
	  // 	  os<<"("<<entry.first.first<<","<<entry.first.second<<")="<<entry.second<<",";
	  // 	}

    os<<"\n";
  }
    
  virtual void print(std::ostream& os) const
  {
    print_incidence(os);
    print_vertices(os);
    print_edges(os);
  }
private:
  size_t nrows_;
  size_t ncols_;
    
  MatrixI incidence_;
  std::vector<VertexT> vertex_values_;
  std::map<std::pair<IndexT, IndexT>, EdgeT> edge_values_;
};

//CSR:
//for matrix A_{mxn} with nnz non-zero entries:
//
//vals[nnz]:    contains the non-zero entries in order left-right, top-down;
//              no entry for rows without non-zeros;
//row_ptr[m+1]: contains poition in "vals" of first non-zero entry for each row;
//              last element is nnz;
//              for empty row i, we repeat info from i+1 in row_ptr
//cols_ind[nnz]:contains column of each non-zero entry in vals;
//              no entry for rows without non-zeros;
/*
  col_ind[j] and vals[j] for j in [row_ptr[i], row_ptr[i+1]-1] represent the column index (unsigned integer) and value of matrix (double) on row i
*/
//
template<typename IndexT, typename VertexT, typename EdgeT>
struct CsrGraph: Graph<IndexT, VertexT, EdgeT>
{
  using Graph<IndexT, VertexT, EdgeT>::get_incidence;
  using Graph<IndexT, VertexT, EdgeT>::get_nrows;
  using Graph<IndexT, VertexT, EdgeT>::get_ncols;
  using Graph<IndexT, VertexT, EdgeT>::get_nnz;
  using Graph<IndexT, VertexT, EdgeT>::get_edges;//not confused by 2 versions of it...
  using Graph<IndexT, VertexT, EdgeT>::get_vertices;
    
  CsrGraph(void):Graph<IndexT, VertexT, EdgeT>()
  {
  }
    
  explicit CsrGraph(Graph<IndexT, VertexT, EdgeT>& g)://g must be non-const...why?
    Graph<IndexT, VertexT, EdgeT>(g.get_incidence())
    //,get_edges()(g.get_edges()) //fails to compile in initialization list...why?
  {
    get_edges() = g.get_edges();//ok!
    get_vertices() = g.get_vertices();
        
    to_csr();
  }

  CsrGraph(const std::vector<EdgeT>& vals,
	   const std::vector<IndexT>& row_ptr,
	   const std::vector<IndexT>& col_ind,
	   const std::vector<VertexT>& vertex_values):
    vals_(vals),
    row_ptr_(row_ptr),
    col_ind_(col_ind)
  {
    from_csr(vertex_values);
  }

  void from_csr(const std::vector<VertexT>& vertex_values)
  {
    ///size_t nnz = col_ind_.size();
    size_t nrows = vertex_values.size();
    get_nrows() = nrows;
    get_ncols() = nrows;

    get_incidence().assign(nrows,std::vector<IndexT>(nrows,IndexT(0)));
    get_vertices() = vertex_values;
      
    for(IndexT i=IndexT(0);i<IndexT(nrows);++i)
      {
		for(IndexT j=row_ptr_[i]; j<row_ptr_[i+1];++j)
		  {
			IndexT k = col_ind_[j];
			EdgeT v = vals_[j];
	      
			get_incidence()[i][k] = 1;
			get_edges().insert(std::make_pair(std::make_pair(i,k),v));
		  }
      }
  }
    
  void to_csr(void)
  {
    size_t nnz = get_nnz();
    size_t nrows = get_nrows();
    size_t ncols = get_ncols();
    //const auto& edges = get_edges();
    const std::map<std::pair<IndexT, IndexT>, EdgeT>& edges = get_edges();
        
    vals_.assign(nnz,EdgeT());
    row_ptr_.assign(nrows+1,IndexT(0));
    row_ptr_[nrows] = IndexT(nnz);
    col_ind_.assign(nnz,IndexT(0));
        
    const MatrixI& A = get_incidence();
    IndexT crt_row_ptr_i(0);
    IndexT crt_nz_i(0);
        
    std::vector<IndexT> all_zeros;
    all_zeros.reserve(nrows);
        
    for(IndexT i=0;i<nrows;++i)
      {
		bool first_nz_inrow = true;
		for(IndexT j=0;j<ncols;++j)
		  {
			if( A[i][j] != IndexT(0) )
			  {
				///std::pair<IndexT,IndexT> key(i,j);//ok
				//std::pair<IndexT,IndexT> key = std::make_pair<IndexT,IndexT>(i, j);//fails...why???
				//see: http://stackoverflow.com/questions/9641960/c11-make-pair-with-specified-template-parameters-doesnt-compile
                    
				std::pair<IndexT,IndexT> key = std::make_pair(i, j);
                    
				typename std::map<std::pair<IndexT, IndexT>, EdgeT>::const_iterator pos = edges.find(key);
				if (pos == edges.end())
				  {
					std::stringstream ss;
					ss << "ERROR: edge("<<i<<","<<j<<") not found.";
					throw std::runtime_error(ss.str());
				  }
				vals_[crt_nz_i] = pos->second;
                    
                    
				if (first_nz_inrow)
				  {
					row_ptr_[crt_row_ptr_i] = crt_nz_i;
					first_nz_inrow = false;
                        
					++crt_row_ptr_i;
				  }
				col_ind_[crt_nz_i] = j;
                    
				++crt_nz_i;
			  }//end if
		  }//end for j
            
		//special cases of a row with all zeros: mark it!
		if (first_nz_inrow)
		  {
			all_zeros.push_back(i);
		  }
      }//end for i
        
    //handle all zero row cases:
    fix_zero_rows(all_zeros, row_ptr_);   
  }
    
  const std::vector<EdgeT>& get_vals(void) const
  {
    return vals_;
  }

  std::vector<EdgeT>& get_vals(void)
  {
    return vals_;
  }
    
  const std::vector<IndexT>& get_row_ptr(void) const
  {
    return row_ptr_;
  }

  std::vector<IndexT>& get_row_ptr(void)
  {
    return row_ptr_;
  }
    
  const std::vector<IndexT>& get_col_ind(void) const
  {
    return col_ind_;
  }

  std::vector<IndexT>& get_col_ind(void)
  {
    return col_ind_;
  }
    
  friend std::ostream& operator<<(std::ostream& os, const CsrGraph& g)
  {
    g.Graph<IndexT, VertexT, EdgeT>::print(os);
    g.print(os);
        
    return os;
  }

  void extract_subgraph(std::vector<IndexT>& vertexSubset, 
			CsrGraph& subgraph) const
  {
    //check if vertexSubset is sorted increasingly:
    //
    
    if( std::adjacent_find(vertexSubset.begin(), vertexSubset.end(), std::greater<IndexT>()) 
		!= vertexSubset.end() )//not sorted in ascending order...
      {
		std::sort(vertexSubset.begin(), vertexSubset.end());
		//#ifdef DEBUG_
		std::copy(vertexSubset.begin(), vertexSubset.end(), std::ostream_iterator<IndexT>(std::cout,","));
		std::cout<<std::endl;
		//#endif
      }
    //#ifdef DEBUG_
    else
      {
		std::cout<<"was sorted...\n";
      }
    //#endif

    std::vector<EdgeT>& vals_subg = subgraph.vals_;
    std::vector<IndexT>& row_ptr_subg = subgraph.row_ptr_;
    std::vector<IndexT>& col_ind_subg = subgraph.col_ind_;

    std::vector<IndexT> all_zeros;

    IndexT last_updated_pos(0);
    //
    size_t nrows_subg = vertexSubset.size();

    row_ptr_subg.assign(nrows_subg+1, IndexT(0));
    all_zeros.reserve(nrows_subg);

    IndexT nz_subg(0);

    for(IndexT i=IndexT(0);i<IndexT(nrows_subg);++i)
      {
		IndexT row_index = vertexSubset[i];
		bool first_nz_inrow = true;

		for(IndexT j=row_ptr_[row_index]; j<row_ptr_[row_index+1];++j)
		  {
			IndexT k = col_ind_[j];
			if( std::binary_search(vertexSubset.begin(), vertexSubset.end(), k) )//in vertex subset!
			  {
				vals_subg.push_back(vals_[j]);
				col_ind_subg.push_back(k);

				++nz_subg;

				if( first_nz_inrow )//update row_ptr_subg
				  {
					row_ptr_subg[i] = last_updated_pos;
					first_nz_inrow = false;
				  }

				++last_updated_pos;
			  }
		  }//end for(j;..)

		//special cases of a row with all zeros: mark it!
		if (first_nz_inrow)
		  {
			all_zeros.push_back(i);
		  }
      }//end for(i;...)

    assert( nz_subg == vals_subg.size() );
    assert( nz_subg == col_ind_subg.size() );
    
    //last entry in row_ptr_subg:
    row_ptr_subg.back() = nz_subg;

    //handle all zero row cases:
    fix_zero_rows(all_zeros, row_ptr_subg);    

    remap_indices(vertexSubset, col_ind_subg);
  }

protected:
  void print(std::ostream& os) const
  {
    os<<"vals: ";
    std::copy(vals_.begin(), vals_.end(), std::ostream_iterator<EdgeT>(os,","));
    os<<"\n";
        
    os<<"row_ptr: ";
    std::copy(row_ptr_.begin(), row_ptr_.end(), std::ostream_iterator<IndexT>(os,","));
    os<<"\n";
        
    os<<"col_ind: ";
    std::copy(col_ind_.begin(), col_ind_.end(), std::ostream_iterator<IndexT>(os,","));
    os<<"\n";
  }

  struct Updater
  {
	explicit Updater(std::vector<IndexT>& row_ptr):
	  m_row_ptr(row_ptr)
	{
	}

	void operator()(const IndexT& i)
	{
	  m_row_ptr[i] = m_row_ptr[i+1];
	}
  private:
	std::vector<IndexT>& m_row_ptr;
  };

  //correct row_ptr: iterate all_zeros from end towards beginning 
  //and correct row_ptr_ at corresponding index
  //
  static void fix_zero_rows(const std::vector<IndexT>& all_zeros,
			    std::vector<IndexT>& row_ptr)
  {
	Updater up(row_ptr);
	std::for_each(all_zeros.rbegin(), all_zeros.rend(), up);
	
    // std::for_each(all_zeros.rbegin(), all_zeros.rend(), [&](const IndexT& i){
	// 	row_ptr[i] = row_ptr[i+1];
    //   });
  }

  struct HashUpdater
  {
	explicit HashUpdater(std::vector<IndexT>& hash):
	  m_hash(hash),
	  m_counter(0)
	{
	}

	void operator()(const IndexT& i)
	{
	  m_hash[i]=m_counter++;
	}
  private:
	std::vector<IndexT>& m_hash;
	IndexT m_counter;
  };

  //assumes src is ordered increasingly
  //
  static void remap_indices(const std::vector<IndexT>& src, 
			    std::vector<IndexT>& index_set)
  {
    IndexT max_entry = src.back();

    //use hash_src vector as hash-table:
    //
    std::vector<IndexT> hash_src(max_entry+1, IndexT(0));
    ///std::iota(hash_src.begin(), hash_src.end(), IndexT(0));//increasing sequence

	HashUpdater hasher(hash_src);
	std::for_each(src.begin(), src.end(), hasher);

    // IndexT counter(0);
    // std::for_each(src.begin(), src.end(), [&](const IndexT& i){
	// 	hash_src[i]=counter++;
    //   });

    size_t set_sz = index_set.size();
    std::vector<IndexT> old_index_set(index_set);
      
    for(IndexT k = 0;k<set_sz;++k)
      {
		index_set[k] = hash_src[old_index_set[k]];
      }
  }
    
private:
  std::vector<EdgeT> vals_;
  std::vector<IndexT> row_ptr_;
  std::vector<IndexT> col_ind_;
};

}//end namespace debug
}//end namespace nvgraph

#endif /* incidence_graph_hxx */
