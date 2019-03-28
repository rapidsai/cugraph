/*
 *  Copyright 2008-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file ainv.inl
 *  \brief Inline file for ainv.h
 */

#include <cusp/blas/blas.h>
#include <cusp/format_utils.h>
#include <cusp/transpose.h>
#include <cusp/multiply.h>
#include <cusp/csr_matrix.h>

#include <map>
#include <vector>

namespace cusp
{
namespace precond
{
namespace detail
{

template<typename T>
bool less_than_abs(const T &a, const T &b)
{
    T abs_a = a < 0 ? -a : a;
    T abs_b = b < 0 ? -b : b;
    return abs_a < abs_b;
}

template<typename IndexType, typename ValueType>
class ainv_matrix_row
{
public:

    struct map_entry {
        ValueType value;
        int heapidx;

        map_entry(const ValueType &v, int i) : value(v), heapidx(i) { }
        map_entry() { }
    };

    struct heap_entry {
        ValueType value;
        typename std::map<IndexType, typename ainv_matrix_row::map_entry>::iterator mapiter;

        heap_entry(const ValueType &v, const typename std::map<IndexType, typename ainv_matrix_row::map_entry>::iterator &i) : value(v), mapiter(i) { }
        heap_entry() { }
    };

    typedef typename std::map<IndexType, typename ainv_matrix_row::map_entry>::const_iterator const_iterator;

private:

    typename std::map<IndexType, typename ainv_matrix_row::map_entry> row_map; // row entries sorted by index
    typename std::vector<typename ainv_matrix_row::heap_entry> row_heap; // row entries sorted by min-abs-val (in a heap)

    void heap_swap(int i, int j)
    {
        // swap the entries
        typename ainv_matrix_row::heap_entry val = this->row_heap[i];
        this->row_heap[i] = this->row_heap[j];
        this->row_heap[j] = val;

        // update the backpointers
        this->row_heap[i].mapiter->second.heapidx = i;
        this->row_heap[j].mapiter->second.heapidx = j;
    }

    void downheap(int i) {
        int child0 = (i+1)*2-1;
        int child1 = (i+1)*2;

        while ((size_t) child0 < this->row_heap.size() || (size_t) child1 < this->row_heap.size()) {
            int min_child = child0; // this will be the child with the lowest value that is in-bounds.
            if ((size_t) child1 < this->row_heap.size() && less_than_abs(this->row_heap[child1].value, this->row_heap[child0].value))
                min_child = child1;
            // if either child is lower, swap with whichever is smaller, otherwise we're done
            if (less_than_abs(this->row_heap[child0].value, this->row_heap[i].value) || ((size_t) child1 < this->row_heap.size() && less_than_abs(this->row_heap[child1].value, this->row_heap[i].value)))
                this->heap_swap(i, min_child);
            else
                break;

            i = min_child;
            child0 = (i+1)*2-1;
            child1 = (i+1)*2;
        }
    }

    void upheap(int i) {
        int parent = (i-1)/2;
        while (i != 0) {
            if (less_than_abs(this->row_heap[i].value, this->row_heap[parent].value))
                this->heap_swap(i, parent);
            else
                break;

            i = parent;
            parent = (i-1)/2;
        }
    }


    void heap_insert(typename ainv_matrix_row::heap_entry val)
    {
        this->row_heap.push_back(val);
        val.mapiter->second.heapidx = (int) this->row_heap.size()-1;
        upheap(this->row_heap.size()-1);
    }

    void heap_pop()
    {
        if (this->row_heap.empty())
            return;

        heap_swap(0, this->row_heap.size()-1);
        //no need to erase the backpointer, since the tree will be updated elsewhere

        this->row_heap.pop_back();

        downheap(0);
    }

    void heap_update(int i, ValueType val)
    {
        ValueType old_val = this->row_heap[i].value;
        this->row_heap[i].value = val;

        if (less_than_abs(val, old_val))
            upheap(i);
        else
            downheap(i);
    }


public:

    typename ainv_matrix_row::const_iterator begin() const {
        return this->row_map.begin();
    }
    typename ainv_matrix_row::const_iterator end()   const {
        return this->row_map.end();
    }
    size_t size() const {
        return this->row_map.size();
    }

    bool has_entry_at_index(IndexType i) {
        return this->row_map.count(i) != 0;
    }

    void mult_by_scalar(ValueType scalar) {
        // since we already have a table of pointers into the map, this is O(n) via pointer chasing
        for (int i=0; (size_t) i < this->row_heap.size(); i++) {
            this->row_heap[i].value *= scalar;
            this->row_heap[i].mapiter->second.value *= scalar;
        }
    }

    void insert(IndexType i, ValueType t) {
        ainv_matrix_row::map_entry me(t, -1);
        ainv_matrix_row::heap_entry he;

        // map::insert returns a pair (iterator, bool), so we can grab the iterator from that
        he.mapiter = this->row_map.insert(std::make_pair(i, me)).first;
        he.value = t;

        this->heap_insert(he);
    }

    ValueType min_abs_value() const {
        return this->row_heap.empty() ? (ValueType)0 : this->row_heap.begin()->value;
    }

    // these are here for the unit test only
    bool validate_heap() const {
        for (int i=0; (size_t) i < this->size(); i++) {
            int child0 = (i+1)*2-1;
            int child1 = (i+1)*2;
            if ((size_t) child0 < this->size() && !less_than_abs(this->row_heap[i].value, this->row_heap[child0].value))
                return false;
            if ((size_t) child1 < this->size() && !less_than_abs(this->row_heap[i].value, this->row_heap[child1].value))
                return false;

        }
        return true;
    }

    // these are here for the unit test only
    bool validate_backpointers() const {
        for (typename ainv_matrix_row::const_iterator iter = this->row_map.begin(); iter != this->row_map.end(); ++iter) {
            if (this->row_heap[iter->second.heapidx].mapiter != iter ||
                    this->row_heap[iter->second.heapidx].value != iter->second.value)
                return false;
        }
        return true;
    }

    void add_to_value(IndexType i, ValueType addend) {
        // update val in map, which is free
        typename std::map<IndexType, typename ainv_matrix_row::map_entry>::iterator map_iter = this->row_map.find(i);
        map_iter->second.value += addend;

        // update val in heap, which requires re-sorting
        this->heap_update(map_iter->second.heapidx, map_iter->second.value);
    }

    void remove_min() {
        if (this->row_heap.empty())
            return;

        typename std::map<IndexType, typename ainv_matrix_row::map_entry>::iterator iter_to_remove = this->row_heap.begin()->mapiter;
        this->heap_pop();
        this->row_map.erase(iter_to_remove);
    }

    void replace_min_if_greater(IndexType i, ValueType t) {
        if (!less_than_abs(t, this->min_abs_value())) {
            remove_min();
            insert(i, t);
        }
    }
}; // end struct ainv_matrix_row

template<typename IndexType, typename ValueType>
void vector_scalar(std::map<IndexType, ValueType> &vec, ValueType scalar)
{
    for (typename std::map<IndexType, ValueType>::iterator vec_iter = vec.begin(); vec_iter != vec.end(); ++vec_iter) {
        vec_iter->second *= scalar;
    }
}


template<typename IndexType, typename ValueType>
void matrix_vector_product(const csr_matrix<IndexType, ValueType, host_memory> &A, const detail::ainv_matrix_row<IndexType, ValueType> &x, std::map<IndexType, ValueType> &b)
{
    b.clear();

    for (typename detail::ainv_matrix_row<IndexType, ValueType>::const_iterator x_iter = x.begin(); x_iter != x.end(); ++x_iter) {
        ValueType x_i  = x_iter->second.value;
        IndexType row = x_iter->first;

        IndexType row_start = A.row_offsets[row];
        IndexType row_end = A.row_offsets[row+1];

        for (IndexType row_j = row_start; row_j < row_end; row_j++) {
            IndexType col = A.column_indices[row_j];
            ValueType Aij = A.values[row_j];

            ValueType product = Aij * x_i;

            // add to b if it's not already in b
            typename std::map<IndexType, ValueType>::iterator b_iter = b.find(col);
            if (b_iter == b.end())
                b[col] = product;
            else
                b_iter->second += product;
        }
    }

}


template<typename IndexType, typename ValueType>
ValueType dot_product(const detail::ainv_matrix_row<IndexType, ValueType> &a, const std::map<IndexType, ValueType> &b)
{
    typename detail::ainv_matrix_row<IndexType, ValueType>::const_iterator a_iter = a.begin();
    typename std::map<IndexType, ValueType>::const_iterator b_iter = b.begin();

    ValueType sum = 0;
    while (a_iter != a.end() && b_iter != b.end()) {
        IndexType a_ind = a_iter->first;
        IndexType b_ind = b_iter->first;
        if (a_ind == b_ind) {
            sum += a_iter->second.value * b_iter->second;
            ++a_iter;
            ++b_iter;
        }
        else if (a_ind < b_ind)
            ++a_iter;
        else
            ++b_iter;
    }

    return sum;
}


template<typename IndexType, typename ValueType>
void vector_add_inplace_drop(detail::ainv_matrix_row<IndexType, ValueType> &result, ValueType mult, const detail::ainv_matrix_row<IndexType, ValueType> &operand, ValueType tolerance, int nonzeros_this_row)
{
    // write into result:
    // result += mult * operand
    // but dropping any terms from (mult * operand) if they are less than tolerance

    for (typename detail::ainv_matrix_row<IndexType, ValueType>::const_iterator op_iter = operand.begin(); op_iter != operand.end(); ++op_iter) {
        IndexType i = op_iter->first;
        ValueType term = mult * op_iter->second.value;
        ValueType abs_term = term < 0 ? -term : term;

        if (abs_term < tolerance)
            continue;

        // We use a combination of 2 dropping strategies: a standard drop tolerance, as well as a bound on the
        // number of non-zeros per row.  if we've already reached that maximum size
        // and this would add a new entry to result, we add it only if it is larger than one of the current entries
        // in which case we remove that element in its place.
        // This idea has been applied to IC factorization, but not to AINV as far as I'm aware.
        // See: Lin, C. and More, J. J. 1999. Incomplete Cholesky Factorizations with Limited Memory.
        //      SIAM J. Sci. Comput. 21, 1 (Aug. 1999), 24-45.

        // can improve this by storing min idx & min_abs_val for each matrix row, and keeping up to date.
        // as new entry is considered, skip if below min_val.  Otherwise, remove entry corresponding to min_val, insert new entry, and search for the new min.
        // best case, this cuts from O(n) to O(1).  Worst case stays as before.
        // even better: could i just use a heap?  i need both the map for fast inserts & deletes, and a heap to maintain lowest entry
        // this makes it O(log n) worst case, i think...
        // idea: instead of using a map for the matrix rows, wrap it in a struct that also maintains a heap of entries by abs_value
        if (result.has_entry_at_index(i))
            result.add_to_value(i, term);
        else {
            if (nonzeros_this_row < 0 || result.size() < (size_t) nonzeros_this_row) {
                // there is an empty slot left, so just insert
                result.insert(i, term);
            }
            else {
                // check if this is larger than one of the existing values.  If so, replace the smallest value.
                result.replace_min_if_greater(i, term);
            }
        }
    }
}

template<typename IndexTypeA, typename ValueTypeA, typename IndexTypeB, typename ValueTypeB, typename MemorySpaceB>
void convert_to_device_csr(const std::vector<detail::ainv_matrix_row<IndexTypeA, ValueTypeA> > &src, cusp::hyb_matrix<IndexTypeB, ValueTypeB, MemorySpaceB> &dst)
{
    // convert wt to csr
    IndexTypeA nnz = 0;
    IndexTypeA n = src.size();

    int i;
    for (i=0; i < n; i++)
        nnz += src[i].size();

    cusp::csr_matrix<IndexTypeA, ValueTypeA, host_memory> host_src(n, n, nnz);

    IndexTypeA pos = 0;
    host_src.row_offsets[0] = 0;

    for (i=0; i < n; i++) {
        typename detail::ainv_matrix_row<IndexTypeA, ValueTypeA>::const_iterator src_iter = src[i].begin();
        while (src_iter != src[i].end()) {
            host_src.column_indices[pos] = src_iter->first;
            host_src.values        [pos] = src_iter->second.value;

            ++src_iter;
            ++pos;
        }
        host_src.row_offsets[i+1] = pos;
    }

    // copy to device & transpose
    dst = host_src;
}



} // end namespace detail


// constructor
template <typename ValueType, typename MemorySpace>
template<typename MatrixTypeA>
nonsym_bridson_ainv<ValueType,MemorySpace>
::nonsym_bridson_ainv(const MatrixTypeA & A, ValueType drop_tolerance, int nonzero_per_row, bool lin_dropping, int lin_param)
    : linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_rows)
{
    typename MatrixTypeA::index_type n = A.num_rows;

    temp1.resize(n);
    temp2.resize(n);

    MatrixTypeA At;
    cusp::transpose(A, At);

    // copy A, At to host
    typename cusp::csr_matrix<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type, host_memory> host_A = A;
    typename cusp::csr_matrix<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type, host_memory> host_At = At;
    cusp::array1d<ValueType, host_memory> host_diagonals(n);

    // perform factorization
    typename std::vector<detail::ainv_matrix_row<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> > wt_factor(n);
    typename std::vector<detail::ainv_matrix_row<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> > z_factor(n);

    typename MatrixTypeA::index_type i,j;
    for (i=0; i < n; i++) {
        wt_factor[i].insert(i, (typename MatrixTypeA::value_type)1);
        z_factor[i].insert(i, (typename MatrixTypeA::value_type)1);
    }

    typename std::map<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> u, l;

    for (j=0; j < n; j++)
    {
        cusp::precond::detail::matrix_vector_product(host_At, wt_factor[j], u);
        cusp::precond::detail::matrix_vector_product(host_A, z_factor[j], l);
        typename MatrixTypeA::value_type p = detail::dot_product(wt_factor[j], l);
        //could also do: typename MatrixTypeA::value_type p = detail::dot_product(z_factor[j], u);
        host_diagonals[j] = (ValueType) (1.0/p);

        // for i = j+1 to n, skipping where u_i == 0
        // this should be a O(1)-time operation, since u is a sparse vector
        for (typename std::map<typename MatrixTypeA::index_type,typename MatrixTypeA::value_type>::const_iterator u_iter = u.upper_bound(j); u_iter != u.end(); ++u_iter) {
            i = u_iter->first;
            int row_count = nonzero_per_row;
            if (lin_dropping) {
                row_count = lin_param + (int) (host_A.row_offsets[i+1] - host_A.row_offsets[i]);
                if (row_count < 1) row_count = 1;
            }

            detail::vector_add_inplace_drop(z_factor[i], -u_iter->second/p, z_factor[j], drop_tolerance, row_count);
        }

        for (typename std::map<typename MatrixTypeA::index_type,typename MatrixTypeA::value_type>::const_iterator l_iter = l.upper_bound(j); l_iter != l.end(); ++l_iter) {
            i = l_iter->first;
            int row_count = nonzero_per_row;
            if (lin_dropping) {
                row_count = lin_param + (int) (host_A.row_offsets[i+1] - host_A.row_offsets[i]);
                if (row_count < 1) row_count = 1;
            }

            detail::vector_add_inplace_drop(wt_factor[i], -l_iter->second/p, wt_factor[j], drop_tolerance, row_count);
        }

    }

    // copy w_factor into w, w_t
    diagonals = host_diagonals;

    // convert wt to csr
    typename cusp::hyb_matrix<int, ValueType, MemorySpace> w;
    detail::convert_to_device_csr(wt_factor, w);
    cusp::transpose(w, w_t);
    detail::convert_to_device_csr(z_factor, z);
}

// linear operator
template <typename ValueType, typename MemorySpace>
template <typename VectorType1, typename VectorType2>
void nonsym_bridson_ainv<ValueType, MemorySpace>
::operator()(const VectorType1& x, VectorType2& y)
{
    cusp::multiply(z, x, temp1);
    cusp::blas::xmy(temp1, diagonals, temp2);
    cusp::multiply(w_t, temp2, y);
}



// constructor
template <typename ValueType, typename MemorySpace>
template<typename MatrixTypeA>
bridson_ainv<ValueType,MemorySpace>
::bridson_ainv(const MatrixTypeA & A, ValueType drop_tolerance, int nonzero_per_row, bool lin_dropping, int lin_param)
    : linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_rows)
{
    typename MatrixTypeA::index_type n = A.num_rows;

    temp1.resize(n);
    temp2.resize(n);

    // copy A to host
    typename cusp::csr_matrix<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type, host_memory> host_A = A;
    cusp::array1d<ValueType, host_memory> host_diagonals(n);


    // perform factorization
    typename std::vector<detail::ainv_matrix_row<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> > w_factor(n);

    typename MatrixTypeA::index_type i,j;
    for (i=0; i < n; i++) {
        w_factor[i].insert(i, (typename MatrixTypeA::value_type)1);
    }

    typename std::map<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> u;

    for (j=0; j < n; j++)
    {
        cusp::precond::detail::matrix_vector_product(host_A, w_factor[j], u);
        typename MatrixTypeA::value_type p = detail::dot_product(w_factor[j], u);
        host_diagonals[j] = (ValueType) (1.0/p);

        // for i = j+1 to n, skipping where u_i == 0
        // this should be a O(1)-time operation, since u is a sparse vector
        for (typename std::map<typename MatrixTypeA::index_type,typename MatrixTypeA::value_type>::const_iterator u_iter = u.upper_bound(j); u_iter != u.end(); ++u_iter) {
            i = u_iter->first;
            int row_count = nonzero_per_row;
            if (lin_dropping) {
                row_count = lin_param + (int) (host_A.row_offsets[i+1] - host_A.row_offsets[i]);
                if (row_count < 1) row_count = 1;
            }

            detail::vector_add_inplace_drop(w_factor[i], -u_iter->second/p, w_factor[j], drop_tolerance, row_count);
        }

    }

    // copy diagonal & w_factor into w, w_t
    diagonals = host_diagonals;
    detail::convert_to_device_csr(w_factor, w);
    cusp::transpose(w, w_t);
}

// linear operator
template <typename ValueType, typename MemorySpace>
template <typename VectorType1, typename VectorType2>
void bridson_ainv<ValueType, MemorySpace>
::operator()(const VectorType1& x, VectorType2& y)
{
    cusp::multiply(w, x, temp1);
    cusp::blas::xmy(temp1, diagonals, temp2);
    cusp::multiply(w_t, temp2, y);
}


// constructor
template <typename ValueType, typename MemorySpace>
template<typename MatrixTypeA>
scaled_bridson_ainv<ValueType,MemorySpace>
::scaled_bridson_ainv(const MatrixTypeA & A, ValueType drop_tolerance, int nonzero_per_row, bool lin_dropping, int lin_param)
    : linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_rows)
{
    typename MatrixTypeA::index_type n = A.num_rows;
    temp1.resize(n);

    // copy A to host
    typename cusp::csr_matrix<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type, host_memory> host_A = A;

    // perform factorization
    typename std::vector<detail::ainv_matrix_row<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> > w_factor(n);

    typename MatrixTypeA::index_type i,j;
    for (i=0; i < n; i++) {
        w_factor[i].insert(i, (typename MatrixTypeA::value_type)1);
    }

    typename std::map<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> u;

    for (j=0; j < n; j++) {
        cusp::precond::detail::matrix_vector_product(host_A, w_factor[j], u);
        typename MatrixTypeA::value_type p = detail::dot_product(w_factor[j], u);

        detail::vector_scalar(u, (typename MatrixTypeA::value_type) (1.0/std::sqrt((ValueType) p)));
        w_factor[j].mult_by_scalar((typename MatrixTypeA::value_type) (1.0/std::sqrt((ValueType) p)));

        // for i = j+1 to n, skipping where u_i == 0
        // this should be a O(1)-time operation, since u is a sparse vector
        for (typename std::map<typename MatrixTypeA::index_type,typename MatrixTypeA::value_type>::const_iterator u_iter = u.upper_bound(j); u_iter != u.end(); ++u_iter) {
            i = u_iter->first;
            int row_count = nonzero_per_row;
            if (lin_dropping) {
                row_count = lin_param + (int) (host_A.row_offsets[i+1] - host_A.row_offsets[i]);
                if (row_count < 1) row_count = 1;
            }
            detail::vector_add_inplace_drop(w_factor[i], -u_iter->second, w_factor[j], drop_tolerance, row_count);
        }

    }

    // copy w_factor into w:
    detail::convert_to_device_csr(w_factor, w);
    cusp::transpose(w, w_t);
}

template <typename ValueType, typename MemorySpace>
template <typename VectorType1, typename VectorType2>
void scaled_bridson_ainv<ValueType, MemorySpace>
::operator()(const VectorType1& x, VectorType2& y)
{
    cusp::multiply(w, x, temp1);
    cusp::multiply(w_t, temp1, y);
}

} // end namespace precond
} // end namespace cusp


