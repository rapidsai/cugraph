/*
 *  Copyright 2008-2014 NVIDIA Corporation
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
#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/temporary_array.h>

#include <cusp/array1d.h>
#include <cusp/exception.h>

#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

// Tables and Hilbert transform codes adapted from HSFC implementation in Zoltan v3.601
static unsigned const int IMAX = ~(0U);
static const int MAXLEVEL_2d = 28; // 56 bits of significance, 28 per dimension
static const int MAXLEVEL_3d = 19; // 56 bits of significance, 18+ per dimension

__constant__ static unsigned const int idata2d[] =  // 2 dimension to nkey conversion
{   0, 3, 1, 2,
    0, 1, 3, 2,
    2, 3, 1, 0,
    2, 1, 3, 0
};

__constant__ static unsigned const int istate2d[] = // 2 dimension to nkey state transitions
{   1, 2, 0, 0,
    0, 1, 3, 1,
    2, 0, 2, 3,
    3, 3, 1, 2
};

__constant__ static unsigned const idata3d [] = {   // 3 dimension to nkey conversion
    0,  7,  3,  4,  1,  6,  2,  5,
    0,  1,  3,  2,  7,  6,  4,  5,
    0,  3,  7,  4,  1,  2,  6,  5,
    2,  3,  5,  4,  1,  0,  6,  7,
    4,  5,  3,  2,  7,  6,  0,  1,
    4,  7,  3,  0,  5,  6,  2,  1,
    6,  7,  5,  4,  1,  0,  2,  3,
    0,  1,  7,  6,  3,  2,  4,  5,
    2,  1,  5,  6,  3,  0,  4,  7,
    6,  1,  5,  2,  7,  0,  4,  3,
    0,  7,  1,  6,  3,  4,  2,  5,
    2,  1,  3,  0,  5,  6,  4,  7,
    4,  7,  5,  6,  3,  0,  2,  1,
    4,  5,  7,  6,  3,  2,  0,  1,
    6,  1,  7,  0,  5,  2,  4,  3,
    0,  3,  1,  2,  7,  4,  6,  5,
    2,  3,  1,  0,  5,  4,  6,  7,
    6,  7,  1,  0,  5,  4,  2,  3,
    2,  5,  1,  6,  3,  4,  0,  7,
    4,  3,  7,  0,  5,  2,  6,  1,
    4,  3,  5,  2,  7,  0,  6,  1,
    6,  5,  1,  2,  7,  4,  0,  3,
    2,  5,  3,  4,  1,  6,  0,  7,
    6,  5,  7,  4,  1,  2,  0,  3
};

__constant__ static unsigned const istate3d [] = { // 3 dimension to nkey state transitions
    1,  6,  3,  4,  2,  5,  0,  0,
    0,  7,  8,  1,  9,  4,  5,  1,
    15, 22, 23, 20,  0,  2, 19,  2,
    3, 23,  3, 15,  6, 20, 16, 22,
    11,  4, 12,  4, 20,  1, 22, 13,
    22, 12, 20, 11,  5,  0,  5, 19,
    17,  0,  6, 21,  3,  9,  6,  2,
    10,  1, 14, 13, 11,  7, 12,  7,
    8,  9,  8, 18, 14, 12, 10, 11,
    21,  8,  9,  9,  1,  6, 17,  7,
    7, 17, 15, 12, 16, 13, 10, 10,
    11, 14,  9,  5, 11, 22,  0,  8,
    18,  5, 12, 10, 19,  8, 12, 20,
    8, 13, 19,  7,  5, 13, 18,  4,
    23, 11,  7, 17, 14, 14,  6,  1,
    2, 18, 10, 15, 21, 19, 20, 15,
    16, 21, 17, 19, 16,  2,  3, 18,
    6, 10, 16, 14, 17, 23, 17, 15,
    18, 18, 21,  8, 17,  7, 13, 16,
    3,  4, 13, 16, 19, 19,  2,  5,
    16, 13, 20, 20,  4,  3, 15, 12,
    9, 21, 18, 21, 15, 14, 23, 10,
    22, 22,  6,  1, 23, 11,  4,  3,
    14, 23,  2,  9, 22, 23, 21,  0
};

__constant__ static const unsigned *d2d[]= {idata2d,  idata2d  +4, idata2d  +8, idata2d  +12};
__constant__ static const unsigned *s2d[]= {istate2d, istate2d +4, istate2d +8, istate2d +12};

__constant__ static const unsigned int *d3d[] =
{   idata3d,      idata3d +8,   idata3d +16,  idata3d +24,
    idata3d +32,  idata3d +40,  idata3d +48,  idata3d +56,
    idata3d +64,  idata3d +72,  idata3d +80,  idata3d +88,
    idata3d +96,  idata3d +104, idata3d +112, idata3d +120,
    idata3d +128, idata3d +136, idata3d +144, idata3d +152,
    idata3d +160, idata3d +168, idata3d +176, idata3d +184
};

__constant__ static const unsigned int *s3d[] =
{   istate3d,      istate3d +8,   istate3d +16,  istate3d +24,
    istate3d +32,  istate3d +40,  istate3d +48,  istate3d +56,
    istate3d +64,  istate3d +72,  istate3d +80,  istate3d +88,
    istate3d +96,  istate3d +104, istate3d +112, istate3d +120,
    istate3d +128, istate3d +136, istate3d +144, istate3d +152,
    istate3d +160, istate3d +168, istate3d +176, istate3d +184
};

struct hilbert_transform_2d : public thrust::unary_function<double,double>
{
    template<typename Tuple>
    __device__
    double operator()(const Tuple& t) const
    {
        const double x = thrust::get<0>(t);
        const double y = thrust::get<1>(t);

        int level;
        unsigned int key[2], c[2], temp, state;

        // convert x,y coordinates to integers in range [0, IMAX]
        c[0] = (unsigned int) (x * (double) IMAX);               // x
        c[1] = (unsigned int) (y * (double) IMAX);               // y

        // use state tables to convert nested quadrant's coordinates level by level
        key[0] = key[1] = 0;
        state = 0;
        for (level = 0; level < MAXLEVEL_2d; level++) {
            temp = ((c[0] >> (30-level)) & 2)    // extract 2 bits at current level
                   | ((c[1] >> (31-level)) & 1);

            // treat key[] as long shift register, shift in converted coordinate
            key[0] = (key[0] << 2) | (key[1] >> 30);
            key[1] = (key[1] << 2) | *(d2d[state] + temp);

            state = *(s2d[state] + temp);
        }

        // convert 2 part Hilbert key to double and return
        return ldexp ((double) key[0], -24)  +  ldexp ((double) key[1], -56);
    }
};

struct hilbert_transform_3d : public thrust::unary_function<double,double>
{
    template<typename Tuple>
    __device__
    double operator()(const Tuple& t) const
    {
        const double x = thrust::get<0>(t);
        const double y = thrust::get<1>(t);
        const double z = thrust::get<2>(t);

        int level;
        unsigned int key[2], c[3], temp, state;

        // convert x,y,z coordinates to integers in range [0, IMAX]
        c[0] = (unsigned int) (x * (double) IMAX);         // x
        c[1] = (unsigned int) (y * (double) IMAX);         // y
        c[2] = (unsigned int) (z * (double) IMAX);     		 // z

        // use state tables to convert nested quadrant's coordinates level by level
        key[0] = key[1] = 0;
        state = 0;
        for (level = 0; level < MAXLEVEL_3d; level++) {
            temp = ((c[0] >> (29-level)) & 4)  // extract 3 bits at current level
                   | ((c[1] >> (30-level)) & 2)
                   | ((c[2] >> (31-level)) & 1);

            // treat key[] as long shift register, shift in converted coordinate
            key[0] = (key[0] << 3) |  (key[1] >> 29);
            key[1] = (key[1] << 3) | *(d3d[state] + temp);

            state = *(s3d[state] + temp);
        }

        // convert 2 part Hilbert key to double and return
        return ldexp ((double) key[0], -25)  +  ldexp ((double) key[1], -57);
    }
};

template <typename DerivedPolicy, typename Array2d, typename Array1d>
void hilbert_curve(cuda::execution_policy<DerivedPolicy>& exec,
                   const Array2d& coord,
                   size_t num_parts,
                   Array1d& parts)
{
    typedef typename Array1d::value_type PartType;
    typedef typename Array2d::const_column_view::iterator Iterator;
    typedef typename Array2d::value_type ValueType;
    typedef typename Array2d::memory_space MemorySpace;

    size_t num_points = coord.num_rows;
    size_t dims = coord.num_cols;

    if( (dims != 2) && (dims != 3) )
        throw cusp::invalid_input_exception("Hilbert curve partitioning only implemented for 2D or 3D data.");

    thrust::pair<Iterator,Iterator> x_iter = thrust::minmax_element(exec, coord.column(0).begin(), coord.column(0).end());
    thrust::pair<Iterator,Iterator> y_iter = thrust::minmax_element(exec, coord.column(1).begin(), coord.column(1).end());

    ValueType xmin = *x_iter.first;
    ValueType xmax = *x_iter.second;
    ValueType ymin = *y_iter.first;
    ValueType ymax = *y_iter.second;

    if( xmin < ValueType(0) || xmax > ValueType(1) || ymin < ValueType(0) || ymax > ValueType(1) )
        throw cusp::invalid_input_exception("Hilbert coordinates should be in the range [0,1]");

    cusp::detail::temporary_array<double, DerivedPolicy> hilbert_keys(exec, coord.num_rows);

    if( dims == 2 )
    {
        thrust::transform(exec,
                          thrust::make_zip_iterator(thrust::make_tuple(coord.column(0).begin(), coord.column(1).begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(coord.column(0).end(), coord.column(1).end())),
                          hilbert_keys.begin(), hilbert_transform_2d());
    }
    else
    {
        thrust::pair<Iterator,Iterator> z_iter = thrust::minmax_element(exec, coord.column(2).begin(), coord.column(2).end());
        ValueType zmin = *z_iter.first;
        ValueType zmax = *z_iter.second;

        if( zmin < ValueType(0) || zmax > ValueType(1) )
            throw cusp::invalid_input_exception("Hilbert coordinates should be in the range [0,1]");

        thrust::transform(exec,
                          thrust::make_zip_iterator(thrust::make_tuple(coord.column(0).begin(), coord.column(1).begin(), coord.column(2).begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(coord.column(0).end(), coord.column(1).end(), coord.column(2).end())),
                          hilbert_keys.begin(), hilbert_transform_3d());
    }

    cusp::detail::temporary_array<PartType, DerivedPolicy> perm(exec, num_points);
    thrust::sequence(exec, perm.begin(), perm.end());
    thrust::sort_by_key(exec, hilbert_keys.begin(), hilbert_keys.end(), perm.begin());

    cusp::detail::temporary_array<PartType, DerivedPolicy> uniform_parts(exec, num_points);
    thrust::transform(exec,
                      thrust::counting_iterator<PartType>(0), thrust::counting_iterator<PartType>(num_points),
                      thrust::constant_iterator<PartType>(num_points/num_parts), uniform_parts.begin(), thrust::divides<PartType>());
    thrust::gather(exec, perm.begin(), perm.end(), uniform_parts.begin(), parts.begin());
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

