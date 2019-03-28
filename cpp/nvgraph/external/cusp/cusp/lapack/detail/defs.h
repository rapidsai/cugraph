/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

/*! \file defs.h
 *  \brief Lapack utility definitions for interface routines
 */

#pragma once

#include <lapacke.h>

namespace cusp
{
namespace lapack
{

struct lapack_format {};

struct upper   : public lapack_format {};
struct lower   : public lapack_format {};
struct unit    : public lapack_format {};
struct nonunit : public lapack_format {};
struct evals   : public lapack_format {};
struct evecs   : public lapack_format {};
struct gen_op1 : public lapack_format {};
struct gen_op2 : public lapack_format {};
struct gen_op3 : public lapack_format {};

template< typename LayoutFormat >
struct Orientation {static const lapack_int type;};
template<>
const lapack_int Orientation<cusp::row_major>::type    = LAPACK_ROW_MAJOR;
template<>
const lapack_int Orientation<cusp::column_major>::type = LAPACK_COL_MAJOR;

template< typename TriangularFormat >
struct UpperOrLower {static const char type;};
template<>
const char UpperOrLower<upper>::type = 'U';
template<>
const char UpperOrLower<lower>::type = 'L';

template< typename DiagonalFormat >
struct UnitOrNonunit {static const char type;};
template<>
const char UnitOrNonunit<unit>::type    = 'U';
template<>
const char UnitOrNonunit<nonunit>::type = 'N';

template< typename JobType >
struct EvalsOrEvecs {static const char type;};
template<>
const char EvalsOrEvecs<evals>::type = 'N';
template<>
const char EvalsOrEvecs<evecs>::type = 'V';

template< typename OpType >
struct GenEigOp {static const char type;};
template<>
const char GenEigOp<gen_op1>::type = 1;
template<>
const char GenEigOp<gen_op2>::type = 2;
template<>
const char GenEigOp<gen_op3>::type = 3;

} // end namespace lapack
} // end namespace cusp

