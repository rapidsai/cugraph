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

// Pagerank solver
// Author: Alex Fender afender@nvidia.com
 
#pragma once
namespace cugraph
{

template <typename IndexType, typename ValueType>
int pagerank (  IndexType n, IndexType e, IndexType *cscPtr, IndexType *cscInd,ValueType *cscVal,
                       ValueType alpha, ValueType *a, bool has_guess, float tolerance, int max_iter, ValueType * &pagerank_vector, ValueType * &residual);

} //namespace cugraph
