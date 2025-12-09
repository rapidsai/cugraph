/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph_c/resource_handle.h>

#include <sstream>

namespace cugraph {
namespace c_api {

// multi_gpu bool dispatcher:
// resolves bool `multi_gpu`
// and using template arguments vertex_t, edge_t, weight_t, store_transpose
// Calls functor
//
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename time_stamp_t,
          bool store_transposed,
          typename functor_t>
auto multi_gpu_dispatcher(bool multi_gpu, functor_t& functor)
{
  if (multi_gpu) {
    return functor.template
    operator()<vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t, store_transposed, true>();
  } else {
    return functor.template
    operator()<vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t, store_transposed, false>();
  }
}

// transpose bool dispatcher:
// resolves bool `store_transpose`
// and using template arguments vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t
// cascades into next level
// multi_gpu_dispatcher()
//
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename time_stamp_t,
          typename functor_t>
auto transpose_dispatcher(bool store_transposed, bool multi_gpu, functor_t& functor)
{
  if (store_transposed) {
    return multi_gpu_dispatcher<vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t, true>(
      multi_gpu, functor);
  } else {
    return multi_gpu_dispatcher<vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t, false>(
      multi_gpu, functor);
  }
}

// edge_time_type type dispatcher:
// resolves time_stamp_t from edge_time_type enum and using template arguments vertex_t, edge_t,
// weight_t, edge_t cascades into next level transpose_dispatcher()
//
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename functor_t>
auto edge_time_type_dispatcher(cugraph_data_type_id_t edge_time_type,
                               bool store_transposed,
                               bool multi_gpu,
                               functor_t& functor)
{
  switch (edge_time_type) {
    case cugraph_data_type_id_t::INT32: {
      using time_stamp_t = int32_t;
      return transpose_dispatcher<vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t>(
        store_transposed, multi_gpu, functor);
    }
    case cugraph_data_type_id_t::INT64: {
      using time_stamp_t = int64_t;
      return transpose_dispatcher<vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t>(
        store_transposed, multi_gpu, functor);
      break;
    }
    case cugraph_data_type_id_t::FLOAT32: {
      throw std::runtime_error(
        "ERROR: Data type FLOAT32 not allowed for edge type (valid types: INT32).");
      break;
    }
    case cugraph_data_type_id_t::FLOAT64: {
      throw std::runtime_error(
        "ERROR: Data type FLOAT64 not allowed for edge type (valid types: INT32).");
      break;
    }

    default: {
      std::stringstream ss;
      ss << "ERROR: Unknown type enum:" << static_cast<int>(edge_time_type);
      throw std::runtime_error(ss.str());
    }
  }
}

// edge_type_type type dispatcher:
// resolves edge_type_t from edge_type_type enum and using template arguments vertex_t, edge_t,
// weight_t cascades into next level edge_time_type_dispatcher()
//
template <typename vertex_t, typename edge_t, typename weight_t, typename functor_t>
auto edge_type_type_dispatcher(cugraph_data_type_id_t edge_type_type,
                               cugraph_data_type_id_t edge_time_type,
                               bool store_transposed,
                               bool multi_gpu,
                               functor_t& functor)
{
  switch (edge_type_type) {
    case cugraph_data_type_id_t::INT32: {
      using edge_type_t = int32_t;
      return edge_time_type_dispatcher<vertex_t, edge_t, weight_t, edge_type_t>(
        edge_time_type, store_transposed, multi_gpu, functor);
    }
    case cugraph_data_type_id_t::INT64: {
      throw std::runtime_error(
        "ERROR: Data type INT64 not allowed for edge type (valid types: INT32).");
      break;
    }
    case cugraph_data_type_id_t::FLOAT32: {
      throw std::runtime_error(
        "ERROR: Data type FLOAT32 not allowed for edge type (valid types: INT32).");
      break;
    }
    case cugraph_data_type_id_t::FLOAT64: {
      throw std::runtime_error(
        "ERROR: Data type FLOAT64 not allowed for edge type (valid types: INT32).");
      break;
    }

    default: {
      std::stringstream ss;
      ss << "ERROR: Unknown type enum:" << static_cast<int>(edge_type_type);
      throw std::runtime_error(ss.str());
    }
  }
}

// weight type dispatcher:
// resolves weight_t from weight_type enum
// and using template arguments vertex_t, edge_t
// cascades into next level
// edge_type_type_dispatcher()
//
template <typename vertex_t, typename edge_t, typename functor_t>
auto weight_dispatcher(cugraph_data_type_id_t weight_type,
                       cugraph_data_type_id_t edge_type_type,
                       cugraph_data_type_id_t edge_time_type,
                       bool store_transposed,
                       bool multi_gpu,
                       functor_t& functor)
{
  switch (weight_type) {
    case cugraph_data_type_id_t::INT32: {
      using weight_t = int32_t;
      return edge_type_type_dispatcher<vertex_t, edge_t, weight_t>(
        edge_type_type, edge_time_type, store_transposed, multi_gpu, functor);
    } break;
    case cugraph_data_type_id_t::INT64: {
      using weight_t = int64_t;
      return edge_type_type_dispatcher<vertex_t, edge_t, weight_t>(
        edge_type_type, edge_time_type, store_transposed, multi_gpu, functor);
    } break;
    case cugraph_data_type_id_t::FLOAT32: {
      using weight_t = float;
      return edge_type_type_dispatcher<vertex_t, edge_t, weight_t>(
        edge_type_type, edge_time_type, store_transposed, multi_gpu, functor);
    } break;
    case cugraph_data_type_id_t::FLOAT64: {
      using weight_t = double;
      return edge_type_type_dispatcher<vertex_t, edge_t, weight_t>(
        edge_type_type, edge_time_type, store_transposed, multi_gpu, functor);
    } break;
    default: {
      std::stringstream ss;
      ss << "ERROR: Unknown type enum:" << static_cast<int>(weight_type);
      throw std::runtime_error(ss.str());
    }
  }
}

// edge type dispatcher:
// resolves edge_t from edge_type enum
// and using template argument vertex_t
// cascades into the next level
// weight_dispatcher();
//
template <typename vertex_t, typename functor_t>
auto edge_dispatcher(cugraph_data_type_id_t edge_type,
                     cugraph_data_type_id_t weight_type,
                     cugraph_data_type_id_t edge_type_type,
                     cugraph_data_type_id_t edge_time_type,
                     bool store_transposed,
                     bool multi_gpu,
                     functor_t& functor)
{
  switch (edge_type) {
    case cugraph_data_type_id_t::INT32: {
      using edge_t = int32_t;
      return weight_dispatcher<vertex_t, edge_t>(
        weight_type, edge_type_type, edge_time_type, store_transposed, multi_gpu, functor);
    } break;
    case cugraph_data_type_id_t::INT64: {
      using edge_t = int64_t;
      return weight_dispatcher<vertex_t, edge_t>(
        weight_type, edge_type_type, edge_time_type, store_transposed, multi_gpu, functor);
    } break;
    case cugraph_data_type_id_t::FLOAT32: {
      throw std::runtime_error("ERROR: FLOAT32 not supported for a vertex type");
    } break;
    case cugraph_data_type_id_t::FLOAT64: {
      throw std::runtime_error("ERROR: FLOAT64 not supported for a vertex type");
    } break;
    default: {
      std::stringstream ss;
      ss << "ERROR: Unknown type enum:" << static_cast<int>(edge_type);
      throw std::runtime_error(ss.str());
    }
  }
}

// vertex type dispatcher:
// entry point,
// resolves vertex_t from vertex_type enum
// and  cascades into the next level
// edge_dispatcher();
//
template <typename functor_t>
auto vertex_dispatcher(cugraph_data_type_id_t vertex_type,
                       cugraph_data_type_id_t edge_type,
                       cugraph_data_type_id_t weight_type,
                       cugraph_data_type_id_t edge_type_type,
                       cugraph_data_type_id_t edge_time_type,
                       bool store_transposed,
                       bool multi_gpu,
                       functor_t& functor)
{
  switch (vertex_type) {
    case cugraph_data_type_id_t::INT32: {
      using vertex_t = int32_t;
      return edge_dispatcher<vertex_t>(edge_type,
                                       weight_type,
                                       edge_type_type,
                                       edge_time_type,
                                       store_transposed,
                                       multi_gpu,
                                       functor);
    } break;
    case cugraph_data_type_id_t::INT64: {
      using vertex_t = int64_t;
      return edge_dispatcher<vertex_t>(edge_type,
                                       weight_type,
                                       edge_type_type,
                                       edge_time_type,
                                       store_transposed,
                                       multi_gpu,
                                       functor);
    } break;
    case cugraph_data_type_id_t::FLOAT32: {
      throw std::runtime_error("ERROR: FLOAT32 not supported for a vertex type");
    } break;
    case cugraph_data_type_id_t::FLOAT64: {
      throw std::runtime_error("ERROR: FLOAT64 not supported for a vertex type");
    } break;
    default: {
      std::stringstream ss;
      ss << "ERROR: Unknown type enum:" << static_cast<int>(vertex_type);
      throw std::runtime_error(ss.str());
    }
  }
}

}  // namespace c_api
}  // namespace cugraph
