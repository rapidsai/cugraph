/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include <cstdint>

namespace cugraph::legacy::ops {
namespace graph {
/**
 * @brief constant to represent invalid node id.
 *
 * @tparam IdxT node index type
 *
 * @note It is important that its bit representation has 1 set everywhere,
 *       no matter which index type is used
 * @note As of now, we expect `IdxT` to be a signed integer type, although
 *       everything should work with unsigned types. In any case, we restrict
 *       the range of valid IDs to [0, 2^(sizeof(IdxT) * 8 - 1) - 2].
 *       This reduced range is due to reserving the most significant bit
 *       for (potential) indicator variables in algos, as well as one
 *       additional value for the invalid ID. This comes naturally when
 *       using signed integer types since the range simply excludes all
 *       negative values as well as the maximal positive value.
 */
template <typename IdxT>
static constexpr IdxT INVALID_ID = IdxT{-1};

/**
 * @brief constant to represent default node type
 *
 * @note We expect node and edge types to be representable with signed 32-bit
 *       integers, meaning that the valid range is [0, 2^31 - 1].
 */
static constexpr int32_t DEFAULT_NODE_TYPE = 0;

/**
 * @brief constant to represent default edge type
 *
 * @note Valid range: [0, 2^31 - 1] (see `DEFAULT_NODE_TYPE`)
 */
static constexpr int32_t DEFAULT_EDGE_TYPE = 0;

/**
 * @brief simple CSC representation of a bipartite graph.
 *
 * @tparam EdgeIdxT edge index type (signed type, see `INVALID_ID`)
 * @tparam NodeIdxT node index type (signed type, see `INVALID_ID`)
 *
 * @note By default, we expect an adjacency matrix representing the in-graph
         in the CSC format: the index into `csc_offsets` represents the
         node ID of a destination node and the values of `csc_indices` are the
         node IDs of source nodes.
 * @note For valid IDs, see `INVALID_ID`.
 * @note This object does NOT own any of the underlying pointers and thus
 *       their lifetime management is left to the calling code.
 */
template <typename EdgeIdxT, typename NodeIdxT = EdgeIdxT>
struct __attribute__((aligned(16))) bipartite_csc {
  using nidx_t = NodeIdxT;
  using eidx_t = EdgeIdxT;
  // handle sub-classes correctly
  bipartite_csc()          = default;
  virtual ~bipartite_csc() = default;

  /**
   * monotonically increasing array with each location pointing to the start
   * offset of the neighborhood of that node in the `indices` array.
   * Together with `indices`, this forms the CSC (in-graph) representation
   * of a graph. It is of length `n_dst_nodes + 1`.
   */
  EdgeIdxT* offsets{nullptr};
  /**
   * contains neighbor indices of every destination node belonging to this object.
   * Together with `offsets`, this forms the CSC (in-graph) representation
   * of a graph. It is of length `n_indices`.
   */
  NodeIdxT* indices{nullptr};

  /**
   * monotonically increasing array with each location pointing to the start
   * offset of the neighborhood of that node in the `rev_indices` array.
   * Together with `rev_indices`, this forms the CSR (out-graph) representation
   * of a graph which corresponds to the "reversed" representation of the CSC.
   * It is of length `n_src_nodes + 1`.
   */
  EdgeIdxT* rev_offsets{nullptr};
  /**
   * contains neighbor indices of every source node belonging to this object.
   * Together with `rev_offsets`, this forms the CSR (out-graph) representation
   * of a graph which corresponds to the "reversed" representation of the CSC.
   * It is of length `n_indices`.
   *
   */
  NodeIdxT* rev_indices{nullptr};

  /**
   * optional map from indexes into `indices` (i.e. the CSC representation)
   * to the indexes of a potentially underlying COO representation or edge
   * embedding table in general. This can be used to avoid having to reorder edge features
   * when we sort edges while creating CSC, or apply other transformations on
   * the graph structure. Its length is `n_indices`.
   *
   * @note This pointer is optional for many operations. If not given, it is
   *       assumed that edge features are ordered the same way as `indices`.
   * @note If provided, we assume that this is a perfect partition of the
   *       indexes in [0, n_indices - 1]. Operations using edge features only
   *       as input can usually work with any index, but operations writing
   *       to the edge features or gradients on edge features assume this
   *       to be a partition unless specified.
   *       Otherwise, be aware that this could lead to data races!
   */
  EdgeIdxT* map_csc_to_coo{nullptr};
  /**
   * optional map from indexes into `indices` (i.e. the CSC representation)
   * to the indexes of a potentially underlying COO representation or edge
   * embedding table in general. This can be used to avoid having to reorder edge features
   * when we sort edges while creating CSC, or apply other transformations on
   * the graph structure. Its length is `n_indices`.
   *
   * @note This is only optional if the reversed graph (i.e. the csc representation)
   * is not used or not used with any input over edge features (including intermediate
   * outputs which are just written for the backward pass).
   * @note If `map_csc_to_coo` is not given, this corresponds to the map between
   * indexes into `rev_indices` to the corresponding positions of `indices` or
   * into edge embedding tables in general.
   * @note We recommend creating this map automatically using `graph::get_reverse_graph`
   * to gurantuee consistency.
   */
  EdgeIdxT* map_rev_to_coo{nullptr};

  /** number of source nodes belonging to this graph */
  NodeIdxT n_src_nodes{0};
  /** number of destination nodes belonging to this graph */
  NodeIdxT n_dst_nodes{0};
  /** total number of edges in this graph (length of `indices` array) */
  EdgeIdxT n_indices{0};
  /** maximum in-degree of all destination nodes in this graph  */
  EdgeIdxT dst_max_in_degree{-1};
};

/**
 * @brief CSC representation of a bipartite heterogeneous graph.
 *
 * @tparam EdgeIdxT edge index type (signed type, see `INVALID_ID`)
 * @tparam NodeIdxT node index type (signed type, see `INVALID_ID`)
 *
 * @note See `bipartite_csc` first. This struct adds information needed for
 *       heterogeneous graphs.
 *
 * @note Only integer node and edge types are supported as of now.
 * @note By default, we expect the graph to be an in-graph. See `bipartite_csc`.
 */
template <typename EdgeIdxT, typename NodeIdxT = EdgeIdxT>
struct bipartite_csc_hg : bipartite_csc<EdgeIdxT, NodeIdxT> {
  using nidx_t = NodeIdxT;
  using eidx_t = EdgeIdxT;
  using base_t = bipartite_csc<EdgeIdxT, NodeIdxT>;
  // handle sub-classes correctly
  bipartite_csc_hg()           = default;
  ~bipartite_csc_hg() override = default;

  explicit bipartite_csc_hg(const base_t& graph) : base_t(graph) {}
  bipartite_csc_hg(const base_t& graph,
                   int32_t _n_node_types,
                   int32_t _n_edge_types,
                   int32_t* _src_node_types,
                   int32_t* _dst_node_types,
                   int32_t* _edge_types)
    : base_t(graph),
      n_node_types(_n_node_types),
      n_edge_types(_n_edge_types),
      src_node_types(_src_node_types),
      dst_node_types(_dst_node_types),
      edge_types(_edge_types)
  {
  }

  /** node types of each inpupt node in the graph (length `n_src_nodes`) */
  int32_t* src_node_types{nullptr};
  /** node types of each output node in the graph (length `n_dst_nodes`) */
  int32_t* dst_node_types{nullptr};
  /**
   * edge types of each edge in the graph. It is of length `n_indices` and is
   * expected to be ordered the same way as `indices`
   */
  int32_t* edge_types{nullptr};

  /** number of node types in this graph */
  int32_t n_node_types{1};
  /** number of edge types in this graph */
  int32_t n_edge_types{1};
};

/**
 * @brief CSC representation of a static graph or sampled subgraph.
 *
 * @tparam EdgeIdxT edge index type (signed type, see `INVALID_ID`)
 * @tparam NodeIdxT node index type (signed type, see `INVALID_ID`)
 *
 * @note CSC follows the convention of referring to the in-graph as CSC
         while the out-graph is referred to as CSR of the underlying adjancency matrix.
 * @note Valid NIDs: [0, 2^(sizeof(NodeIdxT) * 8 - 1) - 2] (see `INVALID_ID`)
 * @note Valid EIDs: [0, 2^(sizeof(EdgeIdxT) * 8 - 1) - 2] (see `INVALID_ID`)
 * @note This object does NOT own any of the underlying pointers and thus
 *       their lifetime management is left to the calling code.
 */
template <typename EdgeIdxT, typename NodeIdxT = EdgeIdxT>
struct csc : bipartite_csc<EdgeIdxT, NodeIdxT> {
  using nidx_t = NodeIdxT;
  using eidx_t = EdgeIdxT;
  using base_t = bipartite_csc<EdgeIdxT, NodeIdxT>;
  // handle sub-classes correctly
  csc()           = default;
  ~csc() override = default;

  /**
   * optional map which maps the node IDs of destination nodes
   * to their corresponding IDs as source nodes. This can be helpful
   * if the sampling algorithm for instance does not gurantuee to
   * always place the destination nodes at the front of the list
   * of source node IDs.
   *
   * @note If this is passed as `nullptr`, it is assumed that the
   * the ID of any of the `N` destination nodes is the same ID out
   * of the range `0, ..., N - 1` among the source nodes.
   */
  NodeIdxT* map_dst_to_src{nullptr};
};

/**
 * @brief CSC representation of a sampled/static, heterogeneous graph.
 *
 * @tparam IdxT node index type
 *
 * @note See `csc` first. This struct adds information needed for
 *       heterogeneous graphs.
 *
 * @note Only integer node and edge types are supported as of now.
 * @note By default, we expect the graph to be an in-graph. See `csc`.
 */
template <typename EdgeIdxT, typename NodeIdxT = EdgeIdxT>
struct csc_hg : csc<EdgeIdxT, NodeIdxT> {
  using nidx_t = NodeIdxT;
  using eidx_t = EdgeIdxT;
  using base_t = csc<EdgeIdxT, NodeIdxT>;
  // handle sub-classes correctly
  csc_hg()           = default;
  ~csc_hg() override = default;

  explicit csc_hg(const base_t& graph) : base_t(graph) {}
  csc_hg(const base_t& graph,
         int32_t _n_node_types,
         int32_t _n_edge_types,
         int32_t* _node_types,
         int32_t* _edge_types)
    : base_t(graph),
      n_node_types(_n_node_types),
      n_edge_types(_n_edge_types),
      node_types(_node_types),
      edge_types(_edge_types)
  {
  }

  /** node types of each node in the graph (length `n_dst_nodes`) */
  int32_t* node_types{nullptr};
  /**
   * edge types of each edge in the graph. It is of length `n_indices` and is
   * expected to be ordered the same way as `indices`
   */
  int32_t* edge_types{nullptr};

  /** number of node types in this graph */
  int32_t n_node_types{1};
  /** number of edge types in this graph */
  int32_t n_edge_types{1};
};

}  // namespace graph

/** Instantiations for different index types (in main namespace) */

/**
 * @brief CSC representation of bipartite graphs for 32b node and edge id's
 */
using bipartite_csc_s32_t = graph::bipartite_csc<int32_t>;
/**
 * @brief CSC representation of bipartite graphs for 32b node and 64b edge id's
 */
using bipartite_csc_s64_s32_t = graph::bipartite_csc<int64_t, int32_t>;
/**
 * @brief CSC representation of bipartite graphs for 64b node and edge id's
 */
using bipartite_csc_s64_t = graph::bipartite_csc<int64_t>;

/**
 * @brief CSC representation of static/sampled graphs for 32b node and edge id's
 */
using csc_s32_t = graph::csc<int32_t>;
/**
 * @brief CSC representation of static/sampled graphs for 32b node and 64b edge id's
 */
using csc_s64_s32_t = graph::csc<int64_t, int32_t>;
/**
 * @brief CSC representation of static/sampled graphs for 64b node and edge id's
 */
using csc_s64_t = graph::csc<int64_t>;

/**
 * @brief base CSC representation of a bipartite, heterogeneous graph (32 id's)
 */
using bipartite_csc_hg_s32_t = graph::bipartite_csc_hg<int32_t>;
/**
 * @brief base CSC representation of a bipartite, heterogeneous graph (64/32 id's)
 */
using bipartite_csc_hg_s64_s32_t = graph::bipartite_csc_hg<int64_t, int32_t>;
/**
 * @brief base CSC representation of a bipartite, heterogeneous graph (64b id's)
 */
using bipartite_csc_hg_s64_t = graph::bipartite_csc_hg<int64_t>;

/**
 * @brief base CSC representation of a static/sampled, heterogeneous graph (32 id's)
 */
using csc_hg_s32_t = graph::csc_hg<int32_t>;
/**
 * @brief base CSC representation of a static/sampled, heterogeneous graph (64/32 id's)
 */
using csc_hg_s64_s32_t = graph::csc_hg<int64_t, int32_t>;
/**
 * @brief base CSC representation of a static/sampled, heterogeneous graph (64b id's)
 */
using csc_hg_s64_t = graph::csc_hg<int64_t>;

}  // namespace cugraph::legacy::ops
