/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Explicit CUDA graphs for the per-level phases. A list of KernelCalls runs as stream launches or
// as a GraphChain: a DAG of kernel nodes built with cudaGraphAddKernelNode (no stream capture, so
// any stream works, including the legacy default stream), instantiated once per call, launched
// on the caller's stream and replayed with updated nodes. With CUDA >= 12.4 a sweep is the body of
// a conditional WHILE node; older toolkits replay the sweep graph once per sweep. The kernels and
// their arguments are the same in every mode, so every mode gives the bitwise identical result;
// replays only start the kernels closer together.

#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

#include <cstring>
#include <type_traits>
#include <vector>

// WHILE nodes need CUDA 12.4 (older toolkits replay the sweep graph per sweep).
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
#define LEIDEN_GRAPH_LOOP 1
#else
#define LEIDEN_GRAPH_LOOP 0
#endif

namespace cugraph::detail::leiden_engine {

constexpr int kCallMaxArgs          = 24;
constexpr std::size_t kCallArgBytes = 512;

// One kernel launch, its arguments stored as the kernel's parameter types.
class KernelCall {
 public:
  template <typename... P, typename... A>
  void set(bool enabled, void (*f)(P...), dim3 grid, dim3 block, unsigned smem, const A&... a)
  {
    static_assert(sizeof...(P) == sizeof...(A),
                  "KernelCall: argument count does not match the kernel");
    static_assert(sizeof...(P) <= kCallMaxArgs, "KernelCall: too many args");
    enabled_        = enabled;
    func_           = reinterpret_cast<const void*>(f);
    grid_           = grid;
    block_          = block;
    smem_           = smem;
    nargs_          = 0;
    std::size_t off = 0;
    (put<std::remove_cv_t<P>>(off, a), ...);
    used_ = static_cast<unsigned short>(off);
  }

  // Overwrites the last argument (same type as the kernel's parameter).
  template <typename T>
  void set_last(const T& v)
  {
    CUGRAPH_EXPECTS(nargs_ >= 1 && off_[nargs_ - 1] + sizeof(T) == used_,
                    "leiden: KernelCall: last argument size mismatch.");
    std::memcpy(store_ + off_[nargs_ - 1], &v, sizeof(T));
  }

  // Same launch (function, geometry and argument bytes).
  bool same(const KernelCall& o) const
  {
    return enabled_ == o.enabled_ && func_ == o.func_ && grid_.x == o.grid_.x &&
           grid_.y == o.grid_.y && grid_.z == o.grid_.z && block_.x == o.block_.x &&
           block_.y == o.block_.y && block_.z == o.block_.z && smem_ == o.smem_ &&
           nargs_ == o.nargs_ && used_ == o.used_ && std::memcmp(store_, o.store_, used_) == 0;
  }

  bool enabled() const { return enabled_; }

  // Kernel-node parameters; `args` (kCallMaxArgs) must outlive the result.
  cudaKernelNodeParams params(void** args) const
  {
    for (int i = 0; i < nargs_; ++i)
      args[i] = const_cast<unsigned char*>(store_ + off_[i]);
    cudaKernelNodeParams p{};
    p.func           = const_cast<void*>(func_);
    p.gridDim        = grid_;
    p.blockDim       = block_;
    p.sharedMemBytes = smem_;
    p.kernelParams   = args;
    p.extra          = nullptr;
    return p;
  }

  void launch(cudaStream_t s) const
  {
    if (!enabled_) return;
    void* args[kCallMaxArgs];
    params(args);
    RAFT_CUDA_TRY(cudaLaunchKernel(func_, grid_, block_, args, smem_, s));
  }

 private:
  template <typename T, typename U>
  void put(std::size_t& off, const U& v)
  {
    const T t = v;  // implicit conversion to the parameter type
    off       = (off + alignof(T) - 1) / alignof(T) * alignof(T);
    CUGRAPH_EXPECTS(off + sizeof(T) <= kCallArgBytes,
                    "leiden: KernelCall: argument storage exceeded.");
    std::memcpy(store_ + off, &t, sizeof(T));
    off_[nargs_++] = static_cast<unsigned short>(off);
    off += sizeof(T);
  }

  bool enabled_     = false;
  const void* func_ = nullptr;
  dim3 grid_{1, 1, 1};
  dim3 block_{1, 1, 1};
  unsigned smem_                                  = 0;
  int nargs_                                      = 0;
  unsigned short used_                            = 0;  // argument bytes
  unsigned short off_[kCallMaxArgs]               = {};
  alignas(16) unsigned char store_[kCallArgBytes] = {};
};

inline void launch_calls(const KernelCall* calls, int count, cudaStream_t s)
{
  for (int i = 0; i < count; ++i)
    calls[i].launch(s);
}

// deps[i]: the earlier calls that call i waits for (stream order satisfies it).
using GraphDeps = std::vector<std::vector<int>>;

#if LEIDEN_GRAPH_LOOP
// Whether this device and driver run WHILE nodes (probed once per device).
inline bool graph_loop_supported()
{
  static thread_local std::vector<signed char> known;
  const int dev = device_info().device;
  if (static_cast<int>(known.size()) <= dev) known.resize(dev + 1, -1);
  if (known[dev] >= 0) return known[dev] != 0;
  bool ok           = false;
  cudaGraph_t g     = nullptr;
  cudaGraphExec_t e = nullptr;
  if (cudaGraphCreate(&g, 0) == cudaSuccess) {
    cudaGraphConditionalHandle h;
    cudaGraphNodeParams cp{};
    cudaGraphNode_t cn;
    ok = cudaGraphConditionalHandleCreate(&h, g, 0, cudaGraphCondAssignDefault) == cudaSuccess;
    if (ok) {
      cp.type               = cudaGraphNodeTypeConditional;
      cp.conditional.handle = h;
      cp.conditional.type   = cudaGraphCondTypeWhile;
      cp.conditional.size   = 1;
#if CUDART_VERSION >= 13000
      ok = cudaGraphAddNode(&cn, g, nullptr, nullptr, 0, &cp) == cudaSuccess;
#else
      ok = cudaGraphAddNode(&cn, g, nullptr, 0, &cp) == cudaSuccess;
#endif
    }
    ok = ok && cudaGraphInstantiate(&e, g, 0) == cudaSuccess;
  }
  if (e) cudaGraphExecDestroy(e);
  if (g) cudaGraphDestroy(g);
  cudaGetLastError();  // a failed probe leaves no sticky error
  known[dev] = ok ? 1 : 0;
  return ok;
}
#else
inline bool graph_loop_supported() { return false; }
#endif

// A DAG of kernel nodes, built on the first launch (or a new topology), then
// replayed with updated arguments and enabled flags. With `loop`, the nodes are
// a WHILE node's body; its handle is the last call's last argument.
class GraphChain {
 public:
  GraphChain()                             = default;
  GraphChain(const GraphChain&)            = delete;
  GraphChain& operator=(const GraphChain&) = delete;
  ~GraphChain() { reset(); }

  void launch(
    const KernelCall* calls, int count, const GraphDeps& deps, cudaStream_t s, bool loop = false)
  {
    in_.assign(calls, calls + count);
    if (!exec_ || static_cast<int>(nodes_.size()) != count || deps_ != deps || loop_ != loop) {
      build(count, deps, loop);
    } else {
      if (loop_) patch_handle();
      update(count);
    }
    RAFT_CUDA_TRY(cudaGraphLaunch(exec_, s));
  }

  // In-flight launches complete first (the runtime defers the free).
  void reset()
  {
    if (exec_) cudaGraphExecDestroy(exec_);
    if (graph_) cudaGraphDestroy(graph_);
    exec_  = nullptr;
    graph_ = nullptr;
    nodes_.clear();
    on_.clear();
    cur_.clear();
    deps_.clear();
    loop_ = false;
  }

 private:
  void patch_handle()
  {
#if LEIDEN_GRAPH_LOOP
    in_.back().set_last(handle_);
#endif
  }

  void build(int count, const GraphDeps& deps, bool loop)
  {
    reset();
    CUGRAPH_EXPECTS(static_cast<int>(deps.size()) == count,
                    "leiden: GraphChain: one dependency list per call.");
    try {
      RAFT_CUDA_TRY(cudaGraphCreate(&graph_, 0));
      cudaGraph_t body = graph_;
      if (loop) {
#if LEIDEN_GRAPH_LOOP
        RAFT_CUDA_TRY(
          cudaGraphConditionalHandleCreate(&handle_, graph_, 1, cudaGraphCondAssignDefault));
        cudaGraphNodeParams cp{};
        cp.type               = cudaGraphNodeTypeConditional;
        cp.conditional.handle = handle_;
        cp.conditional.type   = cudaGraphCondTypeWhile;
        cp.conditional.size   = 1;
        cudaGraphNode_t cn;
#if CUDART_VERSION >= 13000
        RAFT_CUDA_TRY(cudaGraphAddNode(&cn, graph_, nullptr, nullptr, 0, &cp));
#else
        RAFT_CUDA_TRY(cudaGraphAddNode(&cn, graph_, nullptr, 0, &cp));
#endif
        body = cp.conditional.phGraph_out[0];
        patch_handle();
#else
        CUGRAPH_FAIL("leiden: GraphChain: no conditional nodes.");
#endif
      }
      nodes_.assign(static_cast<std::size_t>(count), nullptr);
      on_.assign(static_cast<std::size_t>(count), 1);
      void* args[kCallMaxArgs];
      std::vector<cudaGraphNode_t> dep;
      for (int i = 0; i < count; ++i) {
        const cudaKernelNodeParams p = in_[i].params(args);
        dep.clear();
        for (int d : deps[i]) {
          CUGRAPH_EXPECTS(d >= 0 && d < i, "leiden: GraphChain: bad dependency.");
          dep.push_back(nodes_[d]);
        }
        RAFT_CUDA_TRY(cudaGraphAddKernelNode(&nodes_[i], body, dep.data(), dep.size(), &p));
      }
      RAFT_CUDA_TRY(cudaGraphInstantiate(&exec_, graph_, 0));
      cur_  = in_;
      deps_ = deps;
      loop_ = loop;
      for (int i = 0; i < count; ++i) {
        if (!in_[i].enabled()) {
          RAFT_CUDA_TRY(cudaGraphNodeSetEnabled(exec_, nodes_[i], 0));
          on_[i] = 0;
        }
      }
    } catch (...) {
      reset();
      throw;
    }
  }

  // Only changed nodes are updated (an update costs host time and an upload).
  void update(int count)
  {
    void* args[kCallMaxArgs];
    for (int i = 0; i < count; ++i) {
      const KernelCall& c = in_[static_cast<std::size_t>(i)];
      if (c.same(cur_[static_cast<std::size_t>(i)])) continue;
      cur_[static_cast<std::size_t>(i)] = c;
      if (c.enabled()) {
        const cudaKernelNodeParams p = c.params(args);
        RAFT_CUDA_TRY(cudaGraphExecKernelNodeSetParams(exec_, nodes_[i], &p));
        if (!on_[i]) {
          RAFT_CUDA_TRY(cudaGraphNodeSetEnabled(exec_, nodes_[i], 1));
          on_[i] = 1;
        }
      } else if (on_[i]) {
        RAFT_CUDA_TRY(cudaGraphNodeSetEnabled(exec_, nodes_[i], 0));
        on_[i] = 0;
      }
    }
  }

  cudaGraph_t graph_    = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  std::vector<cudaGraphNode_t> nodes_;
  std::vector<char> on_;
  std::vector<KernelCall> in_;   // the launch being replayed
  std::vector<KernelCall> cur_;  // what the nodes hold
  GraphDeps deps_;
  bool loop_ = false;
#if LEIDEN_GRAPH_LOOP
  cudaGraphConditionalHandle handle_ = 0;
#endif
};

inline GraphDeps chain_deps(int count)
{  // call i waits for i - 1
  GraphDeps d(static_cast<std::size_t>(count));
  for (int i = 1; i < count; ++i)
    d[i].assign(1, i - 1);  // gcc 12: not = {}
  return d;
}

// Launches the calls in order: a chained replay of `graph` or stream launches.
inline void launch_chain(const std::vector<KernelCall>& calls, GraphChain* graph, cudaStream_t s)
{
  const int count = static_cast<int>(calls.size());
  if (count == 0) return;
  if (graph) {
    graph->launch(calls.data(), count, chain_deps(count), s);
  } else {
    launch_calls(calls.data(), count, s);
  }
}

}  // namespace cugraph::detail::leiden_engine
