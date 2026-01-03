# Optimization Proposal: B/C + D Combination and Hash Table Fix

## Current State After Optimization D

| Kernel | Time % | Description |
|--------|--------|-------------|
| `cuco::insert_if_n` | **86.8%** | Hash table insertions for deduplication |
| `transform_v_frontier_e_hypersparse` | 0.3% | Inline temporal filtering (Optimization D) |

## Issue 1: Combining B/C with D for Rolling Window Temporal Sampling

### Why Combine B/C with D?

| Approach | What it does | Time Window | Per-Query Filter |
|----------|--------------|-------------|------------------|
| D alone | Inline temporal filter | None | edges where time < query_vertex_time |
| B/C alone | Pre-filter to window | [window_start, window_end) | None |
| **B/C + D** | Both | [window_start, window_end) AND time < query_vertex_time |

### Use Case: Rolling Window Sampling

For a scenario like "1-year rolling window over 2-year data":
1. **B/C**: Pre-filter to edges in [current_day - 365, current_day)
2. **D**: For each query vertex, further filter to edges with time < vertex_time

### Implementation Plan

```cpp
// In temporal_sampling_impl.hpp

// Step 1: Set window mask (B/C) - O(ΔE) incremental per window slide
if (window_based_sampling) {
  if (first_iteration) {
    // Full window setup using binary search + mask set
    auto [start_idx, end_idx] = compute_window_bounds_binary_search(
      handle, sorted_edge_times, num_edges, window_start, window_end);
    set_mask_from_sorted_range(handle, edge_mask, sorted_edge_indices, start_idx, end_idx);
  } else {
    // Incremental update - only process delta edges
    update_mask_incremental(handle, edge_mask, sorted_edge_indices,
                            leaving_start, leaving_end, entering_start, entering_end);
  }
  
  // Attach window mask to graph view
  temporal_graph_view.attach_edge_mask(window_edge_mask.view());
}

// Step 2: Sample with D (inline temporal filtering) - operates on windowed graph
auto [srcs, dsts, ...] = temporal_sample_edges<...>(
  handle, rng_state, 
  temporal_graph_view,  // Now has window mask attached
  ...,
  edge_start_time_view,
  frontier_vertex_times,  // D: per-vertex temporal filter
  ...);
```

### Expected Benefit

- **B/C overhead**: ~0.2ms per window slide (410K delta edges)
- **D improvement**: Potentially faster since graph is smaller (50% of edges after window)
- **Total**: ~26ms + 0.2ms ≈ 26ms (similar to D alone, but with proper windowing)

---

## Issue 2: Hash Table Bottleneck

### Current Problem

After D, `cuco::insert_if_n` dominates at 86.8% of GPU time. This is used for vertex deduplication:

```cpp
// key_store.cuh line 273
void insert_if(KeyIterator key_first, KeyIterator key_last,
               StencilIterator stencil_first, PredOp pred_op, ...)
{
  size_ += cuco_store_->insert_if(key_first, key_last, stencil_first, pred_op, stream.value());
}
```

### Root Cause Analysis

1. **CG size = 1**: Single-thread probing is inefficient
   ```cpp
   cuco::linear_probing<1, cuco::murmurhash3_32<key_t>>
   ```

2. **Load factor 0.7**: 30% empty slots, but still high collision rate

3. **Hash table for small sets**: For small frontiers, sort+unique is faster

### Proposed Solutions

#### Option A: Increase Cooperative Group Size (NOT VIABLE)

```cpp
// Change from CG size 1 to 4 for parallel probing
cuco::linear_probing<4, cuco::murmurhash3_32<key_t>>
```

**Status**: Tested and failed. The error:
```
"Non-CG operation is incompatible with the current probing scheme"
```

The cuGraph code uses non-CG (single-thread) device-side `find()` operations
that are incompatible with CG size > 1. Fixing this would require changing
all hash table access patterns to use cooperative groups.

#### Option B: Use Binary Search Mode

The `key_store_t` already supports binary search mode:

```cpp
// Change from:
key_store_t<vertex_t, false>  // hash table mode

// To:
key_store_t<vertex_t, true>   // binary search mode (sort + unique)
```

For deduplication, binary search mode would:
1. Sort vertices: O(n log n)
2. Unique: O(n)
3. Binary search for lookups: O(log n) per lookup

This may be faster for smaller frontiers (<1M vertices) due to better cache behavior.

#### Option C: Skip Deduplication When Possible

Check if deduplication is actually needed:

```cpp
// If duplicate vertices don't cause correctness issues, skip dedup
if (!require_strict_deduplication) {
  // Process duplicates, just waste some work
  // Faster than expensive hash table
}
```

#### Option D: Hybrid Approach

Choose strategy based on frontier size:

```cpp
if (frontier_size < threshold) {
  // Small frontier: sort + unique
  thrust::sort(frontier.begin(), frontier.end());
  auto new_end = thrust::unique(frontier.begin(), frontier.end());
  frontier.resize(new_end - frontier.begin());
} else {
  // Large frontier: hash table
  key_store.insert_if(frontier.begin(), frontier.end(), ...);
}
```

### Recommended Implementation Order

1. **First**: Try Option A (CG size change) - minimal code change
2. **Second**: Try Option B (binary search mode) - test performance
3. **Third**: Implement Option D (hybrid) if neither A nor B is sufficient

---

## Summary

| Optimization | Target | Expected Impact | Complexity |
|--------------|--------|-----------------|------------|
| B/C + D combination | Rolling window sampling | Enables proper windowing | Medium |
| Hash CG size increase | `cuco::insert_if_n` | 2-4x hash speedup | Low |
| Binary search mode | Small frontiers | Better cache behavior | Low |
| Hybrid approach | All frontier sizes | Optimal per size | Medium |

## Next Steps

1. [ ] Implement B/C + D combination for rolling window benchmark
2. [ ] Test CG size increase (Option A)
3. [ ] Benchmark binary search mode (Option B) 
4. [ ] Profile and compare all approaches
