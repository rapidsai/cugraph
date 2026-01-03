# Optimization Results: B/C + D Combination and Hash Table Analysis

## Results Achieved

### Performance Summary

| Optimization | Mean Time (ms) | Speedup vs Baseline | Hash Table % | Edge Mask % |
|--------------|----------------|---------------------|--------------|-------------|
| Baseline A | 42.67 | 1.00x | 32.5% | **62.6%** |
| Optimization D | 26.18 | 1.63x | **86.8%** | 0% |
| **Full B+C+D** | **16.09** | **2.65x** | 19.2% | 0% |

### Key Achievements

1. **2.65x speedup** from baseline (42.67ms → 16.09ms)
2. **Edge mask eliminated**: `transform_e_packed_bool` reduced from 62.6% to 0%
3. **Hash table relative reduction**: From 86.8% (after D) to 19.2% (after B+C+D)
4. **C++ integration complete**: `windowed_temporal_sampling_impl.hpp` ready for use

---

## Implementation Details

### B/C + D Combination

Successfully implemented in `windowed_temporal_sampling_impl.hpp`:

```cpp
// Window state for incremental updates (Optimization C)
template <typename edge_t, typename time_stamp_t>
struct window_state_t {
  rmm::device_uvector<edge_t> sorted_edge_indices;
  rmm::device_uvector<time_stamp_t> sorted_edge_times;
  size_t current_start_idx{0};
  size_t current_end_idx{0};
  bool initialized{false};
};

// Main function combining B/C with D
windowed_temporal_neighbor_sample_impl(
  ...,
  std::optional<time_stamp_t> window_start,  // B: Window start
  std::optional<time_stamp_t> window_end,    // B: Window end
  std::optional<window_state_t> window_state, // C: State for incremental
  ...);
```

### What B/C + D Does

| Approach | Time Window | Per-Query Filter |
|----------|-------------|------------------|
| D alone | None | edges where time < query_vertex_time |
| B/C alone | [window_start, window_end) | None |
| **B/C + D** | [window_start, window_end) AND time < query_vertex_time |

---

## Hash Table Analysis

### Current State (After B+C+D)

After all optimizations, `cuco::insert_if_n` is at 19.2% of GPU time:
- Absolute time: 268.78ms over 30 iterations ≈ 8.96ms per iteration
- Potential savings if 2x faster: ~4.5ms per iteration
- Expected additional speedup: 16.09ms → ~11.6ms (1.4x more)

### Why CG Size Increase Is Invasive

**Attempted and failed.** The cuGraph codebase uses device-side hash table operations:

```cpp
// key_store.cuh line 76
__device__ bool contains(key_type key) const { 
  return cuco_store_device_ref.contains(key);  // Requires CG size == 1
}

// key_store.cuh line 93
__device__ void insert(key_type key) { 
  cuco_store_device_ref.insert(key);  // Requires CG size == 1
}
```

For CG size > 1, ALL callers must change to use cooperative group tiles:

```cpp
// Would require cooperative group tile parameter
__device__ bool contains(cg::thread_block_tile<4> tile, key_type key) const { 
  return cuco_store_device_ref.contains(tile, key);
}
```

**This affects 15+ files** across community, structure, sampling, traversal, components.

### Alternative: key_store_cg.cuh

Created `prims/key_store_cg.cuh` with:
- CG-compatible key store (CG size = 4)
- Alternative deduplication via sort + unique
- Can be used incrementally for new code paths

```cpp
// key_store_cg.cuh
template <typename key_t>
class key_store_cg_t {
  // Uses CG size = 4 for parallel probing
  using cuco_set_type = cuco::static_set<key_t, ...,
    cuco::linear_probing<4, cuco::murmurhash3_32<key_t>>,
    ...>;
};

// Alternative: sort + unique for deduplication
template <typename vertex_t>
size_t deduplicate_sort_unique(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& vertices);
```

---

## Future Optimization Opportunities

| Option | Expected Impact | Complexity | Status |
|--------|-----------------|------------|--------|
| Binary search mode | Better for small frontiers | Low | 📝 Proposed |
| Hybrid hash/sort | Optimal per size | Medium | 📝 Proposed |
| Full CG migration | 2-4x hash speedup | High | ⚠️ Invasive |
| Skip dedup when safe | Avoid hash entirely | Low | 📝 Proposed |

### Recommended Next Steps

1. **Profile frontier sizes** to determine if binary search mode would help
2. **Test sort+unique** as alternative to hash table for deduplication
3. **Incremental CG migration** for hot paths only (if needed)

---

## nsys Profile Files

| Configuration | Profile Path |
|---------------|--------------|
| Baseline A | `benchmarks/baseline_A_fixed_profile.nsys-rep` |
| Optimization D | `benchmarks/optimization_D_profile.nsys-rep` |
| Full B+C+D | `benchmarks/optimization_full_BCD_profile.nsys-rep` |

---

## GPU Kernel Breakdown (Full B+C+D)

| Kernel | Time % | Description |
|--------|--------|-------------|
| `DeviceMergeSortMergeKernel` | 28.5% | Sorting operations |
| `cuco::insert_if_n` | 19.2% | Hash table for dedup |
| `cupy_take` | 10.4% | Graph data access (Python) |
| `DeviceRadixSortOnesweep` | 8.2% | Radix sort |
| `transform_v_frontier_e_hypersparse` | **0.1%** | Inline temporal filter (D) |
| `transform_e_packed_bool` | **0%** | **Eliminated** |

---

## References

- [CUDA Programming Guide - Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [cuCollections (cuco)](https://github.com/NVIDIA/cuCollections)
- [Thrust Documentation](https://nvidia.github.io/thrust/)
