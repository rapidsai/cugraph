# SG/MG Split Plan

## Goal

Fully separate the SG and MG architecture so that:

- `libcugraph.so` contains only SG algorithms and SG-facing API surface
- `libcugraph_mg.so` contains only MG algorithms and MG-facing API surface
- `libcugraph_common.so` contains only neutral/shared helpers, data structures, and explicit instantiations
- `libcugraph.so` does not depend on `libcugraph_mg.so`
- `libcugraph_mg.so` does not depend on SG-only implementation details

## Current state

- `libcugraph.so` no longer depends on `libcugraph_mg.so`
- both DSOs depend on `libcugraph_common.so`
- `libcugraph_common.so` owns neutral shared instantiation shims
- SG/MG runtime separation is materially improved

## Work packages

### Stage 1: finish explicit-instantiation cleanup

Status: partially done

Still needed:

- audit remaining `_sg_` / `_mg_` source files that are compiled only because their symbols are transitively required by the other side
- either:
  - move the shared instantiations into neutral `*_common*` files, or
  - prove they are truly SG-only / MG-only and keep them where they are

Deliverable:

- `CUGRAPH_COMMON_SOURCES` contains only neutral files
- no SG/MG-named file is required by the opposite DSO for symbol resolution

### Stage 2: classify top-level algorithm ownership

Need to classify every source in:

- `CUGRAPH_SG_SOURCES`
- `CUGRAPH_MG_SOURCES`

into:

1. SG-only
2. MG-only
3. common helper/instantiation
4. suspicious mixed ownership

Deliverable:

- a clean source inventory
- ideally no `_mg_` file in SG sources and no `_sg_` file in MG sources unless there is a documented reason

### Stage 3: split template-heavy ownership at the header boundary

A lot of mixing comes from:

- template implementations in `.cuh`
- runtime variant dispatch that instantiates both SG/MG flavors in the same TU
- explicit instantiations whose names encode SG/MG but whose symbols are actually shared

Possible tactics:

- create more neutral explicit-instantiation files
- split variant-dispatch entrypoints into SG/MG/common wrappers
- move shared template instantiations out of SG/MG-named `.cu` files

### Stage 4: tighten exported CMake targets

Once source ownership is cleaner:

- make sure `cugraph::cugraph` exports only SG-facing link interface + `cugraph_common`
- make sure `cugraph::cugraph_mg` exports only MG-facing link interface + `cugraph_common`
- keep `cugraph_c` explicitly linking whichever DSOs it needs

### Stage 5: packaging and ABI review

For a real PR:

- confirm `libcugraph_common.so` installs correctly in conda
- confirm wheel bundling behavior is what we want
- check SONAME / runtime lookup assumptions
- document that `libcugraph_common.so` is internal-but-shipped

## Recommended execution order

1. Source inventory pass
2. Finish remaining shared-instantiation cleanup
3. Header/template boundary review
4. CMake/export cleanup
5. Packaging/CI validation

## Immediate next step

Produce a full inventory of current `CUGRAPH_SG_SOURCES` and `CUGRAPH_MG_SOURCES`, highlighting:

- all `_mg_` files in SG
- all `_sg_` files in MG
- all neutral files that maybe belong in common
- any source whose symbols are referenced by the opposite DSO

## Inventory results

Current source-list state after introducing `libcugraph_common.so` and removing the final MG-named files from SG:

- `CUGRAPH_SG_SOURCES`: 149 files
- `CUGRAPH_MG_SOURCES`: 118 files
- `CUGRAPH_COMMON_SOURCES`: 31 files

Naming state:

- SG contains no `_mg_` files
- MG contains no `_sg_` files
- common contains no `_sg_` or `_mg_` files

This establishes a clean source-ownership baseline by filename.

## Common-source buckets

`cpp/CMakeLists.txt` now represents these buckets explicitly as separate source lists:

- `CUGRAPH_COMMON_UTILITY_SOURCES`
- `CUGRAPH_COMMON_DISTRIBUTED_SOURCES`
- `CUGRAPH_COMMON_SAMPLING_SOURCES`
- `CUGRAPH_COMMON_TRAVERSAL_SOURCES`
- `CUGRAPH_COMMON_STRUCTURE_SOURCES`
- `CUGRAPH_COMMON_LINK_PREDICTION_SOURCES`

`CUGRAPH_COMMON_SOURCES` is assembled from those bucketed lists.

### Generic utility instantiations

These are neutral helper instantiations used by both DSOs:

- `src/detail/utility_wrappers_32.cu`
- `src/detail/utility_wrappers_64.cu`
- `src/detail/permute_range_v32.cu`
- `src/detail/permute_range_v64.cu`
- `src/utilities/invert_flags.cu`
- `src/utilities/validation_checks.cu`

### Distributed communication and shuffle primitives

These are common because both SG-facing code paths and MG algorithms use distributed communication/shuffle helpers:

- `src/detail/device_comm_wrapper_common_v32_e32.cu`
- `src/detail/device_comm_wrapper_common_v64_e64.cu`
- `src/utilities/shuffle_properties_common.cu`
- `src/utilities/shuffle_vertices_common_v32.cu`
- `src/utilities/shuffle_vertices_common_v64.cu`
- `src/utilities/shuffle_vertex_pairs_common_v32_e32.cu`
- `src/utilities/shuffle_vertex_pairs_common_v64_e64.cu`

### Graph view / graph structure primitives

These are common because SG and MG both consume graph-view and graph-structure explicit instantiations:

- `src/structure/graph_view_common_v32_e32.cu`
- `src/structure/graph_view_common_v64_e64.cu`
- `src/structure/graph_weight_utils_common_v32_e32.cu`
- `src/structure/graph_weight_utils_common_v64_e64.cu`
- `src/structure/renumber_utils_common_v32_e32.cu`
- `src/structure/renumber_utils_common_v64_e64.cu`

### Sampling shared helpers

These are common because sampling helper implementations are reused across SG- and MG-facing code paths:

- `src/sampling/detail/gather_sampled_properties.cu`
- `src/sampling/detail/update_visited_utils.cu`
- `src/sampling/detail/deduplicate_edges_by_minor.cu`
- `src/sampling/detail/shuffle_and_organize_output.cu`
- `src/sampling/detail/remove_visited_vertices_from_frontier_common_v32_e32.cu`
- `src/sampling/detail/remove_visited_vertices_from_frontier_common_v64_e64.cu`
- `src/sampling/detail/temporal_partition_vertices_v32.cu`
- `src/sampling/detail/temporal_partition_vertices_v64.cu`

### Link prediction shared instantiations

- `src/link_prediction/detail/similarity_common_v32_e32.cu`
- `src/link_prediction/detail/similarity_common_v64_e64.cu`

### Traversal shared instantiations

- `src/traversal/k_hop_nbrs_common_v32_e32.cu`
- `src/traversal/k_hop_nbrs_common_v64_e64.cu`

## Next-stage recommendations

1. Treat the generic utility bucket as stable common infrastructure.
2. Review the distributed communication/shuffle bucket as a possible `distributed_common` sublayer.
3. Review graph-view / graph-structure common files carefully: these are the most semantically MG-flavored common instantiations and are the main candidates for deeper API-boundary cleanup.
4. Keep sampling, link-prediction, and traversal common files as explicit-instantiation owners unless later profiling or ABI review indicates a better split.
5. Avoid chasing template-level ELF symbol duplication unless it maps back to an obvious duplicate instantiation owner.

## Common-layer stability classification

### Stable common infrastructure

These files are neutral by name and behavior. They should remain in `libcugraph_common.so` unless a later split creates a more specific common utility library.

- generic utility instantiations
- sampling shared helpers
- link-prediction shared instantiations
- traversal shared instantiations

These files are common because they own explicit instantiations or helper logic that is naturally reused by both SG and MG paths. No immediate API redesign is needed.

### Distributed common infrastructure

The distributed communication and shuffle bucket is common by behavior but deserves clearer architecture over time. It is not SG or MG algorithm code; it is a distributed primitive layer used by both sides.

Files in this bucket include:

- `device_comm_wrapper_common_*`
- `shuffle_properties_common.cu`
- `shuffle_vertices_common_*`
- `shuffle_vertex_pairs_common_*`

Recommendation: keep these in common, but consider documenting this bucket as a distributed primitive layer rather than generic utilities.

### MG-flavored common instantiations

The graph-view / graph-structure bucket is the most semantically important remaining area. Files like:

- `graph_view_common_*`
- `graph_weight_utils_common_*`

own `multi_gpu=true` instantiations that are consumed by both SG-facing and MG-facing DSOs. These are common by ownership but MG-flavored by template parameters.

This is acceptable in the current architecture because it lets `libcugraph.so` avoid depending on `libcugraph_mg.so`. However, it also marks the main future design question: should SG-facing code consume MG-flavored graph-view functionality at all, or should those call paths be refactored behind a more explicit distributed/common API?

Recommendation: keep these in common for now, but treat them as the main target for a future API-boundary review.

## Current milestone

Current achieved state:

- no `_mg_` files in `CUGRAPH_SG_SOURCES`
- no `_sg_` files in `CUGRAPH_MG_SOURCES`
- no `_sg_` or `_mg_` files in `CUGRAPH_COMMON_SOURCES`
- `libcugraph.so` does not link `libcugraph_mg.so`
- both `libcugraph.so` and `libcugraph_mg.so` link `libcugraph_common.so`

This establishes clean source-list ownership. Remaining work is about semantic API boundaries, not filename cleanup.
