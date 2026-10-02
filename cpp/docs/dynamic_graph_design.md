# Dynamic Graph Storage Design (cuGraph)

Focus of this document: **decisions that translate into code**.

---

## 1. Goals

- Target **large, long-lived graphs**. Building a tiny graph is cheap; on updates, **recreate a static `graph_t` from scratch** instead of using this dynamic graph data structure.
- Vertex and edge insert/delete cost proportional to the insertion/deletion **batch** size, not \(O(V)\) or \(O(E)\).
- \(V_{\max}\) is `std::optional<vertex_t>`. If set, vertex insert **fails if it would make \(V\) exceed \(V_{\max}\)**. If omitted, the vertex set is **fixed at creation** (Hornet): **no vertex insert or delete**. There is **no \(E_{\max}\)**; edge insert fails only when **allocation** fails.
- Peak topology memory about **+10–20% vs a static `graph_t`** when \(V \approx V_{\max}\).
- Reduce **per-vertex (per-major) adjacency metadata** from Hornet’s **32 bytes** to **8 bytes** (§4.1).
- Keep the existing types `graph_t`, `graph_view_t`, `edge_property_t`, `edge_src_property_t`, and `edge_dst_property_t`, and add a new **`vertex_property_t`**. A static `graph_t` and its `graph_view_t` both have a fixed topology. On a **dynamic** graph, `graph_t`'s topology is updatable while a `graph_view_t` is a frozen snapshot. Properties follow the object from which they are constructed (§2.1, §3). For properties created from a dynamic `graph_t`, memory slots can be added or removed as vertices, edges, or edge sources/destinations are inserted or deleted. Properties created from an immutable snapshot become invalid once vertices or edges are added to or removed from the original `graph_t`.
- Reuse the Hornet-inspired **block-array allocator**.
- Largely maintain cuGraph’s existing mechanisms, especially those implemented to support multi-GPU and 2D partitioning.
- **Python / pylibcugraph thin**: create(optional \(V_{\max}\)) and insert/delete wrappers only. Shuffle, renumber, and ownership stay in C/C++.

---

## 2. Graph, IDs, and vertex capacity

### 2.1 `graph_t` vs `graph_view_t`

| Type | Role |
|------|------|
| **`graph_t`** | Owns topology. Static: immutable. Dynamic: batch insert/delete. |
| **`graph_view_t`** | Algorithm view. Static: immutable, a view object of the original `graph_t`. Dynamic: an immutable snapshot of the `graph_t` at the time the view is created. Updating the `graph_t` will invalidate the view object.|

`dynamic` is a non-type template parameter on `graph_t` and `graph_view_t`, default `false`. The static path is the `false` instantiation.

`attach_edge_mask` attaches only to `graph_view_t`, same as static.

`create_graph_snapshot_index` takes a dynamic `graph_view_t const&` and returns a `graph_snapshot_index_t`, the same pattern as an edge mask. Attach it with `attach_graph_snapshot_index` before creating any property from that snapshot: `vertex_property_t`, `edge_property_t`, `edge_src_property_t`, or `edge_dst_property_t`.

Currently, it is a `std::vector` of `rmm::device_uvector`, one device vector (edge offset values) per edge partition; this is used for `edge_property_t` only for now.

- **Dense per-vertex metadata.** Each device vector stores offsets computed from each major’s degree.
- **Chunked metadata (§4.3).** Next step. Store offsets in two levels: an offset for each chunk, then the offsets of the non-zero local degree major vertex IDs within that chunk.

### 2.2 Id space vs `number_of_vertices()`

Static cuGraph: `number_of_vertices() == max_id + 1`, and `local_vertex_partition_range_*` is exactly the local vertex set.

Dynamic with \(V_{\max}\): in \([0, V_{\max})\) (multi-GPU stripe \(L = V_{\max}/P\)) many IDs are **void** (not present), which is not the same as **active isolate** (degree 0). Without \(V_{\max}\), the id space is the vertices given at creation and does not grow or shrink.

| API | Meaning |
|-----|---------|
| `number_of_vertices()` | **Active** vertex count |
| `local_vertex_partition_range_*` | **ID-space** stripe, including voids |

`graph_t` stores a bitmap over the local vertex partition range. A bit is 1 when that local vertex is active and 0 when it is void.

Python and pylibcugraph outputs are **active vertices only** (external IDs). Callers do not see the internal id stripe or the holes in it.

### 2.3 Vertex capacity (single-GPU and multi-GPU)

\(V_{\max}\) is `std::optional<vertex_t>`.

- **Set:** insert vertex takes a free id and marks it active; it **fails at \(V_{\max}\)** even if memory remains. Delete frees adjacency and property slots and returns the id. Idle \(V_{\max}-V\) is acceptable if users set \(V_{\max}\) near peak live \(V\).
- **Omitted:** vertex count is fixed at creation, same as Hornet. `insert_vertices` / `delete_vertices` are not allowed. Edge insert and delete still are.
- **No \(E_{\max}\).** Edge insert fails if the allocator cannot get a block.

**Multi-GPU:** each rank has \(L = V_{\max}/P\) slots when \(V_{\max}\) is set, otherwise \(L\) is the fixed local vertex count. **Home** = `Hash(external vertex id)` (existing `partition_manager` map, including 2D). Internal owner = internal vertex id / \(L\).

Hashing plus a hard local cap \(L\) cannot guarantee a global live count of \(V_{\max}\) without a tail path: some homes fill first. That tail exists only when \(V_{\max}\) is set. The **home** GPU (`Hash(external vertex id)`) keeps `(external vertex id, owning GPU)` pairs for vertices it placed elsewhere. A later shuffle resolves an existing external id by asking that home. `compute_gpu_id_from_ext_*` uses this map; a missing pair means the vertex lives on the home.

Batch vertex updates (only when \(V_{\max}\) is set): user call order is honored. Guide users to **delete before insert** if possible so a mixed batch can reuse slots just freed; if the user inserts first, insert runs first.

**External vs internal IDs:** If \(V_{\max}\) is set, `Hash(external vertex id)` and overflow are how **external** vertices (vertex IDs before renumbering) and edges are shuffled onto GPUs. After renumber, **internal** vertex and edge partitions are equal stripes of \(L\) (internal vertex id / \(L\)). If \(V_{\max}\) is not set, we use the same mechanism used for a static graph for shuffling **external** and **internal** vertices and edges.

---

## 3. Properties and updates

```text
insert_vertices(graph, vertex_ids, live_vertex_properties…, live_edge_src/dst_properties...);
delete_vertices(graph, vertex_ids, live_vertex_properties…, live_edge_src/dst_properties...);
delete_and_insert_vertices(graph, delete_vertex_ids, insert_vertex_ids, live_vertex_properties…, live_edge_src/dst_properties...);

// Edge updates that do not insert vertices (initial API, missing endpoints are disallowed)
insert_edges(graph, src_ids, dst_ids, live_edge_properties...);
delete_edges(graph, src_ids, dst_ids, live_edge_properties...);
delete_and_insert_edges(graph, delete_src_ids, delete_dst_ids, insert_src_ids, insert_dst_ids, live_edge_properties...);

// Edge updates that may insert vertices (possible future overloads, insert missing endpoint vertices)
insert_edges(graph, src_ids, dst_ids, live_vertex_properties…, live_edge_src/dst_properties…, live_edge_properties...);
delete_and_insert_edges(graph, delete_src_ids, delete_dst_ids, insert_src_ids, insert_dst_ids, live_vertex_properties…, live_edge_src/dst_properties…, live_edge_properties...);
```

Only property objects created from the dynamic `graph_t` can be passed here. Their memory slots are added or removed with the graph. Property objects created from a `graph_view_t` cannot be passed.

Vertex updates take every live `vertex_property_t` and every live `edge_src_property_t` / `edge_dst_property_t`. Edge updates that cannot insert vertices take every live `edge_property_t`. An edge-update overload that may insert vertices must also take every live `vertex_property_t` and `edge_src_property_t` / `edge_dst_property_t`. Omitting a required property leaves it out of sync with `graph_t`; using an out-of-sync property object is undefined behavior.

The initial edge-update API **disallows missing endpoints**: a `src` or `dst` that is not an active vertex is an error. The overloads that permit vertex insertion may be added later. They require \(V_{\max}\) and insert missing endpoints before updating the edges.

---

## 4. Adjacency lists

Each vertex (each **major**, once a neighbor list is split across GPUs) has one adjacency list and an 8-byte metadata word. A typical list is one power-of-two block from the `graph_t` layout (§8). A very high local degree uses multiple neighbor blocks (§4.1).

### 4.1 Degree classes

Three adjacency modes. The top 2 bits of the 8-byte word select the mode.

| Tag | Mode |
|-----|------|
| `00` | **Inline.** Neighbors are stored in the remaining 62 bits. |
| `01` | **Single block.** One power-of-two block from the `graph_t` layout (§8). |
| `10` | **Multi-block.** Degree beyond one slab block. The first-level word addresses a power-of-two block of 8-byte metadata words. |

Drop Hornet’s four 8-byte fields (pointer, capacity, size, start). Each direct or second-level metadata entry is **8 bytes (reduced from Hornet’s 32 bytes)**. A single-block address is `base[bin][slab_id] + block_index * sizeof(T) * 2^{bin}`, used length `size`.

**Single-block payload** (62 bits, `slab_size = 2^23`):

| Field | Bits |
|-------|------|
| `tag` | **2** |
| `cap_log` | **5** (the bin; values through 31 represent capacities through \(2^{31}\); this can handle a slab size up to \(2^{31}\)) |
| `block_index` + `size` | **23** (exponent of the slab size) |
| `slab_id` | **34** |

`block_index` and `size` share 23 bits: a larger block widens `size` and shrinks `block_index` (\(2^{23-k}\) blocks in the slab, where \(2^k\) is the size of a single block). **34-bit `slab_id`** ⇒ at most \(2^{23}\times 2^{34}=2^{57}\) elements in one power-of-two bin on one GPU. That local-edge ceiling per bin is enough for practical GPUs for the foreseeable future.

**Inline payload** (62 bits). The initial implementation stores a minor, as cuGraph does today. With \(V_{\max}\), the minor width is \(\lceil\log_2 V_{\max}\rceil\); without it, the width is \(\lceil\log_2 V\rceil\). The inline capacity is how many minors of that width fit in 62 bits. A later change may store a minor offset instead. Then the width is \(\lceil\log_2 L\rceil\) when \(V_{\max}\) is set (\(L = V_{\max}/P\)), and the maximum local minor-range size when \(V_{\max}\) is omitted.

**Multi-block.** The first-level word uses the single-block payload to address a power-of-two block whose elements are 8-byte metadata words. Each of those words is a single-block word and addresses one block of neighbors.

### 4.2 Order inside a neighbor list

**Low-degree majors** keep their neighbor lists packed and sorted after a batch insert or delete.

**High-degree majors**, initial design: **delete first**. Delete **compacts** the neighbor list. Insert **sorts** the new neighbors, then **merges** them into the existing list. On a multi-block list, both steps run only on the affected blocks. We may consider more complex data structures in the future to limit the insertion and deletion cost while avoiding a full neighbor-list scan during search.

### 4.3 Two-level metadata when a neighbor list is partitioned

Enable this layout only when a neighbor list is split across many GPUs and the average vertex degree divided by that partition count is small, so many majors have local degree 0. Otherwise — single-GPU, vertex-owned 1D, or 2D with a larger average local degree — store a direct 8-byte metadata word for every major.

Partition each GPU’s **local major-vertex range** into fixed-size chunks: `chunk = (v - major_first) / chunk_size`, `lane = (v - major_first) % chunk_size`. Two options for a chunk:

**Small chunk (initial).** A bitmap of `chunk_size` bits. A bit is 1 when that vertex has non-zero local degree.

**Larger chunk, mostly local degree 0.** Store the offsets of the non-zero local degree vertices within the chunk, then one 8-byte metadata word per such vertex. The word uses the format in §4.1. An offset is 1 byte when the chunk size is at most 256, and 2 bytes when the chunk size is at most \(2^{16}\).

---

## 5. `vertex_property_t`

A `vertex_property_t` is aligned with the local owned vertex-id stripe. With \(V_{\max}\), its length is \(L = V_{\max}/P\), including slots for void vertices. Without \(V_{\max}\), its length is the fixed local vertex count.

### 5.1 Created from `graph_t` (memory slots can be added or removed)

The property is passed to every vertex insert and delete (§3) so those calls can **add or remove memory slots**. Insert adds a slot for the new vertex. Delete frees that slot for reuse. Note that adding or removing memory slots in `vertex_property_t` is conceptual. In actual implementation, we assign a consecutive array for the entire local vertex partition range. `graph_t` maintains a bitmap marking active vertices. Only memory slots corresponding to active vertices are conceptually available.

### 5.2 Created from `graph_view_t`

On a dynamic graph the view is a snapshot, and the view object becomes invalid once the original `graph_t` is updated. Creating the property requires the `graph_snapshot_index_t` already attached to that view (§2.1). `vertex_property_t` is internally one device vector for the local vertex partition range.

---

## 6. `edge_property_t`

An `edge_property_t` created from `graph_t` can gain or lose memory slots with edge insert and delete. One created from `graph_view_t` cannot gain or lose memory slots (§2.1, §3).

### 6.1 Created from `graph_t` (memory slots can be added or removed)

A live `edge_property_t` has **no metadata of its own**. It shares `graph_t`’s adjacency metadata. The same `(bin, slab_id, block_index)` indexes its value buffer (§8).

Encourage `cuda::std::tuple` instead of creating multiple `edge_property_t` objects. A block element is at least **1 byte**, including a 1-bit value (we do not support packing for dynamic `edge_property_t`).

A live `edge_property_t` created after edges already exist copies the current `graph_t` layout (§8). Later insert and delete stay correct only if they add or remove slots in `graph_t` and that property together.

### 6.2 Created from `graph_view_t`

On a dynamic graph the view is a snapshot: the topology is frozen, and later insert or delete to the original `graph_t` invalidates the `graph_view_t` and any property objects created from the `graph_view_t`. Creating the property requires the `graph_snapshot_index_t` already attached to that view (§2.1).

---

## 7. `edge_src_property_t` and `edge_dst_property_t`

Same split as `edge_property_t`: a property created from `graph_t` is passed on insert/delete (§3) so those calls can add or remove memory slots; a property created from a dynamic `graph_view_t` (a snapshot) becomes invalid once the original `graph_t` gets updated. Creating `edge_src_property_t` or `edge_dst_property_t` from `graph_view_t` requires the `graph_snapshot_index_t` already attached to the `graph_view_t` (§2.1).

By default, the property for majors is one `rmm::device_uvector` per edge partition, covering that partition’s entire major range. The property for minors is a single `rmm::device_uvector` covering the minor range. This is the same for a property created from `graph_t` and one created from a snapshot. Later, a lower-memory layout may store only the existing sources or destinations, using chunking with a bitmap or the unique source or destination vertices.

---

## 8. Allocator

```
graph_t
  └── std::array<std::vector<bit_tree_t>, num_bins_v>     ← the only block layout
        bin k: slabs whose blocks hold 2^k elements
              each slab: bit_tree_t (which blocks are free)

block_array_manager_t<T>                                  ← owned by graph_t (adjacency) and
  └── per bin k: std::vector<block_array_t<T>>               by each live edge_property_t<T>
        block_array_t<T> = one GPU buffer, no bit_tree_t;
        indexed by the same (bin, slab_id) as the graph_t layout
```

Many vertices share a slab; each vertex owns a private power-of-two **block** in that slab. Default slab: `1 << 23` **elements** (\(\times sizeof(T)\) bytes: ~32 MB for a 4-byte `T`, ~8 MB only if `T` is 1 byte). Hornet **rejects** degree ≥ that; we will **chain multiple blocks** for high degree.

**One layout, many buffers.** `graph_t` owns the bit trees. `allocate` / `release` on that layout returns `(bin, slab_id, block_index)`. The adjacency `block_array_manager_t` and the one in every live `edge_property_t` use that same triplet, so their block arrays stay aligned. Neither the manager nor `block_array_t` tracks free blocks.

A live `edge_property_t` created after the graph already has edges copies this layout: the same bins, slab IDs, and occupied blocks. After that, every insert and delete updates `graph_t` and every live `edge_property_t` together. The property has no layout of its own. Destroying `graph_t` makes the property uninterpretable, which is acceptable: the property is meaningless without that graph. Do not share one layout across graphs.

**`slab_id`:** not a device pointer. When the layout creates a new slab in bin \(k\) (`elements_per_block = 2^k`), assign the next integer from a **per-bin counter** (0, 1, 2, …). The per-vertex word stores that id with the bin and `block_index` (§4). Kernels address a slab through `base[k][slab_id]`. Do not reuse an id while any packed word still names it (monotonic IDs by default).

**Bulk allocate** many blocks of one power-of-two size in one call when building a graph with many edges. One-block allocate in a loop is too slow.

---

## 9. `transpose_graph` / `transpose_graph_storage`

Both **build a new** `graph_t`. **For now the output kind matches the input:** static → static, dynamic → dynamic. The same rule applies later to other new-graph ops (`coarsen`, induced subgraph).

| Function | What it does | Output `store_transposed` |
|----------|----------------|---------------------------|
| `transpose_graph` | Topology transpose (in-edges ↔ out-edges) | Same as input |
| `transpose_graph_storage` | Storage only (single-GPU CSR ↔ CSC; multi-GPU CSR+DCSR hybrid ↔ CSC+DCSC hybrid) | Flipped |

Static → static materializes CSR. \(O(E)\) is expected; this is a new graph, not an in-place update.

Dynamic → dynamic keeps the input’s \(V_{\max}\) (including omitted: the vertex set stays fixed). It does **not** preserve occupancy or overflow guests: extract an edgelist and rebuild. Rebuild `edge_property_t`, `vertex_property_t`, and `edge_src_property_t` / `edge_dst_property_t` with that graph.

---

## 10. Implementation plan

1. **Allocator.** Move block ownership to the layout `graph_t` will hold: `bit_tree_t` leaves `block_array_t`, and `block_array_manager_t` holds the typed block arrays (§8).
2. **`vertex_property_t` and vertex primitives.** Add `vertex_property_t` and update the vertex primitives to use it.
3. **`is_dynamic = false`.** Add the non-type template parameter to `graph_t` and `graph_view_t`, default `false` (§2.1). Current instantiations stay the static path.
4. **Edge insert and delete.** Dynamic `graph_t` and `graph_view_t` support edge insert and delete: per-vertex adjacency metadata (§4), and a live `edge_property_t` that shares the `graph_t` layout (§6.1). Add `graph_snapshot_index_t` so a snapshot can create an `edge_property_t` (§2.1, §6.2).
5. **Vertex insert and delete.** Active/void bitmap, optional \(V_{\max}\), and `edge_src_property_t` / `edge_dst_property_t` (§2.3, §7).
6. **Multi-GPU.** Keep the existing shuffle, renumber, and 2D partitioning. Use a direct metadata word per major until a neighbor list is split across GPUs; add the chunked layout only then (§4.3).
7. **Remaining primitives, one by one.** Edge and traversal primitives follow the graph API.
8. **Python.** Thin create and insert/delete wrappers. Shuffle, renumber, and ownership stay in C/C++.
9. **Memory and performance.** Lower-memory source and destination properties, high-degree updates that touch only the affected blocks, and other costs found after the API works.

---

## 11. Research questions

- **Neighbors of a large local-degree vertex (§4.2).** Keeping that neighbor list sorted makes an insertion cost more than the number of inserted neighbors. Leaving it unsorted makes a search scan the entire neighbor list.

---

## 12. References

- Hornet: Busato et al., “Hornet: An Efficient Data Structure for Dynamic Sparse Graphs and Matrices on GPUs.”