# cuGraph Python API and Validation Patterns

These examples are grounded in the current cuGraph source tree. Verify signatures and algorithm support against the installed release. They use the high-level `cugraph` Python API; `pylibcugraph`, `cugraph.dask`, and the NetworkX backend have different contracts.

## Directed graph with stable string IDs and an isolate

```python
import cudf
import cugraph

edges = cudf.DataFrame(
    {
        "src": ["A", "B", "B", "C"],
        "dst": ["B", "C", "D", "D"],
        "latency_ms": [1.0, 2.0, 5.0, 1.0],
    }
)
all_vertices = cudf.Series(["A", "B", "C", "D", "E"])

if edges[["src", "dst"]].isna().any(axis=1).any():
    raise ValueError("null edge endpoint")

G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(
    edges,
    source="src",
    destination="dst",
    edge_attr="latency_ms",
    vertices=all_vertices,
    renumber=True,
)

assert G.is_directed()
assert G.is_weighted()
assert G.number_of_vertices() == all_vertices.nunique()
assert set(G.nodes().to_pandas()) == set(all_vertices.to_pandas())
```

The default renumbering maps the external IDs to contiguous internal IDs and high-level results unrenumber the vertex columns. Keep external IDs as the durable result keys.

## BFS reachability

BFS is the unweighted hop-distance method. Its result contains `vertex`, `distance`, and `predecessor` columns for a `Graph` input.

```python
bfs_result = cugraph.bfs(G, start="A", depth_limit=2)
reachable = cugraph.filter_unreachable(bfs_result)

source_row = reachable[reachable["vertex"] == "A"]
assert len(source_row) == 1
assert int(source_row["distance"].iloc[0]) == 0

# E is an isolate and must not be called reachable from A.
assert not (reachable["vertex"] == "E").any()
```

Do not filter with a hard-coded sentinel. `filter_unreachable` handles the maximum representable distance used by the current BFS/SSSP result dtype.

### Predecessor-edge validation

```python
pred = reachable[
    (reachable["vertex"] != "A") & (reachable["predecessor"] != -1)
][["vertex", "predecessor"]]
pred = pred.rename(columns={"vertex": "dst", "predecessor": "src"})

valid_steps = pred.merge(
    edges[["src", "dst"]].drop_duplicates(),
    on=["src", "dst"],
    how="leftsemi",
)
assert len(valid_steps) == len(pred)
```

This validates one predecessor tree. It does not enumerate all shortest paths or all simple paths.

## Weighted single-source shortest paths

`cugraph.sssp` requires a weighted graph in the current high-level implementation.

```python
sssp_result = cugraph.sssp(G, source="A")
reachable_weighted = cugraph.filter_unreachable(sssp_result)

d = reachable_weighted.set_index("vertex")["distance"]
assert abs(float(d.loc["D"]) - 4.0) <= 1e-6  # A -> B -> C -> D
```

Validate that weights encode additive path cost. If larger values mean stronger affinity rather than greater cost, SSSP on raw affinity is usually the wrong model; define a justified transformation or choose a different method.

The implementation documents unsupported negative-weight cycles. For ordinary distance/cost use, validate finite nonnegative weights unless the use case and algorithm contract explicitly justify otherwise.

## Simple graph versus multigraph

```python
repeated = cudf.DataFrame(
    {
        "src": [1, 1, 2],
        "dst": [2, 2, 3],
        "amount": [4.0, 6.0, 2.0],
    }
)

simple = cugraph.Graph(directed=True)
simple.from_cudf_edgelist(
    repeated, source="src", destination="dst", edge_attr="amount"
)

multi = cugraph.MultiGraph(directed=True)
multi.from_cudf_edgelist(
    repeated, source="src", destination="dst", edge_attr="amount"
)
```

A simple `Graph` drops parallel edges; a `MultiGraph` retains them. Neither behavior decides the business meaning. If repeats represent amounts on one relationship, aggregate explicitly and document the function:

```python
aggregated = (
    repeated.groupby(["src", "dst"], sort=False)["amount"]
    .sum()
    .reset_index()
)
```

If repeats are distinct events whose multiplicity matters, keep a `MultiGraph` only when the selected algorithm supports it. Preserve the raw event-edge table either way.

## Composite vertex identifiers

Do not concatenate multi-part IDs into a string: values such as `("ab", "c")` and `("a", "bc")` can collide, and delimiter escaping merely moves the ambiguity. Pass matching source and destination column lists and retain renumbering:

```python
composite_graph = cugraph.Graph(directed=True)
composite_graph.from_cudf_edgelist(
    edges,
    source=["src_tenant", "src_account"],
    destination=["dst_tenant", "dst_account"],
    renumber=True,
    store_transposed=True,
)
composite_rank = cugraph.pagerank(composite_graph)
```

The current PageRank tests receive unrenumbered composite keys in columns such as `0_vertex` and `1_vertex`, plus `pagerank`. Inspect and rename those result columns explicitly before a business-key handoff; do not join scores back by row order. Multi-column IDs require `renumber=True` in the high-level construction source.

## Undirected topology with edge IDs or types

Automatic symmetrization is not compatible with edge IDs or edge types. If the selected algorithm only needs topology, preserve the typed raw edge table separately and build an unweighted topology graph without those attributes. If metadata must remain in the graph, first create and validate an already-symmetric edge list, set `symmetrize=False`, and verify that the exact graph/attribute combination is supported. Never silently discard edge identity or invent reverse-edge identifiers.

## Components and arbitrary labels

In the current high-level source, `weakly_connected_components` rejects a directed `Graph`, while `strongly_connected_components` accepts directed input. For a `Graph` input, `directed`, `connection`, and `return_labels` cannot be passed to either component call. Build separate directed and undirected graph views deliberately rather than trying to override construction semantics at algorithm call time.

```python
undirected = cugraph.Graph(directed=False)
undirected.from_cudf_edgelist(
    edges,
    source="src",
    destination="dst",
    vertices=all_vertices,
)
components = cugraph.weakly_connected_components(undirected)

assert len(components) == G.number_of_vertices()
assert not components.duplicated(subset=["vertex"]).any()
```

Component IDs are arbitrary. To compare two results, canonicalize each partition to sets of external vertex IDs rather than comparing literal label values.

```python
def canonical_partition(frame):
    pdf = frame[["vertex", "labels"]].to_pandas()
    groups = [frozenset(g["vertex"]) for _, g in pdf.groupby("labels")]
    return frozenset(groups)
```

## PageRank with explicit convergence evidence

```python
rank_graph = cugraph.Graph(directed=True)
rank_graph.from_cudf_edgelist(
    edges,
    source="src",
    destination="dst",
    edge_attr="latency_ms",
    vertices=all_vertices,
    store_transposed=True,
)

pagerank, converged = cugraph.pagerank(
    rank_graph,
    alpha=0.85,
    max_iter=200,
    tol=1e-6,
    fail_on_nonconvergence=False,
)

if not converged:
    raise RuntimeError("PageRank did not converge under the declared settings")
assert len(pagerank) == rank_graph.number_of_vertices()
assert pagerank["pagerank"].notna().all()
assert abs(float(pagerank["pagerank"].sum()) - 1.0) <= 1e-4
```

The edge weight above is mechanically valid but may be semantically wrong for PageRank: latency as a positive transition strength makes high-latency edges more influential. Use weights only when their meaning matches the algorithm, or construct an unweighted/separately transformed graph.

## Louvain output

```python
community_graph = cugraph.Graph(directed=False)
community_graph.from_cudf_edgelist(
    edges,
    source="src",
    destination="dst",
    vertices=all_vertices,
)

membership, modularity = cugraph.louvain(
    community_graph,
    resolution=1.0,
    threshold=1e-7,
)
assert set(membership.columns) >= {"vertex", "partition"}
```

Record `resolution`, `threshold`, `max_level` when supplied, and modularity. A different partition ID is not a changed community; compare memberships after label permutation. A changed membership can be algorithmic variability or sensitivity, not automatically a defect.

## Result handoff checklist

Retain:

- raw and graph-ready edge counts;
- complete vertex count and isolate count;
- directed/weighted/multigraph/symmetrization policy;
- source, destination, weight, edge ID/type columns and dtypes;
- algorithm, parameters, convergence/sampling state, and package version;
- keyed result with no duplicate/missing required vertex IDs;
- miniature-fixture and algorithm-invariant checks;
- any unsupported operation or fallback/dispatch evidence.
