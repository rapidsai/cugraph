---
name: cugraph-graph-analysis
version: "1.0.0"
description: Use for cuGraph construction and analytics where direction, weights, parallel edges, isolates, or vertex identity matter; not table-only aggregation.
license: Apache-2.0
metadata:
  author: "NVIDIA <opensource@nvidia.com>"
  tags:
    - cugraph
    - graph-analytics
    - networkx
    - traversal
    - centrality
---

# cuGraph Graph Analysis

## Purpose

Turn a relational question into a correctly constructed graph, select an algorithm that answers that question, execute it with the appropriate cuGraph interface, and validate the keyed result.

## Prerequisites

- A supported NVIDIA GPU environment with version-matched cuGraph and cuDF packages; consult the RAPIDS installation guide for current platform requirements.
- A source edge definition, stable external vertex IDs, and enough domain context to choose direction, weight, multiplicity, and isolate semantics.
- `nx-cugraph`, `pylibcugraph`, or Dask dependencies only when that interface is selected. No API key is required by cuGraph itself.

## When to use this skill

Use cuGraph when topology matters: multi-hop reachability, shortest paths, connected components, centrality, communities, link similarity/prediction, cores, random walks, or graph sampling.

Do **not** build a graph for a one-hop join, grouped count, or ordinary lookup that cuDF can answer directly. Do not use a structural score as a prediction of future behavior without a separate predictive design. An optimization or action-selection question needs an optimizer after the graph evidence is established.

## Instructions

## 1. Define graph semantics before code

Record these choices:

| Element | Required definition |
|---|---|
| Vertex | Entity type, stable external identifier, and complete required universe |
| Edge | Meaning of `source -> destination`, time/snapshot, and inclusion rule |
| Direction | Directed or undirected, justified by the question |
| Weight | Meaning, units, valid sign/range, and whether larger means closer or farther |
| Multiplicity | Whether repeated interactions are distinct edges or must be aggregated |
| Self-loops | Valid observations, removable noise, or unsupported by the selected algorithm |
| Isolates | Whether entities with no edges belong in the analytical population |
| Output | Algorithm, parameters, result grain, and interpretation |

A field named `parent`, `follows`, or `depends_on` does not by itself prove edge direction. Verify direction on a tiny named chain whose expected reachability is obvious.

Do not project a bipartite or event model into entity-to-entity edges unless the projection rule is part of the analysis. A projection can discard edge type, timing, quantities, and path meaning.

## 2. Select the interface

- **`cugraph` Python API:** preferred for new Python graph analytics using cuDF edge lists.
- **NetworkX with the `nx-cugraph` backend:** preferred when the user wants to retain NetworkX code. Check dispatch and backend coverage rather than assuming every call used the GPU.
- **`pylibcugraph`:** lower-level Python bindings for tighter integration and fewer high-level dependencies; use only when the caller needs its explicit resource/array interface.
- **libcugraph C/C++ or C API:** use for native integration, not as the default answer to a Python request.
- **`cugraph.dask`:** use only for algorithms available in the distributed namespace and a demonstrated capacity or throughput need. Distributed and single-GPU APIs are not interchangeable.

Check the installed release's algorithm matrix and signature before implementation. A symbol in a newer document is not evidence that the current environment supports it.

## 3. Construct the graph without losing meaning

For the high-level Python API:

```python
import cudf
import cugraph

edges = cudf.DataFrame(
    {
        "src": ["A", "B", "C"],
        "dst": ["B", "C", "D"],
        "cost": [1.0, 2.0, 1.5],
    }
)
vertices = cudf.Series(["A", "B", "C", "D", "ISOLATE"])

G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(
    edges,
    source="src",
    destination="dst",
    edge_attr="cost",
    vertices=vertices,
    renumber=True,
)
```

Grounded behavior in the current source tree:

- `Graph()` defaults to undirected; pass `directed=True` intentionally when direction matters.
- `from_cudf_edgelist` defaults to `renumber=True`. External IDs may be strings, non-contiguous, or composite columns. Pass matching source/destination column lists for composite IDs; never make a supposedly unique key by lossy string concatenation. Multi-column IDs require renumbering, and output key columns must be inspected and explicitly renamed before handoff.
- Pass the complete `vertices=` collection on the single-GPU construction path when isolates must be represented.
- An undirected `Graph` symmetrizes input by default. Set `symmetrize=False` only when the edge list is already symmetric and that assertion has been checked. Symmetrization cannot be requested with edge IDs or edge types.
- A simple `Graph` drops parallel edges during construction; a `MultiGraph` retains them. Confirm that the selected algorithm supports the chosen graph type. Aggregating repeated edges is a modeling decision, not a harmless optimization.
- Null source/destination values are rejected. Validate and resolve them before graph construction.

Preserve the raw edge table and any aggregated graph-edge table separately so the transformation can be audited.

## 4. Match the method to the question

| Question | Method family | Important caveat |
|---|---|---|
| Which vertices are reachable within hops? | `bfs` | Unweighted; preserve direction and `depth_limit` |
| What is the least-cost path distance? | `sssp` | Requires a weighted graph; negative weight cycles unsupported |
| Which groups are connected? | weak/strong components | Strong and weak semantics differ; inspect current graph-type restrictions |
| Which nodes are structurally influential? | degree, PageRank, Katz, betweenness, eigenvector | Each defines importance differently; do not blend scores without a model |
| Which communities appear? | Louvain, Leiden, ECG, spectral methods | Algorithm/parameters define the partition; Louvain currently requires undirected input |
| Which pairs have similar neighborhoods? | Jaccard, overlap, Sørensen, cosine | Candidate generation and graph restrictions matter; similarity is not a future-link probability |
| Which local dense structure exists? | core, triangle, k-truss | Self-loop and parallel-edge restrictions vary |
| Which samples feed graph learning? | neighbor sampling, random walks | Sampling is not complete enumeration |

Use BFS for unweighted shortest-hop distance. In this source tree, `sssp` rejects an unweighted `Graph`. The `shortest_path` alias is deprecated as of 25.10; call `sssp` for the weighted high-level path.

Scope all-pairs or path-enumeration work before materializing it. If the user only needs reachable endpoints, do not enumerate every path. A predecessor tree can explain one witness path; it is not the set of all simple paths.

## 5. Validate construction and results

### Construction checks

- Compare input rows with materialized graph edges, accounting explicitly for symmetrization, duplicate removal, aggregation, and self-loop policy.
- Compare the required vertex universe with `G.nodes()` and `G.number_of_vertices()`.
- Check `G.is_directed()`, `G.is_weighted()`, and the chosen simple/multigraph type.
- Verify source, destination, and weight dtypes and restore stable external keys in every output.
- Test a forward chain, reversed chain, isolate, duplicate edge, and self-loop when those cases affect semantics.

### Algorithm-specific checks

- **BFS/SSSP:** source distance is zero; unreachable values are filtered with `cugraph.filter_unreachable` rather than guessed sentinels; each predecessor step is a valid edge in the intended direction; recomputed path weight matches distance.
- **Components:** every required vertex appears once; accepted edge endpoints obey the partition invariant; compare partitions independent of arbitrary component label numbers. A directed `Graph` is invalid for `weakly_connected_components`; construct an explicit undirected view. Do not pass `directed`, `connection`, or `return_labels` when the input is already a `Graph`.
- **PageRank:** inspect convergence behavior, parameters, result coverage, finiteness, and score mass within numerical tolerance. If `fail_on_nonconvergence=False`, unpack `(pagerank, converged)` and reject or visibly qualify non-convergence. `store_transposed=True` avoids rebuilding the preferred representation. Do not use latency, distance, or cost as transition strength merely because it is numeric.
- **Communities:** record algorithm, resolution, stopping parameters, and modularity/quality output. Compare memberships up to label permutation and assess stability where decisions depend on a partition.
- **Centrality/link scores:** validate on hand-computable motifs and report normalization, sampling, direction, and weight use.

See [Python API and validation patterns](references/python-api-patterns.md) for complete examples.

## 6. Interpret without overclaiming

Report the graph snapshot, vertex/edge populations, construction policy, algorithm, parameters, runtime interface, and keyed output. Explain exactly what a score or relationship means.

- Reachability is a possible structural path, not proof of causation, flow, exposure magnitude, or operational impact.
- High centrality is importance under one graph definition, not universal business importance.
- A community is an algorithm-dependent partition, not a true class label.
- Link similarity is descriptive evidence unless a separately evaluated predictive model turns it into a forecast.
- Missing edges can mean no relationship, missing observation, filtering, or unavailable data; state which interpretation is justified.

## Scaling and performance

Keep edge preparation and graph input on GPU with cuDF. Estimate memory from vertices, edges, attributes, graph representation, and algorithm workspaces—not only input file bytes. Benchmark after graph construction and transfers are defined; state whether timings are algorithm-only or end-to-end.

Use distributed cuGraph only when the selected algorithm exists in `cugraph.dask` and the graph or workload warrants it. Validate partitioning and result parity on a smaller graph first. Do not describe multi-GPU dispatch as concurrent speedup without measurement.

## Examples

- For directed hop reachability with stable string IDs and isolates, use the BFS construction and predecessor checks in [Python API and validation patterns](references/python-api-patterns.md#bfs-reachability).
- For weighted routes, first verify that the edge value is additive cost; do not pass a larger-is-better affinity directly to SSSP.
- For repeated event edges, compare a simple graph, multigraph, and explicit aggregation against the declared multiplicity contract.

## Limitations

- Algorithm support varies by release, interface, graph type, direction, weights, and distributed mode; this guide does not replace the installed algorithm matrix.
- The high-level API does not make every projection or edge aggregation semantically valid and does not infer missing business relationships.
- Graph algorithms establish structural properties, not causation, future outcomes, or feasible actions.
- GPU memory requirements include graph representations and algorithm workspaces; fitting the raw edge file in memory is not a sufficient capacity test.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Reverse reachability | Source/destination roles were inverted | Validate a miniature directed chain and rebuild |
| Isolates missing | Vertices came only from edge endpoints | Supply the complete vertex universe on a supported construction path |
| Edge count changed | Undirected symmetrization or simple-graph duplicate removal | Compare raw and materialized edge sets; choose policy intentionally |
| SSSP rejects input | Graph is unweighted | Use BFS for hop distance or supply a meaningful numeric weight |
| Louvain rejects graph | Input is directed | Use an appropriate directed method or justify an undirected projection |
| Labels differ across runs/implementations | Component/community IDs are arbitrary | Compare vertex sets per partition, not literal label integers |
| NetworkX code ran on CPU | Unsupported or unconfigured backend dispatch | Inspect backend configuration/dispatch and installed support |

## Deliverable

Return the graph definition, construction code, algorithm choice, parameters, keyed result, validation evidence, runtime/version information, and interpretation limits. Preserve failed convergence or unsupported-operation evidence rather than silently substituting another algorithm.

## References

- [Python API and validation patterns](references/python-api-patterns.md)
- [cuGraph algorithm matrix](https://docs.rapids.ai/api/cugraph/stable/graph_support/algorithms/)
- [cuGraph Python API](https://docs.rapids.ai/api/cugraph/stable/api_docs/cugraph/)
- [nx-cugraph](https://rapids.ai/nx-cugraph/)
