### nx_cugraph


Whereas previous versions of cuGraph have included mechanisms to make it
trivial to plug in cuGraph algorithm calls. Beginning with version 24.02, nx-cuGraph
is now a [networkX backend](<https://networkx.org/documentation/stable/reference/utils.html#backends>).
The user now need only [install nx-cugraph](<https://github.com/rapidsai/cugraph/blob/branch-24.04/python/nx-cugraph/README.md#install>)
to experience GPU speedups.

Lets look at some examples of algorithm speedups comparing CPU based NetworkX to dispatched versions run on GPU with nx_cugraph.

Each chart has three measurements.
* NX - running the algorithm natively with networkX on CPU.
* nx-cugraph - running with GPU accelerated networkX achieved by simply calling the cugraph backend. This pays the overhead of building the GPU resident object for each algorithm called. This achieves significant improvement but stil isn't compleltely optimum.
* nx-cugraph (preconvert) - This is a bit more complicated since it involves building (precomputing) the GPU resident graph ahead and reusing it for each algorithm.


![Ancestors](../images/ancestors.png)
![BFS Tree](../images/bfs_tree.png)
![Connected Components](../images/conn_component.png)
![Descendents](../images/descendents.png)
![Katz](../images/katz.png)
![Pagerank](../images/pagerank.png)
![Single Source Shortest Path](../images/sssp.png)
![Weakly Connected Components](../images/wcc.png)


The following algorithms are supported and automatically dispatched to nx-cuGraph for acceleration.

#### Algorithms
```
bipartite
 ├─ basic
 │   └─ is_bipartite
 └─ generators
     └─ complete_bipartite_graph
centrality
 ├─ betweenness
 │   ├─ betweenness_centrality
 │   └─ edge_betweenness_centrality
 ├─ degree_alg
 │   ├─ degree_centrality
 │   ├─ in_degree_centrality
 │   └─ out_degree_centrality
 ├─ eigenvector
 │   └─ eigenvector_centrality
 └─ katz
     └─ katz_centrality
cluster
 ├─ average_clustering
 ├─ clustering
 ├─ transitivity
 └─ triangles
community
 └─ louvain
     └─ louvain_communities
components
 ├─ connected
 │   ├─ connected_components
 │   ├─ is_connected
 │   ├─ node_connected_component
 │   └─ number_connected_components
 └─ weakly_connected
     ├─ is_weakly_connected
     ├─ number_weakly_connected_components
     └─ weakly_connected_components
core
 ├─ core_number
 └─ k_truss
dag
 ├─ ancestors
 └─ descendants
isolate
 ├─ is_isolate
 ├─ isolates
 └─ number_of_isolates
link_analysis
 ├─ hits_alg
 │   └─ hits
 └─ pagerank_alg
     └─ pagerank
operators
 └─ unary
     ├─ complement
     └─ reverse
reciprocity
 ├─ overall_reciprocity
 └─ reciprocity
shortest_paths
 └─ unweighted
     ├─ single_source_shortest_path_length
     └─ single_target_shortest_path_length
traversal
 └─ breadth_first_search
     ├─ bfs_edges
     ├─ bfs_layers
     ├─ bfs_predecessors
     ├─ bfs_successors
     ├─ bfs_tree
     ├─ descendants_at_distance
     └─ generic_bfs_edges
tree
 └─ recognition
     ├─ is_arborescence
     ├─ is_branching
     ├─ is_forest
     └─ is_tree
```

#### Generators
```
classic
 ├─ barbell_graph
 ├─ circular_ladder_graph
 ├─ complete_graph
 ├─ complete_multipartite_graph
 ├─ cycle_graph
 ├─ empty_graph
 ├─ ladder_graph
 ├─ lollipop_graph
 ├─ null_graph
 ├─ path_graph
 ├─ star_graph
 ├─ tadpole_graph
 ├─ trivial_graph
 ├─ turan_graph
 └─ wheel_graph
community
 └─ caveman_graph
small
 ├─ bull_graph
 ├─ chvatal_graph
 ├─ cubical_graph
 ├─ desargues_graph
 ├─ diamond_graph
 ├─ dodecahedral_graph
 ├─ frucht_graph
 ├─ heawood_graph
 ├─ house_graph
 ├─ house_x_graph
 ├─ icosahedral_graph
 ├─ krackhardt_kite_graph
 ├─ moebius_kantor_graph
 ├─ octahedral_graph
 ├─ pappus_graph
 ├─ petersen_graph
 ├─ sedgewick_maze_graph
 ├─ tetrahedral_graph
 ├─ truncated_cube_graph
 ├─ truncated_tetrahedron_graph
 └─ tutte_graph
social
 ├─ davis_southern_women_graph
 ├─ florentine_families_graph
 ├─ karate_club_graph
 └─ les_miserables_graph
```

#### Other

```
convert_matrix
 ├─ from_pandas_edgelist
 └─ from_scipy_sparse_array
```
