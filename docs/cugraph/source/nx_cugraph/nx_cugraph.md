### nx_cugraph


Whereas previous versions of cuGraph have included mechanisms to make it
trivial to plug in cuGraph algorithm calls. Beginning with version 24.02, nx-cuGraph
is now a [networkX backend](<https://networkx.org/documentation/stable/reference/utils.html#backends>).
The user now need only [install nx-cugraph](<https://github.com/rapidsai/cugraph/blob/branch-24.08/python/nx-cugraph/README.md#install>)
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
