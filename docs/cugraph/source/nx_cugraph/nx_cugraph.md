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

### Command line example
Open bc_demo.ipy and paste the code below.

```
import pandas as pd
import networkx as nx

url = "https://data.rapids.ai/cugraph/datasets/cit-Patents.csv"
df = pd.read_csv(url, sep=" ", names=["src", "dst"], dtype="int32")
G = nx.from_pandas_edgelist(df, source="src", target="dst")

%time result = nx.betweenness_centrality(G, k=10)
```
Run the command:
```
user@machine:/# ipython bc_demo.ipy
```

You will observe a run time of approximately 7 minutes...more or less depending on your cpu.

Run the command again, this time specifiying cugraph as the NetworkX backend of choice.
```
user@machine:/# NETWORKX_BACKEND_PRIORITY=cugraph ipython bc_demo.ipy
```
This run will be much faster, typically around 20 seconds depending on your GPU.
```
user@machine:/# NETWORKX_BACKEND_PRIORITY=cugraph ipython bc_demo.ipy
```
There is also an option to add caching. This will dramatically help performance when running multiple algorithms on the same graph.
```
NETWORKX_BACKEND_PRIORITY=cugraph CACHE_CONVERTED_GRAPH=True ipython bc_demo.ipy
```

When running Python interactively, cugraph backend can be specified as an argument in the algorithm call.

For example:
```
nx.betweenness_centrality(cit_patents_graph, k=k, backend="cugraph")
```


The latest list of algorithms that can be dispatched to nx-cuGraph for acceleration is found [here](https://github.com/rapidsai/cugraph/blob/main/python/nx-cugraph/README.md#algorithms).
