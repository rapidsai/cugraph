### nx_cugraph


nx-cugraph is a [NetworkX
backend](<https://networkx.org/documentation/stable/reference/utils.html#backends>) that provides GPU acceleration to many popular NetworkX algorithms.

By simply [installing and enabling nx-cugraph](<https://github.com/rapidsai/cugraph/blob/HEAD/python/nx-cugraph/README.md#install>), users can see significant speedup on workflows where performance is hindered by the default NetworkX implementation.  With nx-cugraph, users can have GPU-based, large-scale performance without changing their familiar and easy-to-use NetworkX code.

Let's look at some examples of algorithm speedups comparing NetworkX with and without GPU acceleration using nx-cugraph.

Each chart has three measurements.
* NX - default NetworkX, no GPU acceleration
* nx-cugraph - GPU-accelerated NetworkX using nx-cugraph. This involves an internal conversion/transfer of graph data from CPU to GPU memory
* nx-cugraph (preconvert) - GPU-accelerated NetworkX using nx-cugraph with the graph data pre-converted/transferred to GPU


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

Run the command again, this time specifying cugraph as the NetworkX backend.
```
user@machine:/# NETWORKX_BACKEND_PRIORITY=cugraph ipython bc_demo.ipy
```
This run will be much faster, typically around 20 seconds depending on your GPU.
```
user@machine:/# NETWORKX_BACKEND_PRIORITY=cugraph ipython bc_demo.ipy
```
There is also an option to cache the graph conversion to GPU. This can dramatically improve performance when running multiple algorithms on the same graph.
```
NETWORKX_BACKEND_PRIORITY=cugraph NETWORKX_CACHE_CONVERTED_GRAPHS=True ipython bc_demo.ipy
```

When running Python interactively, the cugraph backend can be specified as an argument in the algorithm call.

For example:
```
nx.betweenness_centrality(cit_patents_graph, k=k, backend="cugraph")
```


The latest list of algorithms supported by nx-cugraph can be found [here](https://github.com/rapidsai/cugraph/blob/main/python/nx-cugraph/README.md#algorithms).
