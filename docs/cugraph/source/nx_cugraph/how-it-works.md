# How it Works

NetworkX has the ability to **dispatch function calls to separately-installed third-party backends**.

NetworkX backends let users experience improved performance and/or additional functionality without changing their NetworkX Python code. Examples include backends that provide algorithm acceleration using GPUs, parallel processing, graph database integration, and more.

While NetworkX is a pure-Python implementation with minimal to no dependencies, backends may be written in other languages and require specialized hardware and/or OS support, additional software dependencies, or even separate services. Installation instructions vary based on the backend, and additional information can be found from the individual backend project pages.


![nxcg-execution-flow](../_static/nxcg-execution-diagram.jpg)

## Enabling nx-cugraph

NetworkX will use `nx-cugraph` as the backend if any of the following are used:

### `NX_CUGRAPH_AUTOCONFIG` environment variable.

The `NX_CUGRAPH_AUTOCONFIG` environment variable can be used to configure NetworkX for full zero code change acceleration using `nx-cugraph`.  If a NetworkX function is called that `nx-cugraph` supports, NetworkX will redirect the function call to `nx-cugraph` automatically, or fall back to either another backend if enabled or the default NetworkX implementation. See the [NetworkX documentation on backends](https://networkx.org/documentation/stable/reference/backends.html) for configuring NetworkX manually.

```
bash> NX_CUGRAPH_AUTOCONFIG=True python my_networkx_script.py
```

### `backend=` keyword argument

To explicitly specify a particular backend for an API, use the `backend=`
keyword argument. This argument takes precedence over the
`NX_CUGRAPH_AUTOCONFIG` environment variable. This requires anyone
running code that uses the `backend=` keyword argument to have the specified
backend installed.

Example:
```python
nx.betweenness_centrality(cit_patents_graph, k=k, backend="cugraph")
```

### Type-based dispatching

NetworkX also supports automatically dispatching to backends associated with
specific graph types. Like the `backend=` keyword argument example above, this
requires the user to write code for a specific backend, and therefore requires
the backend to be installed, but has the advantage of ensuring a particular
behavior without the potential for runtime conversions.

To use type-based dispatching with `nx-cugraph`, the user must import the backend
directly in their code to access the utilities provided to create a Graph
instance specifically for the `nx-cugraph` backend.

Example:
```python
import networkx as nx
import nx_cugraph as nxcg

G = nx.Graph()
...
nxcg_G = nxcg.from_networkx(G)             # conversion happens once here
nx.betweenness_centrality(nxcg_G, k=1000)  # nxcg Graph type causes cugraph backend
                                           # to be used, no conversion necessary
```

## Command Line Example

---

Create `bc_demo.ipy` and paste the code below.

```python
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

CPU times: user 7min 36s, sys: 5.22 s, total: 7min 41s
Wall time: 7min 41s
```

You will observe a run time of approximately 7 minutes...more or less depending on your CPU.

Run the command again, this time specifying cugraph as the NetworkX backend.
```bash
user@machine:/# NX_CUGRAPH_AUTOCONFIG=True ipython bc_demo.ipy

CPU times: user 4.14 s, sys: 1.13 s, total: 5.27 s
Wall time: 5.32 s
```
This run will be much faster, typically around 4 seconds depending on your GPU.
```bash
user@machine:/# NX_CUGRAPH_AUTOCONFIG=True ipython bc_demo.ipy
```
There is also an option to cache the graph conversion to GPU. This can dramatically improve performance when running multiple algorithms on the same graph. Caching is enabled by default for NetworkX versions 3.4 and later, but if using an older version, set `NETWORKX_CACHE_CONVERTED_GRAPHS=True`.
```
user@machine:/# NX_CUGRAPH_AUTOCONFIG=cugraph NETWORKX_CACHE_CONVERTED_GRAPHS=True ipython bc_demo.ipy
```

When running Python interactively, the cugraph backend can be specified as an argument in the algorithm call.

For example:
```python
nx.betweenness_centrality(cit_patents_graph, k=k, backend="cugraph")
```

*Note, the examples above were run using the following specs*:
```{note}
NetworkX 3.4
nx-cugraph 24.10
CPU: Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz 45GB RAM
GPU: NVIDIA Quadro RTX 8000 80GB RAM
```

The latest list of algorithms supported by `nx-cugraph` can be found in the [cugraph documentation](https://github.com/rapidsai/cugraph/blob/HEAD/python/nx-cugraph/README.md#algorithms), or in the next section.
