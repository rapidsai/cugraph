**************************************
NetworkX by calling cuGraph Algorithms
**************************************


*Note: this is a work in progress and will be updatred and changed as we better flesh out
compatibility issues*

Latest Update
#############

Last Update:	March 7th, 2024
Release:		24.04

**CuGraph is now a registered backend for networkX. This is described in the following blog:
`Accelerating NetworkX on NVIDIA GPUs for High Performance Graph Analytics
<https://developer.nvidia.com/blog/accelerating-networkx-on-nvidia-gpus-for-high-performance-graph-analytics/>`_


Easy Path – Use NetworkX Graph Objects, Accelerated Algorithms
##############################################################

Rather than updating all of your existing code, simply update the calls to
graph algorithms by replacing the module name.  This allows all the complicated
ETL code to be unchanged while still seeing significate performance
improvements. Again this will be deprecated since networkX dispatching to nx_cugraph
has many advantages.

improvements.


.. image:: ../images/Nx_Cg_1.png
  :width: 600

It is that easy.  All algorithms in cuGraph support a NetworkX graph object as
input and match the NetworkX API list of arguments.

Currently, cuGraph accepts both NetworkX Graph and DiGraph objects. We will be
adding support for Bipartite graph and Multigraph over the next few releases.

Differences in Algorithms
##########################

Since cuGraph currently does not support attribute rich graphs, those
algorithms that return simple scores (centrality, clustering, etc.) best match
the NetworkX process.  Algorithms that return a subgraph will do so without
any additional attributes on the nodes or edges.

Algorithms that exactly match
*****************************

+-------------------------------+------------------------+
|       Algorithm               |     Differences        |
+===============================+========================+
| Core Number                   | None                   |
+-------------------------------+------------------------+
| HITS 	                        | None                   |
+-------------------------------+------------------------+
| PageRank                      | None                   |
+-------------------------------+------------------------+
| Personal PageRank             | None                   |
+-------------------------------+------------------------+
| Strongly Connected Components | None                   |
+-------------------------------+------------------------+
| Weakly Connected Components   | None                   |
+-------------------------------+------------------------+

|



Algorithms that do not copy over additional attributes
************************************************************************

+-------------------------------+-------------------------------------+
|       Algorithm               |     Differences                     |
+===============================+=====================================+
| K-Truss                       | Does not copy over attributes       |
+-------------------------------+-------------------------------------+
| K-Core                        | Does not copy over attributes       |
+-------------------------------+-------------------------------------+
| Subgraph Extraction           | Does not copy over attributes       |
+-------------------------------+-------------------------------------+

|


Algorithms not in NetworkX
**************************

+--------------------------------------+----------------------------+
|       Algorithm                      |     Differences            |
+======================================+============================+
| Ensemble Clustering for Graphs (ECG) | Currently not in NetworkX  |
+--------------------------------------+----------------------------+
| Force Atlas 2                        | Currently not in NetworkX  |
+--------------------------------------+----------------------------+
| Leiden                               | Currently not in NetworkX  |
+--------------------------------------+----------------------------+
| Louvain                              | Currently not in NetworkX  |
+--------------------------------------+----------------------------+
| Overlap coefficient                  | Currently not in NetworkX  |
+--------------------------------------+----------------------------+
| Spectral Clustering                  | Currently not in NetworkX  |
+--------------------------------------+----------------------------+

|


Algorithm where not all arguments are supported
***********************************************

+----------------------------+-------------------------------------------------+
|       Algorithm            |     Differences                                 |
+============================+=================================================+
|Betweenness Centrality      | weight is currently not supported – ignored     |
|                            | endpoints is currently not supported – ignored  |
+----------------------------+-------------------------------------------------+
|Edge Betweenness Centrality | weight is currently not supported – ignored     |
+----------------------------+-------------------------------------------------+
| Katz Centrality            | beta is currently not supported – ignored       |
|                            | max_iter defaults to 100 versus 1000            |
+----------------------------+-------------------------------------------------+

|

Algorithms where the results are different
******************************************


For example, the NetworkX traversal algorithms typically return a generator
rather than a dictionary.


+----------------------------+-------------------------------------------------+
|       Algorithm            |     Differences                                 |
+============================+=================================================+
| Triangle Counting          | this algorithm simply returns the total number  |
|                            | of triangle and not the number per vertex       |
|                            | (on roadmap to update)                          |
+----------------------------+-------------------------------------------------+
| Jaccard coefficient        | Currently we only do a 1-hop computation rather |
|                            | than an all-pairs.  Fix is on roadmap           |
+----------------------------+-------------------------------------------------+
| Breadth First Search (BFS) | Returns a Pandas DataFrame with:                |
|                            | [vertex][distance][predecessor]                 |
+----------------------------+-------------------------------------------------+
| Single Source              | Returns a Pandas DataFrame with:                |
| Shortest Path (SSSP)       | [vertex][distance][predecessor]                 |
+----------------------------+-------------------------------------------------+

|

Graph Building
##############

The biggest difference between NetworkX and cuGraph is with how Graph objects
are built. NetworkX, for the most part, stores graph data in a dictionary.
That structure allows easy insertion of new records. Consider the following
code for building a NetworkX Graph::

    # Read the node data
    df = pd.read_csv( data_file)

    # Construct graph from edge list.
    G = nx.DiGraph()

    for row in df.iterrows():
        G.add_edge(
            row[1]["1"], row[1]["2"], count=row[1]["3"]
        )


The code block is perfectly fine for NetworkX. However, the process of iterating over the dataframe and adding one node at a time is problematic for GPUs and something that we try and avoid.  cuGraph stores data in columns (i.e. arrays).  Resizing an array requires allocating a new array one element larger, copying the data, and adding the new value.  That is not very efficient.

If your code follows the above model of inserting one element at a time, the we suggest either rewriting that code or using it as is within NetworkX and just accelerating the algorithms with cuGraph.

Now, if your code bulk loads the data from Pandas, then RAPIDS can accelerate that process by orders of magnitude.

.. image:: ../images/Nx_Cg_2.png
  :width: 600

The above cuGraph code will create cuGraph.Graph object and not a NetworkX.Graph object.
