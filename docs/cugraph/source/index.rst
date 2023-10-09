RAPIDS Graph documentation
==========================
*Making graph analytics fast and easy regardless of scale*

RAPIDS Graph covers a range of graph libraries and packages, that includes:


.. list-table:: RAPIDS Graph
   :widths: 25 25 50
   :header-rows: 1

   * - Core
     - GNN
     - Extension
   * - cugraph
     - cugraph-ops
     - cugraph-service
   * - pylibcugraph
     - cugraph-dgl
     - 
   * - libcugraph
     - cugraph-pyg
     -
   * - libcugraph_etl
     - wholegraph
     -



A description of the package are:

* cugraph: GPU-accelerated graph algorithms
* cugraph-ops: GPU-accelerated GNN aggregators and operators
* cugraph-service: multi-user, remote GPU-accelerated graph algorithm service
* cugraph-pyg:  GPU-accelerated extensions for use with the PyG framework
* cugraph-dgl:  GPU-accelerated extensions for use with the DGL framework
* wholegraph: shared memory-based GPU-accelerated GNN training

cuGraph is a library of graph algorithms that seamlessly integrates into the RAPIDS data science ecosystem and allows the data scientist to easily call graph algorithms using data stored in GPU DataFrames, NetworkX Graphs, or even CuPy or SciPy sparse Matrices.

Note: We are redoing all of our documents, please be patient as we update
the docs and links


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   basics/index
   installation/index
   tutorials/index
   graph_support/index
   references/index
   dev_resources/index
   releases/index
   api_docs/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
