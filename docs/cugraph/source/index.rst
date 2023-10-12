RAPIDS Graph documentation
==========================
.. image:: images/cugraph_logo_2.png
   :width: 600

*Making graph analytics fast and easy regardless of scale*


.. list-table:: RAPIDS Graph covers a range of graph libraries and packages, that includes:
   :widths: 25 25 25
   :header-rows: 1

   * - Core
     - GNN
     - Extension
   * - :abbr:`cugraph (Python wrapper with lots of convenience functions)`
     - :abbr:`cugraph-ops (GNN aggregators and operators)`
     - :abbr:`cugraph-service (Graph-as-a-service provides both Client and Server packages)`
   * - :abbr:`pylibcugraph (light-weight Python wrapper with no guard rails)`
     - :abbr:`cugraph-dgl (Accelerated extensions for use with the DGL framework)`
     - 
   * - :abbr:`libcugraph (C++ API)`
     - :abbr:`cugraph-pyg (Accelerated extensions for use with the PyG framework)`
     -
   * - :abbr:`libcugraph_etl (C++ renumbering function for strings)`
     - :abbr:`wholegraph (Shared memory-based GPU-accelerated GNN training)`
     -

..

|
|

cuGraph is a library of graph algorithms that seamlessly integrates into the
RAPIDS data science ecosystem and allows the data scientist to easily call
graph algorithms using data stored in GPU DataFrames, NetworkX Graphs, or 
even CuPy or SciPy sparse Matrices.

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
