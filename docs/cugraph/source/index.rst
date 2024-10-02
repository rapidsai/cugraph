RAPIDS Graph documentation
==========================

.. image:: images/cugraph_logo_2.png
   :width: 600


~~~~~~~~~~~~
Introduction
~~~~~~~~~~~~
cuGraph is a library of graph algorithms that seamlessly integrates into the
RAPIDS data science ecosystem and allows the data scientist to easily call
graph algorithms using data stored in GPU DataFrames, NetworkX Graphs, or even
CuPy or SciPy sparse Matrices. Our major integration effort with NetworkX
allows for **zero code change** GPU acceleration through the use of the
nx-cugraph backend. NetworkX and the nx-cugraph backend offer a seamless
transition to GPU accelerated graph analytics for NetworkX users with access to
a supported GPU.

Getting started with cuGraph

Required hardware/software for cuGraph and `RAPIDS <https://docs.rapids.ai/user-guide>`_
 * NVIDIA GPU, Volta architecture or later, with `compute capability <https://developer.nvidia.com/cuda-gpus> 7.0+`_
 * CUDA 11.2-11.8, 12.0-12.5
 * Python version 3.10, 3.11, or 3.12
 * NetworkX version 3.0 or newer in order to use use the nx-cuGraph backend. NetworkX version 3.4 or newer is recommended. (`see below <#cugraph-using-networkx-code>`).

Installation
The latest RAPIDS System Requirements documentation is located `here <https://docs.rapids.ai/install#system-req>`_.

This includes several ways to set up cuGraph

* From Unix

  * `Conda <https://docs.rapids.ai/install/#conda>`_
  * `Docker <https://docs.rapids.ai/install/#docker>`_
  * `pip <https://docs.rapids.ai/install/#pip>`_


**Note: Windows use of RAPIDS depends on prior installation of** `WSL2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

* From Windows

  * `Conda <https://docs.rapids.ai/install#wsl-conda>`_
  * `Docker <https://docs.rapids.ai/install#wsl-docker>`_
  * `pip <https://docs.rapids.ai/install#wsl-pip>`_


cuGraph Using NetworkX Code

cuGraph is now available as a NetworkX backend using `nx-cugraph <https://rapids.ai/nx-cugraph/>`_.
nx-cugraph offers NetworkX users a **zero code change** option to accelerate
their existing NetworkX code using an NVIDIA GPU and cuGraph.


 Cugraph API Example

 .. code-block:: python

  import cugraph
  import cudf

  # Create an instance of the popular Zachary Karate Club graph
  from cugraph.datasets import karate
  G = karate.get_graph()

  # Call cugraph.degree_centrality
  vertex_bc = cugraph.degree_centrality(G)

There are several resources containing cuGraph examples, `the cuGraph notebook repository <https://github.com/rapidsai/cugraph/blob/main/notebooks/README.md>`_
has many examples of loading graph data and running algorithms in Jupyter notebooks.
The `cuGraph test code <https://github.com/rapidsai/cugraph/tree/main/python/cugraph/cugraph/tests>_` contain python scripts setting up and calling cuGraph algorithms.
A simple example of `testing the degree centrality algorithm <https://github.com/rapidsai/cugraph/blob/main/python/cugraph/cugraph/tests/centrality/test_degree_centrality.py>`_
is a good place to start. Some of these show `multi-GPU tests/examples <https://github.com/rapidsai/cugraph/blob/main/python/cugraph/cugraph/tests/centrality/test_degree_centrality_mg.py>`_ with larger data sets as well.

.. toctree::
   :maxdepth: 2

   top_toc

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
