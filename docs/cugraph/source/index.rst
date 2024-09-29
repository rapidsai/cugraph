RAPIDS Graph documentation
==========================

.. image:: images/cugraph_logo_2.png
   :width: 600


~~~~~~~~~~~~
Introduction
~~~~~~~~~~~~
cuGraph is a library of graph algorithms that seamlessly integrates into the
RAPIDS data science ecosystem and allows the data scientist to easily call
graph algorithms using data stored in GPU DataFrames, NetworkX Graphs, or
even CuPy or SciPy sparse Matrices. A major integration effort with NetworkX is focused on
**zero code change** use of nx-cuGraph as a backend for NetworkX calls. This allows a near seamless
transition to GPU accelerated graph analytics.

Getting started with cuGraph

Required hardware/software for cuGraph and `RAPIDS <https://docs.rapids.ai/user-guide>`_
 * NVIDIA GPU, Volta architecture or later, with `compute capability <https://developer.nvidia.com/cuda-gpus> 7.0+`_
 * CUDA 11.4-11.8, 12.0-12.5
 * Python version 3.10, 3.11, or 3.12
 * NetworkX version 3.0 or newer in order to use use the nx-cuGraph backend. Version 3.3 is required to use `NetworkX Configs <https://networkx.org/documentation/stable/reference/backends.html#module-networkx.utils.configs>`_ `see below <#cugraph-using-networkx-code>`_.

Installation
The latest RAPIDS System Requirements documentation is located `here <https://docs.rapids.ai/install#system-req>`_.

This includes several ways to set up cuGraph

* From Unix

  * `Conda <https://docs.rapids.ai/install/#conda>`_
  * `Docker <https://docs.rapids.ai/install/#docker>`_
  * `pip <https://docs.rapids.ai/install/#pip>`_


**Note: Windows use of RAPIDS depends on prior installation of** `WSL2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

* From windows

  * `Conda <https://docs.rapids.ai/install#wsl-conda>`_
  * `Docker <https://docs.rapids.ai/install#wsl-docker>`_
  * `pip <https://docs.rapids.ai/install#wsl-pip>`_




Build From Source

To build from source, check each RAPIDS GitHub README for set up and build instructions. Further links are provided in the `selector tool <https://docs.rapids.ai/install#selector>`_. If additional help is needed reach out on our `Slack Channel <https://rapids-goai.slack.com/archives/C5E06F4DC>`_.

CuGraph Using NetworkX Code
While the steps above are required to use the full suite of cuGraph graph analytics, cuGraph is now supported as a NetworkX backend using `nx-cugraph <https://docs.rapids.ai/api/cugraph/nightly/nx_cugraph/nx_cugraph/>_.
Nx-cugraph offers those with existing NetworkX code, a **zero code change** option with a growing list of supported algorithms.


 Cugraph API Example
Coming soon !


Until then, `the cuGraph notebook repository <https://github.com/rapidsai/cugraph/blob/main/notebooks/README.md>`_ 
has many examples of loading graph data and running algorithms in Jupyter notebooks.
The `cuGraph test code <https://github.com/rapidsai/cugraph/tree/main/python/cugraph/cugraph/tests>_` contain python scripts setting up and calling cuGraph algorithms. 
A simple example of `testing the degree centrality algorithm <https://github.com/rapidsai/cugraph/blob/main/python/cugraph/cugraph/tests/centrality/test_degree_centrality.py>`_ 
is a good place to start. Some of these show `multi-GPU tests/examples <https://github.com/rapidsai/cugraph/blob/main/python/cugraph/cugraph/tests/centrality/test_degree_centrality_mg.py>`_ with larger data sets  as well.

.. toctree::
  :maxdepth: 1
  :caption: cuGraph documentation Contents:

  basics/index
  nx_cugraph/index
  installation/index
  tutorials/index
  graph_support/index
  wholegraph/index
  references/index
  api_docs/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
