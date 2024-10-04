nx-cugraph
-----------

``nx-cugraph`` is a NetworkX backend that provides **GPU acceleration** to many popular NetworkX algorithms.

By simply `installing and enabling nx-cugraph <https://github.com/rapidsai/cugraph/blob/HEAD/python/nx-cugraph/README.md#install>`_, users can see significant speedup on workflows where performance is hindered by the default NetworkX implementation. 

Users can have GPU-based, large-scale performance **without** changing their familiar and easy-to-use NetworkX code.

.. centered:: Timed result from running the following code snippet (called ``demo.ipy``, showing NetworkX with vs. without ``nx-cugraph``)

.. code-block:: python

    import pandas as pd
    import networkx as nx

    url = "https://data.rapids.ai/cugraph/datasets/cit-Patents.csv"
    df = pd.read_csv(url, sep=" ", names=["src", "dst"], dtype="int32")
    G = nx.from_pandas_edgelist(df, source="src", target="dst")

    %time result = nx.betweenness_centrality(G, k=10)


::

    user@machine:/# ipython demo.ipy
    CPU times: user 7min 36s, sys: 5.22 s, total: 7min 41s
    Wall time: 7min 41s


::

    user@machine:/# NX_CUGRAPH_AUTOCONFIG=True ipython demo.ipy
    CPU times: user 4.14 s, sys: 1.13 s, total: 5.27 s
    Wall time: 5.32 s


.. figure:: ../_static/colab.png
    :width: 200px
    :target: https://nvda.ws/4drM4re

    Try it on Google Colab!


+--------------------------------------------------------------------------------------------------------+
| **Zero Code Change Acceleration**                                                                      |
|                                                                                                        |
| Just set the environment variable ``NX_CUGRAPH_AUTOCONFIG=True`` to enable ``nx-cugraph`` in NetworkX. |
+--------------------------------------------------------------------------------------------------------+
| **Run the same code on CPU or GPU**                                                                    |
|                                                                                                        |
| Nothing changes, not even your `import` statements, when going from CPU to GPU.                        |
+--------------------------------------------------------------------------------------------------------+


``nx-cugraph`` is now Generally Available (GA) as part of the ``RAPIDS`` package.  See `RAPIDS
Quick Start <https://rapids.ai/#quick-start>`_ to get up-and-running with ``nx-cugraph``.

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    how-it-works
    installation
    supported-algorithms
    benchmarks
