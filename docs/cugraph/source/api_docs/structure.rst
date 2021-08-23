=========
Structure
=========
.. currentmodule:: cugraph.structure.graph_classes

Constructor
-----------
.. autosummary::
   :toctree: api/

   Graph


Methods
-------
.. autosummary::
   :toctree: api/

   Graph.add_internal_vertex_id
   Graph.add_nodes_from
   Graph.clear
   Graph.from_cudf_adjlist
   Graph.from_cudf_edgelist
   Graph.from_dask_cudf_edgelist
   Graph.from_numpy_array
   Graph.from_numpy_matrix
   Graph.from_pandas_adjacency
   Graph.from_pandas_edgelist
   Graph.has_isolated_vertices
   Graph.is_bipartite
   Graph.is_directed
   Graph.is_multigraph
   Graph.is_multipartite
   Graph.is_renumbered
   Graph.is_weighted
   Graph.lookup_internal_vertex_id
   Graph.to_directed
   Graph.to_undirected
   Graph.unrenumber

.. currentmodule:: cugraph

Symmetrize
----------
.. autosummary::
   :toctree: api/

   cugraph.structure.symmetrize.symmetrize
   cugraph.structure.symmetrize.symmetrize_ddf
   cugraph.structure.symmetrize.symmetrize_df


Conversion from Other Formats
-----------------------------
.. autosummary::
   :toctree: api/

   cugraph.structure.convert_matrix.from_adjlist
   cugraph.structure.convert_matrix.from_cudf_edgelist
   cugraph.structure.convert_matrix.from_edgelist
   cugraph.structure.convert_matrix.from_numpy_array
   cugraph.structure.convert_matrix.from_numpy_matrix
   cugraph.structure.convert_matrix.from_pandas_adjacency
   cugraph.structure.convert_matrix.from_pandas_edgelist
   cugraph.structure.convert_matrix.to_numpy_array
   cugraph.structure.convert_matrix.to_numpy_matrix
   cugraph.structure.convert_matrix.to_pandas_adjacency
   cugraph.structure.convert_matrix.to_pandas_edgelist