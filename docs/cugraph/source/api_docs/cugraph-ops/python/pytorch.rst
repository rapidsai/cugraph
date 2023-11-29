==========================
PyTorch Autograd Wrappers
==========================

.. currentmodule:: pylibcugraphops.pytorch

Simple Neighborhood Aggregator (SAGEConv)
-----------------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.agg_concat_n2n

Graph Attention (GATConv/GATv2Conv)
-----------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.mha_gat_n2n
   operators.mha_gat_v2_n2n

Heterogenous Aggregator using Basis Decomposition (RGCNConv)
------------------------------------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.agg_hg_basis_n2n_post


Update Edges: Concatenation or Sum of Edge and Node Features
------------------------------------------------------------
.. autosummary::
   :toctree: ../../api/ops

   operators.update_efeat_e2e
   operators.update_efeat_e2e
