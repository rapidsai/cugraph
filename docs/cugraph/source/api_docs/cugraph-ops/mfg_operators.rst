================================
Operators on Message-Flow Graphs
================================

.. currentmodule:: pylibcugraphops

Simple Neighborhood Aggregator (SAGEConv)
-----------------------------------------
.. autosummary::
   :toctree: ../api/ops/

   operators.agg_simple_mfg_n2n_fwd
   operators.agg_simple_mfg_n2n_bwd
   operators.agg_concat_mfg_n2n_fwd
   operators.agg_concat_mfg_n2n_bwd

Graph Attention (GATConv)
-------------------------
.. autosummary::
   :toctree: ../api/ops/

   operators.mha_gat_mfg_n2n_fwd
   operators.mha_gat_mfg_n2n_bwd

Heterogenous Aggregator using Basis Decomposition (RGCNConv)
------------------------------------------------------------
.. autosummary::
   :toctree: ../api/ops/

   operators.agg_hg_basis_mfg_n2n_post_fwd
   operators.agg_hg_basis_mfg_n2n_post_bwd
