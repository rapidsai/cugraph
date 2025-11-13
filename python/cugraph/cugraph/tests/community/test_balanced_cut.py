# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cugraph
from cugraph.datasets import karate


@pytest.mark.sg
def test_spectral_balanced_cut_clustering_deprecation_warning():
    """Test that spectralBalancedCutClustering emits a deprecation warning.
    
    Note: spectralBalancedCutClustering is deprecated in favor of
    spectralModularityMaximizationClustering. Functional tests for spectral
    clustering (including edge cut validation) are in test_modularity.py.
    """
    G = karate.get_graph(
        create_using=cugraph.Graph(directed=False), ignore_weights=True
    )
    warning_msg = (
        "spectralBalancedCutClustering is deprecated and will be removed in a future "
        "release. Use spectralModularityMaximizationClustering instead."
    )

    with pytest.warns(DeprecationWarning, match=warning_msg):
        cugraph.spectralBalancedCutClustering(G, num_clusters=2)
