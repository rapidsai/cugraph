import pytest
import cugraph.tests.utils as utils
import numpy as np


from cugraph.tests.dask.mg_context import (MGContext,
                                           skip_if_not_enough_devices)

# Get parameters from standard betwenness_centrality_test
from cugraph.tests.test_edge_betweenness_centrality import (
    DIRECTED_GRAPH_OPTIONS,
    NORMALIZED_OPTIONS,
    DEFAULT_EPSILON,
    SUBSET_SIZE_OPTIONS,
    SUBSET_SEED_OPTIONS,
)

from cugraph.tests.test_edge_betweenness_centrality import (
    prepare_test,
    calc_edge_betweenness_centrality,
    compare_scores
)

# =============================================================================
# Parameters
# =============================================================================
DATASETS = ['../datasets/karate.csv']
MG_DEVICE_COUNT_OPTIONS = [1, 2, 4]
RESULT_DTYPE_OPTIONS = [np.float64]


@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('weight', [None])
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
@pytest.mark.parametrize('mg_device_count', MG_DEVICE_COUNT_OPTIONS)
def test_mg_edge_betweenness_centrality(graph_file,
                                        directed,
                                        subset_size,
                                        normalized,
                                        weight,
                                        subset_seed,
                                        result_dtype,
                                        mg_device_count):
    prepare_test()
    skip_if_not_enough_devices(mg_device_count)
    with MGContext(mg_device_count):
        sorted_df = calc_edge_betweenness_centrality(graph_file,
                                                     directed=directed,
                                                     normalized=normalized,
                                                     k=subset_size,
                                                     weight=weight,
                                                     seed=subset_seed,
                                                     result_dtype=result_dtype,
                                                     multi_gpu_batch=True)
    compare_scores(sorted_df, first_key="cu_bc", second_key="ref_bc",
                   epsilon=DEFAULT_EPSILON)
