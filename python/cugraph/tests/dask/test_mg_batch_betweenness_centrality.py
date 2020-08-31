import pytest
import cugraph.tests.utils as utils
import numpy as np

from cugraph.tests.dask.mg_context import (MGContext,
                                           skip_if_not_enough_devices)

# Get parameters from standard betwenness_centrality_test
from cugraph.tests.test_betweenness_centrality import (
    DIRECTED_GRAPH_OPTIONS,
    ENDPOINTS_OPTIONS,
    NORMALIZED_OPTIONS,
    DEFAULT_EPSILON,
    SUBSET_SIZE_OPTIONS,
    SUBSET_SEED_OPTIONS,
)

from cugraph.tests.test_betweenness_centrality import (
    prepare_test,
    calc_betweenness_centrality,
    compare_scores
)

# =============================================================================
# Parameters
# =============================================================================
DATASETS = utils.DATASETS_1
MG_DEVICE_COUNT_OPTIONS = [1, 2, 3, 4]
RESULT_DTYPE_OPTIONS = [np.float64]


# FIXME: The following creates and destroys Comms at every call making the
# testsuite quite slow
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('weight', [None])
@pytest.mark.parametrize('endpoints', ENDPOINTS_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
@pytest.mark.parametrize('mg_device_count', MG_DEVICE_COUNT_OPTIONS)
def test_mg_betweenness_centrality(graph_file,
                                   directed,
                                   subset_size,
                                   normalized,
                                   weight,
                                   endpoints,
                                   subset_seed,
                                   result_dtype,
                                   mg_device_count):
    prepare_test()
    skip_if_not_enough_devices(mg_device_count)
    with MGContext(mg_device_count):
        sorted_df = calc_betweenness_centrality(graph_file,
                                                directed=directed,
                                                normalized=normalized,
                                                k=subset_size,
                                                weight=weight,
                                                endpoints=endpoints,
                                                seed=subset_seed,
                                                result_dtype=result_dtype,
                                                multi_gpu_batch=True)
    compare_scores(sorted_df, first_key="cu_bc", second_key="ref_bc",
                   epsilon=DEFAULT_EPSILON)
