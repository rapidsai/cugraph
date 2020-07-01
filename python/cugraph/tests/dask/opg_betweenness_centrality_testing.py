import pytest

from cugraph.dask.core import get_visible_devices

# Get parameters from standard betwenness_centrality_test
from cugraph.tests.test_betweenness_centrality import (
    DIRECTED_GRAPH_OPTIONS,
    ENDPOINTS_OPTIONS,
    NORMALIZED_OPTIONS,
    DEFAULT_EPSILON,
    DATASETS,
    UNRENUMBERED_DATASETS,
    SUBSET_SIZE_OPTIONS,
    SUBSET_SEED_OPTIONS,
    RESULT_DTYPE_OPTIONS
)

from cugraph.tests.test_betweenness_centrality import (
    OPGContext,
    prepare_test,
    calc_betweenness_centrality,
    compare_scores
)

# =============================================================================
# Parameters
# =============================================================================
OPG_DEVICE_COUNT_OPTIONS = [1, 2, 3, 4]


@pytest.mark.parametrize('opg_device_count', OPG_DEVICE_COUNT_OPTIONS)
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('weight', [None])
@pytest.mark.parametrize('endpoints', ENDPOINTS_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_opg_betweenness_centrality(opg_device_count,
                                    graph_file,
                                    directed,
                                    subset_size,
                                    normalized,
                                    weight,
                                    endpoints,
                                    subset_seed,
                                    result_dtype):
    prepare_test()
    visible_devices = get_visible_devices()
    number_of_visible_devices = len(visible_devices)
    if opg_device_count > number_of_visible_devices:
        pytest.skip("Not enough devices available to test OPG")
    with OPGContext(opg_device_count):
        sorted_df = calc_betweenness_centrality(graph_file,
                                                directed=directed,
                                                normalized=normalized,
                                                k=subset_size,
                                                weight=weight,
                                                endpoints=endpoints,
                                                seed=subset_seed,
                                                result_dtype=result_dtype)
    compare_scores(sorted_df, first_key="cu_bc", second_key="ref_bc",
                   epsilon=DEFAULT_EPSILON)
