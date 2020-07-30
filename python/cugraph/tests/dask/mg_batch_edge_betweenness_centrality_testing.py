import pytest

from cugraph.tests.dask.mg_context import (MGContext, get_visible_devices)

# Get parameters from standard betwenness_centrality_test
from cugraph.tests.test_edge_betweenness_centrality import (
    DIRECTED_GRAPH_OPTIONS,
    NORMALIZED_OPTIONS,
    DEFAULT_EPSILON,
    DATASETS,
    SUBSET_SIZE_OPTIONS,
    SUBSET_SEED_OPTIONS,
    RESULT_DTYPE_OPTIONS
)

from cugraph.tests.test_edge_betweenness_centrality import (
    prepare_test,
    calc_edge_betweenness_centrality,
    compare_scores
)

# =============================================================================
# Parameters
# =============================================================================
MG_DEVICE_COUNT_OPTIONS = [1, 2, 3, 4]


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
    visible_devices = get_visible_devices()
    number_of_visible_devices = len(visible_devices)
    if mg_device_count > number_of_visible_devices:
        pytest.skip("Not enough devices available to "
                    "test MG({})".format(mg_device_count))
    with MGContext(mg_device_count) as context:
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
