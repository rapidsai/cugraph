import pytest

from cugraph.dask.core import get_visible_devices

from cugraph.tests.dask.opg_context import (OPGContext, enforce_rescale)

# Get parameters from standard betwenness_centrality_test
from cugraph.tests.test_betweenness_centrality import (
    #DIRECTED_GRAPH_OPTIONS,
    ENDPOINTS_OPTIONS,
    NORMALIZED_OPTIONS,
    DEFAULT_EPSILON,
    DATASETS,
    SUBSET_SIZE_OPTIONS,
    SUBSET_SEED_OPTIONS,
    RESULT_DTYPE_OPTIONS
)

from cugraph.tests.test_betweenness_centrality import (
    prepare_test,
    calc_betweenness_centrality,
    compare_scores
)

# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [True] # FIXME: Undirected Distributed Graph currently not supported
OPG_DEVICE_COUNT_OPTIONS = [1, 2, 3, 4]


# NOTE: This approach implies that the resources are distributed at
# the creation of the (i.e if it is started with 300GB, and 3 workers)
# each worker will get ~100GB, thus after rescaling to a single worker,
# the worker will only have ~100GB and not all 300GB.
@pytest.fixture(scope="module")
def fixture_setup_opg():
    visible_devices = get_visible_devices()
    number_of_visible_devices = len(visible_devices)
    with OPGContext(number_of_devices=number_of_visible_devices) as context:
        cluster = context._cluster
        yield cluster


@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('weight', [None])
@pytest.mark.parametrize('endpoints', ENDPOINTS_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
@pytest.mark.parametrize('opg_device_count', OPG_DEVICE_COUNT_OPTIONS)
def test_opg_betweenness_centrality(fixture_setup_opg,
                                    graph_file,
                                    directed,
                                    subset_size,
                                    normalized,
                                    weight,
                                    endpoints,
                                    subset_seed,
                                    result_dtype,
                                    opg_device_count):
    prepare_test()
    visible_devices = get_visible_devices()
    number_of_visible_devices = len(visible_devices)
    if opg_device_count > number_of_visible_devices:
        pytest.skip("Not enough devices available to "
                    "test OPG({})".format(opg_device_count))
    cluster = fixture_setup_opg
    enforce_rescale(cluster, opg_device_count)
    assert len(cluster.workers) == opg_device_count, \
        "Error on OPG context, mismatch on the number of workers"
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
