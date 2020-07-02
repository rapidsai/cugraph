import pytest

from cugraph.dask.core import get_visible_devices
import time

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
DEFAULT_MAX_ATTEMPT = 100
DEFAULT_WAIT_TIME = 0.5


# NOTE: This only looks for the number of  workers
def enforce_rescale(cluster, scale, max_attempts=DEFAULT_MAX_ATTEMPT,
                    wait_time=DEFAULT_WAIT_TIME):
    cluster.scale(scale)
    attempt = 0
    ready = (len(cluster.workers) == scale)
    while (attempt < max_attempts) and not ready:
        time.sleep(wait_time)
        ready = (len(cluster.workers) == scale)
        attempt += 1
    assert ready, "Unable to rescale cluster to {}".format(scale)


@pytest.fixture(scope="module")
def fixture_setup_opg():
    visible_devices = get_visible_devices()
    number_of_visible_devices = len(visible_devices)
    with OPGContext(number_of_devices=number_of_visible_devices) as context:
        print("[DBG] Started Client: {}", context._client)
        cluster = context._cluster
        yield cluster

@pytest.mark.parametrize('opg_device_count', OPG_DEVICE_COUNT_OPTIONS)
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('directed', DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize('subset_size', SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize('normalized', NORMALIZED_OPTIONS)
@pytest.mark.parametrize('weight', [None])
@pytest.mark.parametrize('endpoints', ENDPOINTS_OPTIONS)
@pytest.mark.parametrize('subset_seed', SUBSET_SEED_OPTIONS)
@pytest.mark.parametrize('result_dtype', RESULT_DTYPE_OPTIONS)
def test_opg_betweenness_centrality(fixture_setup_opg,
                                    opg_device_count,
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

    opg_cluster = fixture_setup_opg[0]
    enforce_rescale(opg_cluster, opg_device_count,
                    DEFAULT_MAX_ATTEMPT,
                    DEFAULT_WAIT_TIME)
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
