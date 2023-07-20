import pytest

import cugraph
from python.cugraph.cugraph.testing.resultset_pr import get_resultset
from cudf.testing import assert_frame_equal

karate_test_data = [
    [0.6, 50, get_resultset],
    [0.6, 75, get_resultset],
    [0.6, 100, get_resultset],
    [0.6, -100, OverflowError],
    [0.75, 50, get_resultset],
    [0.75, 75, get_resultset],
    [0.75, 100, get_resultset],
    [0.85, 50, get_resultset],
    [0.85, 75, get_resultset],
    [0.85, 100, get_resultset],
]

@pytest.fixture(params=[pytest.param(p) for p in karate_test_data])
def test_data(request):
    alpha, max_iter, expected_result = request.param
    breakpoint()
    if (type(expected_result) != type) and callable(expected_result):
        expected_result = expected_result(graph_dataset="karate",
                                          graph_directed=False,
                                          algo="pagerank",
                                          alpha=alpha,
                                          max_iter=max_iter).get_cudf_dataframe()
    return (alpha, max_iter, expected_result)


########################################
def test_pagerank(test_data):
    (alpha, max_iter, expected_result) = test_data
    G = cugraph.experimental.datasets.karate.get_graph()

    if (type(expected_result) == type) and issubclass(expected_result, Exception):
        with pytest.raises(expected_result):
            cugraph.pagerank(G, alpha=alpha, max_iter=max_iter)
    else:
        pr = cugraph.pagerank(G, alpha=alpha, max_iter=max_iter)
        pr = pr.sort_values("vertex", ignore_index=True)
        expected_result = expected_result.sort_values("vertex", ignore_index=True)
        expected_result["pagerank"] = expected_result["pagerank"].astype("float32")
        assert_frame_equal(pr,
                           expected_result,
                           check_like=True,
                           check_dtype=False,
                           atol=1e-2)