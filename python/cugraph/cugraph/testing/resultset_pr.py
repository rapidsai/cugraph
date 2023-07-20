from pathlib import Path

# from cugraph.experimental.datasets import karate
import cudf
# import networkx as nx
import os

class Resultset:
    def __init__(self, data_dictionary):
        self._data_dictionary = data_dictionary

    def get_cudf_dataframe(self):
        return cudf.DataFrame(self._data_dictionary)

_resultsets = {}

results_dir = Path(os.environ.get("RAPIDS_DATASET_ROOT_DIR"))
results_dir = Path("testing/results")

def add_resultset(result_data_dictionary, **kwargs):
    rs = Resultset(result_data_dictionary)
    hashable_dict_repr = tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
    _resultsets[hashable_dict_repr] = rs

def get_resultset(**kwargs):
    hashable_dict_repr = tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
    desired = results_dir / (str(hashable_dict_repr) + ".csv")
    return cudf.read_csv(desired)


results_dir = Path("testing/results")

# Example Code
################################################################################
# Populate with results from running pagerank on karate over all combinations
# of the values for alpha and max_iter below.
"""pdf = karate.get_edgelist().to_pandas().rename(columns={"src": "source",
                                                        "dst": "target"})
Gnx = nx.from_pandas_edgelist(pdf)

alpha_values = [0.6, 0.75, 0.85]
max_iter_values = [50, 75, 100]

for alpha in alpha_values:
    for max_iter in max_iter_values:
        print(f"pagerank: {alpha=}, {max_iter=}")
        results = nx.pagerank(Gnx, alpha=alpha, max_iter=max_iter)
        (vertices, pageranks) = zip(*results.items())
        add_resultset({"vertex": vertices, "pagerank": pageranks},
                      graph_dataset="karate",
                      graph_directed=False,
                      algo="pagerank",
                      alpha=alpha,
                      max_iter=max_iter)
"""
