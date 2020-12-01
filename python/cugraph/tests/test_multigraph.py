import cugraph
import networkx as nx
from cugraph.tests import utils
import pytest
import gc
import pandas as pd
import numpy as np

@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_multigraph(graph_file):
    gc.collect()
    cuM = utils.read_csv_file(graph_file)
    G = cugraph.MultiGraph()
    G.from_cudf_edgelist(cuM, source="0", destination="1", edge_attr="2")

    nxM = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiGraph(),
    )

    assert G.number_of_edges() == Gnx.number_of_edges()
    assert G.number_of_nodes() == Gnx.number_of_nodes()


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_multigraph_sssp(graph_file):
    gc.collect()
    cuM = utils.read_csv_file(graph_file)
    G = cugraph.MultiDiGraph()
    G.from_cudf_edgelist(cuM, source="0", destination="1", edge_attr="2")
    cu_paths = cugraph.sssp(G, 0)
    max_val = np.finfo(cu_paths["distance"].dtype).max
    cu_paths = cu_paths[cu_paths["distance"] != max_val]
    nxM = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiDiGraph(),
    )
    nx_paths = nx.single_source_dijkstra_path_length(Gnx, 0)

    print(cu_paths)
    print(nx_paths)

    cu_dist = cu_paths.sort_values(by='vertex')['distance'].to_array()
    nx_dist = [i[1] for i in sorted(nx_paths.items())]

    assert (cu_dist == nx_dist).all()
