import pytest

import cudf
import dask_cudf

import tempfile
import os

from cugraph.gnn.data_loading.bulk_sampler_io import write_samples

@pytest.mark.mg
def test_bulk_sampler_io():
    results = cudf.DataFrame({
        'sources': [0, 0, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7],
        'destinations': [1, 2, 3, 3, 3, 4, 1, 1, 6, 7, 2, 3],
        'edge_id': None,
        'edge_type': None,
        'weight': None,
        'hop_id': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    })
    results = dask_cudf.from_cudf(results, npartitions=1).repartition(divisions=[0, 8, 11])

    offsets = cudf.DataFrame({
        'offsets': [0, 0],
        'batch_id': [0, 1]
    })
    offsets = dask_cudf.from_cudf(offsets, npartitions=2)
    print(offsets.npartitions)

    tempdir_object = tempfile.TemporaryDirectory()
    write_samples(results, offsets, 1, tempdir_object.name)

    assert len(os.listdir(tempdir_object.name)) == 2

    df = cudf.read_parquet(os.path.join(tempdir_object.name, 'batch=0-0.parquet'))
    assert len(df) == 8

    results = results.compute()
    assert df.sources.values_host.tolist() == results.sources.iloc[0:8].values_host.tolist()
    assert df.destinations.values_host.tolist() == results.destinations.iloc[0:8].values_host.tolist()
    assert df.hop_id.values_host.tolist() == results.hop_id.iloc[0:8].values_host.tolist()
    assert (df.batch_id==0).all()

    df = cudf.read_parquet(os.path.join(tempdir_object.name, 'batch=1-1.parquet'))
    assert len(df) == 4
    assert df.sources.values_host.tolist() == results.sources.iloc[8:12].values_host.tolist()
    assert df.destinations.values_host.tolist() == results.destinations.iloc[8:12].values_host.tolist()
    assert df.hop_id.values_host.tolist() == results.hop_id.iloc[8:12].values_host.tolist()
    assert (df.batch_id==1).all()
