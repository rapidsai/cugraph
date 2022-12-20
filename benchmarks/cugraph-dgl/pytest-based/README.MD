## Run Benchmarks 

#### SG 
```
pytest bench_cugraph_dgl_uniform_neighbor_sample.py -k "SG and fanout_10_25 and rmat_24_4"
```
#### MG 

```
DASK_NUM_WORKERS=2 pytest bench_cugraph_dgl_uniform_neighbor_sample.py -k "MG and fanout_10_25 and rmat_24_16" --benchmark-save='2_rmat_24_16.json'
DASK_NUM_WORKERS=4 pytest bench_cugraph_dgl_uniform_neighbor_sample.py -k "MG and fanout_10_25 and rmat_25_16" --benchmark-save='3_rmat_25_16.json'
DASK_NUM_WORKERS=8 pytest bench_cugraph_dgl_uniform_neighbor_sample.py -k "MG and fanout_10_25 and rmat_26_8" --benchmark-save='8_rmat_26_8.json'
```
