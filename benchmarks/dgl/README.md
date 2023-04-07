# DGL  Benchmarks

## Create Dataset
```
python3 create_dataset.py
```

## For PURE GPU Benchmarks
```
pytest dgl_benchmark.py::bench_dgl_pure_gpu
```

## For UVA Benchmarks
```
pytest dgl_benchmark.py::bench_dgl_uva
```