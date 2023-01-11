#!/bin/bash
python3 test_cugraph_sampling.py --n_workers=2 --scale=25 > sample_benchmark/ucx_scale_25.txt
python3 test_cugraph_sampling.py --n_workers=3 --scale=26 > sample_benchmark/ucx_scale_26.txt
python3 test_cugraph_sampling.py --n_workers=6 --scale=27 > sample_benchmark/ucx_scale_27.txt
python3 test_cugraph_sampling.py --n_workers=11 --scale=28 > sample_benchmark/ucx_scale_28.txt