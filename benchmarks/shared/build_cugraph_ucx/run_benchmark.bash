#!/bin/bash
#python3 test_cugraph_sampling.py --n_workers=2 --scale=25 > scale_25.txt
python3 test_cugraph_sampling.py --n_workers=4 --scale=26 > scale_26.txt
#python3 test_cugraph_sampling.py --n_workers=8 --scale=27 > scale_27.txt
#python3 test_cugraph_sampling.py --n_workers=16 --scale=28 > scale_28.txt