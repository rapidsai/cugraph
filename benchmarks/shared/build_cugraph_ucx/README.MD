# Cugraph with UCX
## Docker Build Instructions
docker build -f cugraph_ucx.dockerfile . -t cugraph_ucx

## Docker Run Instructions
docker run --privileged -it --gpus=all --net=host cugraph_ucx /bin/bash

#### Client Bandwidth Test
python3 test_client_bandwidth.py 

```bash
(base) root@exp02:/home# python3 test_client_bandwidth.py 
2022-12-19 13:31:30,867 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:31:30,867 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
2022-12-19 13:31:30,891 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:31:30,891 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
2022-12-19 13:31:30,918 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:31:30,918 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
Getting 1,000,000 rows  of size 0.01 took = 20.995140075683594 ms
Bandwidth = 0.4763 gb/s
Getting 2,000,000 rows  of size 0.02 took = 18.51017475128174 ms
Bandwidth = 1.0805 gb/s
Getting 4,000,000 rows  of size 0.04 took = 25.659966468811035 ms
Bandwidth = 1.5588 gb/s
Getting 25,000,000 rows  of size 0.28 took = 53.80809307098389 ms
Bandwidth = 5.2037 gb/s
----------------------------------------Completed Test----------------------------------------
```

#### Sampling Test
python3 test_cugraph_sampling.py
```bash
test_client_bandwidth.py  test_cugraph_sampling.py  
(base) root@exp02:/home# python3 test_cugraph_sampling.py 
[1671456769.722931] [exp02:93   :0]          parser.c:1989 UCX  WARN  unused environment variable: UCX_MEMTYPE_CACHE (maybe: UCX_MEMTYPE_CACHE?)
[1671456769.722931] [exp02:93   :0]          parser.c:1989 UCX  WARN  (set UCX_WARN_UNUSED_ENV_VARS=n to suppress this warning)
2022-12-19 13:32:56,228 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:32:56,228 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
2022-12-19 13:32:57,485 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:32:57,485 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
2022-12-19 13:32:57,655 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:32:57,655 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
2022-12-19 13:32:57,878 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:32:57,879 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
2022-12-19 13:32:57,957 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:32:57,957 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
2022-12-19 13:32:57,963 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:32:57,963 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
2022-12-19 13:32:57,973 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:32:57,973 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
2022-12-19 13:32:58,011 - distributed.preloading - INFO - Creating preload: dask_cuda.initialize
2022-12-19 13:32:58,012 - distributed.preloading - INFO - Import preload module: dask_cuda.initialize
Sampling 1,000 took = 69.15879249572754 ms
Sampling 10,000 took = 89.63620662689209 ms
Sampling 100,000 took = 135.9888792037964 ms
----------------------------------------Completed Test----------------------------------------
```