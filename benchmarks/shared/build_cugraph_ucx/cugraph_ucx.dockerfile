# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# rdma build instructions from here:
# https://github.com/rapidsai/dask-cuda-benchmarks/blob/main/runscripts/draco/docker/UCXPy-rdma-core.dockerfile
ARG CUDA_VER=11.2
ARG LINUX_VER=ubuntu20.04
FROM gpuci/miniforge-cuda:$CUDA_VER-devel-$LINUX_VER

RUN apt-get update -y \
    && apt-get --fix-missing upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && apt-get install -y \
        automake \
        dh-make \
        git \
        libcap2 \
        libnuma-dev \
        libtool \
        make \
        pkg-config \
        udev \
        curl \
        librdmacm-dev \
        rdma-core \
    && apt-get autoremove -y \
    && apt-get clean

ARG PYTHON_VER=3.9
ARG RAPIDS_VER=23.02
ARG PYTORCH_VER=1.12.0
ARG CUDATOOLKIT_VER=11.3

RUN conda config --set ssl_verify false
RUN conda install -c gpuci gpuci-tools
RUN gpuci_conda_retry install -c conda-forge mamba


RUN gpuci_mamba_retry install -y -c pytorch -c rapidsai-nightly -c rapidsai -c conda-forge -c nvidia \
    cudatoolkit=$CUDATOOLKIT_VER \
    cugraph=$RAPIDS_VER \
    pytorch=$PYTORCH_VER \
    python=$PYTHON_VER \
    setuptools \
    tqdm


# Build ucx from source with IB support 
# on 1.14.x
RUN conda remove --force -y ucx ucx-proc

ADD build-ucx.sh /root/build-ucx.sh
RUN chmod 744 /root/build-ucx.sh & bash /root/build-ucx.sh


ADD test_client_bandwidth.py  /root/test_client_bandwidth.py
RUN chmod 777 /root/test_client_bandwidth.py
ADD test_cugraph_sampling.py  /root/test_cugraph_sampling.py
RUN chmod 777 /root/test_cugraph_sampling.py

ENV PATH /opt/conda/env/bin:$PATH
WORKDIR /root/
