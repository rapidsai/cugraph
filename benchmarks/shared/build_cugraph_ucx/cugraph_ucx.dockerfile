# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

ARG CUDA_VER=12.8.1
ARG LINUX_VER=ubuntu22.04
FROM nvidia/cuda:${CUDA_VER}-devel-$LINUX_VER

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
        wget \
        librdmacm-dev \
        rdma-core \
    && apt-get autoremove -y \
    && apt-get clean

# Install miniforge and configure
ENV PATH="/opt/conda/bin:$PATH"
RUN wget -qO Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
  && bash Miniforge3.sh -b -p "/opt/conda" \
  && . "/opt/conda/etc/profile.d/conda.sh" \
  && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
  && echo "ssl_verify: true" >> /opt/conda/.condarc \
  && echo "conda activate base" >> ~/.bashrc \
  && conda init \
  && conda activate

ARG PYTHON_VER=3.13
ARG RAPIDS_VER=25.08
ARG PYTORCH_VER=2.5.1

RUN conda config --set ssl_verify false
RUN conda install -c gpuci gpuci-tools
RUN gpuci_conda_retry install -c conda-forge mamba

RUN gpuci_mamba_retry install -y -c pytorch -c rapidsai-nightly -c rapidsai -c conda-forge -c nvidia \
    cugraph=$RAPIDS_VER \
    pytorch=$PYTORCH_VER \
    python=$PYTHON_VER \
    setuptools \
    tqdm

ADD build-ucx.sh /root/build-ucx.sh
RUN chmod 744 /root/build-ucx.sh & bash /root/build-ucx.sh

ADD test_client_bandwidth.py  /root/test_client_bandwidth.py
RUN chmod 777 /root/test_client_bandwidth.py
ADD test_cugraph_sampling.py  /root/test_cugraph_sampling.py
RUN chmod 777 /root/test_cugraph_sampling.py

ENV PATH /opt/conda/env/bin:$PATH
WORKDIR /root/
