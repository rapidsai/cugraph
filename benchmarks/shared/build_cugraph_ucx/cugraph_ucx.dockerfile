# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

RUN gpuci_mamba_retry install -y -c rapidsai-nightly -c rapidsai -c conda-forge \
    cugraph=$RAPIDS_VER \
    pytorch=$PYTORCH_VER \
    python=$PYTHON_VER \
    setuptools \
    tqdm

ADD build-ucx.sh /root/build-ucx.sh
RUN chmod 744 /root/build-ucx.sh & bash /root/build-ucx.sh

ADD test_client_bandwidth.py  /root/test_client_bandwidth.py
RUN chmod 777 /root/test_client_bandwidth.py

ENV PATH /opt/conda/env/bin:$PATH
WORKDIR /root/
