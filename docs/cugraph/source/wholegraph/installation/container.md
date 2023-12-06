# Build Container for WholeGraph
To run WholeGraph or build WholeGraph from source, set up the environment first.
We recommend using Docker images.
For example, to build the WholeGraph base image from the NGC pytorch 22.10 image, you can follow `Dockerfile`:
```dockerfile
FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y lsb-core software-properties-common wget libspdlog-dev

#RUN remove old cmake to update
RUN conda remove --force -y cmake
RUN rm -rf /usr/local/bin/cmake && rm -rf /usr/local/lib/cmake && rm -rf /usr/lib/cmake

RUN apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc && \
    export LSB_CODENAME=$(lsb_release -cs) && \
    apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ ${LSB_CODENAME} main" && \
    apt update && apt install -y cmake

# update py for pytest
RUN pip3 install -U py
RUN pip3 install Cython setuputils3 scikit-build nanobind pytest-forked pytest
```

To run GNN applications, you may also need cuGraphOps, DGL and/or PyG libraries to run the GNN layers.
You may refer to [DGL](https://www.dgl.ai/pages/start.html) or [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
For example, to install DGL, you may need to add:
```dockerfile
RUN pip3 install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```
