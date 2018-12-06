# built from https://github.com/rapidsai/cudf/blob/master/Dockerfile
FROM cudf

ADD src /cugraph/src
ADD include /cugraph/include
ADD cmake /cugraph/cmake
ADD CMakeLists.txt /cugraph/CMakeLists.txt
ADD python /cugraph/python
ADD setup.py /cugraph/setup.py
ADD docs /cugraph/docs

WORKDIR /cugraph/build
RUN source activate cudf && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DNVG_PLUGIN=FALSE && \
    make install && \
    cd .. && \
    python setup.py install
