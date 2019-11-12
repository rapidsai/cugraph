#!/usr/bin/env bash

export BUILD_CUGRAPH=1
export BUILD_LIBCUGRAPH=1

if [[ "$CUDA" == "9.2" ]]; then
    export UPLOAD_CUGRAPH=1
else
    export UPLOAD_CUGRAPH=0
fi

if [[ "$PYTHON" == "3.6" ]]; then
    export UPLOAD_LIBCUGRAPH=1
else
    export UPLOAD_LIBCUGRAPH=0
fi
