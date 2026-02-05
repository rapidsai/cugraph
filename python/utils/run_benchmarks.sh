#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -e

THISDIR=$(dirname $0)
VERSION=${VERSION:=0.0}
UTILS_DIR=${UTILS_DIR:=${THISDIR}}
DATASET_DIR=${DATASET_DIR:=${THISDIR}/../../datasets}
MACHINE_NAME=${MACHINE_NAME:="mymachine"}
CONDA=${CONDA:=conda}

ERROR=0
for ds in ${DATASET_DIR}/csv/undirected/*; do
   echo "================================ ${ds}"
   if [ "${ds}" == "${DATASET_DIR}/csv/undirected/soc-twitter-2010.csv" ]; then
      echo
      echo "SKIPPING ${ds}"
      echo
   else
       python ${UTILS_DIR}/run_benchmarks.py \
              --report_cuda_ver=${CUDA_VERSION} \
              --report_python_ver=${PYTHON_VERSION} \
              --report_os_type=${LINUX_VERSION} \
              --report_machine_name=${MACHINE_NAME} \
              --compute_adj_list \
              \
              --algo=cugraph.bfs \
              --algo=cugraph.sssp \
              --algo=cugraph.jaccard \
              --algo=cugraph.louvain \
              --algo=cugraph.weakly_connected_components \
              --algo=cugraph.overlap \
              --algo=cugraph.triangles \
              --algo=cugraph.spectralBalancedCutClustering \
              --algo=cugraph.spectralModularityMaximizationClustering \
              --algo=cugraph.renumber \
              --algo=cugraph.graph.degree \
              --algo=cugraph.graph.degrees \
              \
              ${ds}
       python ${UTILS_DIR}/run_benchmarks.py \
              --report_cuda_ver=${CUDA_VERSION} \
              --report_python_ver=${PYTHON_VERSION} \
              --report_os_type=${LINUX_VERSION} \
              --report_machine_name=${MACHINE_NAME} \
              --compute_transposed_adj_list \
              \
              --algo=cugraph.pagerank \
              \
              ${ds}
      exitcode=$?
      if (( ${exitcode} != 0 )); then
         ERROR=${exitcode}
         echo "ERROR: ${ds}"
      fi
   fi
   echo
done
for ds in ${DATASET_DIR}/csv/directed/*; do
   echo "================================ ${ds}"
   python ${UTILS_DIR}/run_benchmarks.py \
          --report_cuda_ver=${CUDA_VERSION} \
          --report_python_ver=${PYTHON_VERSION} \
          --report_os_type=${LINUX_VERSION} \
          --report_machine_name=${MACHINE_NAME} \
          --compute_adj_list \
          --digraph \
          \
          --algo=cugraph.bfs \
          --algo=cugraph.sssp \
          --algo=cugraph.overlap \
          --algo=cugraph.renumber \
          --algo=cugraph.graph.degree \
          --algo=cugraph.graph.degrees \
          \
          ${ds}
   python ${UTILS_DIR}/run_benchmarks.py \
          --report_cuda_ver=${CUDA_VERSION} \
          --report_python_ver=${PYTHON_VERSION} \
          --report_os_type=${LINUX_VERSION} \
          --report_machine_name=${MACHINE_NAME} \
          --compute_transposed_adj_list \
          --digraph \
          \
          --algo=cugraph.pagerank \
          \
          ${ds}
   exitcode=$?
   if (( ${exitcode} != 0 )); then
      ERROR=${exitcode}
      echo "ERROR: ${ds}"
   fi
   echo
done
exit ${ERROR}
