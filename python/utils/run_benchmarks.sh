#!/bin/bash
set -e

THISDIR=$(dirname $0)
VERSION=${VERSION:=0.0}
UTILS_DIR=${UTILS_DIR:=${THISDIR}}
DATASET_DIR=${DATASET_DIR:=${THISDIR}/../../datasets}
MACHINE_NAME=${MACHINE_NAME:="mymachine"}
CONDA=${CONDA:=conda}

# To output results for use with ASV, set
# ASV_OUTPUT_OPTION="--update_asv_dir=/asv/cugraph-e2e" (update /asv/cugraph-e2e
# to the desired results dir)
ASV_OUTPUT_OPTION=${ASV_OUTPUT_OPTION:=""}


ERROR=0
for ds in ${DATASET_DIR}/csv/undirected/*; do
   echo "================================ ${ds}"
   if [ "${ds}" == "${DATASET_DIR}/csv/undirected/soc-twitter-2010.csv" ]; then
      echo
      echo "SKIPPING ${ds}"
      echo
   else
      python ${UTILS_DIR}/run_benchmarks.py ${ASV_OUTPUT_OPTION} --auto_csr --report_cuda_ver=${CUDA_VERSION} --report_python_ver=${PYTHON_VERSION} --report_os_type=${LINUX_VERSION} --report_machine_name=${MACHINE_NAME} ${ds}
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
   python ${UTILS_DIR}/run_benchmarks.py ${ASV_OUTPUT_OPTION} --auto_csr --report_cuda_ver=${CUDA_VERSION} --report_python_ver=${PYTHON_VERSION} --report_os_type=${LINUX_VERSION} --report_machine_name=${MACHINE_NAME} --algo=cugraph.pagerank --algo=cugraph.bfs --algo=cugraph.sssp --algo=cugraph.overlap --algo=cugraph.renumber --algo=cugraph.graph.view_adj_list --algo=cugraph.graph.degree --algo=cugraph.graph.degrees --digraph ${ds}
   exitcode=$?
   if (( ${exitcode} != 0 )); then
      ERROR=${exitcode}
      echo "ERROR: ${ds}"
   fi
   echo
done
exit ${ERROR}
