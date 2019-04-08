#!/bin/sh
#Usage sh run_all_tests.sh
#Run all the tests in the current directory (ie. you should copy it in your build/test/ directory).
test="nvgraph_test
csrmv_test
semiring_maxmin_test
semiring_minplus_test
semiring_orand_test
pagerank_test
sssp_test
max_flow_test"

for i in $test
do
./$i
done
