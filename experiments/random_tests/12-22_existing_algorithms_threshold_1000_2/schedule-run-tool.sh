#!/bin/bash

for TOPO in $(ls confs/12-22_existing_algorithms_threshold_1000_2) ; do
    for CONF in $(ls confs/12-22_existing_algorithms_threshold_1000_2/${TOPO}/conf*) ; do
        sbatch experiments/random_tests/12-22_existing_algorithms_threshold_1000_2/run-tool.sh ${CONF}
    done
done
