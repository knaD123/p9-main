#!/bin/bash

for TOPO in $(ls confs_12-22_existing_algorithms_threshold_1000_4) ; do
    for CONF in $(ls confs_12-22_existing_algorithms_threshold_1000_4/${TOPO}/conf*) ; do
        sbatch experiments/random_tests/12-22_existing_algorithms_threshold_1000_4/run-tool.sh ${CONF}
    done
done
