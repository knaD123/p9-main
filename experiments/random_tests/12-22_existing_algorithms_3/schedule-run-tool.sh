#!/bin/bash

for TOPO in $(ls confs/12-22_existing_algorithms_3) ; do
    for CONF in $(ls confs/12-22_existing_algorithms_3/${TOPO}/conf*) ; do
        sbatch experiments/random_tests/12-22_existing_algorithms_3/run-tool.sh ${CONF}
    done
done
