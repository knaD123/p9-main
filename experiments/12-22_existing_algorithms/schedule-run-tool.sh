#!/bin/bash


for TOPO in $(ls confs/12-22_existing_algorithms) ; do
    for CONF in $(ls confs/12-22_existing_algorithms/${TOPO}/conf*) ; do
        sbatch experiments/12-22_existing_algorithms/run-tool.sh ${CONF}
    done
done
