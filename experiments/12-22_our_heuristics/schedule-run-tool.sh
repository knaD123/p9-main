#!/bin/bash


for TOPO in $(ls confs/12-22_our_heuristics) ; do
    for CONF in $(ls confs/12-22_our_heuristics/${TOPO}/conf*) ; do
        sbatch experiments/12-22_our_heuristics/run-tool.sh ${CONF}
    done
done
