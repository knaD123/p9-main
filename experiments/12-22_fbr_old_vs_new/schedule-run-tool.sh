#!/bin/bash


for TOPO in $(ls confs/12-22_fbr_old_vs_new) ; do
    for CONF in $(ls confs/12-22_fbr_old_vs_new/${TOPO}/conf*) ; do
        sbatch experiments/12-22_fbr_old_vs_new/run-tool.sh ${CONF}
    done
done
