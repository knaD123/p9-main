#!/bin/bash


for TOPO in $(ls confs_fbr_old_vs_new) ; do
    for CONF in $(ls confs_fbr_old_vs_new/${TOPO}/conf*) ; do
        sbatch experiments/fbr_old_vs_new/run-tool.sh ${CONF}
    done
done
