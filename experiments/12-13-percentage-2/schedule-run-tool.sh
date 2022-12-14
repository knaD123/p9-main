#!/bin/bash


for TOPO in $(ls confs) ; do
    for CONF in $(ls confs-percentage-2/${TOPO}/conf*) ; do
        sbatch experiments/12-13-percentage-2/run-tool.sh ${CONF}
    done
done
