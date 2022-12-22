#!/bin/bash


for TOPO in $(ls confs) ; do
    for CONF in $(ls confs/percentage/${TOPO}/conf*) ; do
        sbatch experiments/12-13-percentage/run-tool.sh ${CONF}
    done
done
