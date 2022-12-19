#!/bin/bash


for TOPO in $(ls confs) ; do
    for CONF in $(ls confs/${TOPO}/conf*) ; do
        sbatch experiments/12-14/run-tool.sh ${CONF}
    done
done