#!/bin/bash

PD=$(pwd)

for TOPO in $(ls confs) ; do
    for CONF in $(ls confs/${TOPO}/conf*) ; do
        for FAILCHUNK in $(ls confs/${TOPO}/failure_chunks) ; do
            sbatch scripts/run-tool.sh ${TOPO} ${CONF} ${FAILCHUNK}
        done
    done
done