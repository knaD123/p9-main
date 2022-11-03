#!/bin/bash

PD=$(pwd)

for TOPO in $(ls confs) ; do
    for CONF in $(ls confs/${TOPO}/conf*) ; do
        experiments/10-31/run-tool.sh ${CONF}
    done
done
