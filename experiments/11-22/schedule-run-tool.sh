#!/bin/bash

for TOPO in $(ls confs) ; do
    for CONF in $(ls confs/${TOPO}/conf*) ; do
        experiments/11-22/run-tool.sh ${CONF}
    done
done
