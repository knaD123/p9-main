#!/bin/bash

for TOPO in $(ls confs) ; do
    for CONF in $(ls confs/${TOPO}/conf*) ; do
        experiments/12-06/run-tool.sh ${CONF}
    done
done
