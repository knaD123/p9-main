#!/bin/bash

for TOPO in $(ls topologies) ; do
    sbatch run-lp.sh ${TOPO}
    break
done
