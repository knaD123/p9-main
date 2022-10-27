#!/bin/bash

PD=$(pwd)
CONFIG="${1}"
AMOUNT_TO_TEST="${2}"
FILTER="${3}"
count=1

PD=$(pwd)

source ${PD}/venv/bin/activate

#for TOPO in $(ls -d confs/${FILTER}*) ; do
#    if [ $count -le $AMOUNT_TO_TEST ]
#    then
#        for FAILCHUNK in $(ls ${TOPO}/failure_chunks) ; do
#            python3 ${PD}/tool_simulate.py --conf "${TOPO}/conf_${CONFIG}.yml" --failure_chunk_file "${TOPO}/failure_chunks/${FAILCHUNK}"
#        done
#    fi
#   (( count++ ))
#done

for TOPO in $(ls confs) ; do
    if [ $count -le $AMOUNT_TO_TEST ]
    #for CONF in $(ls confs/${TOPO}/conf_inout-disjoint_max*) ; do
    for FAILCHUNK in $(ls confs/${TOPO}/failure_chunks) ; do
        for FLOWS in $(ls confs/${TOPO}/flows) ; do
            sbatch scripts/run-tool.sh ${TOPO} ${CONF} ${FAILCHUNK} ${FLOWS}
            break
        done
    done
    #done
    (( count++ ))
done