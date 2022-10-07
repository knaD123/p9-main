#!/bin/bash

PD=$(pwd)
TOPO="${1}"
CONFIG_FILE="${2}"
count=1

source ~/p8/venv/bin/activate

for FAILCHUNK in $(ls confs/${TOPO}/failure_chunks) ; do
    python3 ${PD}/tool_simulate.py --conf "confs/${TOPO}/conf_${CONFIG_FILE}.yml" --failure_chunk_file "confs/${TOPO}/failure_chunks/${FAILCHUNK}" --result_folder "results/${CONFIG_FILE}/${TOPO}"
done
