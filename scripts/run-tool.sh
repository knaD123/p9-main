#!/bin/bash
PD=$(pwd)

source ~/Documents/github/p9-main/venv/bin/activate

#python3 -m pip install -r requirements.txt

TOPO="$1"
CONFIG="$2"
FAILCHUNK="$3"

for FLOW in $(ls confs/${TOPO}/flows) ; do
  python3 ${PD}/tool_simulate.py --conf ${CONFIG} --flows_file ${FLOW} --failure_chunk_file "confs/${TOPO}/failure_chunks/${FAILCHUNK}" --take_percent 0.2
done