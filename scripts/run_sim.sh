#!/bin/bash

for TOPO in $(ls confs) ; do
  for CONF in $(ls confs/${TOPO}/conf*) ; do
    for FCHUNK in $(ls confs/${TOPO}/failure_chunks) ; do
      sbatch ./tool_simulate --conf ${CONF} --failure_chunk_file ${FCHUNK}
    done
  done
done
