#!/bin/bash
#SBATCH --partition=naples

EXECUTOR="sbatch"
if [ "$1" = "no" ]; then
  EXECUTOR=""
fi

FILTER="$2"

source venv/bin/activate

rm confs/*/conf*

for TOPO in $(ls topologies/${FILTER}*) ; do
  $EXECUTOR experiments/11-22/run-createconfs.sh ${TOPO}
done

