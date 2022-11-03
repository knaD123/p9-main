#!/bin/bash
#SBATCH --partition=dhabi,rome


PD=$(pwd)

EXECUTOR="sbatch"
if [ "$1" = "no" ]; then
  EXECUTOR=""
fi

FILTER="$2"

#source ${PD}/venv/bin/activate

rm confs/*/conf*

for TOPO in $(ls topologies/${FILTER}*) ; do
  $EXECUTOR experiments/10-31/run-createconfs.sh ${TOPO}
done

