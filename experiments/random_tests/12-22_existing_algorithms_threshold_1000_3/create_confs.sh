#!/bin/bash
#SBATCH --partition=naples

source venv/bin/activate

for TOPO in $(ls topologies/zoo_A*) ; do
  sbatch experiments/random_tests/12-22_existing_algorithms_threshold_1000_3/run-createconfs.sh ${TOPO}
done