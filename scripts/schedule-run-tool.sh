#!/bin/bash
#SBATCH --mail-type=NONE # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=amad18@student.aau.dk
#SBATCH --output=/nfs/home/student.aau.dk/amad18/slurm-output/gen-net-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/amad18/slurm-output/gen-net-%j.err
#SBATCH --partition=naples


PD=$(pwd)

#CONF=$1

for TOPO in $(ls confs) ; do
    for CONF in $(ls confs/${TOPO}/conf_inout-disjoint_max*) ; do
        for FAILCHUNK in $(ls confs/${TOPO}/failure_chunks) ; do
            sbatch scripts/run-tool.sh ${TOPO} ${CONF} ${FAILCHUNK}
        done
    done
done