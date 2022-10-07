#!/bin/bash
#SBATCH --mail-type=NONE # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<your-email>
#SBATCH --output=/nfs/home/student.aau.dk/<your-id>/slurm-output/gen-net-%j.out 
#SBATCH --error=/nfs/home/student.aau.dk/<your-id>/slurm-output/gen-net-%j.err
#SBATCH --partition=naples,dhabi,rome
#SBATCH --mem=16G

let "m=1024*1024"
ulimit -v $m

PD=$(pwd)

# # In case you write to auxiliary files, you can work in a temporary folder in /scratch (which is node-local).
# U=$(whoami)
# SCRATCH_DIRECTORY=/scratch/${U}/${SLURM_JOBID}
# mkdir -p ${SCRATCH_DIRECTORY}
# cd ${SCRATCH_DIRECTORY}

source $PD/venv/bin/activate

CONFIG_FILE="${PD}/$1"
OUT_FILE="${PD}/$2"
TOPO_FILE="${PD}/$3"

${PD}/tool_generate.py --conf ${CONFIG_FILE} --output_file ${OUT_FILE} --topology ${TOPO_FILE}

# # Clean up in scratch, if used.
# cd /scratch/${U}
# [ -d "${SLURM_JOBID}" ] && rm -r ${SLURM_JOBID}
