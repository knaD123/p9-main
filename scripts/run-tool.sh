#!/bin/bash
#SBATCH --mail-type=NONE # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=amad18@student.aau.dk
#SBATCH --output=/dev/null
#SBATCH --output=/nfs/home/student.aau.dk/amad18/slurm-output/run-tool-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/amad18/slurm-output/run-tool-%j.err
#SBATCH --partition=naples
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

PD=$(pwd)

source ${PD}/venv/bin/activate

python3 -m pip install -r requirements.txt

TOPO="$1"
CONFIG="$2"
FAILCHUNK="$3"
FLOWS="$4"

#"confs/${TOPO}/conf_${CONFIG}.yml"

python3 ${PD}/tool_simulate.py --conf ${CONFIG} --failure_chunk_file "confs/${TOPO}/failure_chunks/${FAILCHUNK}" --flows_file "confs/${TOPO}/flows/${FLOWS}"
