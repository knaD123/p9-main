#!/bin/bash
#SBATCH --output=/nfs/home/student.aau.dk/amad18/slurm-output/createconfs-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/amad18/slurm-output/createconfs-%j.err
#SBATCH --partition=rome,naples
#SBATCH --time=06:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1

PD=$(pwd)

source ${PD}/venv/bin/activate

TOPO="${1}"

python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 0
