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

CONFIG="$2"

python3 ${PD}/tool_simulate.py --conf ${CONFIG}
