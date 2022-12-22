#!/bin/bash
#SBATCH --mail-type=NONE # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=lkar18@student.aau.dk
#SBATCH --output=/dev/null
#SBATCH --output=/nfs/home/student.aau.dk/lkar18/slurm-output/run-tool-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/lkar18/slurm-output/run-tool-%j.err
#SBATCH --partition=naples
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

source venv/bin/activate

python3 -m pip install -r requirements.txt

CONFIG="$1"

python3 tool_simulate.py --conf ${CONFIG}
