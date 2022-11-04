#!/bin/bash
#SBATCH --mail-type=NONE # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=lkar@student.aau.dk
#SBATCH --output=/nfs/home/cs.aau.dk/lkar18/slurm-output/setup-venv-%j.out
#SBATCH --error=/nfs/home/cs.aau.dk/lkar18/slurm-output/setup-venv-%j.err
#SBATCH --partition=naples,rome,dhabi
#SBATCH --mem=16G

let "m=1024*1024"
ulimit -v $m

PYTHON_PROJECT_FOLDER="~/p9-main"

# Setup a venv (virtual environment) called venv. This creates a folder called 'venv' in the current (project) directory.
python3 -m venv venv
# Activate the venv.
source venv/bin/activate

# Install requirements in the now active venv.
python3 -m pip install -r requirements.txt
