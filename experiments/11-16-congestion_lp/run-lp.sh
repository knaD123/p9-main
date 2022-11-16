#!/bin/bash
#SBATCH --mail-type=NONE # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=lkar18@student.aau.dk
#SBATCH --output=/dev/null
#SBATCH --output=/nfs/home/student.aau.dk/lkar18/slurm-output/run-tool-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/lkar18/slurm-output/run-tool-%j.err
#SBATCH --partition=naples
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

source venv/bin/activate

TOPO="$1"

TOPO_RE='.*zoo_(.*).json'

if [[ $TOPO =~ $TOPO_RE ]] ; then
  DEMAND="demands/"${BASH_REMATCH[1]}"_0000.yml"
fi

python3 congestion_lp.py ${TOPO} ${DEMAND}