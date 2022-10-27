#!/bin/bash
#SBATCH --output=/nfs/home/student.aau.dk/amad18/slurm-output/createconfs-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/amad18/slurm-output/createconfs-%j.err
#SBATCH --partition=naples
#SBATCH --time=06:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1

PD=$(pwd)

source ${PD}/venv/bin/activate

python3 -m pip install -r requirements.txt

TOPO="${1}"

python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --algorithm rsvp-fn
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --algorithm inout-disjoint --path_heuristic shortest_path
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --algorithm inout-disjoint --path_heuristic greedy_min_congestion
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --algorithm inout-disjoint --path_heuristic semi_disjoint_paths


