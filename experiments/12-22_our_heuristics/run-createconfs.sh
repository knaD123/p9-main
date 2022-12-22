#!/bin/bash
#SBATCH --output=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs/%j.out
#SBATCH --error=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs/%j.err
#SBATCH --partition=dhabi
#SBATCH --time=06:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1

source venv/bin/activate

python3 -m pip install -r requirements.txt

TOPO="${1}"

TOPO_RE='.*zoo_(.*).json'

if [[ $TOPO =~ $TOPO_RE ]] ; then
  DEMAND="demands/"${BASH_REMATCH[1]}"_0000.yml"
fi

python3 create_confs.py --topology ${TOPO} --conf confs/12-22_our_heuristics --result_folder results/12-22_our_heuristics --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 4 --path_heuristic shortest_path
python3 create_confs.py --topology ${TOPO} --conf confs/12-22_our_heuristics --result_folder results/12-22_our_heuristics --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 4 --path_heuristic greedy_min_congestion
python3 create_confs.py --topology ${TOPO} --conf confs/12-22_our_heuristics --result_folder results/12-22_our_heuristics --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 4 --path_heuristic semi_disjoint_paths
python3 create_confs.py --topology ${TOPO} --conf confs/12-22_our_heuristics --result_folder results/12-22_our_heuristics --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 4 --path_heuristic nielsens_heuristic
python3 create_confs.py --topology ${TOPO} --conf confs/12-22_our_heuristics --result_folder results/12-22_our_heuristics --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 4 --path_heuristic benjamins_heuristic --extra_hops 5
python3 create_confs.py --topology ${TOPO} --conf confs/12-22_our_heuristics --result_folder results/12-22_our_heuristics --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 4 --path_heuristic essence --population 100 --crossover 0.7 --mutation 0.1 --generations 100
python3 create_confs.py --topology ${TOPO} --conf confs/12-22_our_heuristics --result_folder results/12-22_our_heuristics --demand_file ${DEMAND} --algorithm gft
python3 create_confs.py --topology ${TOPO} --conf confs/12-22_our_heuristics --result_folder results/12-22_our_heuristics --demand_file ${DEMAND} --algorithm rmpls





