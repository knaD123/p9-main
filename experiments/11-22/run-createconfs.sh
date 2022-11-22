#!/bin/bash
#SBATCH --output=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs-%j.err
#SBATCH --partition=naples
#SBATCH --time=06:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1

source venv/bin/activate

python3 -m pip install -r requirements.txt

TOPO="${1}"

TOPO_RE='.*zoo_(.*).json'

if [[ $TOPO =~ $TOPO_RE ]] ; then
  DEMAND="demands/"${BASH_REMATCH[1]}"_0000.yml"
fi

python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm rsvp-fn
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic greedy_min_congestion
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic semi_disjoint_paths
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic benjamins_heuristic --extra_hops 0
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic benjamins_heuristic --extra_hops 1
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic benjamins_heuristic --extra_hops 2
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic benjamins_heuristic --extra_hops 3
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic benjamins_heuristic --extra_hops 10
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic benjamins_heuristic --extra_hops 20


