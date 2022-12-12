#!/bin/bash
#SBATCH --output=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs-%j.err
#SBATCH --partition=naples
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

python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 5 --path_heuristic greedy_min_congestion
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 5 --path_heuristic semi_disjoint_paths
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 5 --path_heuristic nielsens_heuristic
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 5 --path_heuristic benjamins_heuristic --extra_hops 0
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 5 --path_heuristic benjamins_heuristic --extra_hops 5
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 5 --path_heuristic benjamins_heuristic --extra_hops 10
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 25 --path_heuristic greedy_min_congestion
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 25 --path_heuristic semi_disjoint_paths
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 25 --path_heuristic nielsens_heuristic
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 25 --path_heuristic benjamins_heuristic --extra_hops 0
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 25 --path_heuristic benjamins_heuristic --extra_hops 5
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 25 --path_heuristic benjamins_heuristic --extra_hops 10
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm rsvp-fn
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm tba-simple
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --max_memory 5 --algorithm tba-complex
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --max_memory 25 --algorithm tba-complex
python3 create_confs.py --topology ${TOPO} --conf confs --result_folder results --demand_file ${DEMAND} --algorithm rmpls





