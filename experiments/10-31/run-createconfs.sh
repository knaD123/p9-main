#!/bin/bash
#SBATCH --output=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs-%j.err
#SBATCH --partition=rome
#SBATCH --time=06:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1

source venv/bin/activate

TOPO="${1}"

TOPO_RE='.*zoo_(.*).json'

if [[ $TOPO =~ $TOPO_RE ]] ; then
  DEMAND="demands/"${BASH_REMATCH[1]}"_0000.yml"
fi

python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --demand_file ${DEMAND} --algorithm tba-complex
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --demand_file ${DEMAND} --algorithm rsvp-fn
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic shortest_path
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic greedy_min_congestion
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --demand_file ${DEMAND} --algorithm inout-disjoint --path_heuristic semi_disjoint_paths
python3 create_confs.py --keep_failure_chunks --topology ${TOPO} --conf confs --result_folder results --threshold 100 --demand_file ${DEMAND} --algorithm gft
