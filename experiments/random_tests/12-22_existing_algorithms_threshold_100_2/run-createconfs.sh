#!/bin/bash
#SBATCH --output=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs/%j.out
#SBATCH --error=/nfs/home/student.aau.dk/lkar18/slurm-output/createconfs/%j.err
#SBATCH --partition=naples
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1

source venv/bin/activate

python3 -m pip install -r requirements.txt

TOPO="${1}"

TOPO_RE='.*zoo_(.*).json'

if [[ $TOPO =~ $TOPO_RE ]] ; then
  DEMAND="demands/"${BASH_REMATCH[1]}"_0000.yml"
fi

python3 create_confs.py --threshold 100 --random_seed 2 --topology ${TOPO} --conf confs/12-22_existing_algorithms_threshold_100_2 --result_folder results/12-22_existing_algorithms_threshold_100_2 --demand_file ${DEMAND} --algorithm inout-disjoint --max_memory 4 --path_heuristic semi_disjoint_paths
python3 create_confs.py --threshold 100 --random_seed 2 --topology ${TOPO} --conf confs/12-22_existing_algorithms_threshold_100_2 --result_folder results/12-22_existing_algorithms_threshold_100_2 --demand_file ${DEMAND} --algorithm rsvp-fn
python3 create_confs.py --threshold 100 --random_seed 2 --topology ${TOPO} --conf confs/12-22_existing_algorithms_threshold_100_2 --result_folder results/12-22_existing_algorithms_threshold_100_2 --demand_file ${DEMAND} --algorithm tba-simple
python3 create_confs.py --threshold 100 --random_seed 2 --topology ${TOPO} --conf confs/12-22_existing_algorithms_threshold_100_2 --result_folder results/12-22_existing_algorithms_threshold_100_2 --demand_file ${DEMAND} --max_memory 4 --algorithm tba-complex
python3 create_confs.py --threshold 100 --random_seed 2 --topology ${TOPO} --conf confs/12-22_existing_algorithms_threshold_100_2 --result_folder results/12-22_existing_algorithms_threshold_100_2 --demand_file ${DEMAND} --algorithm rmpls
python3 create_confs.py --threshold 100 --random_seed 2 --topology ${TOPO} --conf confs/12-22_existing_algorithms_threshold_100_2 --result_folder results/12-22_existing_algorithms_threshold_100_2 --demand_file ${DEMAND} --algorithm gft2




