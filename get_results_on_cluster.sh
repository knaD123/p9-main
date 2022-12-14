#!/bin/bash
#SBATCH --mail-type=NONE # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=lkar18@student.aau.dk
#SBATCH --output=/dev/null
#SBATCH --output=/nfs/home/student.aau.dk/lkar18/slurm-output/run-tool-%j.out
#SBATCH --error=/nfs/home/student.aau.dk/lkar18/slurm-output/run-tool-%j.err
#SBATCH --partition=naples
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

python3 latexplots.py --input_dir results --output_dir latexfiles --path_stretch --max_congestion --delivered_packet_rate --algorithms "inout-disjoint_max-mem=5_path-heuristic=semi_disjoint_paths, rmpls, rsvp-fn,  tba-simple, tba-complex_max-mem=5"
