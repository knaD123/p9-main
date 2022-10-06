# Code for FBR: Dynamic Memory-Aware Fast Rerouting

This is code for the paper *FBR: Dynamic Memory-Aware Fast Rerouting*. This repository is developed on top of the MPLS-Kit, which is available at https://github.com/juartinv/mplskit.


# How to reproduce the results of the paper
This repository uses python virtual environment and might require the installation: `sudo apt install python3.10-venv`.

To reproduce the results from the paper you can run the script: `./scripts/run_experiments.sh "" all`
This will take a long time to run; see in the next section, how to run a subset of the experiments

This script will set up a virtual python environment and install the required packages. It will create the configurations needed for all topologies in the topologies folder and run all dynamic rerouting algorithms used in the paper. Lastly, it will parse the results which can be seen in the latex folder. Here, the files `memory_failure_data.tex` and `latency_full_median.tex` are the ones in the paper. 


# Options to limit amount of topologies and algorithms
The first argument is a string filter for which topologies to run. For example, `./scripts/run_experiments.sh zoo_A all` would run all the zoo topologies starting with A.
The second argument specifies which algorithms to run. For all algorithms write `all`. If you only want to run e.g. all topologies using rmpls and FBR with memory limit of 4, it would be: `./run_experiments.sh "" rmpls input-disjoint_max-mem=4`.

To recreate some of the results in reasonable time, we suggest you run:
```./run_experiments.sh zoo_A rmpls rsvp-fn tba-simple input-disjoint_max-mem=2 input-disjoint_max-mem=3 input-disjoint_max-mem=4 input-disjoint_max-mem=6 input-disjoint_max-mem=8 input-disjoint_max-mem=16 input-disjoint_max-mem=25 tba-complex_max-mem=2 tba-complex_max-mem=3 tba-complex_max-mem=4 tba-complex_max-mem=6 tba-complex_max-mem=8 tba-complex_max-mem=16 tba-complex_max-mem=25```

The algorithms are: 

 [Config-name]                   | [Name in paper]            |
 |-------------------------------|:--------------------------:|
 | inout-disjoint_max-mem=X      | FBR                        |
 | inout-disjoint-full_max-mem=X | FBR with full backtracking |
 | tba-complex_max-mem=X         | E-CA                       |
 | rmpls                         | R-MPLS                     |
 | gft                           | GFT-CA                     |
 | rsvp-fn                       | RSVP-FN                    |
 | tba-simple                    | B-CA                       |

where X is a integer limit on the number of rules per router per demand.

