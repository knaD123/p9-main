from os import listdir
from os.path import isdir, join
import json
with open("topology_info.json", "r") as f:
    all_topologies = json.load(f).keys()

res_dir = "results"

experiments = [x for x in listdir(res_dir) if isdir(join(res_dir, x))]

unfinished_topologies = []
for topology in all_topologies:
    break_topo = False
    for experiment in experiments:
        for alg in [x for x in listdir(join(res_dir, experiment)) if isdir(join(res_dir, experiment, x))]:
            if not topology in listdir(join(res_dir, experiment, alg)):
                unfinished_topologies.append(topology)
                break_topo = True
                break
        if break_topo:
            break

finished_topologies = [x for x in all_topologies if x not in unfinished_topologies]
print(finished_topologies)
