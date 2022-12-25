from os import listdir
from os.path import isdir, join
import json
with open("topology_info.json", "r") as f:
    all_topologies = json.load(f).keys()

experiments = [x for x in listdir("results") if isdir(join("results", x))]

unfinished_topologies = []
for topology in all_topologies:
    for experiment in experiments:
        if not topology in listdir(join("results", experiment)):
            unfinished_topologies.append(topology)

finished_topologies = [x for x in all_topologies if x not in unfinished_topologies]

print(finished_topologies)

