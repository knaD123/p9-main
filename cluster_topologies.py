import json
import shutil

with open("topology_info.json", "r") as f:
    topo_info = json.load(f)

white_list = []
for topo, info in topo_info.items():
    if info["num_nodes"] < 80 and info["num_edges"] < 80:
        white_list.append(topo + ".json")

for topo in white_list:
    shutil.copy("topologies/"+topo, "cluster_topologies/"+topo)