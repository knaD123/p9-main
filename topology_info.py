import json
import os

dump = {}
for topology in os.listdir("topologies"):
    with open(os.path.join("topologies", topology), "r") as f:
        topology_info = json.load(f)
        relevant_info = {}
        relevant_info["num_nodes"] = len(topology_info["network"]["routers"])

        #Bidirectional edges are counted once
        relevant_info["num_edges"] = len(topology_info["network"]["links"])
        relevant_info["degree"] = relevant_info["num_edges"] / relevant_info["num_nodes"]

    topology_name = topology.split(".json")[0]
    dump[topology_name] = relevant_info

with open("topology_info.json", "w") as f:
    json.dump(dump, f, indent=2)