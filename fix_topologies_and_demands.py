import json
import os
import yaml

for topology in os.listdir("zoo"):
    with open(f"zoo/{topology}", "r") as f:
        topo = json.load(f)
        len_nodes = len(topo["nodes"])
        for i in range(len_nodes):
            topo["nodes"][i] = "a" + topo["nodes"][i]

        for edge in topo["edges"]:
            edge["orig"] = "a" + edge["orig"]
            edge["dest"] = "a" + edge["dest"]

    json_object = json.dumps(topo, indent=2)

    with open(f"zoo/{topology}", "w") as f:
        f.write(json_object)

for demand in os.listdir("demands"):
    print(demand)
    with open(f"demands/{demand}", "r") as f:
        demand_list = yaml.load(f, Loader=yaml.BaseLoader)

    with open(f"demands/{demand}", "w") as f:
        flows = []
        for src, tgt, load in demand_list:
            flows.append(f"[{'a' + src}, {'a' + tgt}, {load}]")
        f.write("[")
        for flow in flows:
            f.write(f"{flow}, ")
        f.write("]")
