import os
import re
import json
for f in os.listdir("zoo_demands"):
    print(f)
    top = re.search("(^[^.]*)", f)[1]
    id = re.search(f"{top}.([^.]*)", f)[1]
    output = f"demands/{top}_{id}.yml"

    with open(f"zoo/{top}.json", "r") as t,open(f"zoo_demands/{f}", "r") as i:
        x = re.sub("\?", "unknown", t.read())
        top = json.loads(x)

        for index, value in enumerate(top["nodes"]):
            new_string = re.sub(",", "", value)
            top["nodes"][index] = new_string

        for index, value in enumerate(top["edges"]):
            new_string = re.sub(",", "", value["orig"])
            top["edges"][index]["orig"] = new_string

            new_string = re.sub(",", "", value["dest"])
            top["edges"][index]["dest"] = new_string

        lines = i.readlines()

    flow_id_to_router_dict = dict()
    curr_id = 0
    for router in top["nodes"]:
        flow_id_to_router_dict[curr_id] = router
        curr_id += 1

    with open(output, "w") as o:
        flows = []
        for l in lines[2:]:
            src, tgt, load  = re.search("^demand_[0-9]* ([0-9]*) ([0-9]*) ([0-9]*)", l).groups()
            flows.append(f"[{flow_id_to_router_dict[int(src)]}, {flow_id_to_router_dict[int(tgt)]}, {load}]")
            pass
        o.write("[")
        for flow in flows:
            o.write(f"{flow}, ")
        o.write("]")
