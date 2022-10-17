import json
import argparse
import os
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="")
parser.add_argument("--input_file", type=str, default="")
parser.add_argument("--output_dir", type=str, default="topologies")

args = parser.parse_args()

def __main__():
    if args.input_dir:
        for file in [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]:
            input_file = os.path.join(args.input_dir, file)
            repetita_to_aalwines(input_file)
    elif args.input_file:
        if os.path.isfile(args.input_file):
            repetita_to_aalwines(args.input_file)

def repetita_to_aalwines(file):
    with open(file, "r") as i:

        x = re.sub("\?", "unknown", i.read())
        input_dict = json.loads(x)

        output_dict = dict()
        net_dict = dict()

        # Generate Aalwines link format

        i_edges = input_dict["edges"]
        o_edges = []

        router_to_degree_dict = {router : (0, 0) for router in input_dict["nodes"]}

        for i_edge in i_edges:
            prev_degree, curr_index = router_to_degree_dict[i_edge["orig"]]
            new_degree = prev_degree + 1
            router_to_degree_dict[i_edge["orig"]] = (new_degree, 0)

            o_edge = dict()
            o_edge["bandwidth"] = i_edge["bdw"]
            o_edge["from_router"] = i_edge["orig"]
            o_edge["to_router"] = i_edge["dest"]
            o_edge["latency"] = i_edge["lat"]
            o_edge["weight"] = i_edge["igp"]
            from_interface = f"i{router_to_degree_dict[i_edge['orig']][1]}"
            to_interface = f"i{router_to_degree_dict[i_edge['dest']][1]}"

            o_edge["from_interface"] = from_interface
            o_edge["to_interface"] = to_interface
            is_bi = False
            for e in o_edges:
                if o_edge["bandwidth"] == e["bandwidth"] and o_edge["from_router"] == e["to_router"] and o_edge[
                    "to_router"] == e["from_router"] and o_edge["latency"] == e["latency"]:
                    e["bidirectional"] = True
                    is_bi = True
                    break
            if is_bi:
                continue

            degree, prev_index = router_to_degree_dict[i_edge["orig"]]
            new_index = prev_index + 1
            router_to_degree_dict[i_edge['orig']] = (degree, new_index)
            degree, prev_index = router_to_degree_dict[i_edge["dest"]]
            new_index = prev_index + 1
            router_to_degree_dict[i_edge['dest']] = (degree, new_index)
            o_edges.append(o_edge)

        net_dict["links"] = o_edges

        net_dict["name"] = input_dict["name"]
        # Create routers
        i_routers = input_dict["nodes"]
        aal_routers = []
        for router in i_routers:
            aal_router = dict()
            aal_router["interfaces"] = [
                {"names": [f"i{i}" for i in range(router_to_degree_dict[router][0])], "routing_table": {}}]
            aal_router["location"] = {"latitude": 0.0, "longitude": 0.0}
            aal_router["name"] = router
            aal_routers.append(aal_router)

        net_dict["routers"] = aal_routers
        output_dict["network"] = net_dict
        json_object = json.dumps(output_dict, indent=2)
        output_file = os.path.join(args.output_dir, f"zoo_{input_dict['name']}.json")
        with open(output_file, "w") as o:
            o.write(json_object)
__main__()