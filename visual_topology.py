import graphviz
import argparse
import os
import json
from mpls_fwd_gen import topology_from_aalwines_json
parser = argparse.ArgumentParser()
parser.add_argument("--topology",type=str, required = True)
args = parser.parse_args()

with open(args.topology, "r") as topo:
    aalwines_top = json.load(topo)

G = graphviz.Graph()
for n in aalwines_top["network"]["routers"]:
    G.node(n["name"])

for edge in aalwines_top["network"]["links"]:
    G.edge(edge["from_router"], edge["to_router"], label=f"{edge['bandwidth']}")

G.render(f"{args.topology}", format="png")
pass




