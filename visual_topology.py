import graphviz
import argparse
import os
from mpls_fwd_gen import topology_from_aalwines_json
parser = argparse.ArgumentParser()
parser.add_argument("--topology",type=str, required = True)
args = parser.parse_args()
top = topology_from_aalwines_json(os.path.join("topologies", args.topology))
G = graphviz.Graph()
for n in top:
    G.node(n)

for (u, v) in top.edges:
    G.edge(u,v)

G.render(f"{args.topology}", format="png")
pass




