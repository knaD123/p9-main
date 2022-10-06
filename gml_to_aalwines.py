import mpls_classes
import networkx as nx
import os
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")

args = parser.parse_args()

gml_files = os.listdir(args.input)
type_to_dir = {"fat": "fat_tree", "xpander": "xpander", "bcube": "bcube"}


for gml in tqdm(gml_files):
    type_dir = type_to_dir[gml.split("_")[0]]
    g = nx.read_gml(f"{args.input}/{gml}", label="id")
    n = mpls_classes.Network(g)
    d = n.to_aalwines_json()
    with open(f"{args.output}/{type_dir}/{gml[:-4]}.json", "w") as f:
        f.write(json.dumps(d, indent="\t"))