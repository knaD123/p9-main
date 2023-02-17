import argparse
import re
import sys

import networkx as nx
import matplotlib.pyplot as plt
import random
import time
# import jsonschema
import json
import math
import copy
from pprint import pprint
from itertools import chain, count, islice
import numpy as np
import yaml
from networkx import Graph
import os
from os import path
import pandas as pd

# Kør det her først
# ~/omnetpp-6.0.1/samples/inet/examples/mpls/arpanet196912/results$ opp_scavetool export -F CSV-R -o scenario_1.csv *.vec

def main(conf):
    data = conf['results']
    data = pd.read_csv(data, converters = {'module': parse_string, 'vectime': parse_ndarray, 'vecvalue': parse_ndarray})

    network_row = data[data["attrname"] == "network"]
    topo_name = network_row["attrvalue"].iloc[0]

    #topo_name = network_name.iloc[0]
    with open(f'omnet_results_parser/link_interfaces/{topo_name}_link_interfaces.json', "r") as f:
         interface_dict = json.load(f)

    # Remove none indeces
    data = data[data['vecvalue'].notnull()]


    data = data[['vectime', 'vecvalue', 'module']]

    vecvalue = data['vecvalue']
    vectime = data['vectime']
    module = data['module']

    prune_vecs = zip(vecvalue, vectime)

    min_len_lst = []
    for (v, t) in prune_vecs:
        min_len_lst.append(min(len(v),len(t)))

    vecvalue = [v[:min_len] for v, min_len in zip(vecvalue, min_len_lst)]
    vectime = [v[:min_len] for v, min_len in zip(vectime, min_len_lst)]

    for i in range(len(vecvalue)):
        plt.plot(vectime[i], vecvalue[i], label = interface_dict[map_string(module.values[i])])

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Utilization over time')
    plt.legend()

    plots_folder = "omnet_results_parser/plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    plt.savefig(f'omnet_results_parser/plots/{topo_name}_{conf["scenario"]}.png')

def map_string(string):
    split = string.split(".")
    dict_key = split[1] + "." + split[2].replace("$o","")
    return dict_key

def parse_if_number(s):
    try: return float(s)
    except: return True if s=="true" else False if s=="false" else s if s else None

def parse_ndarray(s):
    return np.fromstring(s, sep=' ') if s else None

def parse_string(s):
    return str(s) if s else None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, required=True, help="Results in a csv file")
    p.add_argument("--scenario", type=str, required=True, help="Scenario identifier")

    conf = vars(p.parse_args())
    main(conf)