#!/usr/bin/env python3
# coding: utf-8

"""
Script to generate configuration files for the generator/simulator. - v0.1

Copyright (C) 2020-2022.

All rights reserved.

 * This file can not be copied and/or distributed.

All rights reserved.

This program is distributed WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

"""
#Load required libraries
import random
import shutil
import time
import json 
import yaml
import math
import argparse
import sys, os
import networkx as nx
from mpls_fwd_gen import *
from itertools import chain, combinations
from functools import reduce

folder = ""


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def partition(lst, division):
    n = math.ceil(len(lst) / division)
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

def powerset(iterable, m = 0):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(m+1))

def generate_failures_random(G, n, division = None, random_seed = 1):
    # create Failure information from sampling.
    F_list = []
    random.seed(random_seed)

    lis = list(map(lambda x: math.comb(G.number_of_edges(),x) , range(K+1)))

    # Caps failure scenarios to n for all K.
    p = [min(f, n) for f in lis]

    # # compute numbers proportional to failure scenarios per k
    # r = reduce(lambda a,b: a+b, lis)
    # p = list(map(lambda x: math.ceil(n*x/r),lis))
    #
    # excess = reduce(lambda a,b: a+b, p) - n
    # p[-1] -= excess   #adjust.
    edges = list(G.edges())
    _k = K
    if len(edges) < _k:
        _k = len(edges)
    if conf["only_K_failed_links"]:
        while len(F_list) < n:
            random_scenario = tuple(random.sample(edges, _k))
            if random_scenario not in F_list:
                F_list.append(random_scenario)
    else:
        F_list.append([])
        for k in range(1,_k+1):
            for f in range(p[k]):
                failed = set()
                while (len(failed) < k):
                    e = random.choice(edges)
                    if e not in failed:
                        failed.add(e)
                F_list.append(tuple(failed))
    if division:
        P = partition([list(x) for x in F_list], division)
        return P
    return [list(x) for x in F_list]

def generate_failures_all(G, division = None, random_seed = 1):

    # create Failure information from sampling.
    edges = [list(x) for x in G.edges()]
    _k = K
    if len(edges) < _k:
        _k =  len(edges)
    if conf["only_K_failed_links"]:
        F_list = combinations(edges, _k)
    else:
        F_list = list(powerset(edges, m=_k))
    if division:
        P = partition([list(x) for x in F_list], division)
        return P

    return [list(x) for x in F_list]


def generate_conf(n, conf_type: str, topofile = None, random_seed = 1, per_flow_memory = None, path_heuristic = None, extra_hops = None):
    conf_name = conf_type + (f"_max-mem={per_flow_memory}" if per_flow_memory is not None else "") + (f"_path-heuristic={path_heuristic}" if path_heuristic is not None else "") + (f"{extra_hops}" if extra_hops is not None else "")
    base_config = {
    #we need extra configuration here!!!!
        "topology": topofile,
        "random_weight_mode": "equal",
        "random_gen_method": 1,
        "php": False,
        "ldp": False,
        "rsvp_tunnels_per_pair": 1,
        "vpn": False,
        "random_seed": random_seed,
        "result_folder": os.path.join(conf["result_folder"], conf_name, topofile.split('/')[-1].split('.')[0]),
        #"flows": [os.path.join(folder, toponame.split("_")[1] + f"_000{x}.yml") for x in range(0,4)]
        "demands": conf["demand_file"],
        "failure_chunk_file": os.path.join(folder, "failure_chunks", "0.yml")
    }
    if per_flow_memory is not None:
        base_config['per_flow_memory'] = per_flow_memory
    if conf_type == 'rsvp-fn':
        base_config['method'] = 'rsvp'
        base_config['protection'] = 'facility-node'
    elif conf_type == "tba-simple":
        base_config["method"] = "tba"
        base_config['path'] = 'simple'
        base_config['per_flow_memory'] = -1
    elif conf_type == "tba-complex":
        base_config["method"] = "tba"
        base_config['path'] = 'complex'
    elif conf_type == 'tba-multi':
        base_config['method'] = 'tba'
        base_config['path'] = 'multi'
    elif conf_type == 'hd':
        base_config['method'] = 'hd'
    elif conf_type == 'gft':
        base_config['method'] = 'gft'
    elif conf_type == 'cfor-short':
        base_config['method'] = 'cfor'
        base_config['path'] = 'shortest'
    elif conf_type == 'cfor-arb':
        base_config['method'] = 'cfor'
        base_config['path'] = 'arborescence'
    elif conf_type == 'cfor-disjoint':
        base_config['method'] = 'cfor'
        base_config['path'] = 'disjoint'
        base_config['num_down_paths'] = 2
        base_config['num_cycling_paths'] = 0
    elif conf_type == 'kf':
        base_config['method'] = 'kf'
    elif conf_type == 'inout-disjoint':
        base_config['method'] = 'inout-disjoint'
        base_config['backtrack'] = 'partial'
        base_config['path_heuristic'] = path_heuristic
        if extra_hops is not None:
            base_config["extra_hops"] = extra_hops
    elif conf_type == 'inout-disjoint-full':
        base_config['method'] = 'inout-disjoint'
        base_config['backtrack'] = 'full'
        base_config['path_heuristic'] = path_heuristic
        if extra_hops is not None:
            base_config["extra_hops"] = extra_hops
    elif conf_type == 'rmpls':
        base_config['enable_RMPLS'] = True
        base_config['protection'] = None
        base_config['method'] = 'rsvp'
    elif conf_type == 'plinko4':
        base_config['method'] = 'rsvp'
        base_config['protection'] = 'plinko'
    else:
        raise Exception(f"Conf type {conf_type} not known")
    return base_config


if __name__ == "__main__":
    # #general options
    p = argparse.ArgumentParser(description='Command line utility to generate MPLS simulation specifications.')

    p.add_argument("--topology", type=str, help="File with existing topology to be loaded.")

    p.add_argument("--conf", type=str, help="where to store created configurations. Must not exists.")

    p.add_argument("--K", type=int, default = 4, help="Maximum number of failed links.")

    p.add_argument("--only_K_failed_links", action="store_true", help="Only creates failure scenarios with K failed links")

    p.add_argument("--threshold",type=int, default = 1000, help="Maximum number of failures to generate")

    p.add_argument("--division",type=int, default = 100000, help="chunk size; number of failure scenarios per worker.")

    p.add_argument("--random_seed",type=int, default = 1, help="Random seed. Leave empty to pick a random one.")

    p.add_argument("--keep_failure_chunks", action="store_true", default=False, help="Do not generate failure chunks if they already exist")

    p.add_argument("--keep_flows", action="store_true", default=False, help="Do not generate flows if they already exist")

    p.add_argument("--result_folder", type=str, default='results', help="Folder to store results in")

    p.add_argument("--algorithm", required=True, choices=["tba-simple", "tba-complex", "gft", "kf", "rmpls", "plinko4", "inout-disjoint", "cfor", "rsvp-fn", "all"])

    p.add_argument("--path_heuristic", default="shortest_path", choices=["shortest_path", "greedy_min_congestion", "semi_disjoint_paths", "benjamins_heuristic", "nielsens_heuristic"])

    p.add_argument("--extra_hops", type=int)

    p.add_argument("--max_memory", type=int, default=3)

    p.add_argument("--demand_file", type=str, required=True)


    args = p.parse_args()
    conf = vars(args)

    topofile = conf["topology"]
    configs_dir = conf["conf"]
    K = conf["K"]
    #L = conf["L"]
    random_seed = conf["random_seed"]
    division = conf["division"]
    threshold = conf["threshold"]
    # Ensure the topologies can be found:
    assert os.path.exists(topofile)

    # create main folder for our experiments
    os.makedirs(configs_dir, exist_ok = True)

    # Load
    if topofile.endswith(".graphml"):
        gen = lambda x: nx.Graph(nx.read_graphml(x))
    elif topofile.endswith(".json"):
        gen = topology_from_aalwines_json
    else:
        exit(1)

    print(topofile)
    toponame = topofile.split('/')[-1].split(".")[0]
    folder = os.path.join(configs_dir,toponame)
    os.makedirs(folder, exist_ok = True)

    G = gen(topofile)
    n = G.number_of_nodes() * G.number_of_nodes()    #tentative number of LSPs

    #Generate flows
    #flows = []
    #for src in list(G.nodes):
    #    tgt = random.choice(list(set(G.nodes) - {src}))
    #    flows.append((src, tgt))

    #with open(os.path.join(folder, "flows.yml"), "w") as file:
    #    yaml.dump(flows, file, default_flow_style=True, Dumper=NoAliasDumper)

    def create(conf_type, max_memory = None, path_heuristic=None, extra_hops = None):
        dict_conf = generate_conf(n, conf_type = conf_type, topofile = topofile, random_seed = random_seed, per_flow_memory=max_memory, path_heuristic = path_heuristic, extra_hops=extra_hops)
        conf_name = "conf_" + conf_type + (f"_max-mem={max_memory}" if max_memory is not None else "") + (
            f"_path-heuristic={path_heuristic}" if path_heuristic is not None else "") + (f"{extra_hops}" if extra_hops is not None else "") + ".yml"

        path = os.path.join(folder, conf_name)
       # dict_conf["output_file"] = os.path.join(folder, "dp_{}.yml".format(conf_type))
        with open(path, "w") as file:
            documents = yaml.dump(dict_conf, file, Dumper=NoAliasDumper)

    algorithm = conf["algorithm"]
    if algorithm == "all":
        create('rsvp-fn')    # conf file with RSVP(FRR), no RMPLS
        create('tba-simple')
        #    create('hd')
        #    create('cfor-short')
        #    create('cfor-arb')
        create('gft')
        create('kf')
        create('rmpls')
        create('plinko4')

        per_flow_memory = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        heuristics = ["semi_disjoint_paths", "global_weights", "greedy_min_congestion", "shortest_path", "benjamins_heuristic","nielsens_heuristic"]
        for mem in per_flow_memory:
            create('tba-complex', mem)
            for h in heuristics:
                create('inout-disjoint', mem, h)
                create('inout-disjoint-full', mem, h)
    elif algorithm in ['inout-disjoint', 'inout-disjoint-full']:
        if conf["path_heuristic"] == "benjamins_heuristic":
            create(algorithm, max_memory=conf["max_memory"], path_heuristic=conf["path_heuristic"], extra_hops=conf["extra_hops"])
        else:
            create(algorithm, max_memory=conf["max_memory"], path_heuristic=conf["path_heuristic"])

    elif algorithm == "tba-complex":
        create(algorithm, conf["max_memory"])
    else:
        create(algorithm)

    if not (args.keep_failure_chunks and os.path.exists(os.path.join(folder, "failure_chunks"))):
        # Generate failures
        if math.comb(G.number_of_edges(), K) > threshold:
            F_list = generate_failures_random(G, threshold, division = division, random_seed = random_seed)
        else:
            F_list = generate_failures_all(G,  division = division, random_seed = random_seed)

        failure_folder = os.path.join(folder, "failure_chunks")
        os.makedirs(failure_folder, exist_ok = True)
        i = 0
        for F_chunk in F_list:
            pathf = os.path.join(failure_folder, str(i)+".yml")
            i+=1
            with open(pathf, "w") as file:
                file.write(str(F_chunk).replace("'", "").replace("(","[").replace(")", "]"))