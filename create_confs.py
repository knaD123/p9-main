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
# Load required libraries
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
from collections import defaultdict

folder = ""


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def powerset(iterable, m=0):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(m + 1))


def generate_failures(G, threshold=1000, random_seed=1):
    if threshold == 0:
        return [[[]]]
    edges = [list(x) for x in G.edges()]

    _k = K
    F_list = []
    if len(edges) < _k:
        _k = len(edges)
    if conf["only_K_failed_links"]:
        F_list = combinations(edges, _k)
    for k in range(_k + 1):

        number_of_scenarios_of_len_k = math.comb(len(edges), k)
        number_of_choices_to_threshold_ratio = number_of_scenarios_of_len_k / threshold

        if number_of_choices_to_threshold_ratio > 2:
            new_scenarios = []

            while len(new_scenarios) < threshold:
                scenario_to_try = list(random.sample(edges, k))
                scenario_to_try.sort(key=lambda x: f"{x[0]}{x[1]})")

                if scenario_to_try not in new_scenarios:
                    new_scenarios.append(scenario_to_try)

        elif number_of_choices_to_threshold_ratio > 1:
            new_scenarios = random.sample(list(combinations(edges, k)), threshold)

        else:
            new_scenarios = list(combinations(edges, k))

        F_list.extend(new_scenarios)

    return [F_list]

def generate_conf(n, conf_type: str, topofile=None, random_seed=1, per_flow_memory=None, path_heuristic=None,
                  population=None, crossover=None, mutation=None, generations=None,
                  congestion_weight=None, stretch_weight=None, connectedness_weight=None):
    conf_name = conf_type + (f"_max-mem={per_flow_memory}" if per_flow_memory is not None else "") + (
        f"_path-heuristic={path_heuristic}" if path_heuristic is not None else "") + (
                    f"_p={population}" if population is not None else "") + (
                    f"_c={crossover}" if crossover is not None else "") + (
                    f"_m={mutation}" if mutation is not None else "") + (
                    f"_g={generations}" if generations is not None else "") + (
                    f"_uw={congestion_weight}" if congestion_weight is not None else "") + (
                    f"_sw={stretch_weight}" if stretch_weight is not None else "") + (
                    f"_cw={connectedness_weight}" if connectedness_weight is not None else "")
    base_config = {
        # we need extra configuration here!!!!
        "topology": topofile,
        "random_weight_mode": "equal",
        "random_gen_method": 1,
        "php": False,
        "ldp": False,
        "rsvp_tunnels_per_pair": 1,
        "vpn": False,
        "random_seed": random_seed,
        "result_folder": os.path.join(conf["result_folder"], conf_name, topofile.split('/')[-1].split('.')[0]),
        # "flows": [os.path.join(folder, toponame.split("_")[1] + f"_000{x}.yml") for x in range(0,4)]
        "demands": conf["demand_file"],
        "failure_chunk_file": os.path.join(folder, "failure_chunks", "0.yml")
    }
    if per_flow_memory is not None:
        base_config['per_flow_memory'] = per_flow_memory
    if conf_type == 'rsvp_fn':
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
    elif conf_type == 'fbr':
        base_config['method'] = 'fbr'
        base_config['backtrack'] = 'partial'
        base_config['path_heuristic'] = path_heuristic
        if path_heuristic == "nielsens_heuristic" or path_heuristic == "essence":
            base_config["max_utilization"] = conf["max_utilization"]
        if path_heuristic == "essence":
            base_config["population"] = conf["population"]
            base_config["crossover"] = conf["crossover"]
            base_config["mutation"] = conf["mutation"]
            base_config["generations"] = conf["generations"]
        if path_heuristic == "essence_v2":
            base_config["congestion_weight"] = conf["congestion_weight"]
            base_config["stretch_weight"] = conf["stretch_weight"]
            base_config["connectedness_weight"] = conf["connectedness_weight"]
            base_config["population"] = conf["population"]
            base_config["crossover"] = conf["crossover"]
            base_config["mutation"] = conf["mutation"]
            base_config["generations"] = conf["generations"]
    elif conf_type == 'fbr-full':
        base_config['method'] = 'fbr'
        base_config['backtrack'] = 'full'
        base_config['path_heuristic'] = path_heuristic
    elif conf_type == "fbr_sd":
        base_config['method'] = 'fbr_sd'
        base_config['backtrack'] = 'partial'
        base_config['epochs'] = 3
    elif conf_type == "fbr_sd_backtrack":
        base_config['method'] = 'fbr_sd_backtrack'
        base_config['backtrack'] = 'full'
        base_config['epochs'] = 3
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

    p.add_argument("--K", type=int, default=4, help="Maximum number of failed links.")

    p.add_argument("--only_K_failed_links", action="store_true",
                   help="Only creates failure scenarios with K failed links")

    p.add_argument("--threshold", type=int, default=1000, help="Maximum number of failures to generate")

    p.add_argument("--random_seed", type=int, default=1, help="Random seed. Leave empty to pick a random one.")

    p.add_argument("--keep_failure_chunks", action="store_true", default=False,
                   help="Do not generate failure chunks if they already exist")

    p.add_argument("--result_folder", type=str, default='results', help="Folder to store results in")

    p.add_argument("--algorithm", required=True,
                   choices=["rmpls", "fbr", "cfor",
                            "rsvp_fn", "all", "fbr_sd", "fbr_sd_backtrack", "fbr_gmc", "fbr_essence"])

    p.add_argument("--path_heuristic", default="shortest_path",
                   choices=["shortest_path", "greedy_min_congestion", "semi_disjoint_paths", "benjamins_heuristic",
                            "nielsens_heuristic", "essence", "essence_v2", "inverse_cap", "placeholder"])

    p.add_argument("--max_memory", type=int, default=3)

    p.add_argument("--demand_file", type=str, required=True)

    p.add_argument("--conf_name", type=str, default="", help="Custom conf name")

    # ESSENCE GENETIC ALGORITHM ARGUMENTS

    p.add_argument("--max_utilization", type=float, default=10000,
                   help="For Nielsens heuristic. Maximum utilization on every link given 0 failed links.")

    p.add_argument("--population", type=int, default=100, help="Population size")

    p.add_argument("--crossover", type=float, default=0.7, help="crossover for genetic algorithm")

    p.add_argument("--mutation", type=float, default=0.1, help="chance for mutation for genetic algorithm")

    p.add_argument("--generations", type=int, default=100, help="number of generations for genetic algorithm")

    p.add_argument("--congestion_weight", type=float, default=0, help="congestion weight")
    p.add_argument("--stretch_weight", type=float, default=0, help="stretch weight")
    p.add_argument("--connectedness_weight", type=float, default=0, help="connectedness weight")

    args = p.parse_args()
    conf = vars(args)
    topofile = conf["topology"]
    configs_dir = conf["conf"]
    K = conf["K"]
    # L = conf["L"]
    random_seed = conf["random_seed"]
    threshold = conf["threshold"]
    # Ensure the topologies can be found:
    assert os.path.exists(topofile)

    # create main folder for our experiments
    os.makedirs(configs_dir, exist_ok=True)

    gen = topology_from_aalwines_json

    toponame = topofile.split('/')[-1].split(".")[0]
    folder = os.path.join(configs_dir, toponame)
    os.makedirs(folder, exist_ok=True)

    # Set path heuristic if using FBR

    if conf["algorithm"] == "fbr_gmc":
        conf["algorithm"] = "fbr"
        conf["max_memory"] = 3
        conf["path_heuristic"] = "greedy_min_congestion"
    elif conf["algorithm"] == "fbr_essence":
        conf["max_memory"] = 3
        conf["algorithm"] = "fbr"
        conf["path_heuristic"] = "essence"
        conf["population"] = 200
        conf["crossover"] = 0.9
        conf["mutation"] = 0.7
        conf["generations"] = 500
        conf["threshold"] = 100

    G = gen(topofile)
    n = G.number_of_nodes() * G.number_of_nodes()  # tentative number of LSPs

    def create(conf_type, max_memory=None, path_heuristic=None, population=None,
               crossover=None, mutation=None, generations=None, congestion_weight=None, stretch_weight=None,
               connectedness_weight=None):
        dict_conf = generate_conf(n, conf_type=conf_type, topofile=topofile, random_seed=random_seed,
                                  per_flow_memory=max_memory, path_heuristic=path_heuristic,
                                  population=population, crossover=crossover,
                                  mutation=mutation, generations=generations, congestion_weight=congestion_weight,
                                  stretch_weight=stretch_weight, connectedness_weight=connectedness_weight)
        if conf["conf_name"]:
            conf_name = conf["conf_name"]
        else:
            conf_name = "conf_" + conf_type + (f"_random_seed={random_seed}" if random_seed != 1 else "") + (f"_max-mem={max_memory}" if max_memory is not None else "") + (
                f"_path-heuristic={path_heuristic}" if path_heuristic is not None else "") + (
                            f"_p={population}" if population is not None else "") + (
                            f"_c={crossover}" if crossover is not None else "") + (
                            f"_m={mutation}" if mutation is not None else "") + (
                            f"_g={generations}" if generations is not None else "") + (
                            f"_uw={congestion_weight}" if congestion_weight is not None else "") + (
                            f"_sw={stretch_weight}" if stretch_weight is not None else "") + (
                            f"_cw={connectedness_weight}" if connectedness_weight is not None else "") + ".yml"

        path = os.path.join(folder, conf_name)
        # dict_conf["output_file"] = os.path.join(folder, "dp_{}.yml".format(conf_type))
        with open(path, "w") as file:
            documents = yaml.dump(dict_conf, file, Dumper=NoAliasDumper)


    algorithm = conf["algorithm"]
    if algorithm == "all":
        create('rsvp_fn')  # conf file with RSVP(FRR), no RMPLS
        create('tba-simple')
        #    create('hd')
        #    create('cfor-short')
        #    create('cfor-arb')
        create('gft')
        create('kf')
        create('rmpls')
        create('plinko4')

        per_flow_memory = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        heuristics = ["semi_disjoint_paths", "global_weights", "greedy_min_congestion", "shortest_path",
                      "benjamins_heuristic", "nielsens_heuristic", "essence"]
        for mem in per_flow_memory:
            create('tba-complex', mem)
            for h in heuristics:
                create('fbr', mem, h)
                create('fbr-full', mem, h)
    elif algorithm in ['fbr', 'fbr-full']:
        if conf["path_heuristic"] == "nielsens_heuristic":
            create(algorithm, max_memory=conf["max_memory"], path_heuristic=conf["path_heuristic"])
        elif conf["path_heuristic"] == "essence":
            create(algorithm, max_memory=conf["max_memory"], path_heuristic=conf["path_heuristic"],
                   population=conf["population"], crossover=conf["crossover"],
                   mutation=conf["mutation"], generations=conf["generations"])
        elif conf["path_heuristic"] == "essence_v2":
            create(algorithm, max_memory=conf["max_memory"], path_heuristic=conf["path_heuristic"],
                   population=conf["population"], crossover=conf["crossover"],
                   mutation=conf["mutation"], generations=conf["generations"],
                   congestion_weight=conf["congestion_weight"], stretch_weight=conf["stretch_weight"],
                   connectedness_weight=conf["connectedness_weight"])
        else:
            create(algorithm, max_memory=conf["max_memory"], path_heuristic=conf["path_heuristic"])

    elif algorithm == "tba-complex" or algorithm == "fbr_sd" or algorithm == "fbr_sd_backtrack":
        create(algorithm, conf["max_memory"])
    else:
        create(algorithm)

    if not (args.keep_failure_chunks and os.path.exists(os.path.join(folder, "failure_chunks"))):
        # Generate failures
        F_list = generate_failures(G, threshold, random_seed=random_seed)

        failure_folder = os.path.join(folder, "failure_chunks")
        os.makedirs(failure_folder, exist_ok=True)
        i = 0
        for F_chunk in F_list:
            pathf = os.path.join(failure_folder, str(i) + ".yml")
            i += 1
            with open(pathf, "w") as file:
                file.write(str(F_chunk).replace("'", "").replace("(", "[").replace(")", "]"))
