#!/usr/bin/env python3
# coding: utf-8


"""
Script to make simulations on the MPLS data plane. - v0.1

Script to generate configuration files for the generator/simulator. - v0.1

Copyright (C) 2020-2022.

All rights reserved.

 * This file can not be copied and/or distributed.

All rights reserved.

This program is distributed WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

"""

###############

# Load required libraries
import networkx as nx
import random
import time
import json, yaml
import math
from pprint import pprint
from itertools import chain, count
import argparse
import sys, os
from typing import Union, Set, List, Dict, Tuple
import statistics
import shutil

from networkx import has_path

from mpls_fwd_gen import *

stats = {}

def main(conf):
    if conf["random_topology"]:
        G = generate_topology(conf["random_mode"],
                              conf["num_routers"],
                              weight_mode=conf["random_weight_mode"],
                              gen_method=conf["random_gen_method"],
                              visualize=False,
                              display_tables=False,
                              random_seed=conf["random_seed"]
                              )

    else:
        # Load topology
        G = topology_from_aalwines_json(conf["topology"], visualize=False)
        print("*****************************")
        print(G.graph["name"])

    with open(conf["demands"],"r") as file:
        flows_with_load = [[x,y, int(z)] for [x,y,z] in yaml.load(file, Loader=yaml.BaseLoader)]

    #Sort the flows
    flows_with_load = sorted(flows_with_load, key=lambda x: x[2], reverse=True)[:math.ceil(len(flows_with_load) * conf["take_percent"])]
    conf["loads"] = flows_with_load
    #Remove flow load
    flows = [flow[:2] for flow in flows_with_load]

    with open(conf["topology"]) as f:
        topo_data = json.load(f)

    # Load link capacities
    link_caps = {}
    for f in topo_data["network"]["links"]:
        src = f["from_router"]
        tgt = f["to_router"]
        if src != tgt:
            link_caps[(src,tgt)] = f.get("bandwidth", 0)
            if f["bidirectional"]:
                link_caps[(tgt, src)] = f.get("bandwidth", 0)

    conf["link_caps"] = link_caps

    before_fwd_gen = time.time_ns()

    network = generate_fwd_rules(G, conf,
                                 enable_PHP=conf["php"],
                                 numeric_labels=False,
                                 enable_LDP=conf["ldp"],
                                 enable_RMPLS=conf["enable_RMPLS"],
                                 num_lsps=flows,
                                 tunnels_per_pair=conf["rsvp_tunnels_per_pair"],
                                 enable_services=conf["vpn"],
                                 num_services=conf["vpn_num_services"],
                                 PE_s_per_service=conf["vpn_pes_per_services"],
                                 CEs_per_PE=conf["vpn_ces_per_pe"],
                                 random_seed=conf["random_seed"]
                                 )
    ## Generate MPLS forwarding rules

    stats['fwd_gen_time'] = time.time_ns() - before_fwd_gen

    # save config
    net_dict = network.to_aalwines_json()

    output_file = conf["output_file"]
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(net_dict, f, indent=2)
    elif conf["verbose"]:
        print(" ~~~ FORWARDING DATAPLANE  ~~~ ")
        pprint(net_dict)

    if conf["print_flows"]:
        flows = network.build_flow_table()
        queries = []
        for router_name, lbl_items in flows.items():
            for in_label, tup in lbl_items.items():
                good_sources, good_targets, flow_size = tup
                q_targets = ",.#".join(good_targets)
                queries.append(f"<{in_label}> [.#{router_name}] .* [.#{q_targets}] < > 0 OVER\n")

        output_file = conf["output_file"]
        if output_file:
            with open(output_file + ".q", 'w') as f:
                f.writelines(queries)
        else:
            for q in queries:
                print(q)
        return

    result_folder = conf["result_folder"]
    os.makedirs(result_folder, exist_ok=True)
    result_file = os.path.join(result_folder, "default")

    with open(conf['failure_chunk_file'], 'r') as f:
        failed_set_chunk = yaml.safe_load(f)
        chunk_name = conf['failure_chunk_file'].split('/')[-1].split(".")[0]
        result_file = os.path.join(result_folder, chunk_name)

    # Prepare dictionary to store results

    results = dict()
    results["topology"] = conf["topology"]
    # The algorithm used to generate the forwarding tables
    method_dict = {}
    method = conf["method"]
    method_dict["name"] = method
    if method == "inout-disjoint":
        method_dict["path_heuristic"] = conf["path_heuristic"]
        method_dict["mem_per_router_per_flow"] = conf["per_flow_memory"]
    results["method"] = method_dict
    results["flows"] = "flows.yml"
    results["failure_scenarios"] = "failure_scenarios.yml"
    results["flows#"] = len(flows)
    results["failure_scenarios#"] = len(failed_set_chunk)
    results["ft_gen_time"] = stats["fwd_gen_time"]
    results["simulation_time"] = None
    results["runs"] = list()


    time_before_sim = time.time_ns()
    for failed_set in failed_set_chunk:
        simulation(network, failed_set, flows_with_load, link_caps, results)

    results["simulation_time"] = time.time_ns() - time_before_sim

    with open(os.path.join(result_folder, "flows.yml"), "w") as f:
        fl = [f"[{', '.join(map(str, flow))}]" for flow in flows_with_load]
        f.write(f"[{', '.join(fl)}]")

    with open(os.path.join(result_folder, "failure_scenarios.yml"), "w") as f:
        fs_strings = list()
        for fs in failed_set_chunk:
            mapped_links = [f"[{l[0]}, {l[1]}]" for l in fs]
            fs_strings.append("[" + ", ".join(mapped_links) + "]")

        dump = "[" + ", ".join(fs_strings) + "]"
        f.write(dump)

    with open(os.path.join(result_folder, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

def simulation(network, failed_set, flows: List[Tuple[str, str, int]], link_caps, results):
    print("STARTING SIMULATION")
    print(failed_set)

    # Mangle the topology:
    def filter_node(n):
        return True

    F = [tuple(x) for x in failed_set]  # why need this??

    def filter_edge(n1, n2):
        if (n1, n2) in F or (n2, n1) in F:
            return False
        return True

    # Compute subgraph
    view = nx.subgraph_view(network.topology, filter_node=filter_node, filter_edge=filter_edge)
    links = len(network.topology.edges)

    # Instantiate simulator object
    s = Simulator(network, trace_mode="links", restricted_topology=view, random_seed=conf["random_seed_sim"])

    verbose=conf["verbose"]
    s.run(flows, verbose=verbose)



    ### OUTPUT RESULTS

    def util_poly(u):
        return u*u

    def util_exp_2(u):
        return 2**u

    def util_exp_4(u):
        return 4**u

    res_dir = dict()

    (successful_flows, total_flows, codes) = s.success_rate(exit_codes=True)

    failed_traces = []
    for src,tgt,load in flows:
        (traces,succes) = s.trace_routes[src,tgt,load]
        if not succes:
            failed_traces = failed_traces + [(src,tgt), traces]

    routers: List[Router] = list(network.routers.values())
    router_memory = [sum(len(rule) for rule in router.LFIB.values()) for router in routers]
    router_memory_str = str(router_memory).replace(' ', '')

    from statistics import median_low as median
    hops = str(list(s.num_hops.values())).replace(' ', '')

    # Compute:
    # link absolute utilization
    # avg path stretch (weighted by demand size, to reflect average for a single packet)
    # rate of succesfully forwarded packets.

    path_stretch = 0
    normalization_factor = 0
    util_dict_abs = {}
    successful_packets = 0
    possible_packets = 0
    for (src, tgt, load), (trace, res) in s.trace_routes.items():
        if res:
            possible_packets += load
            successful_packets += load
            shortest_path = nx.shortest_path_length(view, source=src, target=tgt)
            normalization_factor += load
            path_stretch += (len(trace) - 1) / shortest_path * load
        elif has_path(view, src, tgt):
            possible_packets += load

        trace_fixed = zip(trace, trace[1:])
        for u, v, in trace_fixed:
            # utilization
            util_dict_abs[(u, v)] = util_dict_abs.get((u, v), 0) + load

    if possible_packets > 0:
        success_rate = successful_packets / possible_packets
    else:
        success_rate = 1

    if normalization_factor:
        path_stretch = path_stretch / normalization_factor

    # Compute relative link utilization
    util_dict_rel = {}
    for link, cap in link_caps.items():
        util_abs = util_dict_abs.get(link, 0)
        util_rel = util_abs / cap
        util_dict_rel[link] = util_rel

    # Compute clean packets
    clean_packets = 0
    for (src, tgt, load), (trace, res) in s.trace_routes.items():
        if res:
            trace_fixed = zip(trace, trace[1:])
            for link in trace_fixed:
                if util_dict_rel[link] > 1:
                    break
            else:
                clean_packets += load



    # Compute median and maximum congestion
    median_cong = median(util_dict_rel.values())
    max_cong = max(util_dict_rel.values())

    util_poly_score = sum([link_caps[link] * util_poly(u) for (link, u) in util_dict_rel.items()])
    util_exp_score_2 = sum([link_caps[link] * util_exp_2(u) for (link, u) in util_dict_rel.items()])
    util_exp_score_4 = sum([link_caps[link] * util_exp_4(u) for (link, u) in util_dict_rel.items()])

    res_dir["failed_links"] = F
    res_dir["failed_links#"] = len(F)
    if possible_packets > 0:
        res_dir["clean_packets_ratio"] = clean_packets / possible_packets
    else:
        res_dir["clean_packets_ratio"] = 1
    res_dir["delivered_packet_rate"] = success_rate
    res_dir["median_congestion"] = median_cong
    res_dir["max_congestion"] = max_cong
    res_dir["path_stretch"] = path_stretch
    res_dir["util_poly_score"] = util_poly_score
    res_dir["util_exp_score_2"] = util_exp_score_2
    res_dir["util_exp_score_4"] = util_exp_score_4


    results["runs"].append(res_dir)

    print(f"SIMULATION FINISHED - FAILED: {s.count_connected - successful_flows}")
    if conf["print_fails"]:
        print("Failed traces:")
        for fails in failed_traces:
            print(fails)

if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Command line utility to generate MPLS forwarding rules.')

    # general options
    p.add_argument("--conf", type=str, default="", help="Load configuration file.")

    group1 = p.add_mutually_exclusive_group()
    group1.add_argument("--topology", type=str, default="", help="File with existing topology to be loaded.")
    # random topology options
    group1.add_argument("--random_topology", action="store_true", help="Generate random topology. Defaults False.")
    p.add_argument("--random_mode", type=str, default="random.log_degree",
                   help="custom, random.log_degree, random.large_degree")
    p.add_argument("--num_routers", type=int, default=6, help="Number of routers in topology.")
    p.add_argument("--random_weight_mode", type=str, default="random", help="random, equal or distance")
    p.add_argument("--random_gen_method", type=int, default=1, help="Method for generating the random topology")

    # Generator options
    p.add_argument("--php", action="store_true", help="Enable Penultimate Hop Popping. Defaults False.")
    p.add_argument("--ldp", action="store_true",
                   help="Enable Label Distribution Protocol (LDP). Defaults False. This doesn't scale well.")
    p.add_argument("--rsvp_num_lsps", type=int, default=2,
                   help="Number of (random) LSPS to compute for RSVP only, if enabled. Defaults to 2")
    p.add_argument("--rsvp_tunnels_per_pair", type=int, default=5,
                   help="Number of (random) tunnels between same endpoints. RSVP only, if enabled. Defaults to 5")
    p.add_argument("--enable_RMPLS", action="store_true",
                   help="Use experimental RMPLS recursive protection (LFIB post processing). Defaults False")

    p.add_argument("--vpn", action="store_true", help="Enable MPLS VPN generic services. Defaults False. ")
    p.add_argument("--vpn_num_services", type=int, default=1,
                   help="Number of (random) MPLS VPN services, if enabled. Defaults to 1")
    p.add_argument("--vpn_pes_per_services", type=int, default=3,
                   help="Number of PE routers allocated to each MPLS VPN service, if enabled. Defaults to 3")
    p.add_argument("--vpn_ces_per_pe", type=int, default=1,
                   help="Number of CE to attach to each PE serving a VPN, if enabled. Defaults to 1")

    p.add_argument("--random_seed", type=int, default=random.randint(0, 99999999),
                   help="Random seed for genrating the data plane. Leave empty to pick a random one.")
    p.add_argument("--random_seed_sim", type=int, default=random.randint(0, 99999999),
                   help="Random seed for simulation execution. Leave empty to pick a random one.")

    method_parser = p.add_subparsers(dest="method", help="Which method for MPLS plane generation")

    hba_parser = method_parser.add_parser("hd")

    tba_parser = method_parser.add_parser("tba")
    tba_parser.add_argument('--max_memory', default=0, help="Max memory allowed on any router")

    gft_parser = method_parser.add_parser("gft")

    inout_disjoint_parser = method_parser.add_parser("inout-disjoint")
    inout_disjoint_parser.add_argument('--epochs', default=1000, help="Epochs")
    inout_disjoint_parser.add_argument('--max_memory', default=0, help="Max memory allowed on any router")


    cfor_parser = method_parser.add_parser("cfor")
    cfor_parser.add_argument('--path', choices=['shortest', 'arborescence', 'disjoint'], default='shortest', help='Method for generating paths between switches on same layer')
    cfor_parser.add_argument('--num_down_paths', default=2, help='How many semi-disjoint paths down in layers')
    cfor_parser.add_argument('--num_cycling_paths', default=4, help='How many semi-disjoint paths between two nodes in the same layer')
    cfor_parser.add_argument('--max_memory', default=0, help="Max memory allowed on any router")

    rsvp_parser = method_parser.add_parser("rsvp")
    rsvp_parser.add_argument("--protection", choices=['none', 'facility-node', 'facility-link'], help="Protection bypasses to use")


    p.add_argument("--output_file", type=str, default="",
                   help="Path of the output file, to store forwarding configuration. Defaults to print on screen.")

    # Simulator options
    p.add_argument("--result_folder", type=str, default="", help="Path to the folder of simulation result files. Defaults to print on screen.")
    p.add_argument("--print_flows", action="store_true", help="Print flows instead of running simulation. Defaults False.")
    p.add_argument("--verbose", action="store_true", help="Remove verbosity")
    p.add_argument("--take_percent", type=float, default=0.20, help="What percentage of biggest flows to take")
    p.add_argument("--print_fails", action="store_true", help="Print failed traces")

    args = p.parse_args()

    print(args)
    conf = vars(args)  # arguments as dictionary
    conf_overrride = conf.copy()
#    known_parameters = list(conf.keys()).copy()

    # Configuration Load
    if args.conf:
        print(f"Reading configuration from {args.conf}. Remaining options will override the configuration file contents.")
        if args.conf.endswith(".yaml") or args.conf.endswith(".yml"):
            with open(args.conf, 'r') as f:
                conf_new = yaml.safe_load(f)
                conf.update(conf_new)
                print("Configuration read from file.")
        elif args.conf.endswith(".json"):
            # try json
            with open(args.conf, 'r') as f:
                conf_new = json.load(f)
                conf.update(conf_new)
                print("Configuration read from file.")
        else:
            raise Exception("Unsupported file format. Must be YAML or JSON.")

    # Deal with the topology
    if args.topology and not args.random_topology:
        print(f"Reading topology from {args.topology}. ")
        conf["topology"] = args.topology

    elif args.random_topology:
        conf["random_topology"] = True
        if args.num_routers < 6:
            raise Exception("You need at least 6 routers.")
        conf["random_mode"] = args.random_mode
        conf["num_routers"] = args.num_routers
        conf["random_weight_mode"] = args.random_weight_mode
        conf["random_gen_method"] = args.random_gen_method


    elif conf["random_topology"]:
        if args.num_routers < 6:
            raise Exception("You need at least 6 routers.")
        conf["random_mode"] = args.random_mode
        conf["num_routers"] = args.num_routers
        conf["random_weight_mode"] = args.random_weight_mode
        conf["random_gen_method"] = args.random_gen_method
    else:
        raise Exception("One (and only one) of topology or random_topology must be selected.")

    if "protection" in conf and conf["protection"] in ["None", "none"]:
        conf["protection"] = None

    # raise errors if there is any unknown entry...
    # for a in conf.keys():
    #     if a not in known_parameters:
    #         raise Exception(f"Unknown argument {a}.")

    # Override config options with cli explicit values
    for sysarg in sys.argv:
        # this argument was explicitly overriden!
        if sysarg.startswith('--'):
            conf[sysarg[2:]] = conf_overrride[sysarg[2:]]

    pprint(conf)
    main(conf)
