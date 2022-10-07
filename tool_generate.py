#!/usr/bin/env python3
# coding: utf-8


"""
Script to generate the MPLS data plane. - v0.1

Script to generate configuration files for the generator/simulator. - v0.1

Copyright (C) 2020-2022.

All rights reserved.

 * This file can not be copied and/or distributed.

All rights reserved.

This program is distributed WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


Arguments
----------
    # # Arguments

    # conf                Configuration file to load, if any. File configurations
                          override defaults, and command line arguments override
                          file configurations.

    # topology            File with existing topology to be loaded. Mutually
                          incompatible with random_topology

    # random_topology     Generate random topology. Defaults to true, and is
                          incompatible with 'topology'

    # random_mode         For random topologies, how to obtain the number of edges.
                          One of: custom, random.log_degree, random.large_degree

    # num_routers         Number of routers in the random topology. Defaults to 6.

    # random_weight_mode  How to generate link weights. One of random, equal or
                          distance.

    # random_gen_method   Method for generating the random topology. 1 or 0.

    # php                 Enable Penultimate Hop Popping. Defaults true.

    # ldp                 Enable Label Distribution Protocol (LDP). Defaults true.

    # rsvp                Enable Resource Reservation Protocol with Traffic
                          Engineering extensions (RSVP-TE). Defaults true.

    rsvp_num_lsps         Number of (random) LSPS to compute for RSVP only,
                          if enabled. Defaults to 2

    rsvp_tunnels_per_pair Number of (random) tunnels between same endpoints.
                          RSVP only, if enabled. Defaults to 5

    vpn                   Enable MPLS VPN generic services. Defaults true.

    vpn_num_services      Number of (random) MPLS VPN services, if enabled.
                          Defaults to 1

    vpn_pes_per_services  Number of PE routers allocated to each MPLS VPN
                          service, if enabled. Defaults to 3

    vpn_ces_per_pe        Number of CE to attach to each PE serving a VPN, if
                          enabled. Defaults to 1

    random_seed           Random seed. Leave empty to pick a random one.

    output_file           Path of the output file, to store forwarding
                          configuration. Defaults to print on screen.


Returns
-------
 Prints or save the computed (AalWiNes) JSON file with MPLS forwarding rules.

"""

###############

#Load required libraries
import networkx as nx
import random
import time
import json, yaml
import math
from pprint import pprint
from itertools import chain, count
import argparse
import sys

from mpls_fwd_gen import *

def main(conf):
    if conf["random_topology"]:
        G = generate_topology(conf["random_mode"],
                          conf["num_routers"],
                          weight_mode = conf["random_weight_mode"],
                          gen_method = conf["random_gen_method"],
                          visualize = False,
                          display_tables = False,
                          random_seed = conf["random_seed"]
                         )

    else:
        # Load topology
        G = topology_from_aalwines_json(conf["topology"], visualize = False)
        print("*************cuack****************")
        print(G.graph["name"])

    flows = []

    ## Generate MPLS forwarding rules
    network = generate_fwd_rules(G, conf,
                                 enable_PHP = conf["php"],
                                 numeric_labels = False,
                                 enable_LDP = conf["ldp"],
                                 enable_RMPLS = conf["enable_RMPLS"],
                                 num_lsps = conf["rsvp_num_lsps"],
                                 tunnels_per_pair = conf["rsvp_tunnels_per_pair"],
                                 enable_services = conf["vpn"],
                                 num_services = conf["vpn_num_services"],
                                 PE_s_per_service = conf["vpn_pes_per_services"],
                                 CEs_per_PE = conf["vpn_ces_per_pe"],
                                 random_seed = conf["random_seed"],
                                 protection=conf["protection"],
                                 enable_tba=conf["tba"],
                                 enable_hd=conf["hd"],
                                 enable_cfor=conf["cfor"],
                                 enable_inout=conf["inout-disjoint"]
                          )

    # save config
    net_dict = network.to_aalwines_json()

    output_file = conf["output_file"]
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(net_dict, f, indent=2)
    else:
        pprint(net_dict)



if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Command line utility to generate MPLS forwarding rules.')

    #general options
    p.add_argument("--conf", type=str, default = "", help="Load configuration file.")

    group1 = p.add_mutually_exclusive_group()
    group1.add_argument("--topology", type=str, default = "", help="File with existing topology to be loaded.")
    #random topology options
    group1.add_argument("--random_topology", action="store_true", help="Generate random topology. Defaults False.")
    p.add_argument("--random_mode", type=str, default = "random.log_degree", help="custom, random.log_degree, random.large_degree")
    p.add_argument("--num_routers",type=int, default = 6, help="Number of routers in topology.")
    p.add_argument("--random_weight_mode", type=str, default = "random", help="random, equal or distance")
    p.add_argument("--random_gen_method",type=int, default = 1, help="Method for generating the random topology")


    # Generator options
    p.add_argument("--php", action="store_true", help="Enable Penultimate Hop Popping. Defaults False.")
    p.add_argument("--ldp", action="store_true", help="Enable Label Distribution Protocol (LDP). Defaults False. This doesn't scale well.")
    p.add_argument("--rsvp", action="store_true", help="Enable Resource Reservation Protocol with Traffic Engineering extensions (RSVP-TE). Defaults False.")
    p.add_argument("--rsvp_num_lsps",type=int, default = 2, help="Number of (random) LSPS to compute for RSVP only, if enabled. Defaults to 2")
    p.add_argument("--rsvp_tunnels_per_pair",type=int, default = 5, help="Number of (random) tunnels between same endpoints. RSVP only, if enabled. Defaults to 5")
    p.add_argument("--enable_RMPLS", action="store_true", help="Use experimental RMPLS recursive protection (LFIB post processing). Defaults False")
    p.add_argument("--protection", type=str, default="facility-node", help="RSVP protection to implement. facility-node (default), facility-link or None ")
    p.add_argument("--tba", action="store_true", help="Use target based arborescences")
    p.add_argument("--hd", action="store_true", help="Use hop distance routing")

    p.add_argument("--vpn", action="store_true", help="Enable MPLS VPN generic services. Defaults False. ")
    p.add_argument("--vpn_num_services",type=int, default = 1, help="Number of (random) MPLS VPN services, if enabled. Defaults to 1")
    p.add_argument("--vpn_pes_per_services",type=int, default = 3, help="Number of PE routers allocated to each MPLS VPN service, if enabled. Defaults to 3")
    p.add_argument("--vpn_ces_per_pe",type=int, default = 1, help="Number of CE to attach to each PE serving a VPN, if enabled. Defaults to 1")

    p.add_argument("--random_seed",type=int, default = random.randint(0, 99999999), help="Random seed. Leave empty to pick a random one.")

    p.add_argument("--output_file",type=str, default = "", help="Path of the output file, to store forwarding configuration. Defaults to print on screen.")

    args = p.parse_args()

    print(args)
    conf = vars(args)  # arguments as dictionary
    conf_overrride = conf.copy()
    known_parameters = list(conf.keys()).copy()

    # Configuration Load
    if args.conf:
        print(f"Reading configuration from {args.conf}. Remaining options will override the configuration file contents.")
        if args.conf.endswith(".yaml") or args.conf.endswith(".yml"):
            with open( args.conf, 'r') as f:
                conf_new = yaml.safe_load(f)
                conf.update(conf_new)
                print("Configuration read from file.")
        elif args.conf.endswith(".json"):
            # try to load json conf file
            with open( args.conf, 'r') as f:
                conf_new = json.load(f)
                conf.update(conf_new)
                print("Configuration read from file.")
        else:
            raise Exception("Unsupported file format. Must be YAML or JSON.")

    # Deal with the topology
    print("whats goiing on here?")
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

    # raise errors if there is any unknown entry...
    for a in conf.keys():
        if a not in known_parameters:
            raise Exception(f"Unknown argument {a}.")

    # Override config options with cli explicit values
    for a in known_parameters:
        for sysarg in sys.argv:
            # this argument was explicitly overriden!
            if sysarg == "--" + a:   #max prio
                conf[a] = conf_overrride[a]

    pprint(conf)
    main(conf)
