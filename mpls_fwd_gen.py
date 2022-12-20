#!/usr/bin/env python3
# coding: utf-8

"""
###################################

     MPLS FORWARDING GENERATOR

Script to generate configuration files for the generator/simulator. - v0.1

Copyright (C) 2020-2022.

All rights reserved.

 * This file can not be copied and/or distributed.

All rights reserved.

This program is distributed WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

###################################
"""
from typing import Union

import networkx as nx
import matplotlib.pyplot as plt
import random
import time
# import jsonschema
import json
import math
import copy
from pprint import pprint
from itertools import chain, count
import numpy as np

import cfor
import hop_distance_client
import grafting_client
import keep_forwarding_client
import target_based_arborescence.tba_client as tba
import inout_disjoint_better
import inout_disjoint_old
from mpls_classes import MPLS_Client

from networkx.algorithms.shortest_paths.weighted import _weight_function, _dijkstra_multisource
from resource import getrusage, RUSAGE_SELF

from mpls_classes import *

global global_conf

def generate_fwd_rules(G, conf, enable_PHP = True, numeric_labels = False, enable_LDP = False, num_lsps = 10, tunnels_per_pair = 3,
                       enable_services = False, num_services = 2, PE_s_per_service = 3, CEs_per_PE = 1,
                       enable_RMPLS = False, random_seed = random.random()
                      ):
    """
    Generates MPLS forwarding rules for a given topology.

    Parameters:

    enable_PHP       : Boolean (defaults True). Activate Penultimate Hop Popping functionality.

    numeric_labels   : Boolean (defaults False). Use numeric labels (int) instead of strings.
                       String type labels are required by rthe current service implementation.

    enable_LDP       : Boolean (defaults False). Sets up Label Distribution Protocol (LDP, RFC5036).
                       A label will be allocated for each link and for each node (loopback address)
                       in the network, in a plain fashion (single IGP area/level). Beware that its
                       computation requires O(n^3) cpu time and O(n^4) memory (At least empirically).

    enable_RSVP      : Boolean (defaults True). Sets up Resource Reservation Protocol (RSVP-TE, RFC4090
                       and others). RSVP defines tunnels (LSPs) between a headend and a tail end allowing
                       for traffic engineering. It requires the configuration of the following additional
                       parameters:

                       num_lsps  : (defaults 10). Number of TE tunnels to compute between different
                                   pairs of head and tailend.


                       tunnels_per_pair: (defaults 3) Number of tunnels with between each pair of
                                   head and tailend. So if num_lsps = 10 and tunnels_per_pair = 3,
                                   a total of 30 tunnels would be generated.

    enable_RMPLS     : Enable additional experimental recursive protection by post-processing the LFIB.


    enable_services  : Boolean (defaults True). Sets up MPLS VPN services. These services abstract away
                       the specific type of service (VPWS, VPLS or VPRN), using a single class to represent
                       them all. Requires addtional parameters:

                       num_services: (defaults 2). Number of VPN services to instantiate.

                       PE_s_per_service: (defaults 3). Number of PE routers per VPN service.

                       CEs_per_PE: (defaults 1). Number of CE routers attached to each service PE.

    """

    # Instantiate the network object with topology graph G
    network = Network(G)

    global global_conf
    global_conf = conf

    for n,r in network.routers.items():
        r.php = enable_PHP
        r.numeric_labels = numeric_labels

    if enable_LDP:
        print("Computing LDP...")
        # TODO: implement LDP over RSVP. Must invert order for calling dijkstra.

        random.seed(random_seed)
        network.compute_dijkstra()
        # Start LDP process on each router
        network.start_client(ProcLDP)

        for n,r in network.routers.items():
            r.clients["LDP"].alloc_labels_to_known_resources()
            # Allocate labels for known interfaces

        print("LDP ready.")

    conf['num_flows'] = num_lsps if not isinstance(num_lsps, list) else len(num_lsps)
    conf['max_memory'] = conf['num_flows'] * conf['per_flow_memory'] if 'per_flow_memory' in conf else None

    print("Computing RSVP...")
    random.seed(random_seed)
    method = conf["method"]
    try:
        protection = conf['protection']
    except:
        protection = None
    if method == 'tba':
        network.start_client(tba.TargetBasedArborescence, **conf)
        protocol_name = "tba"
    elif method == 'cfor':
        network.start_client(cfor.CFor, **conf)
        protocol_name = "cfor"
    elif method == 'hd':
        network.start_client(hop_distance_client.HopDistance_Client, **conf)
        protocol_name = "hop_distance"
    elif method == 'gft':
        network.start_client(grafting_client.Grafting_Client, **conf)
        protocol_name = "gft"
    elif method == 'kf':
        network.start_client(keep_forwarding_client.KeepForwarding, **conf)
        protocol_name = 'kf'
    elif method == 'inout-disjoint':
        network.start_client(inout_disjoint_better.InOutDisjoint, **conf)
        protocol_name = 'inout-disjoint'
    elif method == 'inout-disjoint-old':
        network.start_client(inout_disjoint_old.InOutDisjoint, **conf)
        protocol_name = 'inout-disjoint'
    # Start RSVP-TE process in each router
    elif conf['method'] == 'rsvp' and conf['protection'] is not None and conf['protection'].startswith("plinko"):
            network.start_client(ProcPlinko)
            protocol_name = "ProcPlinko"
    else:
        network.start_client(ProcRSVPTE, **conf)
        protocol_name = "RSVP-TE"

    # compute  lsps
    print(f"num_lsps: {num_lsps}")
    # num_lsps can be:
    #   an integer: so we generate randomly.
    #   or a list of tuples (headend, tailend) in order to generate manually.
    if isinstance(num_lsps, int):

        i = 0  # Counter for pairs of headend, tailend
        err_cnt = 0 # Counter of already allocated headend,tailend
        print_thr = 10/num_lsps

        random.seed(random_seed)
        router_names = list(network.routers.keys())

        tunnels = dict()   # track created (headend, tailend) pairs

        while i < num_lsps:
            success = False

            tailend = None
            headend = None
            # get a tunnel across different nodes
            while tailend == headend:
                headend = router_names[random.randint(0,len(G.nodes)-1)]
                tailend = router_names[random.randint(0,len(G.nodes)-1)]

            if (headend, tailend) not in tunnels.keys():
                tunnels[(headend, tailend)] = 0

            for j in range(tunnels_per_pair):
                tunnels[(headend, tailend)] += 1  #counter to differentiate tunnels on same (h,t) pair

                try:
                    if conf['method'] != 'rsvp':

                        network.routers[tailend].clients[protocol_name].define_demand(headend)
                    else:
                        network.routers[headend].clients[protocol_name].define_lsp(tailend,
                                                                   # tunnel_local_id = j,
                                                                   tunnel_local_id = tunnels[(headend, tailend)],
                                                                   weight='weight',
                                                                   protection=conf['protection'])
                    success = True
                except Exception as e:
                    pprint(e)
                    # We have already generated the tunnels for this pair of headend, tailend.
                    err_cnt += 1
                    if err_cnt > num_lsps*10:
                        # too many errors, break the loop!
                        i = num_lsps
                        break
            if random.random() < print_thr:
                print("Tunnel_{}: from {} to {}".format(i,headend,tailend))
            if success:
                i += 1

    elif isinstance(num_lsps, list):
        #verify that this is a list of pairs:
        assert all(filter(lambda x: isinstance(x,(list,tuple)) and len(x)==2 ,num_lsps))
        c = dict()
        for h,t in num_lsps:
            print(f"Manually build LSP from {h} to {t}")
            if (h,t) not in c.keys():
                c[(h,t)] = 0
            else:
                c[(h,t)] += 1

            if protocol_name in ["tba","hop_distance","cfor","gft","kf","inout-disjoint"]:
                network.routers[t].clients[protocol_name].define_demand(h)
            else:
                network.routers[h].clients[protocol_name].define_lsp(t,
                                                             tunnel_local_id = c[(h,t)],
                                                             weight='weight',
                                                             protection=protection)

    else:
        raise Exception("num_lsps has wrong type (must be list of pairs or integer)")

    for n,r in network.routers.items():
        r.clients[protocol_name].commit_config()

    for n,r in network.routers.items():
        r.clients[protocol_name].compute_bypasses()

    if protection is not None and protection.startswith("plinko"):
        arguments = protection.split("/")
        max_resiliency = 4 #default value
        if len(arguments) > 1:
                max_resiliency = int(arguments[1])

        for t in range(1,max_resiliency+1):
            for n,r in network.routers.items():
                r.clients[protocol_name].add_protections(t)

    for n,r in network.routers.items():
        r.clients[protocol_name].alloc_labels_to_known_resources()

    print(f"RSVP ready (frr variant={protection}).")

    if enable_services:
        print("Computing Services")
        random.seed(random_seed)

        # compute services
        ce_cnum = dict()  #keep track of ce numbering on each node
        for i in range(num_services):
            # choose  routers at random
            if PE_s_per_service > len(list(network.routers.values())):
                PE_s_per_service = len(list(network.routers.values()))
            PEs = random.sample(list(network.routers.values()) ,  PE_s_per_service )

            vpn_name = "Service_{}".format(i)
            for pe in PEs:
                client = pe.get_client(MPLS_Service)
                if not client:
                    client = pe.create_client(MPLS_Service)
                client.define_vpn(vpn_name)
                if pe not in ce_cnum.keys():
                    ce_cnum[pe] = 0
                for _ in range(CEs_per_PE):
                    ce_name = "CE_{}".format(ce_cnum[pe])
                    ce_cnum[pe] += 1
                    client.attach_ce(vpn_name, ce_name)

        for n,r in network.routers.items():
            if "service" in r.clients.keys():
                r.clients["service"].alloc_labels_to_known_resources()

        print("Services ready.")


    if enable_RMPLS:
        print("Enabling RMPLS -- recursive protection")
        random.seed(random_seed)

        # Start RMPLS process in each router
        network.start_client(ProcRMPLS)

        for n,r in network.routers.items():
            r.clients["RMPLS"].compute_bypasses()

        for n,r in network.routers.items():
            r.clients["RMPLS"].alloc_labels_to_known_resources()

        print("RMPLS enabled.")


    # Forwarding phase: fill the LFIB
    print("building LFIB")

    # Build the Label forwarding information base
    network.LFIB_build_orderly()
    print("LFIB ready.")
    print("adapting priorities.")   # let the fwd entries have ordinal priorities instead of integer weights.
    for n,r in network.routers.items():
        r.LFIB_weights_to_priorities()

    print("Refining LFIB")
    for n,r in network.routers.items():
        r.LFIB_refine()

    print("Finished.")

    return network


def topology_from_graphml(filename, visualize = True):
    # simple attempt at loading from a graphml file
    GG = nx.read_graphml(filename)
    G = nx.Graph(GG)

    name = filename.split("/")[-1].split(".")[0]
    G.graph["name"] = net_dict_2["network"]["name"]  # load name

    return G


def topology_from_aalwines_json(filename, visualize = True):
    # load topology from aalwines json format
    with open(filename, 'r') as f:
        net_dict_2 = json.load(f)

    G = nx.Graph()

    if "name" in net_dict_2["network"]:
        G.graph["name"] = net_dict_2["network"]["name"]

    #build alias map a (point each alias to actual router name)
    a = dict()

    #build router names List
    router_names = []
    for r_dict in net_dict_2["network"]["routers"]:
        a[r_dict["name"]] = r_dict["name"]
        router_names.append(r_dict["name"])
        for alias_name in r_dict["alias"] if "alias" in r_dict else []:
            a[alias_name] = r_dict["name"]

    #create edges
    links = net_dict_2["network"]["links"]
    for data in links:
        u = data["from_router"]
        v = data["to_router"]
        if "weight" in data.keys():
            w = data["weight"]
        else:
            w = 1

        #random weights
        G.add_edge(a[u], a[v], weight = w)

    # deal with coordinates
    for r_dict in net_dict_2["network"]["routers"]:
        for r in [r_dict["name"]] + (r_dict["alias"] if "alias" in r_dict else []):
            if r in router_names and "location" in r_dict:
                G.nodes[r]["latitude"] =  r_dict["location"]["latitude"]
                G.nodes[r]["longitude"] =  r_dict["location"]["longitude"]
                break

    return G
