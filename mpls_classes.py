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
from networkx import Graph

from networkx.algorithms.shortest_paths.weighted import _weight_function, _dijkstra_multisource

from typing import *

# Auxiliary functions
def rand_name():
    """
    Generates a random name for general purposes, all lower-case.
    """
    consonants = 'bcdfghjklmnpqrstvwxyz'
    vowels = 'aeiou'
    syllables = list(vowels)


    for c in consonants:
        for v in vowels:
            syllables.append(v+c)
            syllables.append(c+v)
            for cc in consonants:
                syllables.append(c+v+cc)

    syllables = list(set(syllables))
    weights = list()
    for s in syllables:
        if len(s) == 3:
            weights.append(1)
        elif len(s) == 2:
            weights.append(11)
        elif len(s) == 1:
            weights.append(300)

    x = random.choices(syllables, weights = weights, k = random.randint(2,4))
    return("".join(x))

def as_list(x):
    if isinstance(x,list):
        return x
    else:
        return [x]

def dict_filter(_dict, callback):
    '''
    Call boolean function callback(key, value) on all pairs from _dict.
    Keep only True result and return new dictionary.
    '''
    new_dict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in _dict.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            new_dict[key] = value
    return new_dict

def rec_int_val_to_str(d):
    # Goes through a dictionary or list (or combinations thereof)
    # changing numbers into strings.

    if isinstance(d,dict):
        for k,v in d.items():
            d[k] = rec_int_val_to_str(v)
        return d

    elif isinstance(d,list):
        for i in range(len(d)):
            d[i] = rec_int_val_to_str(d[i])
        return d

    elif isinstance(d,str) or isinstance(d,int) or isinstance(d, float):
        return str(d)

    else:
        return d

# generation functions
def gen_connected_random_graph(n, m, seed=0, directed=False, method=0):
    """Returns a random graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges.
    seed : integer (default 0) or random_state.
        Randomness seed.
    directed : bool, optional (default=False)
        If True return a directed graph
    method : integer (default=0).
        Defines the link generation method.
        Method 0 creates random edges between nodes, can't guarantee connectivity.
        Method 1 guarantees connectivity, yet with a non-uniform degree distribution.

    Adapted from a networkx function of the same name.
    """

    if directed:
        G = nx.DiGraph()
        max_edges = n * (n - 1)
    else:
        G = nx.Graph()
        max_edges = n * (n - 1) / 2

    router_names = []
    for i in range(n):
        router_names.append("R{}".format(i))

    G.add_nodes_from(router_names)

    if n == 1:
        return G

    if m >= max_edges:
        G = nx.complete_graph(n, create_using=G)
        G = nx.relabel_nodes(G, lambda i: f"R{i}" , copy=False)
        return G

    random.seed(seed)

    # Generate edges...
    if method == 0:
        # Adds edges randomly between nodes until the graph has m edges
        # Can't ensure graph connectivity.
        edge_count = 0
        while edge_count < m:
            # generate random edge,u,v
            u = router_names[random.randint(0,n-1)]
            v = router_names[random.randint(0,n-1)]
            if u == v or G.has_edge(u, v):
                continue
            else:
                G.add_edge(u, v)
                edge_count = edge_count + 1

    elif method == 1:
        # Go through nodes in ascending order, generates new random links with
        # nodes already seen only. Ensures connectivity, yet the first nodes
        # are more likely to have a greater degree than the last ones.
        edge_count = 0
        usable_range = 0
        while edge_count < m:
            # generate random edge,u,v
            u = router_names[random.randint(0,usable_range)]
            if edge_count < n-1:
                v = router_names[edge_count + 1]
            else:
                v = router_names[random.randint(0,n-1)]

            if u == v or G.has_edge(u, v):
                continue
            else:
                G.add_edge(u, v)
                edge_count = edge_count + 1
                if edge_count < n-1:
                    usable_range += 1

    return G


def generate_topology(mode, n, weight_mode = "random", gen_method = 1,
                      visualize = False, display_tables = False, random_seed = random.random() ):
    """
    Generates a topology, given a number of nodes and a edge generation mode.

    Parameters:
    mode:           Method for generating the number of edges.
                       custom
                       random.log_degree
                       random.large_degree

    n:              Number of nodes.

    weight_mode:    Edge weight value: random, equal or distance

    gen_method:     Method for generating the random topology

    visualize:      Use only with small networks

    display_tables: Use only with small networks

    Returns generated graph G.
    Empirically the time consumed is approximately proportional to the square of n.

    """


    def _weight_fn(mode = "equal", **kwargs):
        if mode == "random":
            return random.randint(a=1,b=10)

        elif mode == "distance":
            # we expect additional args p0 and p1
            return _geo_distance( p0,  p1)

        else:
            return 1

    def _geo_distance( p0,  p1):
        """
        Calculate distance (in kilometers) between p0 and p1.
        Each point must be represented as 2-tuples (longitude, latitude)
        long \in [-180,180], latitude \in [-90, 90]
        """
        def _hav(theta): # Haversine formula
            return (1 - math.cos(theta))/2

        a = math.pi / 180 # factor for converting to radians
        ER = 6371 # 6371 km is the earth radii

        lo_0, la_0 = p0[0] * a, p0[1] * a
        lo_1, la_1 = p1[0] * a, p1[1] * a
        delta_la, delta_lo = la_1 - la_0, lo_1 - lo_0
        sum_la, sum_lo = la_1 + la_0, lo_1 + lo_0

        h =  _hav(delta_la) + (1 - _hav(delta_la) - _hav(sum_la))*_hav(sum_lo)
        theta = 2*math.sqrt(h)    # central angle in radians

        return ER * theta

    if mode.startswith("random"):
        #number of nodes
        random.seed(random_seed)

        # uneven range for number of links
        if mode == "random.large_degree":
            # avg. degree proportional to n*n
            beta_l = n/4 + 1/2
            beta_u = n/2

        elif mode == "random.log_degree":
            # attempts to get an average degree proportional to log of (n-1)*log n
            beta_l = 0.5*math.log2(n)  # belongs to [1,n/2]
            beta_u = 1.16*math.log2(n)    # belongs to [1,n/2] and is ge than beta_l 1.16

        lower_b = int(beta_l*(n-1))
        upper_b = int(beta_u*(n-1))

        #pprint((lower_b, upper_b))
        e = random.randint(lower_b, upper_b)
        print(f"Number of edges: {e}")
        G = gen_connected_random_graph(n, e, method = gen_method, seed = random_seed)

        pos = nx.spring_layout(G)

        for u,v in G.edges():
            if weight_mode == "distance":
                G[u][v]["weight"] = _weight_fn(weight_mode,
                                            p0 = (u["longitude"], u["latitude"]),
                                            p1 = (v["longitude"], v["latitude"]))
            else:
                G[u][v]["weight"] = _weight_fn(weight_mode)


    elif mode == "custom":
        # custom graph used for testing.
        G = nx.Graph()
        G.add_edge("R0", "R1", weight=_weight_fn(weight_mode))
        G.add_edge("R1", "R2", weight=_weight_fn(weight_mode))
        G.add_edge("R1", "R3", weight=_weight_fn(weight_mode))
        G.add_edge("R2", "R4", weight=_weight_fn(weight_mode))
        G.add_edge("R3", "R4", weight=_weight_fn(weight_mode))
        G.add_edge("R2", "R5", weight=_weight_fn(weight_mode))
        G.add_edge("R2", "R6", weight=_weight_fn(weight_mode))
        G.add_edge("R5", "R7", weight=_weight_fn(weight_mode))
        G.add_edge("R6", "R7", weight=_weight_fn(weight_mode))
        G.add_edge("R7", "R8", weight=_weight_fn(weight_mode))

    if visualize:
        try:
            pos
        except NameError:
            pos = nx.spring_layout(G)

        fig, ax = plt.subplots(figsize=(12, 7))
        nx.draw_networkx_nodes(G, pos, node_size=250, node_color="#210070", alpha=0.9)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color="m")

        labels = nx.get_edge_attributes(G,'weight')
        label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
        nx.draw_networkx_labels(G, pos, font_size=14, bbox=label_options)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    return G


class MPLS_Client(object):
    """
    Abstract class with minimal functionality and methods any client must implement.

    An MPLS client is a FEC manager process on a router. It is resposible for:

    - Computing the resulting outcome of a given MPLS control plane protocol
    - Request labels for its FECs on the router's LIB
    - Provide functions to compute appropiate routing entries for LFIB contruction

    For initialization indicate the router.
    """

    #static attribute: start_mode
    # if auto, the network will try to allocate labels to local resources right after initialization.
    start_mode = 'manual'
    EXPLICIT_IPV4_NULL = 0
    EXPLICIT_IPV6_NULL = 2
    IMPLICIT_NULL = 3

    def __init__(self, router, build_order = 100, **kwargs):
        self.router: Router = router
        self.build_order = build_order
        self.LOCAL_LOOKUP = router.LOCAL_LOOKUP
        self.comm_count = 0
        self.do_rule_reduction = kwargs['rule_reduction'] if 'rule_reduction' in kwargs else False

    # Common functionality
    def get_comm_count(self):
        return self.comm_count

    def alloc_labels_to_known_resources(self):
        # Asks the router for allocation of labels for each known FEC (resource)
        for fec in self.known_resources():
             self.LIB_alloc(fec)

    def get_remote_label(self, router_name, fec, do_count=True):
        # Gets the label allocated by router <router_name> to the FEC <fec>
        router = self.router.network.routers[router_name]
        owner = router.get_FEC_owner(fec)
        if router.php and owner and owner.self_sourced(fec):
            return self.IMPLICIT_NULL
        else:
            if do_count and router_name != self.router.name:
                self.comm_count += 1
            return self.router.network.routers[router_name].get_label(fec)

    def get_local_label(self, fec):
        # Gets the local label allocated to the FEC <fec>
        return self.get_remote_label(self.router.name, fec)

    def get_fec_by_name_matching(self, name_substring):
        # Return generator of all locally known FEC whose name contains name_substring
        return [ fec for fec in self.router.LIB.keys() if name_substring in fec.name ]

    def LIB_alloc(self, FEC, literal = None):
        # Wrapper for calling the routerś LIB_alloc function.
        return self.router.LIB_alloc(self, FEC, literal = literal)

    # Abstract functions to be implemented by each client subclass.
    def LFIB_compute_entry(self, fec, single = False):
        # Each client must provide an generator to compute routing entries given the fec.
        # optional parameter "single" forces the function to return just one routing entry.
        # returns tuple (label, routing_entry)
        # routing entries have format:
        #  routing_entry = { "out": next_hop_iface, "ops": [{"pop":"" | "push": remote_label | "swap": remote_label}], "weight": cost  }
        # A few rules regarding the ops:
        #
        # Rule 1:          NOp := [{"push":x}, {"pop":""}]  # should never appear in an entry.
        # Rule 2: {"swap": x } := [{"pop":""}, {"push":x}]  # can be analized with just 2 operations.
        # Corollary 1: All ops can be written with just pop and push operations
        # Corollary 2: All ops must have a form like:  [{"pop":""}^n, prod_{i=1}^{m} {"push": x_i}]
        #              which in turn can have at most one swap operation at the deepest inspected stack
        #              level, so (if m > 1 and n>1):
        #                 [{"pop":""}^{n-1}, {"swap": x_1} ,prod_{i=2}^{m} {"push": x_i}]
        pass

    def LFIB_refine(self, label):
        # Some process might require a refinement of the LFIB.
        pass

    def LFIB_fullrefine(self):
        if not self.do_rule_reduction:
            return

        to_remove: Set[Tuple[str, str]] = set()
        to_remove_aux: Set[str] = set()
        to_rerefine: Set[Router] = set()

        for i, (l1, r1) in enumerate(self.router.LFIB.items()):
            if l1 in to_remove_aux:
                continue
            for l2, r2 in islice(self.router.LFIB.items(), i + 1, len(self.router.LFIB)):
                if r1 == r2:
                    to_remove.add((l1, l2))
                    to_remove_aux.add(l2)

                    #print(f'Removed tau({self.router.name}, {l2})')

                    for n in self.router.topology.neighbors(self.router.name):
                        neighbour: Router = self.router.network.routers[n]

                        for l3, r3 in neighbour.LFIB.items():
                            for rd in r3:
                                if rd['out'] == self.router.name and [{'swap': l2}] == rd['ops']:
                                    pre_rd = str(rd)
                                    rd['ops'] = [{'swap': l1}]
                                    #print(f'Changed tau({neighbour.name}, {l3}) entry from {pre_rd} to {rd}')
                                    to_rerefine.add(neighbour)

        label_to_fec: Dict[str, oFEC] = {v['local_label']: fec for fec, v in self.router.LIB.items()}

        for l1, l2 in to_remove:
            label_to_fec[l1].value["ingress"] = label_to_fec[l1].value["ingress"] + label_to_fec[l2].value['ingress']
            del self.router.LFIB[l2]

        for neighbour in to_rerefine:
            neighbour.clients[self.protocol].LFIB_fullrefine()

    def known_resources(self):
        # Returns a generator to iterate over all resources managed by the client.
        # Each client must provide an implementation.
        pass

    def self_sourced(self, FEC):
        # Returns True if the FEC is sourced or generated by this process.
        pass

class ProcLDP(MPLS_Client):
    """
    Class implementing a Label Distribution Protocol (LDP) client.

    This protocol is used for allocation of labels to (some) IP prefixes
    installed on the routing table. Its main purpose is to allow immediate
    forwarding after the IGP has solved shortest path routes in the network.
    It builds its LSPs on a hop-by-hop basis.
    Supports ECMP, in which case it is out of its scope to decide which packets
    should use which path.

    Manages IP prefixes as resources/FECs.
    The fec_types it handles are "link" and "bback"

    Requires access to the router's routing table, nexthop information, the
    linkstate database and the remote labels to generate routing entries.

    LDP FRR is not implemented yet.

    For initialization indicate the router.
    """

    start_mode = "auto" # the network will try to allocate labels immediately.
    protocol = "LDP"

    def __init__(self, router):
        super().__init__(router)
        self.build_order = 200

    def known_resources(self):
        # LDP, known resources are all IP prefixes in the topology
        # We assume that each node in the topology has at least one
        # loopback interface with an IP address, and each link between
        # nodes also have an IP prefix.

        G = self.router.topology
        # Return FECs for each loopback interface in the network.
        for node in G.nodes():
            yield oFEC("loopback","lo_{}".format(node), node)

        # Return FECs for each link in the network.
        for edge in G.edges():
            fec = edge
            # Use a canonical ordering in the undirected edges vertices.
            if edge[0] > edge[1]:
                fec = (edge[1],edge[0])  # always ordered.
            yield oFEC("link", "link_{}_{}".format(fec[0],fec[1]),fec)


    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is an IP prefix that we want to reach, so the computation
        # will require access to remote lables and to routing information.

        router = self.router
        network = router.network

        # Get root router object, the router we want to reach
        if  fec.fec_type == "loopback":
            root = network.routers[fec.value]  # a loopback fec value is its router name.

        elif fec.fec_type == "link":
            e = fec.value   # a link fec is a (canonically) ordered 2-tuple of nodes.
            e0, e1 = network.routers[e[0]], network.routers[e[1]]
            # the root has to be the closest node in the link.
            if e0.dist[router.name] <= e1.dist[router.name]:  #dist( router-> e0) vs. dist( router-> e1)
                root = e0
            else:
                root = e1

        #access routing information on how to reach the root router.
        if router.name == root.name:
            # access labels
            local_label = self.get_local_label(fec)
            # build routing entry
            routing_entry = { "out": router.LOCAL_LOOKUP, "ops": [{"pop": ""}], "weight": 0 }
            yield (local_label, routing_entry)

        for predecessor_name in root.pred[router.name]:  #for each next_hop from router towards root...
            # access labels
            local_label = self.get_local_label(fec)
            remote_label = self.get_remote_label(predecessor_name, fec)
            cost = root.dist[router.name]   #IGP distance from router towards root
            # build routing entry
            if remote_label == self.IMPLICIT_NULL:
                routing_entry = { "out": predecessor_name, "ops": [{"pop": ""}], "weight": cost }
            else:
                routing_entry = { "out": predecessor_name, "ops": [{"swap": remote_label}], "weight": cost }
            yield (local_label, routing_entry)
            if single:
                break  #compute for one predecessor, no load balancing.

    def self_sourced(self, fec):
        # Returns True if the FEC is sourced or generated by this process.
        router = self.router
        network = router.network
        self_sourced = False
        # Get root router object, the router we want to reach
        if  fec.fec_type == "loopback" and router.name == fec.value:
            # a loopback fec value is its router name.
            self_sourced = True

        elif fec.fec_type == "link" and router.name in fec.value:
            # a link fec is a (canonically) ordered 2-tuple of node names.
            self_sourced = True

        return self_sourced

# Classes
class Network(object):
    """
    Class keeping track of the topology and all routers in the network.
    It provides helper methods to call router functions network-wide:
    shortest-path computation, create and start client processes start, LFIB table build and
    visualization of LIB and LFIB tables.

    """
    def __init__(self, topology, name=None):

        # load network topology (a networkx graph)
        self.topology: Graph = topology
        if not name:
            try:
                self.name = topology.graph["name"]
            except:
                self.name = rand_name()
        else:
            self.name = name

        self.service_registry = dict()   # hash table with sets of PE routers for each MPLS service.
        # create and keep track of all routers in the network
        routers = dict()
        self.routers: Dict[str, Router] = routers

        for n in topology.nodes():
            r = Router(self, n)
            routers[n] = r

    def compute_dijkstra(self, weight="weight"):
        # compute the shortest-path directed acyclic graph for each node (how to reach it)
        for n,r in self.routers.items():
            # compute the shortest-path directed acyclic graph for router n
            # this can be computationally expensive!
            r.compute_dijkstra(weight=weight)

    def start_client(self, client_class, **kwargs):
        # Create client_class clients on each router first,
        # then proceed to label allocation if required.
        for n,r in self.routers.items():
            # instantiate a client_class client on each router.
            client = r.create_client(client_class, **kwargs)

        if client_class.start_mode == 'auto':
            for n,r in self.routers.items():
                # Allocate labels for known managed resources
                client.alloc_labels_to_known_resources()


    def _get_build_order(self):
        # helper function to get the order of construction for forwarding rules
        prio = dict()
        for router_name, router in self.routers.items():
            for client_name, client in router.clients.items():
                if client.build_order not in prio.keys():
                        prio[client.build_order] = set()
                prio[client.build_order].add(router)

        order = sorted(list(prio.keys()))

        for o in order:
            yield (o,prio[o])

    def LFIB_build(self):
        # helper function to generate LFIB entries on each router.
        for router_name, router in self.routers.items():
            router.LFIB_build()

    def LFIB_build_orderly(self):
        # helper function to generate LFIB entries on each router, respecting priorities
        for build_order, router_list in self._get_build_order():
            print("Now building order {}".format(build_order))
            #for router_name, router in router_list:
            for router in router_list:
                router.LFIB_build(build_order = build_order)

    def get_number_of_rules(self):
        N = 0
        # helper function to get the total number of rules in the network
        for router_name, router in self.routers.items():
            N += router.get_number_of_rules()
        return N

    def get_number_of_routing_entries(self):
        N = 0
        # helper function to get the total number of routing entries in the network
        for router_name, router in self.routers.items():
            N += router.get_number_of_routing_entries()
        return N

    def get_comm_count(self):
        N = 0
        # helper function to get the (min) number of communication exchanges among routers
        for router_name, router in self.routers.items():
            N += router.get_comm_count()
        return N

    def to_aalwines_json(self, weigth = "weight"):
        # This function generates a dictionary compatible with AalWiNes JSON schema.
        net_dict = {"network":  {"links": [], "name": self.name, "routers": []} }

        # Topology
        links = net_dict["network"]["links"]
        G = self.topology
        for from_router,to_router in G.edges():
            links.append({
                "bidirectional": True,
                "from_interface": to_router,
                "from_router": from_router,
                "to_interface": from_router,
                "to_router": to_router,
                "weight": G[from_router][to_router][weigth]
              })

        for router_name in G.nodes():
            router = self.routers[router_name]
            links.append({
                "from_interface": router.LOCAL_LOOKUP,
                "from_router": router_name,
                "to_interface": router.LOOPBACK,
                "to_router": router_name
              })

        # Forwarding rules
        for idx, router in self.routers.items():
            r_dict = router.to_aalwines_json()
            net_dict["network"]["routers"].append(r_dict)

        return net_dict

    def build_flow_table(self, flows: List[Tuple[str, str, int]], verbose = False):
        # Build dict of flows for each routable FEC the routers know, in the sense
        # of only initiating packets that could actually be generated from the router.

        # classify according to fec_type
        print(f"Computing flows for simulation." )

        labeled_flows = dict()

        for src_router, tgt_router, load in flows:
            if src_router not in labeled_flows:
                labeled_flows[src_router] = dict()
            if verbose:
                print(f"\n processing flow {src_router} {tgt_router}")

            for fec in self.routers[src_router].LIB:
                if verbose:
                    print(fec.name)
                if fec.fec_type.startswith("bypass"):
                    continue

                if fec.fec_type.startswith("plinko"):
                    continue

                elif fec.fec_type == "loopback":
                    good_sources = set(self.routers)
                    if fec.name.endswith(src_router):
                        good_sources.difference_update([src_router])
                    good_targets = [fec.value]

                elif fec.fec_type == "link":
                    good_sources = set(self.routers)
                    if "_"+src_router+"_" in fec.name or fec.name.endswith("_"+src_router):
                        good_sources.difference_update([src_router])
                    good_targets = list(fec.value)

                elif fec.fec_type == "TE_LSP":
                    good_sources = [fec.value[0]]   #the headend router
                    good_targets = [fec.value[1]]   #the tailend router

                elif fec.fec_type == "vpn_endpoint":
                    vpn_name = fec.value[0]
                    tgt_pe = fec.value[1]
                    tgt_ce = fec.value[2]
                    good_targets = [tgt_pe]   #actually we don't have implemented a way of checking delivery to a CE
                    good_sources = []
                    for srv_inst in self.routers[src_router].get_FEC_owner(fec).locate_service_instances(vpn_name):
                        good_sources.append(srv_inst.router.name)

                elif fec.fec_type == "arborescence":
                    if fec.value[3]:
                        good_sources = list(fec.value[2])
                        good_targets = [fec.value[0]]
                    else:
                        continue

                elif fec.fec_type == "hop_distance":
                    good_sources = [fec.value[0]]
                    good_targets = [fec.value[1]]

                elif fec.fec_type == "cfor":
                    if 'iteration' in fec.value and fec.value["iteration"] == 1:
                        good_sources = fec.value["ingress"]
                        good_targets = fec.value["egress"]
                    else:
                        continue

                elif fec.fec_type == 'kf':
                    good_sources = [fec.value['ingress']]
                    good_targets = [fec.value['egress']]

                elif fec.fec_type == "gft":
                    good_sources = fec.value['ingress']
                    good_targets = [fec.value['egress']]

                elif fec.fec_type == "inout-disjoint":
                    if 'path_index' in fec.value and fec.value["path_index"] == 0:
                        good_sources = fec.value["ingress"]
                        good_targets = fec.value["egress"]
                    else:
                        continue

                else:
                    continue

                if src_router not in good_sources or tgt_router not in good_targets:
                    continue  # this router can be the source of a packet to this FEC

                in_label = self.routers[src_router].get_label(fec)
                # Don't even try if this is not in the LFIB
                if in_label not in self.routers[src_router].LFIB:
                    continue

                # I have good_sources and good_targets in memory currently...
                labeled_flows[src_router][in_label] = ([src_router],[tgt_router], load)
                break # Successfully found flow
            else:
                print(f"ERROR: Could not find flow from {src_router} to {tgt_router}", file=sys.stderr)

        return labeled_flows


    def visualize(self, router_list=None):
        # helper function to visualize LIB and LFIB tables of some router.
        # example:   router_list = [0,1,2]   , just provide the identifiers.

        if router_list:
            routers =  { r: self.routers[r] for r in router_list }
        else:
            routers = self.routers

        print(f"Number of forwarding rules in the Network:{self.get_number_of_rules()}")
        print("-"*20)
        for n,r in routers.items():
            # print LIB
            if r.LIB:
                print("{}.LIB".format(n))
                for a,b in r.LIB.items():
                    print("{}: {}".format(str(a), b))
                print()

            # print LFIB
            if r.LFIB:
                print("{}.LFIB".format(n))
                for a,b in r.LFIB.items():
                    print("{}: {}".format(str(a), b))
                print("-"*20)

    def visualize_fec_by_name(self, fec_name):
        for n,r in self.routers.items():

            if r.LIB:
                LIB_entries_by_name = [ (fec,entry) for (fec,entry) in r.LIB.items() if fec_name in fec.name]
                if LIB_entries_by_name:
                    local_labels =  [i[1]['local_label'] for i in LIB_entries_by_name]
                    print("{}.LIB".format(n))
                    for a,b in LIB_entries_by_name:
                        pprint("{}: {}".format(str(a), b))
                    print()

                    if r.LFIB:
                        print("{}.LFIB".format(n))
                        for ll in local_labels:
                            if ll in r.LFIB.keys():
                                pprint("{}: {}".format(str(ll), r.LFIB[ll]))
                    print("-"*20)

class Label_Manager(object):
    """
    A class that handles requests for new MPLS labels.
    """
    def __init__(self, first_label=16, max_first_label = 9223372036854775807, final_value = 9223372036854775807, seed=0, numeric_labels=True):

        self.first_label = first_label
        self.max_first_label = max_first_label
        self.final_value = final_value

        # check input
        if first_label > max_first_label:
            raise Exception("Max first label can't  be smaller than first label.")
        elif max_first_label > final_value:
            raise Exception("Max first label can't be larger than final label.")
        elif first_label > final_value:
            raise Exception("First label can't be larger than final label.")

        random.seed(seed)
        start = random.randint(first_label,max_first_label)

        self._cur_label = count(start=start) # basic label alloc manager

    def next_label(self):
        # Allocates a new label in the MPLS range
        candidate_label = next(self._cur_label)
        if candidate_label < self.final_value:
            return candidate_label
        else:
            raise Exception("MPLS labels depleted in router {} {}".format(self.name))


class Router(object):
    """
    A class for (MPLS) router. Composed by:

    - A Label Information Base (LIB), that allocates a local MPLS label to resources identified by a FEC.
    - A Label Forwarding Information Base (LFIB), that keeps forwarding instructions for each (local) MPLS label.
    - A table keeping track of registered MPLS clients.

    The router knows its interfaces from the network topology it belongs to.
    It is also responsible for computation of the shortest-path from other routers towards itself, that is,
    the directed acyclic graph for node n (how to reach this node).
    """

    def __str__(self):
        return self.name

    def __init__(self, network, name, alternative_names=[], location = None, php = False, dist=None, pred=None, paths=None, first_label=16,
                 max_first_label = 90000, seed=0, numeric_labels=True):
        self.name = name
        self.network = network
        self.topology: Graph = network.topology
        self.location = location  # "location": {"latitude": 25.7743, "longitude": -80.1937 }
        self.alternative_names = alternative_names
        self.PHP = php   # Activate Penultimate Hop Popping
        self.numeric_labels = numeric_labels

        # Define the LOCAL_LOOKUP interface: the interface to which we should send packets that
        # require further processing.
        # case 1: After the are stripped of all their labels and must be forwarded outside of
        #         the MPLS domain.
        # case 2: If they require to be matched again agansts the LFIB after a succesful local pop().
        #         This is the case for recursive MPLS forwarding.
        #
        self.LOCAL_LOOKUP = "local_lookup"

        # LOOPBACK interface. Interface that implements recursive forwarding: will receive the packets send to
        # LOCAL_LOOKUP.
        self.LOOPBACK = "loop_back"

        # Initialize tables
        self.LFIB: Dict[str, List[Dict[str, Any]]] = dict()
        self.LIB: Dict[oFEC, Dict[str, Any]] = dict()
        self.clients = dict()
        self.label_managers = dict()
        self.main_label_manager = Label_Manager(first_label=16,
                                                max_first_label = 90000,
                                                final_value = 1048576,
                                                seed=0,
                                                numeric_labels=True)

        # Attributes to store shortest path spanning tree
        self.dist = dist                       # Distance from other nodes
        self.paths = paths                     # Paths from other nodes
        self.pred = pred                       # Predecesors (upstream nodes) for each router towards this
        self.succ = self.get_successors(pred)  # next-hops

        random.seed(seed)
        start = random.randint(first_label,max_first_label)
        # For debugging or demo purposes, you may comment the previous line and uncomment the next one.
        #start = 100000+100*int(self.name[1:])
        self._cur_label = count(start=start) # basic label alloc manager, could be replaced by instance of Label_Manager

    def get_location(self):
        # get geographical location of router.
        #{"latitude": latitude, "longitude": longitude}
        if self.location:
            return self.location

        G = self.topology

        try:
            location = {"latitude": G.nodes[self.name]["latitude"],
                    "longitude": G.nodes[self.name]["longitude"]}
            self.location = location
            return self.location

        except:
            return None

    def next_label(self):
        # Allocates label in the MPLS range
        candidate_label = next(self._cur_label)
        if candidate_label < 1048576:
            return candidate_label
        else:
            raise Exception("MPLS labels depleted in router {}".format(self.name))


    def compute_dijkstra(self, weight="weight"):
        # Compute the shortest-path directed acyclic graph in-tree (how to reach this node)
        # Relies on networkx._dijkstra_multisource()

        # Initializations
        paths = {self.name: [self.name]}  # dict of paths from each node
        pred = {self.name: []}    # dict of predecessors from each node

        # compute paths and distances
        G = self.topology
        weight = _weight_function(G, weight)
        dist = _dijkstra_multisource( G, {self.name}, weight, pred=pred, paths=paths )

        # store results for this router.
        self.dist = dist  # dict of distances from each node
        self.pred = pred
        self.succ = self.get_successors(pred) # dict of successors from each node
        self.paths = paths

    def get_successors(self, pred):
        # Given the predecesors it the router in-tree, return the dictionary of succesors (next-hops)
        if not pred:
            return None

        succ = dict()

        for router in pred.keys():
            succ[router] = list()
            for r, predecessor_list in pred.items():
                if router in predecessor_list:
                    succ[router].append(r)

        return succ

    def get_interfaces(self, outside_interfaces = False):
        # Returns generator of router interfaces (identified as router names).
        # If outside_interfaces is False (default) it will return all enabled MPLS
        # interfaces (internal interfaces). If outside_interfacs is True,
        # it will only return non-MPLS interfaces, for example the ones used on
        # VPN services as attachement circuits (AC).

        #return chain(iter([self.name]), self.topology.neighbors(self.name))
        if not outside_interfaces:
            yield self.name

            for neigh in self.topology.neighbors(self.name):
                yield neigh

        else:
            # add all service interfaces:
            if "service" in self.clients:
                for vpn_name, vpn in self.clients["service"].services.items():
                    for ce in vpn["ces"]:
                        yield ce

    def get_client(self, mpls_client):
        # Returns registered client of class mpls_client if it existes, None otherwise
        if mpls_client.protocol in self.clients.keys():
            return self.clients[mpls_client.protocol]
        else:
            return None

    def create_client(self, client_class, **kwargs):
        #Creates and registers in this router an MPLS client of class client_class.
        #If a client of same class is already registered return error
        #Only one client per protocol.
        if not issubclass(client_class, MPLS_Client):
            raise Exception("A subclass of MPLS_Client is required.")

        if self.get_client(client_class):
            raise Exception("Client already exists...")

        new_client = client_class(self, **kwargs)  # we refer this client to ourselves.
        self.register_client(new_client)
        return new_client

    def register_client(self, mpls_client):
        # Adds the mpls_client instance to the router's registry.
        # Only one client per protocol is accepted, will override a previous entry.
        self.clients[mpls_client.protocol] = mpls_client
        self.label_managers[mpls_client.protocol] = self.main_label_manager

    def remove_client(self, client_class):
        if not issubclass(client_class, MPLS_Client):
            raise Exception("A subclass of MPLS_Client is required.")

        if self.get_client(client_class):
            del self.clients[mpls_client.protocol]
        raise  Exception("No registered client of class to remove...".format(type(client_class)))


    def get_FEC_owner(self, FEC):
        # Return the process owning the FEC.
        if FEC not in self.LIB.keys():
            return None
        else:
            return self.LIB[FEC]["owner"]


    def LIB_alloc(self, process, FEC, literal = None):
        # Allocates a local MPLS label to the requested FEC and return the allocated label.
        # If already allocated, return the label.
        # If literal is not None and self.numeric is False, then literal must be a NON-NUMERIC
        # string to be directly allocated to the LIB (Main use: MPLS VPN service interfaces)

        lm = self.label_managers[process.protocol]

        if FEC not in self.LIB.keys():
            # Allocate if it hasn't been yet
            if self.numeric_labels:
                self.LIB[FEC] = {"owner": process, "local_label": lm.next_label() }
            elif literal and isinstance(literal, str) and not literal.isnumeric():
                self.LIB[FEC] = {"owner": process, "local_label": literal }
            else:
                self.LIB[FEC] = {"owner": process, "local_label": str(lm.next_label()) }
            return self.LIB[FEC]["local_label"]

        #FEC is already in binding database.
        if self.get_FEC_owner(FEC) != process:
            return "The FEC is under control of other mpls client process {}".format(self.LIB[FEC]["owner"])

        # We just return the allocated label.
        return self.get_label(FEC)

    def get_FEC_from_label(self, label):
        # Return the FEC corresponding to a label.
        candidates =  [z[0] for z in self.LIB.items() if z[1]['local_label'] == label]  # 0 or 1 candidates only
        if candidates:
            return candidates[0]
        else:
            return None

    def get_label(self, FEC):
        # Get the label allocated to a FEC
        if FEC in self.LIB.keys():
            return self.LIB[FEC]["local_label"]
        return None

    def get_routing_entry(self, label):
        # Get the routing entry allocated to a local label
        if label in self.LFIB.keys():
            return self.LFIB[label]
        return None

    def get_routing_entries_from_FEC(self, FEC):
        # Get the label allocated to a (locally registered FEC)
        if FEC in self.LIB.keys():
            return self.get_routing_entry(self.LIB[FEC]["local_label"])

    def LFIB_alloc(self, local_label, routing_entry):
        # Append a routing entry for a local label in
        # the LFIB (Label Forwarding Information Database).
        if local_label in self.LFIB.keys():
            if routing_entry not in self.LFIB[local_label]:
                self.LFIB[local_label].append(routing_entry)
        else:
            self.LFIB[local_label] = [routing_entry]

    def LFIB_build(self, build_order = None):
        # Proc to build the LFIB from the information handled by each MPLS client
        # build_order = None (default) means attempt to build from all MPLS clients (ignore dependencies)

        for mpls_client_name, mpls_client in self.clients.items():    #iterate through all registered clients
            if not build_order or build_order == mpls_client.build_order:
                for fec in mpls_client.known_resources():  #iterate through the client resources
                    for label, routing_entry in mpls_client.LFIB_compute_entry(fec):  # for each computed entry...
                        if fec in self.LIB.keys() and self.LIB[fec]["owner"].protocol == mpls_client_name:
                            # If fec exists and is managed by this client, allocate the routing entry.
                            self.LFIB_alloc(label, routing_entry)

    def LFIB_weights_to_priorities(self):
        # Compute the priority of each route entry, remove weight, cast label as string...
        for label in self.LFIB.keys():
            metrics = sorted(set([x['weight'] for x in self.LFIB[label]]))
            for entry in self.LFIB[label]:
                entry['priority'] = metrics.index(entry["weight"])
                entry.pop("weight")

    def LFIB_refine(self):
        # Proc to refine (post-process) the LFIB
        for mpls_client_name, mpls_client in self.clients.items():    #iterate through all registered clients
            mpls_client.LFIB_fullrefine()
            for label in self.LFIB.keys():
                new_rules = mpls_client.LFIB_refine(label)
                if new_rules:
                    for new_rule in new_rules:
                        self.LFIB_alloc(label,new_rule)

    def get_number_of_rules(self):
        return len(self.LFIB.keys())

    def get_number_of_routing_entries(self):
        return np.sum([len(v) for v in self.LFIB.values()])

    def get_comm_count(self):
        return np.sum([s.get_comm_count() for s in self.clients.values()])

    def self_sourced(self, fec):
        # finds and calls right mpls client self_sourced() function
        owner = get_FEC_owner(fec)
        if not owner:
            return False
        try:
            return owner.self_sourced(fec)
        except:
            return False


    def to_aalwines_json(self):
        # Generates an aalwines json schema compatible view of the router.
        r_dict= {"interfaces": [], "name": str(self.name)}

        if self.alternative_names is list and len(self.alternative_names) > 0:
            r_dict["alias"] = [str(a) for a in self.alternative_names]

        # Now comes the first problem, as the FLIB is ordered by incoming interface...
        # We will just copy it on each interface.

        def _call(t):
            if isinstance(t[0],str) and t[0].startswith("NL_"):
                return False
            return True

        def _call2(t):
            if isinstance(t[0],str) and t[0].startswith("NL_"):
                return False
            return True

        regular_LFIB = dict_filter(self.LFIB, _call)
        service_LFIB = dict_filter(self.LFIB, lambda t: not _call(t))

        ifaces = []
        for x in self.get_interfaces():
            if x == self.name:
                iface = "i"+str(x)
            else:
                iface = str(x)
            ifaces.append(iface)
            # cases not yet implemented:
            #   - entries outputing to LOCAL_LOOKUP
            #   - multiple loopback interfaces


        ifaces.append(self.LOOPBACK)
        r_dict["interfaces"].append({"names":ifaces, "routing_table": regular_LFIB})

        r_dict["interfaces"].append({"name":self.LOCAL_LOOKUP, "routing_table": {}   })

        # process outside interfaces (non-MPLS)
        for x in self.get_interfaces(outside_interfaces = True):
            iface = str(x)
            # get service for this interface...

            s = self.get_client(MPLS_Service)
            vdict = s.get_service_from_ce(iface)
            vpn_name = list(vdict.keys())[0]
            def _call3(t):
                if isinstance(t[0],str) and vpn_name in t[0]:
                    return True
                return False

            srv_iface_LFIB = dict_filter(service_LFIB, _call3)

            rt = {"null": []}
            for _,re_list in srv_iface_LFIB.items():
                for re in re_list:
                    rt["null"].append(re)
            r_dict["interfaces"].append({"name":iface, "routing_table": rt })

        loc = self.get_location()
        if loc:
            r_dict["location"] = dict()
            r_dict["location"]["latitude"] = round(loc["latitude"], 4)     #fixed precision
            r_dict["location"]["longitude"] = round(loc["longitude"], 4)
        return r_dict

class oFEC(object):
    """
    Models a Forwarding Equivalence Class.
    This involves all packets that should be forwarded in the same fashion in the MPLS network.
    Commonly used to represent a network resource or group of resources.
    Examples: An IP prefix, a TE tunnel/LSP, a VRF, etc.

    FEC objects are managed by a MPLS client process and registered on a router's LIB.
    Required information for initialization:

    - fec_type: string, mandatory.
                Any value representing the fec type.

    - name:     string, mandatory.
                The name to identify this FEC in particular.

    - value:    arbitrary, optional. Defaults to None.
                Any kind of information that makes sense for this given FEC.
                Can be used for additional information, metadata, providing context, etc.

    Both 'name' and 'fec_values' are considered immutable so they must never be changed.
    The class allows to check equality of two oFEC objects directly.
    """

    def __init__(self, fec_type, name, value = None):
        self.fec_type = fec_type
        self.name = name
        self.value = value

    def __hash__(self):
        return hash((self.fec_type, self.name, self.value if not isinstance(self.value, dict) else 0))

    def __eq__(self, other):
        return isinstance(other, oFEC) and self.value == other.value and self.name == other.name and self.fec_type == other.fec_type

    def __str__(self):
        return "{}".format(self.name)

    def __repr__(self):
        return self.__str__()

class ProcRSVPTE(MPLS_Client):
    """
    Class implementing a Resource Reservation Protocol with Traffic Engineering
    extensions (RSVP-TE) client.

    This protocol is used for negotiation and setup of tunnels in the MPLS network.
    The tunnels could then be used as virtual interfaces for traffic forwarding, or as
    building blocks for other applications (e.g. L3VPNs)

    Tunnels provide traffic steering capabilities, as the netadmin can setup
    a tunnel with restrictions on which links can be used, what routers should the
    path go through or avoid, and even account for previously allocates bandwidth.
    Using the IGP's  Traffic Engineering database it can compute a constrained
    shortest path (CSPF) from the headend router (responsible for the tunnel itself)
    towards the tailend router.

    NOTE: currently only shortest path computation is supported!

    Manages the tunnels as resources/FECs.

    Provides FRR functionality by default according to RFC4090 with many-to-one (facility)
    option. This means that every router on the network will try to protect all TE tunnels
    against failure of the next downstream node, and if that is unfeasible, against failure
    of the downstream link. The operatoin implies pushing an new label to the stack.

    The fec_types it handles are "TE_LSP" and "bypass" (for FRR bypass lsps).

    Requires access to the router's routing table, nexthop information, the
    linkstate database to generate routing entries. Labels, both local and remotes, are
    allocated only on demand.

    This class has the following structures:

    headended_lsps:
        List of TE tunnels that are requested to start on this node.
        This is esentially configuration for the setup of new tunnels.
        Entry format:
            tunnel name => tunnel FEC , path as list of nodes.

    requested_lsps:
        Has an entry for every TE tunnel that traverses, starts or ends in this tunnel.
        Entries in this table can be generated from other routers.
        Entry format:
            tunnel name => {'FEC':fec, 'tuple': (<link or 3-tuple of nodes to be FRR protected>)}

    requested_bypasses:
        Has an entry for every link of 3-tuple of nodes for which a FRR protection bypass
        lsp must be computed. Entries in this table can be generated from other routers.
        Entry format:
            <link or 3-tuple of nodes to be FRR protected> => {'FEC': fec, 'next_hop': next_hop }


    For initialization indicate the router.
    """

    start_mode = 'manual'   # we must wait until all RSPV clients are initializated
                            # before starting to negotiate tunnels.
    protocol = "RSVP-TE"

    def __init__(self, router, max_hops = 3, **kwargs):
        super().__init__(router, **kwargs)
#         self.protocol = "RSVP-TE"
        self.build_order = 100

        self.bypass_cost = 16385          # Cost/weight allocated to bypass routes.
        self.frr_max_hops = max_hops      # Max number of additionals hops a FRR bypass may have.
        self.headended_lsps = dict()      # Table of tunnels configured to start in this router.
        self.requested_lsps = dict()      # Table of tunnels passing through this router.
        self.requested_bypasses = dict()  # Table of FRR backup paths passing through this router.


    def define_lsp(self, tailend, tunnel_local_id = 0, weight='weight',protection = None, **kwargs):
        # Store a request for a new tunnel; a headended LSP.
        # Compute the main path for it.
        # kwargs are reserved for future use, to pass restrictions for LSP creation.

        # Allowed protections:
        #  None: no protection at all.
        #  "facility-link": attempt only many-to-one link protection.
        #  "facility-node": First attempt many-to-one node protection, if imposible then link.
        #  "one-to-one": (WARNING: TBI) 1 to 1 protection, towards the tailend, trying to avoid overlapping with original LSP.

        # define tunnel name
        lsp_name = "lsp_from_{}_to_{}-{}".format(self.router.name, tailend, tunnel_local_id)
        # Check existence:
        if lsp_name in self.headended_lsps.keys():
            raise Exception("Requested creation of preexisting tunnel {}!. \
                            You might want to use a different tunnel_local_id.".format(lsp_name))

        G = self.router.topology

        constraints = None  #for future use
        if constraints:
            # Constraints can be a tuple of function allowing to build a view (subgraph)
            filter_node, filter_edge = constraints
            # Compute subgraph
            G = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
            # There could also be the 'subgraph_view' object itself
            # To consider link's delay, loss, price, or color, the topology itself should be already
            # labeled and the filter functions leverage that information.

        try:
            headend = self.router.name
            # compute spath: find the shortest path from headend to tailend.
            spath = nx.shortest_path(G, headend, tailend, weight=weight) #find shortest path, stored as list.
            length = len(spath) #numer of routers in path, including extremes.

            # Placeholder: attempt to find secondary path (To be implemented)
            sec_path = None
            # The idea is to (pre)compute a secondary path for the tunnel, sharing as few
            # links as possible with the primary path in order to be used as alternative
            # to FRR when the failure is adjacent to the headend, or as best possible path
            # after a failure FRR kicked in.
            #
            # A pseudocode idea follows, untested:
            # F = list(zip(spath[:-1],spath[1:]))  # transform the patyh from list of routers to list of links.
            # num_hops = len(F)
            # for i in range(num_hops-1):
            #     F = F[:-1]  # I avoid the furthest link if I have to choose.
            #     def filter_node_sec(n):
            #         #return False if n in L else True
            #         return True
            #     def filter_edge_sec(n1,n2):
            #         if (n1,n2) in F or (n2,n1) in F:
            #             return False
            #         return True
            #
            #     GG = nx.subgraph_view(G, filter_node = filter_node_sec, filter_edge = filter_edge_sec)
            #     try:
            #         sec_path = nx.shortest_path(GG, headend, tailend, weight=weight) #find shortest path, stored as list.
            #     except NetworkXNoPath:
            #         continue   #No path found, try with fewer constraints.

            cost = 0
            for i in range(length-1):
                cost += G.edges[spath[i],spath[i+1]][weight]

            #create a FEC object for this tunnel and store it in the pending configuration .
            self.headended_lsps[lsp_name] = { 'FEC': oFEC("TE_LSP",
                                                          lsp_name,
                                                          (self.router.name, tailend, protection)
                                                         ),
                                              'path': spath,
                                              'sec_path': sec_path,
                                              'cost': cost }
            return lsp_name

        except nx.NetworkXNoPath as err_nopath:
            return None

    def commit_config(self):
        # Iterates over the headended_lsps requests to request the
        # corresponding entries for lsps and bypases on the routers
        # along the main LSP path.
        #
        # This function should be executed on all nodes in the
        # network before actually asking for known resources.

        network = self.router.network

        for lsp_name, lsp_data in self.headended_lsps.items():

            G = self.router.topology   #change G to add restrictions...
            spath = lsp_data['path']   #shortest path result, as list of hops..
            sec_path = lsp_data['sec_path']   #alternative shortest path result, as list of hops..
            paths = [spath,sec_path]
            fec = lsp_data['FEC']      #an oFEC object with type="TE_LSP"
            protection = fec.value[2]  #protection mode

            length = len(spath)

            # create entry in local requested_lsps. value should be spath (a list object)
            self.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': (spath[0],spath[1]), 'path': spath}
            # a bad patch (not 100% RFC compliant) but if we don't make it, poor success
            if ((spath[0],spath[1]) not in self.requested_bypasses.keys()) and protection:
                self.requested_bypasses[(spath[0],spath[1])] = dict()

            #create entry in tailend
            tailend_pRSVP = network.routers[spath[-1]].clients["RSVP-TE"]
            self.comm_count += 1
            tailend_pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'path': spath}

            # create entry in requested_lsps of downstream nodes.
            # Value should be a 3-tuple or a 2-tuple only.
            # iterate on upstream direction.
            for i in range(length-2, 0, -1):
                # compute protections only for intermediate nodes.
                PLR = spath[i]       #PLR: Point of Local Repair
                pRSVP = network.routers[PLR].clients["RSVP-TE"]

                if i == length-2 or protection == "facility-link":
                    #penultimate node in LSP, circumvent the last edge
                    MP = spath[i+1]  # MP: Merge Point
                    protected_tuple = (PLR, MP) # the last edge

                else:
                    #circumvent next node
                    facility = spath[i+1] #original next node
                    MP = spath[i+2]
                    protected_tuple = (PLR,facility,MP)


                # create a protection for all LSPs that traverse the protected_tuple
                if protection in ["facility-link", "facility-node"]:
                    self.comm_count += 1
                    pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': protected_tuple,  'path': spath}

                    # create entry in requested_bypasses (if it doent's exist) of
                    # downstream nodes.
                    # Key should be the corresponding protected 3-tuple or 2-tuple.
                    # bypass_name = "bypass_{}_{}_{}".format(*triplet)
                    if protected_tuple not in pRSVP.requested_bypasses.keys():
                        pRSVP.requested_bypasses[protected_tuple] = dict()

                elif protection == "one-to-one":
                    # use tailend as MP
                    self.comm_count += 1
                    pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': (PLR,spath[-1]),  'path': spath}

                    # create entry in requested_bypasses (if it doent's exist) of
                    # downstream nodes.
                    # Key should be the corresponding protected 3-tuple or 2-tuple.
                    # bypass_name = "bypass_{}_{}_{}".format(*triplet)
                    if protected_tuple not in pRSVP.requested_bypasses.keys():
                        pRSVP.requested_bypasses[protected_tuple] = dict()

                elif not protection:
                    # use tailend as MP
                    self.comm_count += 1
                    pRSVP.requested_lsps[lsp_name] = {'FEC':fec, 'tuple': protected_tuple,  'path': spath}



    def compute_bypasses(self):
        # Compute all requested bypasses LSPs from the formation of LSPs during commit_config().
        #
        # This function must only be executed only execute after all RSVP-TE nodes
        # have executed and completed commit_config().
        # This function must be executed on all nodes in the network before actually requesting for known
        # resources.

        if not self.requested_bypasses:
            # No ProcRSVPTE requested bypasses for this router.
            return None

        def _get_path(view, head, tail, weight='weight', max_hops = self.frr_max_hops):
            # Auxiliary function.
            # Returns a shortest path from node "head" to node "tail" on
            # subgraph "view" of the full topology if exists. Subject to
            # having no more than "max_hops" hops.
            try:
                bypass_path = nx.shortest_path(view, head, tail, weight=weight)

                if bypass_path and len(bypass_path) >= max_hops + 2:
                    #bypass too large
                    bypass_path = None

            except nx.NetworkXNoPath as err_nopath:
                #No path!
                bypass_path = None

            finally:
                return bypass_path

        network = self.router.network
        # Iterate through the tuples to protect.
        bypass_path = None
        for protected_tuple in self.requested_bypasses.keys():

            G = self.router.topology   #change G to add constraints
            PLR = protected_tuple[0]   # Point of Local Repair
            MP = protected_tuple[-1]   # Merge Point

            if len(protected_tuple) == 3:
            #Node failure / NextNextHop  protection
                facility = protected_tuple[1] # potentially failed node

                # Auxiliary filter functions for this case
                def filter_node(n):
                    # Remove potentially failed node from the graph.
                    return True if n != facility else False
                def filter_edge(n1,n2):
                    return True

                # Compute subgraph
                view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
                # Get bypass path
                bypass_path = _get_path(view, PLR, MP, weight='weight', max_hops = self.frr_max_hops)

                if not bypass_path:
                    # if we can't protect the next node, attempt to protect the link.
                    MP = facility  # We will merge on the next node downstream instead of skipping it.


            if len(protected_tuple) == 2 or not bypass_path:
            #Link failure / NextHop  protection

                # Auxiliary filter functions for this case
                def filter_node(n):
                    return True
                def filter_edge(n1,n2):
                    # Remove potentially failed link from the graph.
                    return False if (n1,n2) == (PLR, MP) or (n2,n1) == (PLR, MP) else True

                # Compute subgraph
                view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
                # Get bypass path
                bypass_path = _get_path(view, PLR, MP, weight='weight', max_hops = self.frr_max_hops)

                if not bypass_path:
                    # We could not find a valid protection, move on to next request.
                    # Nevertheless an empty entry would remain in requested_bypasses.
                    continue


            # Found an usable bypass path! Create a FEC object for it.
            bypass_name = "bypass_{}".format("_".join([str(item) for item in protected_tuple ]))
            bypass = oFEC("bypass", bypass_name, protected_tuple)

            # Iterate through the bypass path
            length = len(bypass_path)
            for i in range(0, length):
                # Note: we compute protections only for intermediate nodes.
                #Get current router in path, and its RSVP process.
                curr_router = bypass_path[i]
                pRSVP = network.routers[curr_router].clients["RSVP-TE"]

                # Get its nexthop, it is needed to set up the path.
                if i+1 < length:
                    next_hop = bypass_path[i+1]
                else:
                    #for the MP, we will need a local label.
                    next_hop = None

                # Insert a bypass request on the router.
                if protected_tuple not in pRSVP.requested_bypasses.keys():
                    pRSVP.requested_bypasses[protected_tuple] = dict()  #initialize if necesary

                pRSVP.requested_bypasses[protected_tuple] = {'FEC': bypass,
                                                             'next_hop': next_hop,
                                                             'bypass_path': bypass_path }
                if curr_router != self.router.name:
                    self.comm_count += 1


    def known_resources(self):
        # Returns a generator to iterate over all tunnels and bypasses (RSVP-TE resources).

        if not self.requested_lsps and not self.requested_bypasses:
            # ProcRSVPTE in not initialized.
            return None

        #First return the bypases
        for bypass_tuple, bypass_data in self.requested_bypasses.items():
            if 'FEC' not in bypass_data.keys():
                # The requested bypass was impossible/unsolvable.
                continue
            fec = bypass_data['FEC']
            yield fec

        # return the tunnel LSPs
        for lsp_name, lsp_data in self.requested_lsps.items():
            fec = lsp_data['FEC']
            yield fec


    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is a tunnel or bypass identifier
        # The next_hop is already in the path information, but we will
        # require access to the next-hop allocated labels.

        router = self.router
        network = router.network

        # Starting with bypasses
        if  fec.fec_type == "bypass":
            if not fec.value:
                return   # this case should never happen.

            # recover the intended protected tuple.
            tuple_data = fec.value
            if router.name not in tuple_data:
                #intermediate router, process the bypass like a regular LSP
                next_hop_name = self.requested_bypasses[tuple_data]['next_hop']

                next_hop =  network.routers[next_hop_name]
                local_label = self.get_local_label(fec)
                remote_label = self.get_remote_label(next_hop_name, fec, do_count = False) #We cached it
                cost = self.bypass_cost

                # A swap op will do.
                if remote_label == self.IMPLICIT_NULL:
                    routing_entry = { "out": next_hop.name, "ops": [{"pop": ""}], "weight": cost  }
                else:
                    routing_entry = { "out": next_hop.name, "ops": [{"swap": remote_label}], "weight": cost  }
                yield (local_label, routing_entry)


            elif router.name == tuple_data[-1]:
                #this is the MP for the requested bypass. No next_hop, just pop.
                local_label = self.get_local_label(fec)
                routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                yield (local_label, routing_entry)

            #there is another possibility: that the router is the intermediate node. If this is the case it is because only a
            # link protection could be found  #FIX attempt number 1. #DEADLINE
            elif len(tuple_data) == 3 and router.name == tuple_data[1]:
                    #this is the MP for the requested bypass. No next_hop, just pop.
                    local_label = self.get_local_label(fec)
                    routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                    yield (local_label, routing_entry)



        if  fec.fec_type == "TE_LSP":
            headend, tailend, protection = fec.value

            # if 'tuple' not in self.requested_lsps[fec.name].keys():  #There should be a tidier way
            if router.name == tailend:
                #Then I am the tailend for this LSP.
                local_label = self.get_local_label(fec)
                routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                yield (local_label, routing_entry)

            else:
                # Regular case
                # Recover the protected tuple (subpath).
                tuple_data = self.requested_lsps[fec.name]['tuple']
                if len(tuple_data) == 3:     # Node protection
                    _, next_hop_name, MP_name = tuple_data
                elif len(tuple_data) == 2:   # Link protection
                    _, next_hop_name = tuple_data
                    MP_name = next_hop_name

                # Gather next_hop forwarding information
                next_hop =  network.routers[next_hop_name]
                local_label = self.get_local_label(fec)
                remote_label = self.get_remote_label(next_hop_name, fec, do_count = False) #we cached it
                headend_name = fec.value[0]
                headend_proc = network.routers[headend_name].get_client(type(self))
                cost = headend_proc.headended_lsps[fec.name]["cost"]

                # A swap is enough to build the main tunnel
                if remote_label == self.IMPLICIT_NULL:
                    main_entry = { "out": next_hop.name, "ops": [{"pop": ""}], "weight": cost  }
                else:
                    main_entry = { "out": next_hop.name, "ops": [{"swap": remote_label}], "weight": cost  }

                yield (local_label, main_entry)

                # If there is no bypass requested for the tuple/subpath, we are done.
                if not protection or tuple_data not in self.requested_bypasses.keys():
                # if tuple_data not in self.requested_bypasses.keys():
                    return


                # FRR -- Fast Re Route requirements.

                # Check feasability.
                bypass_data = self.requested_bypasses[tuple_data]
                if 'FEC' not in bypass_data.keys():
                    # The requested bypass is impossible/unsolvable, we are done.
                    return

                # Feasability confirmed, gather forwarding information
                bypass_fec = bypass_data['FEC']
                bypass_next_hop_name = bypass_data['next_hop']
                bypass_next_hop = network.routers[bypass_next_hop_name]
                bypass_path = bypass_data["bypass_path"]

                if MP_name != bypass_path:
                    MP_name = bypass_path[-1]  #force it
                # We need to know the label that the MP expects
                merge_label = self.get_remote_label(MP_name, fec)
                # And the bypass label for the FRR next hop
                bypass_label = self.get_remote_label(bypass_next_hop_name, bypass_fec, do_count = False) #we cahced it
                # Fix the entry priority
                cost = self.bypass_cost

                # This is a backup entry. It must push down a label on the stack to
                # encapsulate forwarding through the bypass path.

                backup_entry = { "out": bypass_next_hop.name, "weight": cost  }
                if merge_label == self.IMPLICIT_NULL and bypass_label == self.IMPLICIT_NULL:
                    r_info = [{"pop": ""}]
                elif merge_label == self.IMPLICIT_NULL:
                    r_info = [{"swap": bypass_label} ] #we don't push an IMPNULL
                else:
                    r_info = [{"swap": merge_label}, {"push": bypass_label} ]

                backup_entry["ops"] = r_info

                yield (local_label, backup_entry)

    def self_sourced(self, fec):
        # Returns True if this router is the tailend/mergepoint for this FEC.
        router = self.router
        network = router.network

        self_sourced = False

        if  fec.fec_type == "bypass":
            tuple_data = fec.value
            if router.name == tuple_data[-1]:
                # I am the MP.
                self_sourced = True

        elif fec.fec_type == "TE_LSP" and 'tuple' not in self.requested_lsps[fec.name].keys():
            # Then i am the tailend router. (There should be a tidier way to check this)
            self_sourced = True

        return self_sourced







########################
## WARNING! EXPERIMENTAL 4

class ProcRMPLS(ProcRSVPTE):

    protocol = "RMPLS"

    def __init__(self, router, max_hops = 999999999):
        super().__init__(router, max_hops)
        self.build_order = 10000
        self.bad_edge_pairs = []
        self.bad_edge_computations_covered = [] #We cache which edges we have done computation for
        self.path_cache = dict()
        print(f"Creating RMPLS process in router {router} )")

    def compute_bypasses(self):

        print(f"I am router {self.router.name} and will compute RMPLS bypasses for my own edges!")

        def _get_path(view, head, tail, weight='weight', max_hops = self.frr_max_hops):
            # Auxiliary function.
            # Returns a shortest path from node "head" to node "tail" on
            # subgraph "view" of the full topology if exists. Subject to
            # having no more than "max_hops" hops.
            try:
                bypass_path = nx.shortest_path(view, head, tail, weight=weight)

                if bypass_path and len(bypass_path) >= max_hops + 2:
                    #bypass too large
                    bypass_path = None

            except nx.NetworkXNoPath as err_nopath:
                #No path!
                bypass_path = None

            finally:
                return bypass_path

        network = self.router.network
        G = network.topology

        # for e in G.edges():
        for neig in G.neighbors(self.router.name):
            # Auxiliary filter functions for this case
            e = (self.router.name, neig )
            def filter_node(n):
                # Remove potentially failed node from the graph.
                # return False if n in L else True
                return True

            def filter_edge(n1,n2):
                # Remove potentially failed link from the graph.
                # return False if (n1,n2) == (PLR, MP) or (n2,n1) == (PLR, MP) else True
                return False if ((n1,n2) == e or (n2,n1) == e) else True

            def plug(e, bypass_path):
                if not bypass_path:
                    print(f"No valid rmpls bypass for edge {e}, evaluated False!")
                    return

                pprint(f"candidate rmpls bypass_path computed at {self.router.name} to protect {e}")
                bypass_name = "bypass_rmpls_{}".format("_".join([str(item) for item in e ]))
                bypass = oFEC("bypass_rmpls", bypass_name, e)

                # Iterate through the bypass path
                length = len(bypass_path)
                for i in range(0, length):
                    curr_router = bypass_path[i]
                    pRSVP = network.routers[curr_router].clients["RMPLS"]
                    if i+1 < length:
                        next_hop = bypass_path[i+1]
                    else:
                        #for the MP, we will need a local label.
                        next_hop = None

                    # Insert a bypass request on the router.
                    # not pRSVP.requested_bypasses[protected_tuple]
                    if e not in pRSVP.requested_bypasses.keys():
                        pRSVP.requested_bypasses[e] = dict()  #initialize if necesary

                    pRSVP.requested_bypasses[e] = {'FEC': bypass,
                                                   'next_hop': next_hop,
                                                   'bypass_path': bypass_path }
                    if curr_router != self.router.name:
                        self.comm_count += 1

            # Compute subgraph
            view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
            # Get bypass path
            bypass_path = _get_path(view, e[0], e[1], weight='weight', max_hops = self.frr_max_hops)

            plug(e, bypass_path)

    def known_resources(self):
        # Returns a generator to iterate over all tunnels and bypasses (RSVP-TE resources).

        if not self.requested_bypasses:
            # RMPLS in not initialized.
            return None

        for bypass_tuple, bypass_data in self.requested_bypasses.items():
            # for bypass_data in  self.requested_bypasses[bypass_tuple].items():
            if 'FEC' not in bypass_data.keys():
                # The requested bypass was impossible/unsolvable.
                continue
            fec = bypass_data['FEC']
            yield fec

    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is a tunnel or bypass identifier
        # The next_hop is already in the path information, but we will
        # require access to the next-hop allocated labels.

        router = self.router
        network = router.network
        print(f"Start RMPLS first step LFIB_compute_entry (router {router.name})")
        # Starting with bypasses
        try:
            if fec not in used_ifaces.keys():
                used_ifaces[fec] = []   #for each FEC an interface can appear in at most a single entry.
        except:
            used_ifaces = dict()  #initialize
            used_ifaces[fec] = []


        if  fec.fec_type == "bypass_rmpls":
            if not fec.value:
                return   # this case should never happen.

            # recover the intended protected tuple.
            tuple_data = fec.value

            if (router.name not in tuple_data) or router.name == tuple_data[0]:
                #intermediate router or PLR,

                data = self.requested_bypasses[tuple_data]
                bypass_path = data['bypass_path']
                next_hop_name = data['next_hop']
                if next_hop_name in used_ifaces[fec]:
                    # interface fec already used for this FEC, we can't allocate it again.
                    return

                next_hop =  network.routers[next_hop_name]

                local_label = self.get_local_label(fec)
                remote_label = self.get_remote_label(next_hop_name, fec, do_count = False) # We already know this
                cost = self.bypass_cost

                # A swap op will do in the normal case.
                if router.name == tuple_data[0]:
                    # re_ops = [{"push": remote_label} ]
                    re_ops = [{"swap": remote_label} ]
                elif remote_label == self.IMPLICIT_NULL:
                    re_ops = [{"pop": ""}]
                else:
                    re_ops = [{"swap": remote_label}]

                routing_entry = { "out": next_hop_name, "ops": re_ops, "weight": cost  }
                used_ifaces[fec].append(next_hop_name)
                yield (local_label, routing_entry)

            elif router.name == tuple_data[-1]:
                #this is the MP for the requested bypass. No next_hop, just pop.
                local_label = self.get_local_label(fec)
                routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
                yield (local_label, routing_entry)

    def get_protection_path(self, edge):
        # NOTE: This is a non-local access. TODO: Distribute this information in an earlier phase.
        try:
            if edge not in self.path_cache.keys():
                self.comm_count += 1
                pRMPLS = self.router.network.routers[edge[0]].clients["RMPLS"]
                self.path_cache[edge] = pRMPLS.requested_bypasses[edge]["bypass_path"]
            return self.path_cache[edge]
        except:
            return None

    def get_protection_path_edges(self, edge):
        router_path = self.get_protection_path(edge)
        if not router_path:
            return None
        path = list(zip(router_path[:-1], router_path[1:]))
        return path

    def bad_edge_dfs(self, current_edge, present_set, visited):
        # First check feasibility
        if current_edge in present_set or len(present_set.intersection(visited)) > 0:
            # 'Good' loop i.e. it will not occur, since a link was assumed to be both active and failed
            return None
        # Check if loop was detected
        if current_edge in visited:
            return [current_edge]
        # Continue search on children

        path = self.get_protection_path_edges(current_edge)
        if not path:
            return None
        new_present = set()
        for e in path:
            if (current_edge,e) not in self.bad_edge_pairs:
                back_track = self.bad_edge_dfs(e, present_set.union(new_present), visited.union({current_edge}))
                if back_track is not None:
                    if len(back_track) < 2 or back_track[0] != back_track[-1]: # back_track not yet contains full cycle
                        back_track = [current_edge] + back_track # add current to back_track
                    return back_track
            new_present.add(e)
        return None

    def bad_edge_dfs_go(self, edge):
        cycle = self.bad_edge_dfs(edge, set(), set())
        if cycle is not None:
            assert(len(cycle) > 0 and cycle[0] == cycle[-1])
            potential_bad_edge_pairs = list(zip(cycle[:-1], cycle[1:])) # We need to remove at least one of these edge pairs to break the cycle.
            if len(potential_bad_edge_pairs) == 2: # If cycle has length 2, we can safely remove both edge-pairs.
                assert(len(set(self.bad_edge_pairs).intersection(set(potential_bad_edge_pairs))) == 0)
                self.bad_edge_pairs += potential_bad_edge_pairs # Store and use in future computations
            else: # For longer cycles, we only remove one edge-pair to break cycle, as these pairs can be good in some other scenarios.
                chosen_bad_edge_pair = sorted(potential_bad_edge_pairs)[0] # Take smallest pair, given some global edge order.
                assert(chosen_bad_edge_pair not in self.bad_edge_pairs)
                self.bad_edge_pairs.append(chosen_bad_edge_pair) # Store and use in future computations
            return True
        return False

    def compute_bad_edge_pairs_from(self, edge):
        # This function computes the pairs of links (A,B) such that when on the bypass_path for A, if B also fails, we will not enter B's bypass_path like we normally would with RMPLS.
        # We do this to avoid forwarding loops. The algorithm is designed to guarantee that no forwarding loops can occur when using the computer bad_edge_pairs.
        if edge not in self.bad_edge_computations_covered: # Only compute once for each starting edge
            self.bad_edge_computations_covered.append(edge)
            while True: # Keep adding bad edge pairs as long as cycles are found.
                change = self.bad_edge_dfs_go(edge)
                if not change:
                    break

    def LFIB_refine(self, label):
        # Some process might require a refinement of the LFIB.
        r = self.router
        entries = r.get_routing_entry(label)

        fec_0 = r.get_FEC_from_label(label)
        if not fec_0:
            return []
        protected_edge = fec_0.value


        Q = sorted(entries, key = lambda x:x['priority'])
        max_prio = max([v['priority'] for v in Q]) + 1
        additional_entries = []
        for q in Q:

            e_1 = (r.name, q["out"])
            e_2 = (q["out"], r.name)
            if e_1 in self.requested_bypasses:
                if "FEC" not in self.requested_bypasses[e_1]:
                    #Couldn't find a protection or alternative path.
                    continue
                fec = self.requested_bypasses[e_1]["FEC"]
                bypass_path = self.requested_bypasses[e_1]["bypass_path"]
                now_failing_edge = e_1
            elif e_2 in self.requested_bypasses:
                if "FEC" not in self.requested_bypasses[e_2]:
                    #Couldn't find a protection or alternative path.
                    continue
                fec = self.requested_bypasses[e_2]["FEC"]
                bypass_path = self.requested_bypasses[e_2]["bypass_path"]
                now_failing_edge = e_2 #Not sure why e_2 is here, but meh..
            else:
                continue

            self.compute_bad_edge_pairs_from(now_failing_edge)

            protection_label = r.get_label(fec)   #fixme: get_local_label?
            new_ops = q["ops"] + [{"push": protection_label}]
            if (protected_edge, now_failing_edge) not in self.bad_edge_pairs:   # New Loop protection.
                routing_entry = { "out": r.LOCAL_LOOKUP, "ops":new_ops , "priority": q["priority"] + max_prio  }
                print(f"MPLS: add entry {routing_entry}")
                additional_entries.append(routing_entry)

        return additional_entries

## WARNING! EXPERIMENTAL 4
########################

########################
## WARNING! EXPERIMENTAL 5

class ProcPlinko(ProcRSVPTE):
    """
    Class implementing Plinko on top of RSVP RSVP-TE client.

    Manages the tunnels as resources/FECs.

    For initialization indicate the router.
    """

    start_mode = "manual"   # we must wait until all clients are initializated
                            # before starting to negotiate tunnels.
    protocol = "ProcPlinko"

    def __init__(self, router, max_hops = 9999999999999999):
        super().__init__(router)
#         self.protocol = "RSVP-TE"
        self.build_order = 200

        self.bypass_cost = 16385       # Cost/weight allocated to bypass routes.
        self.frr_max_hops = max_hops   # Max number of additionals hops a FRR bypass may have.
        self.headended_lsps = dict()   # Tunnels starting from this node.
        self.routes = dict()           # store the route objects here.


    def define_lsp(self, tailend, tunnel_local_id = 0, weight='weight',protection = "plinko", **kwargs):
        # Store a request for a new tunnel; a headended LSP.
        # Compute the main path (0-resilient) for it.

        # define tunnel name
        lsp_name = "lsp_from_{}_to_{}-{}".format(self.router.name, tailend, tunnel_local_id)

        #Check existence:
        if lsp_name in self.headended_lsps.keys():
            raise Exception("Requested creation of preexisting tunnel {}!. \
                            You might want to use a different tunnel_local_id.".format(lsp_name))

        G = self.router.topology    # change G to add restrictions...

        try:
            headend = self.router.name
            # compute spath: find the shortest path from headend to tailend.
            spath = nx.shortest_path(G, headend, tailend, weight=weight) #find shortest path, stored as list.
            length = len(spath) #numer of routers in path, including extremes.

            cost = 0   #path cost
            for i in range(length-1):
                cost += G.edges[spath[i],spath[i+1]][weight]

            #create a FEC object for this tunnel and store it in the pending configuration.
            self.headended_lsps[lsp_name] = { 'FEC': oFEC("TE_LSP",
                                                          lsp_name,
                                                          (self.router.name, tailend, protection)
                                                         ),
                                              'path': spath,
                                              'cost': cost,
                                              'r': 0,       # resiliency level.
                                              'F': set() }  # Failed edges set.
            #data is tuple (headend, tailend), to ease recognition.
            return lsp_name

        except nx.NetworkXNoPath as err_nopath:
            return None

    def commit_config(self):
        # Iterates over the headended_lsps requests to request the
        # corresponding entries for lsps and bypases on the routers
        # along the main LSP path.
        #
        # This function should be executed on all nodes in the
        # network before actually asking for known resources.

        network = self.router.network

        for lsp_name, lsp_data in self.headended_lsps.items():

            G = self.router.topology   #change G to add restrictions...
            spath = lsp_data['path']   #shortest path result, as list of hops..
            fec = lsp_data['FEC']      #an oFEC object with type="TE_LSP"
            r = lsp_data['r']
            F = lsp_data['F']
            cost = lsp_data['cost']

            # create entries on all routers on path
            for v in spath:
                pRSVP = network.routers[v].clients[self.protocol]
                pRSVP.routes[lsp_name] = { 'FEC': fec,
                                            'path': spath,
                                            'cost': cost,
                                            'r': r,
                                            'F': F,
                                            'match_on_FEC':  None }
                if v != self.router.name:
                    self.comm_count += 1

    def add_protections(self, resiliency, weight='weight'):
        # Compute all requested bypasses LSPs from the formation of LSPs during commit_config().
        #
        # This function must only be executed only execute after all RSVP-TE nodes
        # have executed and completed commit_config().
        # This function must be executed on all nodes in the network before actually requesting for known
        # resources.

        assert resiliency > 0

        def _get_path(view, head, tail, weight='weight', max_hops = self.frr_max_hops):
            # Auxiliary function.
            # Returns a shortest path from node "head" to node "tail" on
            # subgraph "view" of the full topology if exists. Subject to
            # having no more than "max_hops" hops.
            try:
                bypass_path = nx.shortest_path(view, head, tail, weight=weight)

                if bypass_path and len(bypass_path) >= max_hops + 2:
                    #bypass too large
                    bypass_path = None

            except nx.NetworkXNoPath as err_nopath:
                #No path!
                bypass_path = None

            finally:
                return bypass_path

        network = self.router.network
        me = self.router.name

        # scan local routes and
        # only protect paths from the previous resiliency level!
        to_be_protected = [x for x in self.routes.items() if x[1]['r'] == resiliency-1 ]
        for lsp_name, route_data in to_be_protected:
            fec = route_data["FEC"]
            path = route_data["path"]
            F = route_data["F"]

            pos = path.index(me)   # position of self in path

            if pos == len(path)-1:
                continue   #done, reached the tailend

            tailend = path[-1]
            e = (path[pos],path[pos+1])  # next edge in the path.
            if e in F or (path[pos+1],path[pos]) in F:
                return #WE had already considered this failed link! THis should never happen. Raise alarm?

            new_F = F.union([e]) #new failed set, must pass as list to preserve tuple

            # compute the protection
            G = self.router.topology

            # Auxiliary filter functions for this case
            def filter_node(n):
                return True
            def filter_edge(n1,n2):
                # Remove failed links from the graph.
                return False if (n1,n2) in new_F or (n2,n1) in new_F else True

            # Compute subgraph
            view = nx.subgraph_view(G, filter_node = filter_node, filter_edge = filter_edge)
            # Get bypass path
            prot_path = _get_path(view, me, tailend, weight='weight', max_hops = self.frr_max_hops)

            if not prot_path:
                # We could not find a valid protection, move on to next request.
                # Nevertheless an empty entry would remain in requested_bypasses.
                continue


            if lsp_name.startswith("plinko"):
                prot_name = f"{lsp_name}-p{resiliency}-h{self.router.name}"
            else:
                prot_name = f"plinko-{lsp_name}-p{resiliency}-h{self.router.name}"

            cost = 0   #path cost
            for i in range(len(prot_path)-1):
                cost += G.edges[prot_path[i],prot_path[i+1]][weight]

            prot_fec = oFEC("plinko_resilient", prot_name, (me, tailend, "plinko") )

            for v in prot_path:
                pRSVP = network.routers[v].clients[self.protocol]
                mofec = None
                if v == me:
                    mofec = route_data['match_on_FEC'] # the lowest resiliency being protected here...
                    if not mofec:
                        mofec = fec

                pRSVP.routes[prot_name] = { 'FEC': prot_fec,
                                            'path': prot_path,
                                            'cost': cost,
                                            'r': resiliency,         # resiliency level.
                                            'F': new_F,              # Failed edges set.
                                            'match_on_FEC': mofec }   # What are we protecting?
                if v != self.router.name:
                    self.comm_count += 1

    def known_resources(self):
        # Returns a generator to iterate over all routes.

        if not self.routes:
            # ProcRSVPTE in not initialized.
            return None

        for route_data in self.routes.values():
            yield route_data['FEC']


    def LFIB_compute_entry(self, fec, single = False):
        # Return a generator for routing entries for the requested FEC.
        # The FEC is a tunnel or bypass identifier
        # The next_hop is already in the path information, but we will
        # require access to the next-hop allocated labels.

        router = self.router
        network = router.network

        # Start with bypasses
        route_name = fec.name
        headend, tailend, protection = fec.value
        route_data = self.routes[route_name]
        local_label = self.get_local_label(fec)
        rpath = route_data["path"]
        pos = rpath.index(router.name)
        r = route_data["r"]   #resiliency

        if router.name == tailend:
            routing_entry = { "out": self.LOCAL_LOOKUP, "ops": [{"pop": "" }], "weight": 0  }
            yield (local_label, routing_entry)

        else:
            next_hop_name = rpath[pos+1]
            remote_label = self.get_remote_label(next_hop_name, fec, do_count = False) # Already cached
            if router.name == headend:
                if remote_label == self.IMPLICIT_NULL:
                    routing_entry = { "out": next_hop_name, "ops": [{"pop": ""}], "weight": r  }
                else:
                    routing_entry = { "out": next_hop_name, "ops": [{"swap": remote_label}], "weight": r  }
                if r > 0:
                    # we need the label of the route boing protected
                    local_label = self.get_local_label(route_data['match_on_FEC'])
                    yield (local_label, routing_entry)
                else:
                    yield (local_label, routing_entry)
            else:
                # A swap op will do.
                if remote_label == self.IMPLICIT_NULL:
                    routing_entry = { "out": next_hop_name, "ops": [{"pop": ""}], "weight": r  }
                else:
                    routing_entry = { "out": next_hop_name, "ops": [{"swap": remote_label}], "weight": r  }

                yield (local_label, routing_entry)



    def self_sourced(self, fec):
        # Returns True if this router is the tailend/mergepoint for this FEC.
        router = self.router
        network = router.network

        self_sourced = False

        route_name = fec.name
        headend, tailend, protection = fec.value
        #route = self.routes[route_name]
        if router.name == tailend:
            self_sourced = True

        return self_sourced

## WARNING! EXPERIMENTAL 5
########################

class ProcStatic(MPLS_Client):
    """
    Class implementing static MPLS forwarding entries, or to override labels previously
    allocated by other MPLS protocols.

    This is the more flexible yet administratively expensive option to build MPLS
    forwarding; there is no automatic interaction between routers for this mechanism,
    so the admin is responsible for building hop by hop tunnels.

    It can also 'hijack' local labels owned by other protocols to implement hacks.

    Manages the tunnels as resources/FECs.

    This class has the following structures:

    static_mpls_rules:
        List of TE tunnels that are requested to start on this node.
        This is esentially configuration for the setup of new tunnels.
        Entry format:
            (static) lsp name => static FEC, 'label_override': <local label to be instered or overriden>

    The static FEC is of type "STATIC_LSP" and includes:
        - lsp_name -> the FEC name
        - value -> manual routing information.0

    For initialization indicate the router.
    """

    start_mode = 'manual'  # All static labels must be allocated manually.
    protocol = "STATIC"

    def __init__(self, router):
        super().__init__(router)
#         self.protocol = "STATIC"
        self.static_mpls_rules = dict()
        self._cur_count = count()  #Counter for autogenerated names of static entries.

    def define_lsp(self, lsp_name, ops, outgoing_interface, incoming_interface = None,
                   priority=0, label_override = None):
        #
        # Define a static MPLS entry.
        # Inputs:
        # - lsp_name: the FEC name. Must be unique per router. To autogenerate, set lsp_name="".
        # - ops: The full ordererd set of MPLS stack manipulation for a matching packet.
        # - outgoing_interface: Next hop interface.
        # - incoming_interface: For per interface label space. Defaults to None (per-platform label space)
        # - priority: routing entry priority.
        # - label_override: the local label to be inserted. If it is already allocated, then it
        #                   will be overriden. Leave empty is a fresh new label must be allocated.

        # TODO: Add verifications here.

        # Create an automatic incremental local lsp name.
        if lsp_name == "":
            lsp_name = "static_lsp_{}_{}".format(self.router.name, next(self._cur_count))

        #compose the routing entry.
        data = {'ops':ops, 'out': outgoing_interface, 'priority': priority}
        if not incoming_interface:
            data['in'] = incoming_interface

        # Insert the request on the table
        self.static_mpls_rules[lsp_name] = { 'FEC': oFEC("STATIC_LSP", lsp_name, value = data),
                                             'label_override': label_override}
        return lsp_name

    def known_resources(self):
        # Returns a generator to iterate over all requested static MPLS entries.
        if not self.static_mpls_rules:
            # If there are no requests, we are done.
            return None

        for lsp_name, lsp_data in self.static_mpls_rules.items():
            fec = lsp_data['FEC']
            yield fec


    def alloc_labels_to_known_resources(self):
        # Requests the router for allocation of labels for each known FEC (resource).
        # The inherited functionality is not enough for the case of label override.

        for fec in self.known_resources():
            if self.static_mpls_rules[fec.name]['label_override']:
                # Get requested label
                local_label = self.static_mpls_rules[fec.name]['label_override']

                # If exists, delete previously allocated FEC for that label.
                prev_FEC = self.router.get_FEC_from_label(local_label)
                if prev_FEC:
                    del self.router.LIB[prev_FEC]

                # manual forced entry in LIB (a hack)
                self.router.LIB[fec] = {'owner': self.process, 'local_label': local_label }

            else:
                # Regular case
                self.LIB_alloc(fec)


    def LFIB_compute_entry(self, fec, single = False):
        # Each client must provide an implementation.
        # Return a generator for routing entries for the
        # requested static FEC: returns a generator of
        # (label, routing_entry).

        router = self.router
        network = router.network

        if not fec.value:
            return

        # Compose the routing entry from the FEC information.
        ops = fec.value['ops']
        out = fec.value['out']
        priority = fec.value['priority']
        routing_entry = { "out": next_hop.name, "ops": ops, "priority": priority  }

        # If required append incoming interface condition.
        if 'in' in fec.value.keys():
            incoming_interface = fec.value['in']
            routing_entry['incoming'] = incoming_interface

        # Get local label from LIB
        local_label = self.get_local_label(fec)

        yield (local_label, routing_entry)


class MPLS_Service(MPLS_Client):
    """
    Class implementing a MPLS VPN service.

    This class is a skeleton for services allocating labels to different
    virtual private network schemas.

    Manages tuples of VPNs and attached CEs as resources/FECs.
    The fec_types might reflect different offerings.

    Requires access to the router's routing table, nexthop information,
    the linkstate database, local LDP/RSVP-TE and the remote labels to
    generate routing entries.

    Examples of services that might be emulated include point-to-point
    VPNs (pseudowires) as well as VPLS (Martini draft), yet this is not
    implemented yet.

    For initialization indicate the router.
    """

    start_mode = 'manual' # the network will try to allocate labels immediately.
    protocol = "service"


    def __init__(self, router, max_hops = 3):
        super().__init__(router)
        self.build_order = 1000

        # 'services' keys are services names, their values are dicts containing
        # service type, list of locally attached CEs and an optional service
        # description. vnp_name => {ces:[], vpn_type: , vpn_descr: "" }
        self.services = dict()
        self.fec_type = "vpn_endpoint"

    def define_vpn(self, vpn_name, vpn_type="default", vpn_descr=""):

        if vpn_name in self.services:
            raise Exception("VPN {} already defined".format(vpn_name))
        self.services[vpn_name] = {"ces":[], "vpn_type":vpn_type, "vpn_descr": vpn_descr, "ack": False}


    def remove_vpn(self, vpn_name, vpn_type="default", vpn_descr=""):
        if vpn_name not in self.services:
            raise Exception("Can't remove a VPN (){}) that is not yet defined".format(vpn_name))
        del self.services[vpn_name]

    def attach_ce(self, vpn_name, ce):
        if vpn_name not in self.services:
            raise Exception("VPN {} is not defined.".format(vpn_name))
        self.services[vpn_name]["ces"].append(ce)

    def dettach_ce(self, vpn_name, ce):
        if vpn_name not in self.services:
            raise Exception("VPN {} is not defined.".format(vpn_name))
        self.services[vpn_name]["ces"].remove(ce)

    def get_service_from_ce(self,ce):
        def _call(t):
            return True if ce in t[1]["ces"] else False

        return dict_filter(self.services, _call)

    def locate_service_instances(self, vpn_name):
        network = self.router.network

        # Filter routers that have services 'vpn_name' instantiated
        s_objs = [s.clients[self.protocol] for s in network.routers.values() if self.protocol in s.clients.keys()]
        s_instances = [s for s in s_objs if vpn_name in s.services.keys() ]

        return s_instances


    def known_resources(self):
        # Return FECs for each attached circuit (pe,ce) of each vpn instantiated on this router.
        for vpn_name, vpn_data in self.services.items():
            for service in self.locate_service_instances(vpn_name):  #this includes me!!
                vpn_data = service.services[vpn_name]
                if not vpn_data["ack"]:
                    service.services[vpn_name]["ack"] = True
                    self.comm_count += 1

                pe = service.router.name
                for ce in vpn_data["ces"]:
                    yield oFEC(self.fec_type,"vpn_ep_{}_{}_{}".format(vpn_name, pe, ce), (vpn_name, pe, ce))

    def alloc_labels_to_known_resources(self):
        # Asks the router for allocation of labels for each known FEC (resource)
        for fec in self.known_resources():
            if self.self_sourced(fec):
                self.LIB_alloc(fec)
            else:
                # Don't allocate an actual label, directly the name.
                # This introduces a non-determinism (indirection)
                # that has to be solved by the FIB (e.g: on VPLS,
                # by ading MAC learning)
                # self.LIB_alloc(fec, literal = "NL_"+fec.name)   # NL_ stands for NO LABEL, null.

                # what if I just assume that some process will map packets to
                # the correconding FEC (service, router, ce) and here I just
                # care about reacheability
                self.LIB_alloc(fec)

    def LFIB_compute_entry(self, fec):
        # Each client must provide an generator to compute routing entries given the fec.
        # returns tuple (label, routing_entry)
        # routing entries have format:
        #  routing_entry = { "out": next_hop_iface, "ops": [{"push"|swap": remote_label}], "weight": cost  }

        # only PEs are concerned!
        # 1. Gather all PEs where the service is instantiated: SPEs (Service PEs)
        if fec.fec_type != self.fec_type:
            # invalid fec type, nothing to do!
            return

        vpn_name, pe, ce = fec.value
        network = self.router.network

        if vpn_name not in self.services.keys():
            # vpn not instantiated here
            return

        pe_router = network.routers[pe]
        pe_service_proc = pe_router.clients[self.protocol]
        if vpn_name not in pe_service_proc.services.keys():
            # vpn not instantiated on pe router (should not happen...)
            return

        # 2. Find local LSP to each PE
        local_label = self.get_local_label(fec)
        if pe == self.router.name  and self.self_sourced(fec):
            # compute entries for locally attached service interfaces (local ACs)
            routing_entry = { "out": ce, "ops": [{"pop": "" }], "weight": 0  }
            yield (local_label, routing_entry)

        else:
            # compute entries for remotely attached service interfaces (remote ACs)

            service_label = self.get_remote_label(pe, fec)
            tunnel_label = None

            # try RSVP-TE:
            lsp_name = "lsp_from_{}_to_{}".format(self.router.name, pe)
            candidate_tunnel_fec = self.get_fec_by_name_matching(lsp_name)

            if candidate_tunnel_fec:
                # get list of fecs with lower weight (might be many)
                refined = []
                curr_weight = 9999999999999   # arbitrary initial high value
                for cfec in candidate_tunnel_fec:
                    best = None
                    clabel = self.get_local_label(cfec)
                    for routing_entry in as_list(self.router.get_routing_entry(clabel)):
                        if routing_entry["weight"] < curr_weight:
                            best = cfec
                            curr_weight = routing_entry["weight"]
                    refined.append((best,curr_weight))

                curr_weight = min(refined,key=lambda x:x[1])[1]
                refined = [ tupl[0] for tupl in refined if  tupl[1] == curr_weight]

                # Choose first tunnel. (This introduces a non-determinism, I could have done other things...)
                tunnel_label = self.get_local_label(refined[0])

            if not tunnel_label:
                # try LDP:
                candidate_tunnel_fec = oFEC("loopback","lo_{}".format(pe), pe)
                tunnel_label = self.get_local_label(candidate_tunnel_fec)

            if not tunnel_label:
                # ?I don't know how to reach pe_router inside MPLS
                return


            # 3. For each local fwd rule, add a new one swapping!! the service label.
            for routing_entry in as_list(self.router.get_routing_entry(tunnel_label)):
                new_ops = routing_entry["ops"].copy()
                if "swap" in new_ops[0].keys():
                    lsp_label = new_ops[0]["swap"]
                    new_ops[0] = {"push": lsp_label} # recall: swap = pop + push, drop the pop.
                    new_ops.insert(0, {"swap": service_label})
                elif "pop" in new_ops[0].keys() and self.router.php:
                    # packet coming from AC (attached CE) don't have MPLS labels, drop the pop op.
                    # We know this can happen with PHP enabled.
                    new_ops[0] = {"swap": service_label} # replace the pop.
                else:
                    new_ops.insert(0, {"swap": service_label})

                new_routing_entry = { "out":routing_entry["out"],
                                      "ops": new_ops,
                                      "weight": routing_entry["weight"]  }

                yield (local_label, new_routing_entry )

    def self_sourced(self, fec):
        # Returns True if the FEC is sourced or generated by this process.
        self_sourced = False
        if  fec.fec_type == self.fec_type and type(fec.value) is tuple and len(fec.value) == 3:
            vpn_name, pe, ce = fec.value
            if vpn_name in self.services.keys():
                if pe == self.router.name and ce in self.services[vpn_name]["ces"]:
                    self_sourced = True
        return self_sourced

    def get_remote_label(self, router_name, fec):
        # Gets the label allocated by router <router_name> to the FEC <fec>
        router = self.router.network.routers[router_name]
        owner = router.get_FEC_owner(fec)
        if router.php and owner and owner.self_sourced(fec) and fec.fec_type != self.fec_type:
            # for services we ignore PHP!
            return self.IMPLICIT_NULL
        else:
            return self.router.network.routers[router_name].get_label(fec)


################################################################################
### SIMULATION FUNCTIONALITIES AND CLASSES

class MPLS_packet(object):
    """
    Class intended for packet tracing, for debugging or testing purposes.

    Each packet has a stack,
    and records its paths while it is forwarded through the network in variables
    "traceroute" and "link_traceroute".

    Params:
    - network             : The network object on which the packet will live.
    - init_router         : Initial router (on th eMPLS domain) for the packet.
    - init_stack          : (default empty list). The stack the packet has initially.
    - mode                : 'packet'      -- simulates a single packet forwarding. default.
                            'pathfinder'  -- identifies all possible paths in a given topology. Costly.
    - restricted_topology : Topology with missing links/nodes for the simulation.
    - verbose             : (default False). Outpt lots of information.

    This is just a proof of concept, must be further developed.
    """

    def __init__(self, network, init_router, targets, init_stack = [], restricted_topology = None, mode="packet", max_ttl = 2550, verbose = False):
        self.network = network
        self.ttl = max_ttl
        if restricted_topology is not None:
            self.topology = restricted_topology
        else:
            self.topology = network.topology

        self.stack = init_stack
        self.init_stack = init_stack.copy()

        self.mode = mode   # options: packet, pathfinder.

        if isinstance(init_router, str):
            self.init_router = network.routers[init_router]
        elif isinstance(init_router, Router):
            self.init_router = init_router
        else:
            raise Exception("Unknown init_router")

        self.traceroute = [self.init_router]
        self.link_traceroute = [(self.init_router,None)]
        self.trace: list[tuple[str, str, str]] = [("",self.init_router.name, "|".join(self.init_stack))]
        self.state = "uninitialized"
        self.verbose = verbose
        self.alternatives = []
        self.cause = ""
        self.exit_code = None     # 0 means success, None is not finished yet, other are errors.
        self.success = None
        self.targets = targets
        self.num_hops = 0
        self.num_local_lookups = 0

        def except_false(f):
            try:
                return f()
            except:
                return False
        self.is_connected = any([except_false(lambda: nx.has_path(self.topology, self.init_router.name, tgt)) for tgt in self.targets])

    def info(self):
        print(".....INFO.....")
        print("Packet from {} with initial stack: [{}]" .format(self.init_router.name, "|".join(self.init_stack)))
        print("   current state: {}".format(self.state))
        print("   current cause: {}".format(self.cause))
        print("   current path: {}".format(self.traceroute))
        print("   current stack: {}".format("|".join(self.stack)))
        print("...INFO END...")

    def get_next_hop(self, outgoing_iface, verbose = True):
        # this implementation depends explicitly on interfaces named after the connected router
        curr_r = self.traceroute[-1]
        if verbose:
            print("Into get_next_hop")
            print(f" current_router {curr_r.name}")
            self.info()
            print(f"neighbors: {list(self.topology.neighbors(curr_r.name))}")
            print(f"Is {outgoing_iface} a neighbor?")
            print(outgoing_iface in self.topology.neighbors(curr_r.name))

            print("edges:")
            pprint(self.topology.edges())

            print(f"Let's see if we have a edge towards {outgoing_iface} ")
            print(  self.topology.has_edge(curr_r.name,outgoing_iface))
        if outgoing_iface in self.topology.neighbors(curr_r.name) and self.topology.has_edge(curr_r.name,outgoing_iface):
            return self.network.routers[outgoing_iface]
        else:
            return None

    def step(self):


        # Simulate one step: send the packet to the next hop.
        if self.state == "uninitialized":
            raise Exception("Can't move an UnInitialized packet.")

        if not self.stack:
            self.state = "finished"
            self.cause = " FORWARDING Complete: empty stack."
            self.exit_code = 0
            if self.verbose:
                print(self.cause)
                print("Current router: {}".format(self.traceroute[-1].name))
                print("Current outmost label: {}".format(None))
                print("stack: {}".format([]))
                print()
            if self.mode == "pathfinder":
                return (True, [])
            return True

        curr_r = self.traceroute[-1]
        outer_lbl = self.stack[-1]
        self.num_hops += 1

        if self.verbose:
            print()
            print("Current router: {}".format(curr_r.name))
            print("Current outmost label: {}".format(outer_lbl))

        # check and update time-to-live
        if self.ttl <= 0:
            self.state = "finished"
            self.cause = " FORWARDING Aborted: Time to live expired!"
            self.exit_code = 1
            if self.verbose:
                print(self.cause)
            if self.mode == "pathfinder":
                return (False, [])
            return False  # Time to live expired; possible loop.
        else:
           self.ttl -= 1

        if self.trace[-1] in self.trace[:-1]:
            self.state = "finished"
            self.cause = " FORWARDING Aborted: Loop detected!"
            self.exit_code = 5
            if self.verbose:
                if self.targets is not None:
                    print("WAS CONNECTED" if self.is_connected else "WAS NOT CONNECTED")
                print(self.cause)
            if self.mode == "pathfinder":
                return (False, [])
            return False  # Time to live expired; possible loop.

        # gather the forwarding rules.
        if outer_lbl in curr_r.LFIB:
            rules = curr_r.LFIB[outer_lbl]
        else:
        # if there are no rules
            self.state = "finished"
            self.cause = " FORWARDING Complete: No available forwarding rules at router {} for label {}, yet MPLS stack is not empty".format(curr_r.name, outer_lbl)
            self.exit_code = 2
            if self.verbose:
                if self.targets is not None:
                    print("WAS CONNECTED" if self.is_connected else "WAS NOT CONNECTED")
                print(self.cause)
            if self.mode == "pathfinder":
                return (False, [])
            return False  # Return false because there are no rules here yet the MPLS stack is not empty

        # get all available priorities in the rules
        priorities = sorted(list(set([x['priority'] for x in rules ])))

        # code split here...
        rule_list = []   # store here the acceptable forwarding rules.

        # now consider only the routes from higher to lower priority
        for prio in priorities:
            f_rules = list(filter(lambda x: x["priority"] == prio, rules))
            if len(f_rules) > 1:
                random.shuffle(f_rules)   #inplace random reorder, allows balancing, non determinism.

            for candidate in f_rules:
                # determine validity of outgoing interface and next_hop
                outgoing_iface = candidate["out"]
                next_hop = self.get_next_hop(outgoing_iface, verbose = False)

                # if it points to LOCAL_LOOKUP then that's it, that SHOULD NOT be balanced.
                # if not, but next_hop is reacheable in our topology, we also found a valid rule.
                if outgoing_iface == curr_r.LOCAL_LOOKUP:
                    #we found an usable route!
                    rule_list = [candidate]
                    break
                elif next_hop:
                    rule_list.append(candidate)
                elif not next_hop and outer_lbl:
                    #MPLS stack still not empty, maybe a VPN service CE?

                    try:
                        srvs = curr_r.clients["service"].services
                        y = [x['ces'] for x in srvs.values()]
                        if outgoing_iface in chain(*y):  #check local CEs
                            candidate["x-leaving"] = True
                            rule_list.append(candidate)
                    except:
                        # try next candidates
                        continue
                else:
                    # Exiting the MPLS domain.
                    self.state = "finished"
                    self.cause = " FORWARDING Complete: Exiting the MPLS domain at router {} for label {}".format(curr_r.name, outer_lbl)
                    self.exit_code = 3
                    if self.verbose:
                        if self.targets is not None:
                            print("WAS CONNECTED" if self.is_connected else "WAS NOT CONNECTED")
                        print(self.cause)
                    if self.mode == "pathfinder":
                        return (True, [])
                    return True  # I ended up considering this condition as a SUCCESS.

            if rule_list:
                # if this priority level has any usable rule we are done and move on.
                break

        # If we couldn't find a rule that is the end of the story.
        if not rule_list:
            self.state = "finished"
            self.cause = " FORWARDING Complete: Can't find a forwarding rule at router {} for label {}.".format(curr_r.name, outer_lbl)
            self.exit_code = 4
            if self.verbose:
                if self.targets is not None:
                    print("WAS CONNECTED" if self.is_connected else "WAS NOT CONNECTED")
                print(self.cause)
            if self.mode == "pathfinder":
                return (False, [])
            return False

        p_list = []
        for i in range(len(rule_list)):

            if i < len(rule_list) - 1:
                # I will advance the copies before myself... but in packet mode
                if self.mode == "packet":
                    continue
                p = copy.deepcopy(self)
            else:
                p = self

            rule = rule_list[i]
            outgoing_iface = rule["out"]
            next_hop = p.get_next_hop(outgoing_iface, verbose = False)

            if "x-leaving" in candidate.keys():
                # Exiting the MPLS domain to  a customr's CE
                candidate.pop("x-leaving")
                outgoing_iface = curr_r.LOCAL_LOOKUP

            if outgoing_iface == curr_r.LOCAL_LOOKUP:
                p.cause = " FORWARDING recurrent: attempt to process next level on {}".format(curr_r.name)
                self.num_local_lookups += 1
                self.exit_code = 5
                if p.verbose:
                    print(p.cause)
                # Recycle. We won't record and just run again on the same router, after processing the stack

            else:
                p.traceroute.append(next_hop)
                p.link_traceroute[-1] = (p.link_traceroute[-1][0], outgoing_iface)
                p.link_traceroute.append((next_hop, None))
                p.trace.append((curr_r.name, "|".join(self.stack), next_hop.name))


            # Processing the stack
            ops_list = rule["ops"]
            for op in ops_list:
                if "pop" in op.keys():
                    p.stack.pop()
                elif "push" in op.keys():
                    p.stack.append(op["push"])
                elif "swap" in op.keys():
                    p.stack.pop()
                    p.stack.append(op["swap"])
                else:
                    raise Exception("Unknown MPLS operation")

            if p.verbose:
                print(ops_list)
                print("matching rules: {}".format(len(rules)))
                print("fwd:")
                print(rules)
                if next_hop:
                    print("NH: {} ".format(next_hop.name))
                print("stack: {}".format(p.stack))
                print()

            p_list.append(p)

        if self.mode == "pathfinder":
            return (True, p_list)
        return True

    def initialize(self):
        if self.state == "uninitialized":
            self.state = "live"
            return True
        elif self.state == "live":
            # already initilizated.
            return True
        return False

    def fwd(self, random_seed = None):
        # Control high level forwarding logic.
        # we make sure that we can forward the packet.
        if not self.initialize():
            # We can't move forward, the packet is Finished.
            return False

        if random_seed:
            # We received a specific seed, so we reset.
            # Other wise just proceed.
            random.seed(self.random_seed)

        while not self.state == "finished":
            res = self.step()

        self.success = res   # store the result in the packet.
        return res  # we return the result of the last step before finish



class Simulator(object):
    """
    Class intended for running simulations.

    This is just a proof of concept, must be further developed.
    """
    def __init__(self, network, trace_mode = "router", restricted_topology = None, random_seed = random.random()):

        self.network = network
        self.traces = dict()
        # Map from load to list of switches
        self.trace_routes = dict()
        self.loads = dict()
        self.trace_mode= trace_mode
        self.random_seed = random_seed
        self.count_connected = 0
        self.looping_links = 0
        self.num_hops: dict[Tuple[str,str], int] = {}
        self.num_ll: dict[Tuple[str, str], int] = {}

        if restricted_topology is not None:
            self.topology = restricted_topology
            self.initial_links = len(self.topology.edges)
        else:
            self.topology = network.topology

    def run_blind(self):
        # Forward a packet for each label on LFIB entry
        random.seed(self.random_seed)
        for router_name, r in self.network.routers.items():
            self.traces[router_name] = dict()
            for in_label in r.LFIB:
                p = MPLS_packet(self.network, init_router = router_name, targets = None, init_stack = [in_label], verbose = True)
                res = p.fwd()
                self.traces[router_name][in_label] = [{"trace": p, "result": res}]

    def run(self, flows, verbose = False):
        loop_links = set()
        # Forward a packet for each flow in the 'flows' list, and return results and stats.

        # classify according to fec_type
        print(f"running simulation with seed {self.random_seed}" )
        random.seed(self.random_seed)

        labeled_flows = self.network.build_flow_table(flows)
        if verbose:
            pprint(labeled_flows)

        for router_name, lbl_items in labeled_flows.items():
            self.traces[router_name] = dict()

            for in_label, tup in lbl_items.items():
                good_sources, good_targets, load= tup
                if verbose:
                    print(f"\n processing router {router_name} with flow {good_sources} to {good_targets}")

                p = MPLS_packet(self.network, init_router = router_name, targets = good_targets, init_stack = [in_label],
                                verbose = verbose, restricted_topology = self.topology)
                res = p.fwd()

                # Traces, and whether the destination was reached.
                self.trace_routes[(good_sources[0], good_targets[0], load)] = ([router.name for router in p.traceroute], res)
                 

                last_router_name = p.traceroute[-1].name

                if res and last_router_name not in good_targets:
                    res = False

                if res:
                    self.num_hops[(tup[0][0], tup[1][0])] = p.num_hops
                    self.num_ll[(tup[0][0], tup[1][0])] = p.num_local_lookups
                elif p.is_connected:
                    from get_results import inf
                    self.num_hops[(tup[0][0], tup[1][0])] = inf
                    self.num_ll[(tup[0][0], tup[1][0])] = inf
                else:
                    self.num_hops[(tup[0][0], tup[1][0])] = -1
                    self.num_ll[(tup[0][0], tup[1][0])] = -1


                if verbose:
                    print(f"label: {in_label}, Initial result: {res}")

                self.traces[router_name][in_label] = [{"trace": p, "result": res}]

                self.count_connected += p.is_connected

                if not res and verbose:
                    print(" ##### DEBUG INFO ###")
                    pprint(f"Router: {router_name} Lbl: {in_label}")
                    print("Good Sources:")
                    pprint(good_sources)
                    print("Good Targets:")
                    pprint(good_targets)
                    pprint(last_router_name)
                    #pprint(f"{fec.name}/{fec.fec_type}/{fec.value}")
                    pprint(self.decode_trace(p.traceroute))
                    pprint(f"Result: {res}")
                    print(" ####################")

                if not res:
                    breakloop = False
                    for i in range(0, len(p.trace)):
                        for j in range(i+1, len(p.trace)):
                            if p.trace[i] == p.trace[j]:
                                loop_links = loop_links.union({(x[0], x[2]) for x in p.trace[i:j+1]})
                                breakloop = True
                                break

                            if breakloop:
                                break

        # if loop_links:
        #     def filter_edge(n1, n2):
        #         if (n1, n2) in loop_links or (n2, n1) in loop_links:
        #             return False
        #         return True
        #
        #     def filter_node(n):
        #         return True
        #
        #     view = nx.subgraph_view(self.topology, filter_edge=filter_edge, filter_node=filter_node)
        #     self.topology = view
        #     self.traces = dict()
        #     self.run(flows)

        self.looping_links = self.initial_links - len(self.topology.edges)



    def decode_trace(self, trace):
        l = []
        for i in trace:
            if isinstance(i, Router):
                l.append(i.name)
            elif isinstance(i, tuple) and isinstance(i[0], Router):
                l.append((i[0].name, i[1]))
        return l

    def print_traces(self, store = False):
        # Print traces from the simulations run.
        output = ""
        for router_name, r in self.network.routers.items():
            if router_name not in self.traces.keys():
                continue
            t = self.traces[router_name]
            for in_label in r.LFIB:
                if in_label not in t:
                    continue
                for entry in t[in_label]:
                    p = entry["trace"]
                    res = entry["result"]
                    if self.trace_mode == "router":
                        #link_traceroute
                        s = f"{res};{p.exit_code};{router_name};{in_label};{self.decode_trace(p.traceroute)};"
                    elif self.trace_mode == "links":
                        s = f"{res};{p.exit_code};{router_name};{in_label};{self.decode_trace(p.link_traceroute)};"

                    if store:
                        output += "\n"+ s
                    else:
                        pprint(s)
        if store:
            return output


    def success_rate(self, exit_codes = False):
        # Find ratio of succesful traces.
        success = failures = 0
        codes = [0] * 6  # exit code counter list

        for t_by_router, rtraces  in self.traces.items():
            for in_label, xlist  in rtraces.items():
                for p in xlist:
                    if p["result"]:
                        success += 1
                    else:
                        failures += 1

                    if exit_codes:
                        c = p["trace"].exit_code
                        codes[c] += 1

        total = success + failures
        success_ratio = success/total
        if exit_codes:
            return (success,total, codes)

        return (success_ratio,total)
