import itertools

import networkx.exception

from mpls_classes import *
from functools import *
from networkx import shortest_path
import networkx as nx
from collections import defaultdict
import os

from itertools import islice

from ForwardingTable import ForwardingTable

from typing import Dict, Tuple, List, Callable

def label(ingress, egress, path_index: int):
    return oFEC("inout-disjoint", f"{ingress}_to_{egress}_path{path_index}",
                {"ingress": ingress, "egress": [egress], "path_index": path_index})

def create_label_generator(f):
    return (label(f[0], f[1], i) for i, _ in enumerate(iter(int, 1)))

def compute_memory_usage(network, flows, path_encoder, _flow_to_paths_dict) -> Dict:
    memory_usage = {r: 0 for r in network.routers}

    ft = ForwardingTable()
    for f in flows:
        ft.extend(path_encoder(_flow_to_paths_dict[f], create_label_generator(f)))

    for (router, _), rules in ft.table.items():
        memory_usage[router] += len(rules)

    return memory_usage

def shortest_path(self, network: Network, flows: List[Tuple[str, str, int]], epochs: int, total_max_memory: int,
                                     path_encoder: Callable[[List[str], Generator], ForwardingTable]) -> Dict[
    Tuple[str, oFEC], List[Tuple[int, str, oFEC]]]:

    forwarding_table = ForwardingTable()
    flow_to_paths = defaultdict(list)
    G = network.topology.to_directed()

    for src, tgt in G.edges:
        G[src][tgt]["weight"] = 1

    for src, tgt, load in self.loads:
        path_gen = nx.shortest_simple_paths(G, src, tgt, weight="weight")
        try:
            path = next(path_gen)
        except StopIteration as e:
            raise Exception("Bug")
        flow_to_paths[(src, tgt)].append(path)

    for f in flows:
        # remove duplicate labels
        encoded_path = path_encoder(flow_to_paths[f], create_label_generator(f))
        # encoded_path.to_graphviz(f'ft {f[0]} -> {f[1]}', network.topology)
        forwarding_table.extend(encoded_path)

    return forwarding_table.table

def greedy_min_congestion(self, network: Network, flows: List[Tuple[str, str, int]], epochs: int, total_max_memory: int,
                                     path_encoder: Callable[[List[str], Generator], ForwardingTable]) -> Dict[
    Tuple[str, oFEC], List[Tuple[int, str, oFEC]]]:

    forwarding_table = ForwardingTable()
    flow_to_paths = defaultdict(list)
    weighted_graph = network.topology.to_directed()

    # Absolute link utilization under current path set
    link_to_util = {e: 0 for e in self.link_caps.keys()}

    for src, tgt, load in sorted(self.loads, key=lambda x: x[2], reverse=True):
        # Select greedily the path that imposes least max utilization (the utilization of the most utilized link)
        imposed_util_abs = dict()
        for (u, v), util in link_to_util.items():
            imposed_util_abs[(u, v)] = util + load

        imposed_util_rel= dict()
        for (u, v), util_abs in imposed_util_abs.items():
            link_cap = self.link_caps[(u,v)]
            imposed_util_rel[(u, v)] = util_abs / link_cap

        # We order links by how much relative utilization it has were the flow to go through it
        ordered_by_util = list(imposed_util_rel.items())
        ordered_by_util.sort(key=lambda x: x[1])

        edge_to_weight = dict()
        prev_util = ordered_by_util[0][1]
        next_weight = 1
        for i, (edge, util) in enumerate(ordered_by_util):
            if util > prev_util:
                next_weight = sum(edge_to_weight.values()) + 1
            edge_to_weight[edge] = next_weight
            prev_util = util

        def update_weights(G, edge_to_weight):
            for (src, tgt), weight in edge_to_weight.items():
                G[src][tgt]["weight"] = weight
            return G

        weighted_graph = update_weights(weighted_graph, edge_to_weight)

        path_gen = nx.shortest_simple_paths(weighted_graph, src, tgt, weight="weight")
        try:
            path = next(path_gen)
        except StopIteration as e:
            raise Exception("Bugg")

        flow_to_paths[(src,tgt)].append(path)
        # Update util use
        for v1, v2 in zip(path[:-1], path[1:]):
            link_to_util[(v1, v2)] += load

    for f in flows:
        # remove duplicate labels
        encoded_path = path_encoder(flow_to_paths[f], create_label_generator(f))
        # encoded_path.to_graphviz(f'ft {f[0]} -> {f[1]}', network.topology)
        forwarding_table.extend(encoded_path)

    return forwarding_table.table

def global_weights_heuristic(self, network: Network, flows: List[Tuple[str, str, int]], epochs: int, total_max_memory: int,
                                     path_encoder: Callable[[List[str], Generator], ForwardingTable]) -> Dict[
    Tuple[str, oFEC], List[Tuple[int, str, oFEC]]]:

    forwarding_table = ForwardingTable()
    flow_to_paths_dict = {f: [] for f in flows}
    weighted_graph = network.topology.copy().to_directed()

    reset_weights(weighted_graph, 0)

    unfinished_flows = flows.copy()
    flow_to_misses = {f: 0 for f in flows}  # Number of consecutive times path was not added for flow
    i = 0
    total_paths_used = 0
    while len(unfinished_flows) > 0:
        # select the next ingress router to even out memory usage
        flow = unfinished_flows[i % len(unfinished_flows)]
        ingress_router, egress_router = flow

        path = nx.dijkstra_path(weighted_graph, ingress_router, egress_router, weight="weight")

        try_paths = {}
        for f, paths in flow_to_paths_dict.items():
            try_paths[f] = paths.copy()
            if f == flow:
                if path not in try_paths[f]:
                    try_paths[f].append(path)

        # see if adding this path surpasses the memory limit
        router_memory_usage = compute_memory_usage(network, flows, path_encoder, try_paths)
        max_memory_reached = any(router_memory_usage[r] > total_max_memory for r in network.routers)

        # update weights in the network to change the shortest path'
        update_weights(weighted_graph, path, lambda x: x)

        i += 1
        if not max_memory_reached and path not in flow_to_paths_dict[flow]:
            flow_to_paths_dict[flow].append(path)
            flow_to_misses[flow] = 0
            total_paths_used += 1
        else:
            flow_to_misses[flow] += 1
            if flow_to_misses[flow] > epochs:
                unfinished_flows.remove(flow)
                i -= 1  # Undo increment

    for f in flows:
        # remove duplicate labels
        encoded_path = path_encoder(flow_to_paths_dict[f], create_label_generator(f))
        # encoded_path.to_graphviz(f'ft {f[0]} -> {f[1]}', network.topology)
        forwarding_table.extend(encoded_path)

    return forwarding_table.table

def semi_disjoint_paths(self, network: Network, flows: List[Tuple[str, str, int]], epochs: int, total_max_memory: int,
                                     path_encoder: Callable[[List[str], Generator], ForwardingTable]) -> Dict[
    Tuple[str, oFEC], List[Tuple[int, str, oFEC]]]:

    forwarding_table = ForwardingTable()
    flow_to_paths_dict = {f: [] for f in flows}
    flow_to_weight_graph_dict = {f: network.topology.copy().to_directed() for f in flows}

    for _, weighted_graph in flow_to_weight_graph_dict.items():
        reset_weights(weighted_graph, 0)

    unfinished_flows = flows.copy()
    flow_to_misses = {f: 0 for f in flows}  # Number of consecutive times path was not added for flow
    i = 0
    total_paths_used = 0

    while len(unfinished_flows) > 0:
        # select the next ingress router to even out memory usage
        flow = unfinished_flows[i % len(unfinished_flows)]
        ingress_router, egress_router = flow

        path = nx.dijkstra_path(flow_to_weight_graph_dict[flow], ingress_router, egress_router, weight="weight")

        try_paths = {}
        for f, paths in flow_to_paths_dict.items():
            try_paths[f] = paths.copy()
            if f == flow:
                if path not in try_paths[f]:
                    try_paths[f].append(path)

        # see if adding this path surpasses the memory limit
        router_memory_usage = compute_memory_usage(network, flows, path_encoder, try_paths)
        max_memory_reached = any(router_memory_usage[r] > total_max_memory for r in network.routers)

        # update weights in the network to change the shortest path
        update_weights(flow_to_weight_graph_dict[flow], path, lambda x: x * 2 + 1)

        i += 1
        if not max_memory_reached and path not in flow_to_paths_dict[flow]:
            flow_to_paths_dict[flow].append(path)
            flow_to_misses[flow] = 0
            total_paths_used += 1
        else:
            flow_to_misses[flow] += 1
            if flow_to_misses[flow] > epochs:
                unfinished_flows.remove(flow)
                i -= 1  # Undo increment

    for f in flows:
        # remove duplicate labels
        encoded_path = path_encoder(flow_to_paths_dict[f], create_label_generator(f))
        # encoded_path.to_graphviz(f'ft {f[0]} -> {f[1]}', network.topology)
        forwarding_table.extend(encoded_path)

    return forwarding_table.table


def reset_weights(G: Graph, value):
    for u, v, d in G.edges(data=True):
        d["weight"] = value


def update_weights(G: Graph, path, update_func):
    for v1, v2 in zip(path[:-1], path[1:]):
        # weight = G[v1][v2]["weight"]

        # if weight <= 0:
        #     G[v1][v2]["weight"] = 1
        # else:
        G[v1][v2]["weight"] = update_func(G[v1][v2]["weight"])


def encode_paths_full_backtrack(paths: List[str], label_generator: Iterator[oFEC]):
    ft = ForwardingTable()

    if len(paths) == 0:
        return ft

    # Order the path to (greedily) maximize prefix overlap between adjacent
    new_paths = [paths[0]]
    paths.remove(paths[0])
    while len(paths) > 0:
        best = max(paths, key=lambda p: len(os.path.commonprefix([p, new_paths[-1]])))
        new_paths.append(best)
        paths.remove(best)
    paths = new_paths

    path_labels = [next(label_generator) for _ in paths]
    backtracking_labels = [
        oFEC(fec.fec_type, fec.name + '_backtrack', {'egress': fec.value['egress'], 'ingress': fec.value['ingress']})
        for fec in path_labels]

    # The number of nodes path_i has in common with path_{i-1}
    paths_common_prefix_with_previous = [1] + [len(os.path.commonprefix([paths[i - 1], paths[i]])) for i in
                                               range(1, len(paths))]

    forward_paths = []
    backward_paths = []

    for i, path in enumerate(paths):
        # Encode forward path
        common_prev = paths_common_prefix_with_previous[i]
        forward_path = path[common_prev - 1:]
        forward_paths.append(forward_path)
        for s, t in zip(forward_path[:-1], forward_path[1:]):
            ft.add_rule((s, path_labels[i]), (1, t, path_labels[i]))

        if i < len(paths) - 1:
            # Encode backtracking path
            common_next = paths_common_prefix_with_previous[i + 1]
            backtrack_path = path[common_next - 1:-1][::-1]
            backward_paths.append(backtrack_path)

            if len(backtrack_path) == 1:
                # If the backtrack path is empty, add a pseudo edge self loop to bounce to next path
                backtrack_path = backtrack_path + backtrack_path

            for j, (s, t) in enumerate(zip(backtrack_path[:-1], backtrack_path[1:])):
                # The first node in the backtrack will have a direct 'path to backtrack' edge, here we create self-loops
                # for the others. The last self loop (the one ending at the node intersection with next path) will jump
                # straight to path label
                # ft.add_rule(
                #     (t, path_labels[i]),
                #     (2, t, backtracking_labels[i] if j != len(backtrack_path) - 2 else path_labels[i+1] )
                # )

                ft.add_rule(
                    (s, backtracking_labels[i] if j != 0 else path_labels[i]),
                    (2, t, backtracking_labels[i] if j != len(backtrack_path) - 2 else path_labels[i + 1])
                )

    # Encode bounce self-loops
    for i, fpath in enumerate(forward_paths[:-1]):
        for v in fpath[:-1]:
            # Find the next backtrack path that will backtrack trough this node
            bounce_to = next((k for k, bpath in list(enumerate(backward_paths))[i:] if v in bpath), None)

            if bounce_to is not None and v != backward_paths[bounce_to][0]:
                if v == backward_paths[bounce_to][-1]:
                    to_label = path_labels[bounce_to + 1]
                else:
                    to_label = backtracking_labels[bounce_to]
                ft.add_rule(
                    (v, path_labels[i]),
                    (2, v, to_label)
                )

    return ft


def encode_paths_quick_next_path(paths: List[str], label_generator: Iterator[oFEC]):
    ft = ForwardingTable()

    if len(paths) == 0:
        return ft

    path_labels = [next(label_generator) for _ in paths]

    for i, path in enumerate(paths):
        is_last_path = i == (len(paths) - 1)

        # for each edge in path
        for s, t in zip(path[:-1], path[1:]):
            # create simple forwarding using the path label
            ft.add_rule((s, path_labels[i]), (1, t, path_labels[i]))

            # handle bouncing to next path
            if not is_last_path:
                ft.add_rule((s, path_labels[i]), (2, s, path_labels[i + 1]))

                # create backtracking rules for next subpath
                if t not in paths[i + 1]:
                    ft.add_rule((t, path_labels[i + 1]), (1, s, path_labels[i + 1]))

    return ft


class InOutDisjoint(MPLS_Client):
    protocol = "inout-disjoint"

    def __init__(self, router: Router, **kwargs):
        super().__init__(router, **kwargs)

        # The demands where this router is the tailend
        self.demands: dict[str, tuple[str, str]] = {}

        # Partial forwarding table containing only rules for this router
        self.partial_forwarding_table: dict[tuple[str, oFEC], list[tuple[int, str, oFEC]]] = {}

        self.epochs = kwargs['epochs']
        self.per_flow_memory = kwargs['per_flow_memory']

        back_tracking_methods = {
            'full': encode_paths_full_backtrack,
            'partial': encode_paths_quick_next_path
        }
        self.backtracking_method = back_tracking_methods[kwargs["backtrack"]]

        path_heuristics = {
            'global_weights': global_weights_heuristic,
            'semi_disjoint_paths': semi_disjoint_paths,
            'greedy_min_congestion': greedy_min_congestion,
            'shortest_path': shortest_path,
        }

        self.path_heuristic = semi_disjoint_paths
        self.loads = kwargs["loads"]
        self.link_caps = kwargs["link_caps"]
        if "path_heuristic" in kwargs:
            self.path_heuristic = path_heuristics[kwargs["path_heuristic"]]

    def LFIB_compute_entry(self, fec: oFEC, single=False):
        for priority, next_hop, swap_fec in self.partial_forwarding_table[(self.router.name, fec)]:
            local_label = self.get_local_label(fec)
            assert (local_label is not None)

            if next_hop in fec.value["egress"]:
                yield (local_label, {'out': next_hop, 'ops': [{'pop': ''}], 'weight': priority})
            else:
                remote_label = self.get_remote_label(next_hop, swap_fec)
                assert (remote_label is not None)

                yield (local_label, {'out': next_hop if next_hop != self.router.name else self.LOCAL_LOOKUP,
                                     'ops': [{'swap': remote_label}], 'weight': priority})

    # Defines a demand for a headend to this one
    def define_demand(self, headend: str):
        self.demands[f"{len(self.demands.items())}_{headend}_to_{self.router.name}"] = (headend, self.router.name)

    def commit_config(self):
        # Only one router should generate dataplane!
        if self.router.name != min(rname for rname in self.router.network.routers):
            return

        network = self.router.network

        flows = [(headend, tailend) for tailend in network.routers for headend in
                 map(lambda x: x[0], network.routers[tailend].clients[self.protocol].demands.values())]

        ft = self.path_heuristic(self, self.router.network, flows, self.epochs,
                                              self.per_flow_memory * len(flows), self.backtracking_method)

        for (src, fec), entries in ft.items():
            src_client: InOutDisjoint = self.router.network.routers[src].clients["inout-disjoint"]

            if (src, fec) not in src_client.partial_forwarding_table:
                src_client.partial_forwarding_table[(src, fec)] = []

            src_client.partial_forwarding_table[(src, fec)].extend(entries)

    def compute_bypasses(self):
        pass

    def LFIB_refine(self, label):
        pass

    def known_resources(self):
        for _, fec in self.partial_forwarding_table.keys():
            yield fec

    def self_sourced(self, fec: oFEC):
        return 'inout-disjoint' in fec.fec_type and fec.value["egress"][0] == self.router.name


