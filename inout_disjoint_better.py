import itertools

import networkx.exception

from mpls_classes import *
from functools import *
from networkx import shortest_path
import networkx as nx
from collections import defaultdict
import os

from itertools import islice, cycle

from ForwardingTable import ForwardingTable

from typing import Dict, Tuple, List, Callable

def label(ingress, egress, path_index: int):
    return oFEC("inout-disjoint", f"{ingress}_to_{egress}_path{path_index}",
                {"ingress": ingress, "egress": [egress], "path_index": path_index})

def create_label_generator(f):
    return (label(f[0], f[1], i) for i, _ in enumerate(iter(int, 1)))

def reset_weights(G: Graph, value):
    for u, v, d in G.edges(data=True):
        d["weight"] = value

def update_weights(G: Graph, path, update_func):
    for v1, v2 in zip(path[:-1], path[1:]):
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

def semi_disjoint_paths(client):
    flow_to_graph = {f: client.router.network.topology.to_directed() for f in client.flows}
    for graph in flow_to_graph.values():
        for edge in graph.edges:
            graph[edge[0]][edge[1]]["weight"] = 1

    path_gens = {(src,tgt): nx.shortest_path(flow_to_graph[(src,tgt)], src, tgt, weight="weight") for src, tgt in client.flows}
    for src, tgt, load in cycle(sorted(client.loads, key=lambda x: x[2], reverse=True)):
        if len(path_gens) < 1:
            break
        if (src, tgt) in path_gens:
            try:
                path = next(path_gens[(src, tgt)])
                for v1, v2 in zip(path[:-1], path[1:]):
                    w = flow_to_graph[(src,tgt)][v1][v2]["weight"]
                    w = w * 2 + 1
                    flow_to_graph[(src,tgt)][v1][v2]["weight"] = w
                yield ((src, tgt), path)

            except StopIteration:
                # No more paths
                path_gens.pop((src, tgt))
                continue

def greedy_min_congestion(client):
    G = client.router.network.topology.to_directed()

    # Absolute link utilization under current path set
    link_to_util = {e: 0 for e in client.link_caps.keys()}

    path_gens = {(src,tgt): nx.shortest_simple_paths(G, src, tgt, weight="weight") for src, tgt in client.flows}
    for src, tgt, load in cycle(sorted(client.loads, key=lambda x: x[2], reverse=True)):
        if len(path_gens) < 1:
            break
        if (src, tgt) in path_gens:
            # Select greedily the path that imposes least max utilization (the utilization of the most utilized link)
            imposed_util =  dict()
            for (u, v), util in link_to_util.items():
                imposed_util[(u, v)] = (util + load) / client.link_caps[(u,v)]

            # We or der links by how much relative utilization it has were the flow to go through it
            ordered_links = sorted(list(imposed_util.items()), key=lambda x: x[1])

            # We set the weights. Link weight > sum of link weights for all links with less utilization
            weight = 0
            weight_sum = 0
            prev_util = 0

            for (u, v), curr_util in ordered_links:
                if curr_util > prev_util:
                    weight = weight_sum + 1
                G[u][v]["weight"] = weight
                weight_sum += weight
                prev_util = curr_util

            try:
                path = next(path_gens[(src, tgt)])
                for v1, v2 in zip(path[:-1], path[1:]):
                    link_to_util[(v1, v2)] += load
                yield ((src, tgt), path)
                # Update util use

            except StopIteration:
                # No more paths
                path_gens.pop((src, tgt))
                continue

def shortest_paths(client):
    loads = sorted(client.loads, key=lambda x: x[2], reverse=True)
    G = client.router.network.topology.to_directed()
    for src, tgt in G.edges:
        G[src][tgt]["weight"] = 1

    path_gens = {(src,tgt): nx.shortest_simple_paths(G, src, tgt, weight="weight") for src, tgt in client.flows}


    for src, tgt, load in cycle(loads):
        if len(path_gens) < 1:
            break
        if (src,tgt) in path_gens:
            try:
                path = next(path_gens[(src, tgt)])
                yield ((src,tgt), path)
            except StopIteration as e:
                path_gens.pop((src,tgt))
                continue

class InOutDisjoint(MPLS_Client):
    protocol = "inout-disjoint"

    def __init__(self, router: Router, **kwargs):
        super().__init__(router, **kwargs)

        # The demands where this router is the tailend
        self.demands: dict[str, tuple[str, str]] = {}

        # Partial forwarding table containing only rules for this router
        self.partial_forwarding_table: dict[tuple[str, oFEC], list[tuple[int, str, oFEC]]] = {}

        self.mem_limit_per_router_per_flow = kwargs['per_flow_memory']

        self.epochs = kwargs['epochs']
        back_tracking_methods = {
            'full': encode_paths_full_backtrack,
            'partial': encode_paths_quick_next_path
        }
        self.backtracking_method = back_tracking_methods[kwargs["backtrack"]]

        path_heuristics = {
            'shortest_path': shortest_paths,
            'greedy_min_congestion': greedy_min_congestion,
            'semi_disjoint_paths': semi_disjoint_paths,
        }

        "self.path_heuristic = semi_disjoint_paths"
        self.loads = kwargs["loads"]
        self.link_caps = kwargs["link_caps"]
        if "path_heuristic" in kwargs:
            self.path_heuristic = path_heuristics[kwargs["path_heuristic"]]

    def mem_exceeded(self, flow_to_paths):
        mem_use = {r: 0 for r in self.router.network.routers}

        ft = ForwardingTable()
        for f in self.flows:
            ft.extend(self.backtracking_method(flow_to_paths[f], create_label_generator(f)))

        for (router, _), rules in ft.table.items():
            mem_use[router] += len(rules)

        return any(mem_use[r] > self.mem_limit_per_router for r in self.router.network.routers)

    def compute_forwarding_table(self):
        flow_to_paths = defaultdict(list)

        total_yields = self.mem_limit_per_router * 2
        path_heuristic = self.path_heuristic(self)
        yields = 0
        while yields < total_yields:
            try:
                # Generate next path
                flow, path = next(path_heuristic)
                yields += 1
            except:
                # No more paths can be generated
                break
            flow_to_paths[flow].append(path)

            if self.mem_exceeded(flow_to_paths):
                del flow_to_paths[flow][-1]

        ft = ForwardingTable()
        for f in self.flows:
            # remove duplicate labels
            encoded_path = self.backtracking_method(flow_to_paths[f], create_label_generator(f))
            # encoded_path.to_graphviz(f'ft {f[0]} -> {f[1]}', network.topology)
            ft.extend(encoded_path)

        return ft

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

        self.flows = [(headend, tailend) for tailend in network.routers for headend in
                 map(lambda x: x[0], network.routers[tailend].clients[self.protocol].demands.values())]

        self.mem_limit_per_router = self.mem_limit_per_router_per_flow * len(self.flows)

        ft = self.compute_forwarding_table()

        for (src, fec), entries in ft.table.items():
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


