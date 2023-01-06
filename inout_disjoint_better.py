import itertools
import math

import networkx.exception
from mpls_classes import *
from functools import *
from networkx import shortest_path, diameter
import networkx as nx
from collections import defaultdict
import os
from ortools.linear_solver import pywraplp
import random
#from random import *

from itertools import islice, cycle

from ForwardingTable import ForwardingTable
from benjamins_heuristic_file import initializenetwork, pathfind
from typing import Dict, Tuple, List, Callable
from essence import *
from heuristics.inverse_cap import inverse_cap
from heuristics.placeholder import placeholder
from benj_heuristic import *

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
            graph[edge[0]][edge[1]]["weight"] = 0

    for src, tgt, load in client.loads:
        unique_paths = []
        while True:
            path = nx.shortest_path(flow_to_graph[(src, tgt)], src, tgt, weight="weight")
            for v1, v2 in zip(path[:-1], path[1:]):
                w = flow_to_graph[(src, tgt)][v1][v2]["weight"]
                w = w * 2 + 1
                flow_to_graph[(src, tgt)][v1][v2]["weight"] = w
            pathdict[(src, tgt)].append(path)
            if path not in unique_paths:
                unique_paths.append(path)
            if len(unique_paths) == client.mem_limit_per_router_per_flow or pathdict[(src, tgt)].count(path) == 3:
                pathdict[(src, tgt)] = unique_paths
                break

    path_indices = {}
    for src, tgt, load in client.loads:
        path_indices[src, tgt] = 0
        yield ((src, tgt), pathdict[src, tgt][0])

    for i in range(client.mem_limit_per_router_per_flow):
        for src, tgt, load in client.loads:
            path_index = path_indices[src, tgt]
            if path_index >= len(pathdict[src, tgt]):
                continue
            path_indices[src, tgt] += 1
            yield ((src, tgt), pathdict[src, tgt][path_index])

def greedy_min_congestion(client):
    G = client.router.network.topology.to_directed()

    # Absolute link utilization under current path set
    link_to_util = {e: 0 for e in client.link_caps.keys()}

    for src, tgt, load in sorted(client.loads, key=lambda x: x[2], reverse=True) * client.mem_limit_per_router_per_flow * 2:
        # Select greedily the path that imposes least max utilization (the utilization of the most utilized link)
        imposed_util = dict()
        for (u, v), util in link_to_util.items():
            imposed_util[(u, v)] = (util + load) / client.link_caps[(u, v)]

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

        path = nx.shortest_path(G, src, tgt, weight="weight")
        for v1, v2 in zip(path[:-1], path[1:]):
            link_to_util[(v1, v2)] += load
        yield ((src, tgt), path)


def shortest_paths(client):
    G = client.router.network.topology.to_directed()
    for src, tgt in G.edges:
        G[src][tgt]["weight"] = 1

    path_gens = {(src, tgt): nx.shortest_simple_paths(G, src, tgt, weight="weight") for src, tgt in client.flows}

    for src, tgt, _ in sorted(client.loads, key=lambda x: x[2], reverse=True) * client.mem_limit_per_router_per_flow * 2:
        if len(path_gens) < 1:
            break
        if (src, tgt) in path_gens:
            try:
                path = next(path_gens[(src, tgt)])
                yield ((src, tgt), path)
            except StopIteration as e:
                path_gens.pop((src, tgt))
                continue


def benjamins_heuristic(client):
    # We set number of extra hops allowed
    k = client.kwargs["extra_hops"]

    G = client.router.network.topology
    nx.set_node_attributes(G, 0, "jumpsfromtarget")

    nx.set_edge_attributes(G, 0, "usage")

    for (src, tgt), cap in client.link_caps.items():
        G[src][tgt]["weight"] = cap

    pathdict = dict()

    for src, tgt, load in client.loads:
        pathdict[(src, tgt)] = []

    for src, tgt, load in client.loads:
        paths = pathfind((src,tgt,load), G, k)
        nodepaths = linktonode(kpaths(paths,1,G,(src,tgt,load),k))
        pathdict[src,tgt] = nodepaths

    path_indices = {}
    for src, tgt, load in client.loads:
        path_indices[src, tgt] = 0
        yield ((src, tgt), pathdict[src, tgt][0])

    for i in range(client.mem_limit_per_router_per_flow):
        for src, tgt, load in client.loads:
            path_index = path_indices[src, tgt]
            if path_index >= len(pathdict[src, tgt]):
                continue
            path_indices[src, tgt] += 1
            yield ((src, tgt), pathdict[src, tgt][path_index])

# Insert lowest utility path first in semi disjoint paths
def lowestutilitypathinsert(client, pathdict):
    G = client.router.network.topology.to_directed()

    # Absolute link utilization under current path set
    link_to_util = {e: 0 for e in client.link_caps.keys()}

    for src, tgt, load in sorted(client.loads, key=lambda x: x[2], reverse=True):
        # Select greedily the path that imposes least max utilization (the utilization of the most utilized link)
        imposed_util = dict()
        for (u, v), util in link_to_util.items():
            imposed_util[(u, v)] = (util + load) / client.link_caps[(u, v)]

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

        path = nx.shortest_path(G, src, tgt, weight="weight")
        if path in pathdict[src, tgt]:
            pathdict[src, tgt, load].remove(path)
        pathdict[src, tgt] = [path] + pathdict[src, tgt]
    return pathdict


def find_unused_paths(paths, G, src, tgt):
    graph_copy = G.copy()
    paths_to_add = []

    for path in paths:
        for x, y in zip(path[::], path[1::]):
            if graph_copy.has_edge(x, y):
                graph_copy.remove_edge(x, y)

    for (edgesrc, edgetgt) in graph_copy.edges:
        first = nx.shortest_path(G, src, edgesrc)
        last = nx.shortest_path(G, edgetgt, tgt)
        paths_to_add.append(first + last)
    return paths_to_add


def congestion_lp(graph, capacities, demands):  # Inputs networkx directed graph, dict of capacities, dict of demands
    def demand(i, d):
        if demands[d][0] == i:  # source
            return 1
        elif demands[d][1] == i:  # destination
            return -1
        else:
            return 0  # intermediate

    def fortz_and_thorup(u):
        if u <= 1 / 20:
            return u * 0.1
        if u <= 1 / 10:
            return u * 0.3 - 0.01
        if u <= 1 / 6:
            return u * 1 - 0.08
        if u <= 1 / 3:
            return u * 2 - 0.24666
        if u <= 1 / 2:
            return u * 5 - 1.24666
        if u <= 2 / 3:
            return u * 10 - 3.74666
        if u <= 9 / 10:
            return u * 20 - 10.41333
        if u <= 1:
            return u * 70 - 55.41333
        if u <= 11 / 10:
            return u * 500 - 485.41333
        else:
            return u * 5000 - 5435.41333

    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Flow variables for solver
    f = {(i, j, d): solver.IntVar(0, 1, "{}->{}=>{}->{}".format(i, j, demands[d][0], demands[d][1])) for
         (i, j) in graph.edges for d in range(len(demands))}

    # l = {(i, j): solver.NumVar(0, solver.infinity(), "load:{}->{}".format(i, j)) for (i, j) in graph.edges}

    func = {(i, j): solver.NumVar(0, solver.infinity(), "func:{}->{}".format(i, j)) for (i, j) in graph.edges}

    for (i, j) in graph.edges:
        solver.Add(func[i, j] >= (sum(demands[d][2] * f[i, j, d] for d in range(len(demands)))))
        solver.Add(func[i, j] >= 3 * (sum(demands[d][2] * f[i, j, d] for d in range(len(demands)))) - (
                0.666 * capacities[i, j]))
        solver.Add(func[i, j] >= 10 * (sum(demands[d][2] * f[i, j, d] for d in range(len(demands)))) - (
                5.333 * capacities[i, j]))
        solver.Add(func[i, j] >= 70 * (sum(demands[d][2] * f[i, j, d] for d in range(len(demands)))) - (
                59.333 * capacities[i, j]))
        solver.Add(func[i, j] >= 500 * (sum(demands[d][2] * f[i, j, d] for d in range(len(demands)))) - (
                489.333 * capacities[i, j]))
        solver.Add(func[i, j] >= 5000 * (sum(demands[d][2] * f[i, j, d] for d in range(len(demands)))) - (
                6489.333 * capacities[i, j]))

    # Max utilization
    # for (i, j) in graph.edges:
    #   solver.Add(max_utilization <= ((sum(demands[d][2] * f[i, j, d] for d in range(len(demands))))/capacities[i,j]))

    # Flow conservation constraints: total flow balance at node i for each demand d
    # must be 0 if i is an intermediate node, 1 if i is the source of demand d, and
    # -1 if i is the destination.
    for i in graph.nodes:
        for d in range(len(demands)):
            solver.Add(sum(f[i, j, d] for j in graph.nodes if (i, j) in graph.edges) -
                       sum(f[j, i, d] for j in graph.nodes if (i, j) in graph.edges) ==
                       demand(i, d))

    fortzfunc = sum((func[i, j]) for (i, j) in graph.edges)
    solver.Minimize(fortzfunc)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        # create dictionary to be returned to andreas's heuristic
        pathdict = dict()

        for src, tgt, load in demands:
            pathdict[(src, tgt)] = []

        for d in range(len(demands)):
            src, tgt, load = demands[d]
            pathtoadd = []
            for (i, j) in graph.edges:
                if f[i, j, d].SolutionValue() > 0:
                    pathtoadd.append((i, j))

            new_pta = [src]

            for i in range(len(pathtoadd)):
                for s, d in pathtoadd:
                    if s == new_pta[-1]:
                        new_pta.append(d)
                    else:
                        continue

            pathdict[src, tgt] = [new_pta]

        return pathdict
    else:
        print(solver.Objective().Value())
        print('The problem does not have an optimal solution.')
        pathdict = dict()

        for src, tgt, load in demands:
            pathdict[(src, tgt)] = []
        return pathdict


def nielsens_heuristic(client):
    G = client.router.network.topology.to_directed()
    flow_to_graph = {f: client.router.network.topology.to_directed() for f in client.flows}
    for graph in flow_to_graph.values():
        for edge in graph.edges:
            graph[edge[0]][edge[1]]["weight"] = 1

    # pathdict = dict()
    pathdict = congestion_lp(G, client.link_caps, client.loads)

    #for src, tgt, load in client.loads:
    #   pathdict[(src,tgt,load)] = []

    for src, tgt, load in sorted(client.loads, key=lambda x: x[2], reverse=True) * client.mem_limit_per_router_per_flow:
        path = nx.shortest_path(flow_to_graph[(src, tgt)], src, tgt, weight="weight")
        for v1, v2 in zip(path[:-1], path[1:]):
            w = flow_to_graph[(src, tgt)][v1][v2]["weight"]
            w = w * 2 + 1
            flow_to_graph[(src, tgt)][v1][v2]["weight"] = w
        pathdict[src, tgt].append(path)

    '''
    new_pathdict = dict()

    for src, tgt, load in client.loads:
        new_pathdict[(src, tgt, load)] = []

    # Remove duplicate paths
    for src, tgt, load in pathdict.keys():
        for elem in pathdict[(src, tgt, load)]:
            if elem not in new_pathdict[(src, tgt, load)]:
                new_pathdict[(src, tgt, load)].append(elem)

    for src, tgt, load in client.loads:
        pathdict[src, tgt, load] = new_pathdict[src, tgt, load]

    # pathdict = lowestutilitypathinsert(client, pathdict)
    pathdict = prefixsort(client, pathdict)

    # Find unused paths probably deprecatable
    for src, tgt, load in sorted(client.loads, key=lambda x: x[2], reverse=True):
        unused_paths = find_unused_paths(pathdict[src,tgt,load], G, src, tgt)
        if unused_paths:
            pathdict[src,tgt,load].append(find_unused_paths(pathdict[src,tgt,load], G, src, tgt))
    '''

    pathdict = prefixsort(pathdict)

    path_indices = {}
    for src, tgt, load in client.loads:
        path_indices[src, tgt] = 0
        yield ((src, tgt), pathdict[src, tgt][0])

    for i in range(client.mem_limit_per_router_per_flow):
        for src, tgt, load in client.loads:
            path_index = path_indices[src, tgt]
            if path_index >= len(pathdict[src, tgt]):
                continue
            path_indices[src, tgt] += 1
            yield ((src, tgt), pathdict[src, tgt][path_index])


class InOutDisjoint(MPLS_Client):
    protocol = "inout-disjoint"

    def __init__(self, router: Router, **kwargs):
        super().__init__(router, **kwargs)

        # The demands where this router is the tailend
        self.demands: dict[str, tuple[str, str]] = {}

        # Partial forwarding table containing only rules for this router
        self.partial_forwarding_table: dict[tuple[str, oFEC], list[tuple[int, str, oFEC]]] = {}

        self.mem_limit_per_router_per_flow = kwargs['per_flow_memory']

        back_tracking_methods = {
            'full': encode_paths_full_backtrack,
            'partial': encode_paths_quick_next_path
        }
        self.backtracking_method = back_tracking_methods[kwargs["backtrack"]]

        path_heuristics = {
            'shortest_path': shortest_paths,
            'greedy_min_congestion': greedy_min_congestion,
            'semi_disjoint_paths': semi_disjoint_paths,
            'benjamins_heuristic': benjamins_heuristic,
            'nielsens_heuristic': nielsens_heuristic,
            'essence': essence,
            'essence_v2': essence_v2,
            'inverse_cap': inverse_cap,
            'placeholder': placeholder
        }

        "self.path_heuristic = semi_disjoint_paths"
        self.loads = kwargs["loads"]
        self.link_caps = kwargs["link_caps"]
        self.kwargs = kwargs
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

        total_yields = self.mem_limit_per_router * len(self.router.network.routers)
        path_heuristic = self.path_heuristic(self)
        yields = 0
        while yields < total_yields:
            # Generate next path
            flow, path = next(path_heuristic, (None,None))
            yields += 1
            # If path is already encoded we do not encode it again
            if path in flow_to_paths[flow]:
                continue
            # Break when generator is empty
            if path == None:
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
