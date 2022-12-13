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

from itertools import islice, cycle

from ForwardingTable import ForwardingTable
from benjamins_heuristic_file import initializenetwork, pathfind
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

    # We hardcode number of iterations for the time being
    iterations = 3

    for src, tgt, load in sorted(client.loads, key=lambda x: x[2], reverse=True) * iterations:
        path = nx.shortest_path(flow_to_graph[(src, tgt)], src, tgt, weight="weight")
        for v1, v2 in zip(path[:-1], path[1:]):
            w = flow_to_graph[(src, tgt)][v1][v2]["weight"]
            w = w * 2 + 1
            flow_to_graph[(src, tgt)][v1][v2]["weight"] = w
        yield ((src, tgt), path)


def greedy_min_congestion(client):
    G = client.router.network.topology.to_directed()

    # Absolute link utilization under current path set
    link_to_util = {e: 0 for e in client.link_caps.keys()}

    for src, tgt, load in cycle(sorted(client.loads, key=lambda x: x[2], reverse=True)):
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

    for src, tgt, _ in cycle(sorted(client.loads, key=lambda x: x[2], reverse=True)):
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
    for (src, tgt), cap in client.link_caps.items():
        G[src][tgt]["weight"] = cap

    demands, grapher = initializenetwork(G, sorted(client.loads, key=lambda x: x[2], reverse=True))
    demand_to_paths = dict()

    def benja_path_to_juan_path(benja_path, src):
        juan_path = []

        # Add first node
        first_link = benja_path[0]
        node1 = first_link.node1.identity
        node2 = first_link.node2.identity

        if node1 == src:
            juan_path.append(node1)
        elif node2 == src:
            juan_path.append(node2)
        else:
            raise Exception()

        for link in benja_path:
            node1 = link.node1.identity
            node2 = link.node2.identity
            if juan_path[-1] == node1:
                juan_path.append(node2)
            elif juan_path[-1] == node2:
                juan_path.append(node1)
            else:
                raise Exception()

        return juan_path

    for d in demands:
        paths = pathfind(d, grapher, k)
        demand_to_paths[(d.source.identity, d.target.identity)] = [benja_path_to_juan_path(x, d.source.identity) for x
                                                                   in paths]

    demand_num = len(demand_to_paths.items())
    i = 0
    for (src, tgt), paths in cycle(demand_to_paths.items()):
        if len(paths) > 0:
            i = 0
            path = paths[0]
            paths.remove(path)
            yield ((src, tgt), path)
        else:
            i += 1
        # If all paths have been yielded
        if i == demand_num:
            break


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
        if path in pathdict[src, tgt, load]:
            pathdict[src, tgt, load].remove(path)
        pathdict[src, tgt, load] = [path] + pathdict[src, tgt, load]
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


# Shitter bruteforce algo
def prefixsort(client, pathdict):
    new_pathdict = dict()
    for src, tgt, load in client.loads:
        new_pathdict[(src, tgt, load)] = []
        new_pathdict[src, tgt, load].append(pathdict[src, tgt, load][0])

    for i in range(len(client.loads)):
        for src, tgt, load in client.loads:
            if new_pathdict[src, tgt, load][-1] in pathdict[src, tgt, load]:
                pathdict[src, tgt, load].remove(new_pathdict[src, tgt, load][-1])
            if pathdict[src, tgt, load] != []:
                max_common_prefix_path = max(pathdict[src, tgt, load],
                                             key=lambda x: common_prefix_length(new_pathdict[src, tgt, load][-1], x))
                new_pathdict[src, tgt, load].append(max_common_prefix_path)

    pathdict = new_pathdict

    return pathdict


def common_prefix_length(path1, path2):
    prefixlen = 0
    if path1 == path2:
        return len(path1)
    for i in range(len(path1)):
        if path1[i] == path2[i]:
            prefixlen += 1
        else:
            return prefixlen


# Limit number of hops for packets
def max_hops(max_stretch, pathdict, client, graph):
    new_pathdict = dict()
    for src, tgt, load in client.loads:
        max_hops_for_demand = math.floor(((len(shortest_path(graph, src, tgt)))-1) * max_stretch)
        new_pathdict[(src, tgt, load)] = []

        if max_hops_for_demand >= len(pathdict[src, tgt, load][0]):
            new_pathdict[src, tgt, load].append(pathdict[src, tgt, load][0])
            # -2 because first node does not count as a hop and assume that last link fails so it does not make the hop
            max_hops_for_demand -= len(pathdict[src, tgt, load][0]) - 2
        else:
            for path in pathdict[src,tgt,load]:
                if max_hops_for_demand >= len(path):
                    new_pathdict[src, tgt, load].append(path)
                    # -2 because first node does not count as a hop and assume that last link fails so it does not make the hop
                    max_hops_for_demand -= len(path) - 2
                    break

        for path in pathdict[src, tgt, load]:
            if path in new_pathdict[src,tgt,load]:
                continue
            if max_hops_for_demand >= ((len(new_pathdict[src, tgt, load][-1]) - common_prefix_length(new_pathdict[src, tgt, load][-1], path) - 1) + (len(path) - (common_prefix_length(new_pathdict[src, tgt, load][-1], path)))):
                max_hops_for_demand = max_hops_for_demand - ((len(new_pathdict[src, tgt, load][-1]) - common_prefix_length(new_pathdict[src, tgt, load][-1],path) - 1) + (len(path) - (common_prefix_length(new_pathdict[src, tgt, load][-1], path)) - 1))
                new_pathdict[src, tgt, load].append(path)

    return new_pathdict


def congestion_lp(graph, capacities, demands, max_stretch):  # Inputs networkx directed graph, dict of capacities, dict of demands
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

    # max stretch constraint
    for d in range(len(demands)):
        src,tgt,load = demands[d]
        max_hops = math.floor(len(shortest_path(graph,src,tgt))-1) * max_stretch
        solver.Add(sum(f[i, j, d] for (i, j) in graph.edges) <= max_hops)

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
    #for (i, j) in graph.edges:
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
            pathdict[(src, tgt, load)] = []

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

            pathdict[src, tgt, load] = [new_pta]

        return pathdict
    else:
        print(solver.Objective().Value())
        print('The problem does not have an optimal solution.')
        pathdict = dict()

        for src, tgt, load in demands:
            pathdict[(src, tgt, load)] = []
        return pathdict


def nielsens_heuristic(client):
    G = client.router.network.topology.to_directed()
    flow_to_graph = {f: client.router.network.topology.to_directed() for f in client.flows}
    for graph in flow_to_graph.values():
        for edge in graph.edges:
            graph[edge[0]][edge[1]]["weight"] = 1

    # pathdict = dict()
    pathdict = congestion_lp(G, client.link_caps, client.loads, client.kwargs['max_stretch'])

    # for src, tgt, load in client.loads:
    #    pathdict[(src,tgt,load)] = []

    for src, tgt, load in sorted(client.loads, key=lambda x: x[2],
                                 reverse=True) * client.mem_limit_per_router_per_flow * 2:
        path = nx.shortest_path(flow_to_graph[(src, tgt)], src, tgt, weight="weight")
        for v1, v2 in zip(path[:-1], path[1:]):
            w = flow_to_graph[(src, tgt)][v1][v2]["weight"]
            w = w * 2 + 1
            flow_to_graph[(src, tgt)][v1][v2]["weight"] = w
        pathdict[(src, tgt, load)].append(path)

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
    '''
    for src, tgt, load in sorted(client.loads, key=lambda x: x[2], reverse=True):
        unused_paths = find_unused_paths(pathdict[src,tgt,load], G, src, tgt)
        if unused_paths:
            pathdict[src,tgt,load].append(find_unused_paths(pathdict[src,tgt,load], G, src, tgt))
    '''

    pathdict = max_hops(client.kwargs["max_stretch"], pathdict, client, G)

    for src, tgt, load in sorted(client.loads, key=lambda x: x[2], reverse=True):
        for path in pathdict[src, tgt, load]:
            yield ((src, tgt), path)


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
