import itertools

import networkx.exception

from mpls_classes import *
from functools import *
from networkx import shortest_path
import networkx as nx

from ForwardingTable import ForwardingTable

from itertools import islice

from typing import Dict, Tuple, List, Callable


def generate_pseudo_forwarding_table(network: Network, ingress: [str], egress: str, num_down_paths: int = 1, num_cycling_paths: int = 1) -> Dict[Tuple[str, oFEC], List[Tuple[int, str, oFEC]]]:
    def label(switch: str, iteration: int):
        return oFEC("cfor", f"{ingress}_to_{egress}_at_{switch}_it_{iteration}", {"ingress": ingress, "egress": [egress], "iteration": iteration, "switch": switch})

    edges: set[tuple[str, str]] = set([(n1, n2) for (n1, n2) in network.topology.edges if n1 != n2] \
                                      + [(n2, n1) for (n1, n2) in network.topology.edges if n1 != n2])
    network.compute_dijkstra(weight=1)

    max_ingress_distance = max([network.routers[v].dist[egress] for v in ingress])
    layers: dict[int, list[str]] = {layer: [] for layer in range(0, max_ingress_distance + 1)}

    for v in network.routers.values():
        dist = v.dist[egress]
        if dist <= max_ingress_distance:
            layers[dist].append(v.name)

    forwarding_table = ForwardingTable()

    for layer in layers.values():
        layer.sort()

    for i in range(1, len(layers)):
        for j in range(0, len(layers[i])):
            weight_graph = network.topology.copy()
            for u, v, d in weight_graph.edges(data=True):
                d["weight"] = 1

            v = layers[i][j]

            down_switches_direct = set()
            down_switch_to_outgoing_label_it1 = {}
            down_switch_to_outgoing_label_it2 = {}
            for v_down in filter(lambda edge: edge[0] == v and edge[1] in layers[i - 1], edges):
                down_switches_direct.add(v_down[1])
                forwarding_table.add_rule((v, label(v, 1)), (0, v_down[1], label(v_down[1], 1)))
                forwarding_table.add_rule((v, label(v, 2)), (0, v_down[1], label(v_down[1], 2)))
                weight_graph[v][v_down[1]]["weight"] *= 64

            down_switches_additional = layers[i-1][:num_down_paths]
            for v_i, v_down in enumerate(down_switches_additional):
                if v_i >= num_down_paths or v_i >= len(layers[i-1]):
                    break

                down_switch_to_outgoing_label_it1[v_down] = label(v_down, 1)
                down_switch_to_outgoing_label_it2[v_down] = label(v_down, 2)

            tgt_to_graph = {}
            subtract_switches = set()
            for k in range(0, i-2):
                subtract_switches = subtract_switches.union(set(layers[k]))
            subgraph_switches = set(network.topology.nodes).difference(subtract_switches)
            for tgt in down_switches_additional:
                tgt_to_graph[tgt] = weight_graph.subgraph(subgraph_switches.add(tgt))

            forwarding_table.extend(disjoint_paths_generator_mult_tgts(tgt_to_graph, v, down_switches_additional, label(v, 1), down_switch_to_outgoing_label_it1, 1, "godown", 2))
            forwarding_table.extend(disjoint_paths_generator_mult_tgts(tgt_to_graph, v, down_switches_additional, label(v, 2), down_switch_to_outgoing_label_it2, 1, "godown", 2))

            if len(layers[i]) == 1:
                continue
            if num_cycling_paths == 0:
                continue
            # Start calculating cycling path between switches in given layer
            # Create subgraph omitting switches that are lower wrt. layer level
            subtract_switches = set()
            for k in range(0, i):
                subtract_switches = subtract_switches.union(set(layers[k]))
            subgraph_switches = set(weight_graph.nodes).difference(subtract_switches)
            subgraph = weight_graph.subgraph(subgraph_switches)

            # Find the next switch to route to in cycling path
            is_last_switch = v == layers[i][-1]
            v_next = layers[i][0]
            if not is_last_switch:
                v_next = layers[i][j+1]

            # Generate path between two switches
            if not is_last_switch:
                sub_ft = disjoint_paths_generator(subgraph, v, v_next, label(v, 1), label(v_next, 1), 3, "subpath", num_cycling_paths)
                sub_ft.extend(disjoint_paths_generator(subgraph, v, v_next, label(v, 2), label(v_next, 2), 3, "subpath", num_cycling_paths))
            else:
                sub_ft = disjoint_paths_generator(subgraph, v, v_next, label(v, 1), label(v_next, 2), 3, "subpath", num_cycling_paths)

            forwarding_table.extend(sub_ft)

    return forwarding_table.table


def shortest_path_generator(G: Graph, src: str, tgt: str, ingoing_label, outgoing_label):
    ft = ForwardingTable()
    if src == tgt:
        return ft

    try:
        path = list(shortest_path(G, src, tgt, weight=1))
    except networkx.exception.NetworkXNoPath:
        return ft

    for src, tgt in zip(path[:-2], path[1:-1]):
        ft.add_rule((src, ingoing_label), (2, tgt, ingoing_label))

    src, tgt = path[-2:]
    ft.add_rule((src, ingoing_label), (2, tgt, outgoing_label))
    return ft

def arborescence_path_generator(G: Graph, src: str, tgt: str, ingoing_label: oFEC, outgoing_label: oFEC):
    from target_based_arborescence.arborescences import find_arborescences

    ft = ForwardingTable()
    arborescences = find_arborescences(G, tgt)

    try:
        if not nx.has_path(G, src, tgt):
            return ft
    except:
        return ft

    if src == tgt or not any([len(arb) > 0 for arb in arborescences]):
        return ft

    fec_arbs = [(oFEC('cfor_arb', ingoing_label.name + f"_to_{tgt}_arb{i}{ab}", {'egress':ingoing_label.value['egress']}), a) for ab, (i, a)  in itertools.product(['a', 'b'], enumerate(arborescences))]

    # Create ingoing local lookup rule
    ft.add_rule((src, ingoing_label), (2, src, fec_arbs[0][0]))

    for i, (fec, a) in enumerate(fec_arbs):
        bounce_fec = None if i >= len(fec_arbs) - 1 else fec_arbs[i + 1][0]

        # Add outgoing local lookup rules
        ft.add_rule((tgt, fec), (0, tgt, outgoing_label))

        for s, t in a:
            ft.add_rule((s, fec), (1, t, fec))
            if bounce_fec is not None:
                ft.add_rule((s, fec), (2, s, bounce_fec))

    return ft


def disjoint_paths_generator(G: Graph, src: str, tgt: str, ingoing_label, outgoing_label, priority, type, num_paths):
    # Try to use underlying auxiliary graph for all pairs edge_disjoint_paths
    ft = ForwardingTable()
    if src == tgt:
        return ft
    if num_paths < 1:
        return ft

    try:
        dist_paths: List[List] = compute_disjoint_paths_by_shortest_path_weight_increase(G, src, tgt, num_paths, reset_graph_weights=False)
    except networkx.exception.NetworkXNoPath:
        return ft

    path_labels: list[oFEC] = []
    for i in range(len(dist_paths)):
        path_labels.append(oFEC("cfor", f"{ingoing_label.name}_{type}_{i}", {'ingress': ingoing_label.value['ingress'], 'egress': ingoing_label.value['egress']}))


    # Initially, try subpath 0
    ft.add_rule((src, ingoing_label), (priority, src, path_labels[0]))

    # If at tgt from any subpath, go to outgoing_label
    for l in path_labels:
        ft.add_rule((tgt, l), (0, tgt, outgoing_label))

    for i, path in enumerate(dist_paths):

        # for each edge in path
        for s, t in zip(path[:-1], path[1:]):
            # create forwarding using the path label
            ft.add_rule((s, path_labels[i]), (1, t, path_labels[i]))

            # if not last subpath
            if i < len(path_labels) - 1:
                # if link failed, bounce to next subpath
                ft.add_rule((s, path_labels[i]), (2, s, path_labels[i+1]))

                # create backtracking rules for next subpath
                if t not in dist_paths[i+1]:
                    ft.add_rule((t, path_labels[i+1]), (1, s, path_labels[i+1]))

    return ft


def disjoint_paths_generator_mult_tgts(tgt_to_graph: Dict[str, Graph], src, tgts, ingoing_label, tgt_to_outgoing_label_dict, priority, type, num_paths):
    # Try to use underlying auxiliary graph for all pairs edge_disjoint_paths
    ft = ForwardingTable()
    if num_paths < 1:
        return ft

    for j, tgt in enumerate(tgts):
        try:
            if j == 0:
                dist_paths: List[List] = compute_disjoint_paths_by_shortest_path_weight_increase(tgt_to_graph[tgt], src, tgt, num_paths,
                                                                                                 reset_graph_weights=True)
            else:
                dist_paths: List[List] = compute_disjoint_paths_by_shortest_path_weight_increase(tgt_to_graph[tgt], src, tgt, num_paths,
                                                                                                 reset_graph_weights=False)
        except networkx.exception.NetworkXNoPath:
            return ft

        path_labels: list[oFEC] = []
        for i in range(len(dist_paths)):
            path_labels.append(oFEC("cfor", f"{ingoing_label.name}_{type}_{i}", {'ingress': ingoing_label.value['ingress'], 'egress': ingoing_label.value['egress']}))

        # Initially, try subpath 0
        ft.add_rule((src, ingoing_label), (priority+j, src, path_labels[0]))

        # If at tgt from any subpath, go to outgoing_label
        for l in path_labels:
            ft.add_rule((tgt, l), (0, tgt, tgt_to_outgoing_label_dict[tgt]))

        for i, path in enumerate(dist_paths):
            # for each edge in path
            for s, t in zip(path[:-1], path[1:]):
                # create forwarding using the path label
                ft.add_rule((s, path_labels[i]), (1, t, path_labels[i]))

                # if not last subpath
                if i < len(path_labels) - 1:
                    # if link failed, bounce to next subpath
                    ft.add_rule((s, path_labels[i]), (2, s, path_labels[i+1]))

                    # create backtracking rules for next subpath
                    if t not in dist_paths[i+1]:
                        ft.add_rule((t, path_labels[i+1]), (1, s, path_labels[i+1]))

    return ft

def compute_disjoint_paths_by_shortest_path_weight_increase(G: Graph, src: str, tgt: str, num_paths, reset_graph_weights=True) -> List[List[str]]:
    weight_graph = G.copy()
    if reset_graph_weights:
        for u, v, d in weight_graph.edges(data=True):
            d["weight"] = 1
    paths = []

    for _ in range(num_paths):
        path = shortest_path(weight_graph, src, tgt, "weight")
        if path not in paths:
            paths.append(path)

        for i in (range(len(path) - 1)):
            weight_graph[path[i]][path[i+1]]["weight"] *= 64

    return paths


class CFor(MPLS_Client):
    protocol = "cfor"

    def __init__(self, router: Router, **kwargs):
        super().__init__(router, **kwargs)

        # The demands where this router is the tailend
        self.demands: dict[str, tuple[str, str]] = {}

        # Partial forwarding table containing only rules for this router
        self.partial_forwarding_table: dict[tuple[str, oFEC], list[tuple[int, str, oFEC]]] = {}

        self.path_generator = {
            'shortest': shortest_path_generator,
            'arborescence': arborescence_path_generator,
            'disjoint': disjoint_paths_generator
        }[kwargs['path']]
        self.num_down_paths = kwargs['num_down_paths']
        self.num_cycling_paths = kwargs['num_cycling_paths']

    def LFIB_compute_entry(self, fec: oFEC, single=False):
        for priority, next_hop, swap_fec in self.partial_forwarding_table[(self.router.name, fec)]:
            local_label = self.get_local_label(fec)
            assert(local_label is not None)

            if next_hop in fec.value["egress"]:
                yield (local_label, {'out': next_hop, 'ops': [{'pop': ''}], 'weight': priority})
            else:
                remote_label = self.get_remote_label(next_hop, swap_fec)
                assert(remote_label is not None)

                yield (local_label, {'out': next_hop if next_hop != self.router.name else self.LOCAL_LOOKUP, 'ops': [{'swap': remote_label}], 'weight': priority})

    # Defines a demand for a headend to this one
    def define_demand(self, headend: str):
        self.demands[f"{len(self.demands.items())}_{headend}_to_{self.router.name}"] = (headend, self.router.name)

    def commit_config(self):
        headends = list(map(lambda x: x[0], self.demands.values()))
        if len(headends) == 0:
            return
        ft = generate_pseudo_forwarding_table(self.router.network, headends, self.router.name, self.num_down_paths, self.num_cycling_paths)

        for (src, fec), entries in ft.items():
            src_client: CFor = self.router.network.routers[src].clients["cfor"]

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
        return 'cfor' in fec.fec_type and fec.value["egress"][0] == self.router.name
