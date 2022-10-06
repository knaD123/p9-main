import queue
from functools import cmp_to_key
from typing import Union, Set, List, Dict, Tuple

import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt

#import graphviz as gv

from mpls_classes import MPLS_Client, Network, oFEC, Router
from target_based_arborescence.arborescences import find_arborescences


def find_distance_edges(network: Network, ingress: str, egress: str) -> List[List[Tuple[str, str]]]:
    edges: set[tuple[str, str]] = set([(n1, n2) for (n1, n2) in network.topology.edges if n1 != n2] \
                                      + [(n2, n1) for (n1, n2) in network.topology.edges if n1 != n2])

    layers: list[list] = list(list())  # result

    frontiers: set = set()  # set containing next layer switches
    frontiers.add(egress)  # initialise frontiers to egress, then we traverse the network backwards from there

    i = 0  # number indicating the current layer
    while len(frontiers) != 0:
        layers.append(list())
        next_frontiers: set = set()

        for v in frontiers:
            incoming_edges = set(filter(lambda e: e[1] == v, edges))
            if len(incoming_edges) == 0:
                continue

            # add sources of incoming edges to as next frontiers
            next_frontiers = next_frontiers.union(set(map(lambda e: e[0], incoming_edges)))
            edges = edges - incoming_edges  # we remove already analyzed edges so they are not reconsidered
            layers[i].extend(list(incoming_edges))

        frontiers = next_frontiers.copy()
        i += 1

    # remove last layer if there is no edges in it
    if len(layers[:0]) == 0:
        layers.remove(layers[:0])

    # Ensure switches only have 1 outgoing rule in the same layer
    # We sort layers by source and remove edges if src is same as last element
    new_layers = list(list())
    for layer in layers:
        layer = sorted(layer, key=cmp_to_key(lambda e1, e2: e1[0] < e2 [0] if e1[0] != e2[0] else e1[1] < e2[1]))
        fixed_layer = layer.copy()

        last: int = layer[0][0]
        for src2, tgt2 in layer[1:]:
            if last == src2:
                fixed_layer.remove((src2, tgt2))
            else:
                last = src2

        new_layers.append(fixed_layer)
    layers = new_layers.copy()

    # remove edges that have egress switch as src
    layers = [[(s,t) for s,t in l if s != egress] for l in layers]


    # Remove loops
    E: list[tuple[str, str, int]] = []
    for i, layer in enumerate(layers):
        for src, tgt in layer:
            E.append((src, tgt, i))
    V = set([s for s,_,_ in E] + [t for _,t,_ in E])

    while True:
        cycles = find_cycles(V, E)
        if len(cycles) == 0:
            break

        demote_or_remove_loops(V, ingress, E, cycles)

    max_layer = max(E, key=lambda x: x[2])[2]
    layers = [[(s,t) for (s,t,lp) in E if lp == l] for l in range(max_layer + 1)]

    return layers


def find_cycles(vertices: Set[str], E: List[Tuple[str, str, int]]) -> List[List[str]]:
    cycles: list[list[str]] = []
    missing = vertices.copy()

    def DFS_cycle(path: List[Tuple[str, int]]):
        missing.discard(path[-1][0])
        v, layer = path[-1]
        for src, tgt, l in E:
            if src == v and l >= layer - 1:
                if (tgt, l) in path:
                    idx = path.index((tgt, l))
                    cycles.append([v for v, l in path[idx:]])
                else:
                    DFS_cycle(path + [(tgt, l)])

    while len(missing) > 0:
        v = missing.pop()
        DFS_cycle([(v, 0)])

    # Canonize the cycles
    def canonize(cycle: List[str]):
        min_element = min(cycle)
        min_idx = cycle.index(min_element)
        return cycle[min_idx:] + cycle[:min_idx]

    return [list(x) for x in set(tuple(canonize(cycle)) for cycle in cycles)]

def demote_or_remove_loops(vertices: Set[str], ingress: str, E: List[Tuple[str, str, int]], cycles: List[List[str]]):
    # Minimum number of failures required to reach this vertex
    min_failure_reach: dict[str, int] = {ingress: 0}

    edge_to_layer = {(s,t): l for s,t,l in E}

    for s,t,l in E:
        assert edge_to_layer[(s,t)] == l

    unfinised = set(vertices)
    last_unfinished = set()
    while unfinised != last_unfinished:
        last_unfinished = unfinised.copy()

        for v, f_v in min_failure_reach.copy().items():
            outgoing_edges = [(s,t,l) for s,t,l in E if s == v]
            outgoing_edges.sort(key=lambda x: x[2])

            for f_e, (_, t, _) in enumerate(outgoing_edges):
                if t not in min_failure_reach or f_e + f_v < min_failure_reach[t]:
                    min_failure_reach[t] = f_e + f_v
                    unfinised.add(t)

            unfinised.discard(v)

    # Set unreachable to "infinity"
    min_failure_reach.update({v: 1_000_000 for v in unfinised})

    # Minimum number of failures required to use this edge
    min_failure_edge: dict[tuple[str, str], int] = {}

    for v in vertices:
        outgoing_edges = [(s, t, l) for s, t, l in E if s == v]
        outgoing_edges.sort(key=lambda x: x[2])

        for f, (s,t,l) in enumerate(outgoing_edges):
            min_failure_edge[(s,t)] = min_failure_reach[v] + f

    def can_promote_to(edge: Tuple[str, str]) -> Union[None, int]:
        ln = edge_to_layer[edge]

        # We need to go at least 2 layers above the layer of the next edge in the cycle to break the loop
        pos_layers = [lp + 1 for (sp, tp, lp) in E if sp == edge[1] and lp >= ln]
        if len(pos_layers) > 0:
            return pos_layers[0]
        else:
            return None


    cycles.sort(key=len)
    for cycle in cycles:
        # Check that we did not fix this cycle already
        p = cycle[-1]
        cl = 0
        for n in cycle:
            l = edge_to_layer.get((p,n), None)
            if l is None or not (l >= cl - 1):
                break
            p = n
            cl = l
        else:
            def e_comp(e1: Tuple[str, str], e2: Tuple[str, str]):
                if min_failure_edge[e1] == min_failure_edge[e2]:
                    if edge_to_layer[e1] == edge_to_layer[e2]:
                        cpt1 = can_promote_to(e1)
                        cpt2 = can_promote_to(e2)
                        return (10000 if cpt1 is None else cpt1) <= (10000 if cpt2 is None else cpt2)
                    else:
                        return edge_to_layer[e1] > edge_to_layer[e2]
                else:
                    return min_failure_edge[e1] > min_failure_edge[e2]

            max_fail_edge: tuple[str, str] = (cycle[-1], cycle[0])

            p = cycle[0]
            for n in cycle[1:]:
                if e_comp((p,n), max_fail_edge):
                    max_fail_edge = (p,n)
                p = n

            # Remove the edge
            l = edge_to_layer[max_fail_edge]
            pl = can_promote_to(max_fail_edge)

            E.remove((max_fail_edge[0], max_fail_edge[1],l))
            edge_to_layer.pop(max_fail_edge, None)
            #print(f"Removed {max_fail_edge}")

            if pl is not None:
                E.append((max_fail_edge[0], max_fail_edge[1], pl))
                edge_to_layer[max_fail_edge] = pl
                #print(f"Promoted {max_fail_edge} from {l} to {pl}")


class HopDistance_Client(MPLS_Client):
    protocol = "hop_distance"

    def __init__(self, router: Router, **kwargs):
        super().__init__(router)

        # The demands where this router is the tailend
        self.demands: dict[str, tuple[str, str]] = {}

        # [(headend, [fecs_for_layer_i])]
        self.headend_layers: list[tuple[str, list[oFEC]]] = []

        # Incoming FECs
        self.incoming_fecs: list[oFEC] = []

        # The next_hop and next_fec for this router in some FEC (not only those FECs that are tailend here)
        self.demand_fec_layer_next_hop: dict[str, dict[oFEC, str]] = {}

    # Abstract functions to be implemented by each client subclass.
    def LFIB_compute_entry(self, fec: oFEC, single=False):
        # if fec not in self.incoming_fecs:
        #     return

        if fec.value[1] == self.router.name:
            return

        demand = fec.value[3]
        for next_hop_fec, next_hop in self.demand_fec_layer_next_hop.get(demand, {}).items():
            if next_hop_fec.value[2] >= fec.value[2] - 1:
                local_label = self.get_local_label(fec)
                remote_label = self.get_remote_label(next_hop, next_hop_fec)
                assert(local_label is not None)
                if next_hop == fec.value[1]:
                    yield (local_label, {"out": next_hop, "ops": [{"pop": ""}], "weight": next_hop_fec.value[2]})
                else:
                    assert(remote_label is not None)
                    yield (local_label, {"out": next_hop, "ops": [{"swap": remote_label}], "weight": next_hop_fec.value[2]})

    # Defines a demand for a headend to this one
    def define_demand(self, headend: str):
        self.demands[f"{len(self.demands.items())}_{headend}_to_{self.router.name}"] = (headend, self.router.name)

    def commit_config(self):
        for demand, (ingress, egress) in self.demands.items():
            # Find the distance layers
            distance_edges = find_distance_edges(self.router.network, ingress, egress)

            # Create graph for debugging
            # g = gv.Digraph(format="svg")
            #
            # for i, layer in enumerate(distance_edges):
            #     for s,t in layer:
            #         g.edge(s,t, str(i))
            #
            # g.node(ingress, ingress, color="red")
            # g.node(egress, egress, color="blue")
            #
            # g.render(f"hop_distance_{demand}", "gen")

            for i, layer in enumerate(distance_edges):
                # For each layer, create a fec that represents that layer
                layer_fec = oFEC("hop_distance", f"{demand}_l{i}", (ingress, egress, i, demand))

                # Add the next_hop information to the routers involved
                for (src, tgt) in layer:
                    src_router: Router = self.router.network.routers[src]
                    src_client: HopDistance_Client = src_router.clients["hop_distance"]

                    if demand not in src_client.demand_fec_layer_next_hop:
                        src_client.demand_fec_layer_next_hop[demand] = {}

                    src_client.demand_fec_layer_next_hop[demand][layer_fec] = tgt

                    tgt_router: Router = self.router.network.routers[tgt]
                    tgt_client: HopDistance_Client = tgt_router.clients["hop_distance"]

                    tgt_client.incoming_fecs.append(layer_fec)


    def compute_bypasses(self):
        pass

    def LFIB_refine(self, label):
        pass

    def known_resources(self):
        for _, fec_dict in self.demand_fec_layer_next_hop.items():
            for fec, _ in fec_dict.items():
                yield fec
        for fec in self.incoming_fecs:
            yield fec

    def self_sourced(self, fec: oFEC):
        return fec.fec_type == 'hop_distance' and fec.value[0] == self.router.name
