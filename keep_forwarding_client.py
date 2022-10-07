import itertools

import networkx as nx
import networkx.exception

from mpls_classes import *
from functools import *
from networkx import shortest_path

from ForwardingTable import ForwardingTable

from typing import Dict, Tuple, List, Callable


def undirect(e: Tuple[str, str]):
    return (min(e[0], e[1]), max(e[0], e[1]))

def build_kf_traversal(topology: Graph) -> Dict[Tuple[str,str], List[str]]:
    G: nx.DiGraph = topology.copy().to_directed()
    G.clear_edges()
    G.add_weighted_edges_from(map(lambda e: (e[0], e[1], 0), topology.to_directed().edges))

    used_nodes = set()

    edge_use_count = {undirect(e): 0 for e in G.edges}

    s = list(G.edges())[0]
    traversal = []
    while any(c == 0 for _,c in edge_use_count.items()):
        if (s[1], s[0]) in G.edges:
            G[s[1]][s[0]]['weight'] = 1
        cycle = [s[0]] + nx.shortest_path(G, s[1], s[0], weight='weight')
        cycle = list(zip(cycle[:-1], cycle[1:]))

        i = next((i for i, e in enumerate(traversal) if e[1] == s[0]), 0)
        traversal = traversal[:i+1] + cycle + traversal[i+1:]

        used_nodes.update({t for _,t in cycle})

        for e in cycle:
            edge_use_count[undirect(e)] += 1

            G.remove_edge(e[0], e[1])

        # Find next node s.t. it has a remaining edge
        for n in used_nodes:
            p = next((e for e in G.edges(n) if edge_use_count[undirect(e)] == 0), None)
            if p is not None:
                s = p
                break
        else:
            assert(len(used_nodes) == G.number_of_nodes())

    trav_dict: dict[Tuple[str, str], list[str]] = {e: [] for e in topology.edges}
    undirected_traversal = list(map(lambda e: undirect(e), traversal))
    undirected_traversal = list(dict.fromkeys(undirected_traversal)) # Remove duplicates while keeping order

    for e in topology.to_directed().edges:
        ue = undirect(e)
        idx = undirected_traversal.index(ue)
        e_traversal = undirected_traversal[idx+1:] + undirected_traversal[:idx+1]

        trav_dict[e] = list(map(lambda ep: ep[0] if ep[1] == e[1] else ep[1], filter(lambda ep: e[1] == ep[0] or e[1] == ep[1], e_traversal)))

    return trav_dict


def generate_pseudo_forwarding_table(network: Network, ingress: str, egress: str) -> ForwardingTable:
    router_to_label = {r: oFEC('kf', f'{ingress}_to_{egress}_last_at_{r}', {'egress': egress, 'ingress': ingress}) for r in network.routers.keys()}

    edges: set[tuple[str, str]] = set([(n1, n2) for (n1, n2) in network.topology.edges if n1 != n2] \
                                      + [(n2, n1) for (n1, n2) in network.topology.edges if n1 != n2] \
                                      + [(n,n) for n in network.topology.nodes])
    network.compute_dijkstra(weight=1)
    D: dict[str, int] = {r: network.routers[r].dist[egress] for r in network.routers.keys()}

    kf_traversal = build_kf_traversal(network.topology)

    def true_sink(e: Tuple[str, str], start=None):
        if start is None:
            start = e[1]
        elif e[1] == start:
            return start

        v, u = e
        u_degree = network.topology.degree[u]
        if u_degree > 2:
            return u
        elif u_degree == 2:
            u_edges = list(network.topology.edges(u))
            return true_sink([(s,t) for s,t in u_edges if t != v][0], start)
        else:
            return v

    ft = ForwardingTable()

    for src, tgt in edges:
        if tgt == egress:
            continue

        priority = 0
        def add_ordered_rules(edges: List[Tuple[str, str]]):
            nonlocal priority
            edges.sort(key=lambda x: network.topology.degree[x[1]], reverse=True)

            for _, t in edges:
                ft.add_rule((tgt, router_to_label[src]), (priority, t, router_to_label[tgt]))
                priority = priority + 1

        out_edges = [(s,t) for s,t in edges if s == tgt]
        if D[src] > D[tgt]:
            add_ordered_rules([(s,t) for s,t in out_edges if D[t] < D[s]])
            add_ordered_rules([(s,t) for s,t in out_edges if D[t] == D[s] and tgt != t])
            add_ordered_rules([(s,t) for s,t in out_edges if D[t] > D[s] and true_sink((s,t)) != true_sink((src, tgt))])
            add_ordered_rules([(s,t) for s,t in out_edges if D[t] > D[s] and network.topology.degree[t] > 2])
        elif D[src] < D[tgt]:
            add_ordered_rules([(s,t) for s,t in out_edges if D[t] < D[s] and true_sink((s,t)) != true_sink((tgt, src))])
            add_ordered_rules([(s,t) for s,t in out_edges if D[t] == D[s] and tgt != t])
            add_ordered_rules([(s,t) for s,t in out_edges if D[t] > D[s]])
        else:
            add_ordered_rules([(s,t) for s,t in out_edges if D[t] < D[s]])

            if src == tgt:
                s,t = next((s,t) for s,t in kf_traversal.keys() if t == tgt)
            else:
                s,t = (src,tgt)

            for nh in kf_traversal[(s,t)]:
                ft.add_rule((tgt, router_to_label[src]), (priority, nh, router_to_label[tgt]))
                priority += 1

            add_ordered_rules([(s,t) for s,t in out_edges if D[t] > D[s]])

        ft.add_rule((tgt, router_to_label[src]), (priority, src, router_to_label[tgt]))

    return ft


class KeepForwarding(MPLS_Client):
    protocol = "kf"

    def __init__(self, router: Router, **kwargs):
        super().__init__(router)

        # The demands where this router is the tailend
        self.demands: dict[str, tuple[str, str]] = {}

        # Partial forwarding table containing only rules for this router
        self.partial_forwarding_table: dict[tuple[str, oFEC], list[tuple[int, str, oFEC]]] = {}


    def LFIB_compute_entry(self, fec: oFEC, single=False):
        for priority, next_hop, swap_fec in self.partial_forwarding_table[(self.router.name, fec)]:
            local_label = self.get_local_label(fec)
            assert(local_label is not None)

            if fec.value["egress"] == next_hop:
                yield (local_label, {'out': next_hop, 'ops': [{'pop': ''}], 'weight': priority})
            else:
                remote_label = self.get_remote_label(next_hop, swap_fec)
                assert(remote_label is not None)

                yield (local_label, {'out': next_hop if next_hop != self.router.name else self.LOCAL_LOOKUP, 'ops': [{'swap': remote_label}], 'weight': priority})

    # Defines a demand for a headend to this one
    def define_demand(self, headend: str):
        self.demands[f"{len(self.demands.items())}_{headend}_to_{self.router.name}"] = (headend, self.router.name)

    def commit_config(self):
        for demand, (ingress, egress) in self.demands.items():
            ft = generate_pseudo_forwarding_table(self.router.network, ingress, egress)

            #ft.to_graphviz(f'kf_{ingress}_{egress}', self.router.network.topology)
            #Clean the forwarding table
            for _, rules in ft.table.items():
                seen = set()
                remove = set()
                for rule in rules:
                    if rule[1] in seen:
                        remove.add(rule)
                    seen.add(rule[1])

                for rule in remove:
                    rules.remove(rule)


            for (src, fec), entries in ft.table.items():
                src_client: KeepForwarding = self.router.network.routers[src].clients["kf"]

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
        return 'kf' in fec.fec_type and fec.value["egress"] == self.router.name
