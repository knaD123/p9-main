from typing import Tuple

from mpls_classes import oFEC

import networkx as nx

import graphviz as gv


class ForwardingTable:
    def __init__(self):
        self.table: dict[tuple[str, oFEC], list[tuple[int, str, oFEC]]] = {}

    def add_rule(self, key: Tuple[str, oFEC], value: Tuple[int, str, oFEC]):
        if not self.table.keys().__contains__(key):
            self.table[key] = []
        self.table[key].append(value)

    def extend(self, other):
        for lhs, rhs_list in other.table.items():
            for rhs in rhs_list:
                self.add_rule(lhs, rhs)

    def to_graphviz(self, name: str, G: nx.Graph = nx.Graph()):
        fecs = set(map(lambda x: x[1], self.table.keys()))
        fec_to_color = dict(zip(fecs, (len(fecs)//12 +1)*['blue','red','green','magenta','orange','cyan','indigo','gold','aquamarine','olive','pink','lightsalmon','maroon']))
        g = gv.Digraph(format='svg')

        for s,t in G.edges:
            g.edge(s,t, dir='none', color='black')

        for (s, f1), entries in self.table.items():
            for p, t, f2 in entries:
                g.edge(s, t, label=str(p), arrowtail="dot", dir="both", color=f"{fec_to_color[f2]}:{fec_to_color[f1]}")

        #g.render(f'ft_{name}', 'gen')
