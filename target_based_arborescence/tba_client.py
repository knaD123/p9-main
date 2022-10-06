import itertools
from typing import Union, Tuple, List

from mpls_classes import MPLS_Client, Network, oFEC, Router
from target_based_arborescence.arborescences import find_arborescences, complex_find_arborescence, \
    multi_create_arborescences

import graphviz as gv


class TargetBasedArborescence(MPLS_Client):
    protocol = "tba"

    def __init__(self, router: Router, **kwargs):
        super().__init__(router, **kwargs)

        # The demands where this router is the tailend
        self.demands: dict[str, Tuple[str, str]] = {}

        # The arborescences that are rooted in this router
        self.rooted_arborescences: List[List[Tuple[str, str]]] = []

        # The FECs this router is a non-tailend part of. fec_name -> (fec, next_hop, bounce_fec_name)
        self.arborescence_next_hop: dict[str, Tuple[oFEC, List[Tuple[Union[str, None], int]], str]] = {}

        self.arborescence_finder = {
            'simple': find_arborescences,
            'complex': complex_find_arborescence,
            'multi': multi_create_arborescences
        }[kwargs['path']]

        self.memory_per_flow = kwargs['per_flow_memory']


    # Abstract functions to be implemented by each client subclass.
    def LFIB_compute_entry(self, fec: oFEC, single=False):
        _, next_hops, bounce_fec_name = self.arborescence_next_hop[fec.name]
        for next_hop, p in next_hops:
            local_label = self.get_local_label(fec)
            assert(local_label is not None)

            # If final hop, pop the label
            if next_hop != None:
                if next_hop == fec.value[0]:
                    main_entry = {"out": next_hop, "ops": [{"pop" : ""}], "weight" : p}
                else:
                    remote_label = self.get_remote_label(next_hop, fec)
                    assert(remote_label is not None)
                    main_entry = {"out": next_hop, "ops": [{"swap" : remote_label}], "weight" : p}
                yield (local_label, main_entry)

            if bounce_fec_name is not None:
                bounce_fec, _, _ = self.arborescence_next_hop[bounce_fec_name]
                remote_bounce_label = self.get_remote_label(self.router.name, bounce_fec)
                assert(remote_bounce_label is not None)

                bounce_entry = {"out": self.LOCAL_LOOKUP, "ops": [{"swap" : remote_bounce_label}], "weight" : 3}
                yield (local_label, bounce_entry)


    # Defines a demand for a headend to this one
    def define_demand(self, headend: str):
        self.demands[f"{len(self.demands.items())}_{headend}_to_{self.router.name}"] = (headend, self.router.name)

    def commit_config(self):
        useable_memory = self.memory_per_flow * len(self.demands)

        if len(self.demands) == 0:
            return

        headends = tuple(set(map(lambda x: x[0], self.demands.values()))) #lists cannot be hashed :|
        self.rooted_arborescences = self.arborescence_finder(self.router.network.topology, self.router.name, useable_memory)

        #Create graph for debugging
        g = gv.Digraph(format="svg")

        '''
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        for arb, color in zip(self.rooted_arborescences, colors):
            for src, tgt, _ in arb:
                g.edge(src, tgt, color=color)

        g.render(f"arborescence_{self.router.name}", "gen")'''

        fec_arbors: list[tuple[oFEC, list[tuple[str, str, int]]]] =\
            [(oFEC("arborescence", f"{self.router.name}_{i}", (self.router.name, i, headends, i == 0)), a)
                for i, a in enumerate(self.rooted_arborescences)]

        for i, (fec, a) in enumerate(fec_arbors):
            def find_bounce_fec(v: str) -> str:
                k = i#(i * 31) % len(fec_arbors)
                for j in list(range(k + 1, len(fec_arbors))) + list(range(0, k + 1)):
                    if any(s == v for s, t, _ in fec_arbors[j][1]):
                        return fec_arbors[j][0].name
                assert("Could not bounce to any arborescence??")

            def add_rule(src, tgt, p):
                src_router: TargetBasedArborescence = self.router.network.routers[src].clients["tba"]
                if fec.name not in src_router.arborescence_next_hop:
                    src_router.arborescence_next_hop[fec.name] = (fec, [], bounce_fec_name)

                src_router.arborescence_next_hop[fec.name][1].append((tgt, p))

            # Loop over all edges in arborescence
            for src, tgt, p in a:
                # Add an arborescence next-hop for this FEC to the routers in the arborescence
                bounce_fec_name = find_bounce_fec(src)
                add_rule(src, tgt, p)

            # Loop over all vertices NOT source of any edge in arborescence
            not_in_arb = set(self.router.network.topology.nodes()) - set(map(lambda e: e[0], a))
            for src in not_in_arb:
                bounce_fec_name = find_bounce_fec(src)

                # This has no next-hop, but should have bounce fec
                add_rule(src, None, -1)



    def compute_bypasses(self):
        pass

    def LFIB_refine(self, label):
        pass

    def known_resources(self):
        for _, v in self.arborescence_next_hop.items():
            yield v[0]

    def self_sourced(self, fec: oFEC):
        return fec.fec_type == "arborescence" and fec.value[0] == self.router.name

