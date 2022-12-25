import networkx as nx
import itertools

def inverse_cap(client):
    network = client.router.network.topology.to_directed()

    for (u,v), cap in client.link_caps.items():
        network[u][v]["weight"] = 1 / cap
    for src, tgt, load in itertools.cycle(sorted(client.loads, key=lambda x: x[2], reverse=True)):
        path = nx.shortest_path(network, src, tgt, weight="weight")
        yield ((src, tgt), path)