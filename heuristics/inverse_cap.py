import networkx as nx
import itertools

def inverse_cap(client):
    network = client.router.network.topology.to_directed()
    for (u,v), cap in client.link_caps.items():
        network[u][v]["weight"] = 1 / cap

    demands = client.loads

    for src, tgt, load in demands:
        path = nx.shortest_path(network, src, tgt, weight="weight")
        yield ((src, tgt), path)



