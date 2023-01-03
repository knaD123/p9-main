import math
from networkx import shortest_path

def prefixsort(pathdict):
    new_pathdict = dict()
    for src, tgt in pathdict.keys():
        new_pathdict[(src, tgt)] = []
        new_pathdict[src, tgt].append(pathdict[src, tgt][0])
        del pathdict[src, tgt][0]

    for src, tgt in pathdict.keys():
        while pathdict[src,tgt] != []:
            max_common_prefix_path = max(pathdict[src, tgt],
                                         key=lambda x: common_prefix_length(new_pathdict[src, tgt][-1], x))
            if common_prefix_length(new_pathdict[src, tgt][-1], max_common_prefix_path) == 1:
                max_common_prefix_path = min(pathdict[src, tgt], key=len)
            new_pathdict[src, tgt].append(max_common_prefix_path)
            pathdict[src, tgt].remove(new_pathdict[src, tgt][-1])

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
        max_hops_for_demand = math.floor(((len(shortest_path(graph, src, tgt))) - 1) * max_stretch)
        new_pathdict[(src, tgt)] = []

        if max_hops_for_demand >= len(pathdict[src, tgt][0]) - 1:
            new_pathdict[src, tgt].append(pathdict[src, tgt][0])
            # -2 because first node does not count as a hop and assume that last link fails so it does not make the hop
            max_hops_for_demand -= len(pathdict[src, tgt][0]) - 2
        else:
            for path in pathdict[src, tgt]:
                if max_hops_for_demand >= len(path) - 1:
                    new_pathdict[src, tgt].append(path)
                    # -2 because first node does not count as a hop and assume that last link fails so it does not make the hop
                    max_hops_for_demand -= len(path) - 2
                    break

        for path in pathdict[src, tgt]:
            if path in new_pathdict[src, tgt]:
                continue
            if max_hops_for_demand >= (
                    (len(new_pathdict[src, tgt][-1]) - common_prefix_length(new_pathdict[src, tgt][-1], path) - 1) + (
                    len(path) - (common_prefix_length(new_pathdict[src, tgt][-1], path)))):
                max_hops_for_demand = max_hops_for_demand - ((len(new_pathdict[src, tgt][-1]) - common_prefix_length(
                    new_pathdict[src, tgt][-1], path) - 1) + (len(path) - (
                    common_prefix_length(new_pathdict[src, tgt][-1], path)) - 1))
                new_pathdict[src, tgt].append(path)
    return new_pathdict