## This is a sample Python script.
#import node as nd
import networkx as nx
#import matplotlib.pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#consider thinking about this as a lambda function for your framework type of thing
def valuation(demand, link, G):
    return ((demand[2] + G.edges[link[0],link[1]]['usage'])/G.edges[link[0],link[1]]['cap'])

def initializenetwork():
    G = nx.DiGraph()
    G.add_nodes_from([(1, {'jumpsfromtarget': 0}), (2, {'jumpsfromtarget': 0}), (3, {'jumpsfromtarget': 0})])
    G.add_nodes_from([(4, {'jumpsfromtarget': 0}), (5, {'jumpsfromtarget': 0}), (6, {'jumpsfromtarget': 0})])
    G.add_nodes_from([(7, {'jumpsfromtarget': 0}), (8, {'jumpsfromtarget': 0}), (9, {'jumpsfromtarget': 0})])
    G.add_nodes_from([(10, {'jumpsfromtarget': 0}), (11, {'jumpsfromtarget': 0}), (12, {'jumpsfromtarget': 0})])

    G.add_edges_from([(1, 2, {'cap': 1000, 'usage': 0}), (1, 3, {'cap': 200, 'usage': 0}), (2, 4, {'cap': 1000, 'usage': 0}), (2, 5, {'cap': 1000, 'usage': 0})])
    G.add_edges_from([(3, 5, {'cap': 500, 'usage': 0}), (5, 6, {'cap': 370, 'usage': 0}), (5, 7, {'cap': 1120, 'usage': 0}), (3, 7, {'cap': 120, 'usage': 0})])
    G.add_edges_from([(7, 8, {'cap': 880, 'usage': 0}), (8, 9, {'cap': 560, 'usage': 0}), (9, 12, {'cap': 1000, 'usage': 0}), (10, 12, {'cap': 420, 'usage': 0})])
    G.add_edges_from([(11, 12, {'cap': 720, 'usage': 0}), (5, 9, {'cap': 1000, 'usage': 0}), (7, 10, {'cap': 600, 'usage': 0}), (9, 10, {'cap': 660, 'usage': 0})])

    G.add_edges_from([(2, 1, {'cap': 1000, 'usage': 0}), (3, 1, {'cap': 200, 'usage': 0}), (4, 2, {'cap': 1000, 'usage': 0}), (5, 2, {'cap': 1000, 'usage': 0})])
    G.add_edges_from([(5, 3, {'cap': 500, 'usage': 0}), (6, 5, {'cap': 370, 'usage': 0}), (7, 5, {'cap': 1120, 'usage': 0}), (7, 3, {'cap': 120, 'usage': 0})])
    G.add_edges_from([(8, 7, {'cap': 880, 'usage': 0}), (9, 8, {'cap': 560, 'usage': 0}), (12, 9, {'cap': 1000, 'usage': 0}), (12, 10, {'cap': 420, 'usage': 0})])
    G.add_edges_from([(12, 11, {'cap': 720, 'usage': 0}), (9, 5, {'cap': 1000, 'usage': 0}), (10, 7, {'cap': 600, 'usage': 0}), (10, 9, {'cap': 660, 'usage': 0})])

    D = []
    Demand1 = (1, 12, 500)
    Demand2 = (7, 3, 420)
    D.append(Demand1)
    D.append(Demand2)
    return D, G

def pathfind(demand, grapher, extrasteps = 0):
    bestpaths = []
    removedPath = []

    srilist = []
    srilist.append(demand[1])

    shortestrouteindex2(demand[0], demand[1], grapher, srilist, removedPath)
    paths = pathfindrecursion(demand[0], demand[1], grapher, extrasteps)

    for node in grapher:
        grapher.nodes[node]["jumpsfromtarget"] = 0

    while paths != []:
        value = None
        bestpath = None

        for path in paths:
            tempvalue = 0
            for link in path:
                tempvalue += valuation(demand, link, grapher)

            if value == None or tempvalue < value:
                tempRemoved = path.copy()
                bestpath = path
                value = tempvalue

        removedPath += tempRemoved
        bestpaths.append(bestpath.copy())

        shortestrouteindex2(demand[0], demand[1], grapher, srilist, removedPath)
        paths = pathfindrecursion(demand[0], demand[1], grapher, extrasteps, removedPath)
        for node in grapher:
            grapher.nodes[node]['jumpsfromtarget'] = 0

    for link in bestpaths[0]:
        grapher.edges[link[0], link[1]]['usage'] += demand[2]

    return bestpaths

#only checks each node once but has a memory overhead
def shortestrouteindex2(source, target, graph, nodes=[], removed =[], i = 0):
    i+=1
    nnodes = []
    for node in nodes:
        for link in graph[node]:
            if (link, node) not in removed:
                if link != target and graph.nodes[link]['jumpsfromtarget'] == 0:
                    graph.nodes[link]['jumpsfromtarget'] = i
                    nnodes.append(link)

    if nnodes !=[]:
        shortestrouteindex2(source, target, graph, nnodes, removed, i)
    return

def pathfindrecursion(source, target, graph, i=0, removedlinks =[], removednodes =[], currentpath =[], c=None):
    paths = []
    if c == None:
        removednodes = []
        c = graph.nodes[source]['jumpsfromtarget']-1
        removednodes.append(source)

    if source == target:
        paths.append(currentpath)
        return paths
    else:
        for link in graph[source]:
            if (source, link) not in removedlinks:
                if not (link in removednodes):
                    if (graph.nodes[link]['jumpsfromtarget'] - (c)) - i <= 0:
                        r1 = removednodes.copy()
                        r1.append(link)
                        p1 = currentpath.copy()
                        p1.append((source, link)) #check if this either puts all the paths in a shared list or in individual local lists
                        paths += pathfindrecursion(link, target, graph, i - (graph.nodes[link]['jumpsfromtarget']-c), removedlinks, r1, p1, graph.nodes[link]['jumpsfromtarget']-1)

    return paths

#todo: fix paranthesis values such that they don't affect the first path
def printtestvalues(test, demand, grapher):
    counter = 0
    print("test on route from", demand[0], "to", demand[1], "______________________________________")
    print()
    for t in test:
        highestutilization = 0.0
        highestutilizationS = 0.0
        counter +=1
        print("route "+ str(counter), "______________")
        print()
        for p in t:
            print("from "+str(p[0]) + " to "+ str(p[1]))
            print("Link capacity of " +str(grapher.edges[p[0],p[1]]['cap']))
            print("Link usage of "+str(grapher.edges[p[0],p[1]]['usage'])+ "("+str(grapher.edges[p[0],p[1]]['usage']+demand[2])+")")
            print("Link utilization of "+str(grapher.edges[p[0],p[1]]['usage']/grapher.edges[p[0],p[1]]['cap'])+ "("+str((demand[2]+grapher.edges[p[0],p[1]]['usage'])/grapher.edges[p[0],p[1]]['cap'])+")")
            if grapher.edges[p[0],p[1]]['usage']/grapher.edges[p[0],p[1]]['cap'] > highestutilization:
                highestutilization = grapher.edges[p[0],p[1]]['usage']/grapher.edges[p[0],p[1]]['cap']
            if (grapher.edges[p[0],p[1]]['usage']+demand[2])/grapher.edges[p[0],p[1]]['cap'] > highestutilizationS:
                highestutilizationS = (grapher.edges[p[0],p[1]]['usage']+demand[2])/grapher.edges[p[0],p[1]]['cap']
            print()
        print("max utilization of this route is "+str(highestutilization)+ "("+str(highestutilizationS)+")")
        print()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    demands, G = initializenetwork()

    test = pathfind(demands[0], G, 0)
    test2 =pathfind(demands[1], G, 0)

    printtestvalues(test, demands[0], G)
    printtestvalues(test2, demands[1], G)

    #[[(1, 2), (2, 5), (5, 9), (9, 12)], [(1, 3), (3, 5), (5, 9), (9, 12)],---- [(1, 3), (3, 7), (7, 10), (10, 12)]]
    #[[(1, 3), (3, 7), (7, 10), (10, 12)]]

    #[[(7, 3)]]
    #[[(7, 5), (5, 3)]]

    #[[(7, 8), (8, 9), (9, 5), (5, 2), (2, 1), (1, 3)], [(7, 10), (10, 9), (9, 5), (5, 2), (2, 1), (1, 3)]]
    #[]
    #[[(1, 2), (2, 5), (5, 9), (9, 12)], [(1, 3), (3, 5), (5, 9), (9, 12)]]
    #[]
