## This is a sample Python script.
#import node as nd
import networkx as nx
#import matplotlib.pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#consider thinking about this as a lambda function for your framework type of thing
def valuation(demand, G, paths):
    value = None
    bestpath = []
    tempRemoved = []

    for path in paths:
        tempvalue = 0
        for link in path:
            tempvalue += ((demand[2] + G.edges[link[0],link[1]]['usage'])/G.edges[link[0],link[1]]['weight'])

        if value == None or tempvalue < value:
            tempRemoved = path.copy()
            bestpath = path
            value = tempvalue

    return bestpath, tempRemoved

def initializenetwork():
    G = nx.DiGraph()
    G.add_nodes_from([(1, {'jumpsfromtarget': 0}), (2, {'jumpsfromtarget': 0}), (3, {'jumpsfromtarget': 0})])
    G.add_nodes_from([(4, {'jumpsfromtarget': 0}), (5, {'jumpsfromtarget': 0}), (6, {'jumpsfromtarget': 0})])
    G.add_nodes_from([(7, {'jumpsfromtarget': 0}), (8, {'jumpsfromtarget': 0}), (9, {'jumpsfromtarget': 0})])
    G.add_nodes_from([(10, {'jumpsfromtarget': 0}), (11, {'jumpsfromtarget': 0}), (12, {'jumpsfromtarget': 0})])

    G.add_edges_from([(1, 2, {'weight': 1000, 'usage': 0}), (1, 3, {'weight': 200, 'usage': 0}), (2, 4, {'weight': 1000, 'usage': 0}), (2, 5, {'weight': 1000, 'usage': 0})])
    G.add_edges_from([(3, 5, {'weight': 500, 'usage': 0}), (5, 6, {'weight': 370, 'usage': 0}), (5, 7, {'weight': 1120, 'usage': 0}), (3, 7, {'weight': 120, 'usage': 0})])
    G.add_edges_from([(7, 8, {'weight': 880, 'usage': 0}), (8, 9, {'weight': 560, 'usage': 0}), (9, 12, {'weight': 1000, 'usage': 0}), (10, 12, {'weight': 420, 'usage': 0})])
    G.add_edges_from([(11, 12, {'weight': 720, 'usage': 0}), (5, 9, {'weight': 1000, 'usage': 0}), (7, 10, {'weight': 600, 'usage': 0}), (9, 10, {'weight': 660, 'usage': 0})])

    G.add_edges_from([(2, 1, {'weight': 1000, 'usage': 0}), (3, 1, {'weight': 200, 'usage': 0}), (4, 2, {'weight': 1000, 'usage': 0}), (5, 2, {'weight': 1000, 'usage': 0})])
    G.add_edges_from([(5, 3, {'weight': 500, 'usage': 0}), (6, 5, {'weight': 370, 'usage': 0}), (7, 5, {'weight': 1120, 'usage': 0}), (7, 3, {'weight': 120, 'usage': 0})])
    G.add_edges_from([(8, 7, {'weight': 880, 'usage': 0}), (9, 8, {'weight': 560, 'usage': 0}), (12, 9, {'weight': 1000, 'usage': 0}), (12, 10, {'weight': 420, 'usage': 0})])
    G.add_edges_from([(12, 11, {'weight': 720, 'usage': 0}), (9, 5, {'weight': 1000, 'usage': 0}), (10, 7, {'weight': 600, 'usage': 0}), (10, 9, {'weight': 660, 'usage': 0})])

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
        bestpath, tempRemoved =valuation(demand, grapher, paths)

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

#make sure no identical backup paths are made
def kpaths(paths, k, graph, demand, extrasteps = 0):

    extrapoPaths =[]
    srilist = []
    srilist.append(demand[1])

    for path in paths:
        extrapoPaths.append(path)
        tempList = []
        tempList2 = []
        for link in path:
            paffa = tempList.copy()
            tempList.append(link)
            tempList2.append(link)
            tempList2.append((link[1],link[0]))
            i =0
            while i <k:
                kaffa = paffa.copy()
                i +=1
                shortestrouteindex2(link[0], demand[1], graph, srilist, tempList2)
                p = pathfindrecursion(link[0], demand[1], graph, extrasteps, tempList2)
                for node in graph:
                    graph.nodes[node]['jumpsfromtarget'] = 0

                tu, ti = valuation(demand, graph, p)

                if tu != []:
                    kaffa += tu
                    if kaffa not in extrapoPaths and kaffa not in paths:
                        extrapoPaths.append(kaffa)



    return extrapoPaths

def linktonode(paths):
    pucks = []
    for path in paths:

        puck = []
        for link in path:
            if puck == []:
                puck.append(link[0])
            puck.append(link[1])
        pucks.append(puck)

    return pucks




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
            print("Link capacity of " +str(grapher.edges[p[0],p[1]]['weight']))
            print("Link usage of "+str(grapher.edges[p[0],p[1]]['usage'])+ "("+str(grapher.edges[p[0],p[1]]['usage']+demand[2])+")")
            print("Link utilization of "+str(grapher.edges[p[0],p[1]]['usage']/grapher.edges[p[0],p[1]]['weight'])+ "("+str((demand[2]+grapher.edges[p[0],p[1]]['usage'])/grapher.edges[p[0],p[1]]['weight'])+")")
            if grapher.edges[p[0],p[1]]['usage']/grapher.edges[p[0],p[1]]['weight'] > highestutilization:
                highestutilization = grapher.edges[p[0],p[1]]['usage']/grapher.edges[p[0],p[1]]['weight']
            if (grapher.edges[p[0],p[1]]['usage']+demand[2])/grapher.edges[p[0],p[1]]['weight'] > highestutilizationS:
                highestutilizationS = (grapher.edges[p[0],p[1]]['usage']+demand[2])/grapher.edges[p[0],p[1]]['weight']
            print()
        print("max utilization of this route is "+str(highestutilization)+ "("+str(highestutilizationS)+")")
        print()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    demands, G = initializenetwork()

    test = pathfind(demands[0], G, 5)
    test2 =pathfind(demands[1], G, 0)

    trial = kpaths(test,2,G,demands[0],5)

    print(linktonode(trial))


   # for demand in Demands:
    #    paths = pathfind(demand, graph, extrahops)
     #   fullpaths = kpaths(paths, k, graph, demand, extrahops)
