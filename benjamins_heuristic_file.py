## This is a sample Python script.
import node as nd
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def valuation(demand, link):
    return (demand.load + link.usage)/link.capacity

def initializenetwork(G, demands):
    grapher = nd.Graph()

    #Convert to different graph format
    for node in G.nodes:
        _node = nd.Node(node)
        grapher.nodes.append(_node)
    for src, tgt in G.edges:
        node1 = grapher.get_node(src)
        node2 = grapher.get_node(tgt)
        node1.newLink(node2, G[src][tgt]["weight"])

    D = []
    for src, tgt, load in demands:
        node1 = grapher.get_node(src)
        node2 = grapher.get_node(tgt)
        D.append(nd.Demand(node1, node2, load))

    return D, grapher

def pathfind(demand, grapher, extrasteps = 0):
    bestpaths = []
    removedPath = []

    shortestrouteindex(demand.source, demand.target)
    paths = pathfindrecursion(demand.source, demand.target, [], [], extrasteps)
    grapher.cleanup()

    while paths != []:
        value = None
        bestpath = None

        for path in paths:
            tempvalue = 0
            tempPath = []

            for link in path:
                tempvalue += valuation(demand, link)
                tempPath.append(link)

            if value == None or tempvalue < value:
                tempRemoved = tempPath
                bestpath = path
                value = tempvalue

        removedPath += tempRemoved
        bestpaths.append(bestpath.copy())

        shortestrouteindex(demand.source, demand.target, removedPath)
        paths = pathfindrecursion(demand.source, demand.target, removedPath, [], extrasteps)
        grapher.cleanup()

    for link in bestpaths[0]:
        link.usage += demand.load
    return bestpaths

def shortestrouteindex(source, target, removednodes =[], i = 0):
    i += 1
    target.jumpsfromtarget = i
    for link in target.clinks:
        if link not in removednodes:
            if (link.nextnode(target).jumpsfromtarget == 0 or link.nextnode(target).jumpsfromtarget > i):
                if target != source:
                    shortestrouteindex(source, link.nextnode(target), removednodes, i)
    return

def pathfindrecursion(source, target, removednodes =[], currentpath =[], i=0, c=None):
    paths = []
    if c == None:
        c = source.jumpsfromtarget
    if source == target:
        paths.append(currentpath)
        return paths
    else:
        #print(c)
        for link in source.clinks:
            if not (link in removednodes):
                if (link.nextnode(source).jumpsfromtarget - (c-1)) - i <= 0:
                    r1 = removednodes.copy()
                    r1.append(link)
                    p1 = currentpath.copy()
                    p1.append(link) #check if this either puts all the paths in a shared list or in individual local lists
                    paths += pathfindrecursion(link.nextnode(source), target, r1, p1, i -(link.nextnode(source).jumpsfromtarget-c), c-1)

    return paths

#todo: fix paranthesis values such that they don't affect the first path
def printtestvalues(test, demand):
    counter = 0
    for t in test:
        highestutilization = 0.0
        highestutilizationS = 0.0
        counter +=1
        print("route "+ str(counter))
        print()
        for p in t:

            print("from "+str(p.node1.identity) + " to "+ str(p.node2.identity))
            print("Link capacity of " +str(p.capacity))
            print("Link usage of "+str(p.usage)+ "("+str(p.usage+demand.load)+")")
            print("Link utilization of "+str(p.usage/p.capacity)+ "("+str((demand.load+p.usage)/p.capacity)+")")
            if p.usage/p.capacity > highestutilization:
                highestutilization = p.usage/p.capacity
            if (p.usage+demand.load)/p.capacity > highestutilizationS:
                highestutilizationS = (p.usage+demand.load)/p.capacity
            print()
        print("max utilization of this route is "+str(highestutilization)+ "("+str(highestutilizationS)+")")
        print("/cut1")
        print()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    network, demands, grapher = initializenetwork()

    #test =pathfind(demands[0], grapher, 0)
    test2 =pathfind(demands[1], grapher, 5)

    #printtestvalues(test, demands[0])
    printtestvalues(test2, demands[1])
    #printtestvalues(test2)


