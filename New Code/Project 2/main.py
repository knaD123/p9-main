## This is a sample Python script.
import node as nd
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def valuation(demand, link):
    return (demand.load + link.usage)/link.capacity

def initializenetwork():
    grapher = nd.Graph()
    n1 = nd.Node(1)
    grapher.nodes.append(n1)
    n2 = nd.Node(2)
    grapher.nodes.append(n2)
    n3 = nd.Node(3)
    grapher.nodes.append(n3)
    n4 = nd.Node(4)
    grapher.nodes.append(n4)
    n5 = nd.Node(5)
    grapher.nodes.append(n5)
    n6 = nd.Node(6)
    grapher.nodes.append(n6)
    n7 = nd.Node(7)
    grapher.nodes.append(n7)
    n8 = nd.Node(8)
    grapher.nodes.append(n8)
    n9 = nd.Node(9)
    grapher.nodes.append(n9)
    n10 = nd.Node(10)
    grapher.nodes.append(n10)
    n11 = nd.Node(11)
    grapher.nodes.append(n11)
    n12 = nd.Node(12)
    grapher.nodes.append(n12)

    n1.newLink(n2, 1000)
    n1.newLink(n3, 200)
    n2.newLink(n4, 1000)
    n2.newLink(n5, 1000)
    n3.newLink(n5, 500)
    n5.newLink(n6, 370)
    n5.newLink(n7, 1120)
    n3.newLink(n7, 120)
    n7.newLink(n8, 880)
    n8.newLink(n9, 560)
    n9.newLink(n12, 1000)
    n10.newLink(n12, 420)
    n11.newLink(n12, 720)
    n5.newLink(n9, 1000)
    n7.newLink(n10, 600)
    n9.newLink(n10, 660)
    n12.newLink(n11, 1000)

    d = []
    d.append(nd.Demand(n1, n12, 500))
    d.append(nd.Demand(n7, n3, 420))
    return n1, d, grapher

def pathfind(demand, grapher, extrasteps = 0):
    bestpaths = []
    removedPath = []

    srilist = []
    srilist.append(demand.target)

    shortestrouteindex2(demand.source, demand.target, srilist)
    paths = pathfindrecursion(demand.source, demand.target, extrasteps)
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

        shortestrouteindex2(demand.source, demand.target, srilist, removedPath)
        paths = pathfindrecursion(demand.source, demand.target, extrasteps, removedPath)
        grapher.cleanup()

    for link in bestpaths[0]:
        link.usage += demand.load
    return bestpaths

#only checks each node once but has a memory overhead
def shortestrouteindex2(source, target, nodes=[], removed =[], i = 1):
    i+=1
    nnodes = []
    for node in nodes:
        for link in node.clinks:
            if link not in removed:
                if link.nextnode(node) != target and link.nextnode(node).jumpsfromtarget == 0:
                    link.nextnode(node).jumpsfromtarget = i
                    nnodes.append(link.nextnode(node))
    if nnodes !=[]:
        shortestrouteindex2(source, target, nnodes, removed, i)
    return

#has better memory overhead, but may check shared paths multiple times
def shortestrouteindex(source, target, removednodes =[], i=0):
    i += 1
    target.jumpsfromtarget = i
    for link in target.clinks:
        if link not in removednodes:
            if (link.nextnode(target).jumpsfromtarget == 0 or link.nextnode(target).jumpsfromtarget > i):
                if target != source:
                    shortestrouteindex(source, link.nextnode(target), removednodes, i)
    return

def pathfindrecursion(source, target, i=0, removedlinks =[], removednodes =[], currentpath =[], c=None):
    paths = []
    if c == None:
        c = source.jumpsfromtarget
    if source == target:
        paths.append(currentpath)
        return paths
    else:
        #print(c)
        for link in source.clinks:
            if link not in removedlinks:
                if not (link.nextnode(source) in removednodes):
                    if (link.nextnode(source).jumpsfromtarget - (c-1)) - i <= 0:
                        r1 = removednodes.copy()
                        r1.append(link.nextnode(source))
                        p1 = currentpath.copy()
                        p1.append(link) #check if this either puts all the paths in a shared list or in individual local lists
                        paths += pathfindrecursion(link.nextnode(source), target, i -(link.nextnode(source).jumpsfromtarget-c), removedlinks, r1, p1, c-1)

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

            print("from "+str(p.source_node.identity) + " to "+ str(p.target_node.identity))
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

    test =pathfind(demands[0], grapher, 0)
    test2 =pathfind(demands[1], grapher, 5)

    printtestvalues(test, demands[0])
    printtestvalues(test2, demands[1])
    #printtestvalues(test2)


