## This is a sample Python script.
import node as nd
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def valuation(demand, link):
    return (demand.load + link.usage)/link.capacity

def initializenetwork():
    n1 = nd.Node(1)
    n2 = nd.Node(2)
    n3 = nd.Node(3)
    n4 = nd.Node(4)
    n5 = nd.Node(5)
    n6 = nd.Node(6)
    n7 = nd.Node(7)
    n8 = nd.Node(8)
    n9 = nd.Node(9)
    n10 = nd.Node(10)
    n11 = nd.Node(11)
    n12 = nd.Node(12)

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
    return n1, d

def pathfind(d):

    bestpaths = []
    removed2 = []

    pathy = pathfindrecursion(d.source, d.target)

    while pathy != []:
        value = None
        bestpath = None

        for p in pathy:
            currentvalue = 0
            temp = []

            for plink in p:
                currentvalue += valuation(d, plink)
                temp.append(plink)

            if value == None or currentvalue < value:
                tempRemoved = temp
                bestpath =p
                value = currentvalue

        removed2 += tempRemoved
        bestpaths.append(bestpath.copy())
        pathy = pathfindrecursion(d.source, d.target, removed2)

    for pip in bestpaths[0]:
        pip.usage += d.load
    return bestpaths

def pathfindrecursion(n, t, f =[], p=[]):

    paths = []
    if n == t:
        paths.append(p)
        return paths
    else:
        for link in n.clinks:
            if not (link in f):
                f1 = f.copy()
                f1.append(link)
                p1 = p.copy()
                p1.append(link) #check if this either puts all the paths in a shared list or in individual local lists
                if n == link.node1:
                    paths += pathfindrecursion(link.node2, t, f1, p1)
                else:
                    paths += pathfindrecursion(link.node1, t, f1, p1)
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
    network, demands = initializenetwork()

    test =pathfind(demands[0])
    test2 =pathfind(demands[1])

    printtestvalues(test, demands[0])
    printtestvalues(test2, demands[1])
    #printtestvalues(test2)

