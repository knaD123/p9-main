## This is a sample Python script.
import node as nd
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def valuation(demand, link):
    return (demand.load + link.usage)/link.capacity

def initializenetwork():
    ns = nd.Node(1)
    n1 = nd.Node(2)
    n2 = nd.Node(3)
    n3 = nd.Node(4)
    n4 = nd.Node(5)
    n5 = nd.Node(6)
    n6 = nd.Node(7)
    n7 = nd.Node(8)
    n8 = nd.Node(9)
    n9 = nd.Node(10)
    n10 = nd.Node(11)
    nt = nd.Node(12)

    ns.newLink(n1, 1000)
    ns.newLink(n2, 200)
    n1.newLink(n3, 1000)
    n1.newLink(n4, 1000)
    n2.newLink(n4, 500)
    n4.newLink(n5, 370)
    n4.newLink(n6, 1120)
    n2.newLink(n6, 120)
    n6.newLink(n7, 880)
    n7.newLink(n8, 560)
    n8.newLink(nt, 1000)
    n9.newLink(nt, 420)
    n10.newLink(nt, 720)
    n4.newLink(n8, 1000)
    n6.newLink(n9, 600)
    n8.newLink(n9, 660)
    ns.newLink(n10, 1000)

    d = []
    d.append(nd.Demand(ns, nt, 50))
    d.append(nd.Demand(n6, n2, 42))
    return ns, d

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
    for pop in bestpaths:
        for pip in pop:
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    network, demands = initializenetwork()

    test =pathfind(demands[0])
    test2 =pathfind(demands[1])
    print(test)
    for t in test:
        for p in t:
            print(p.capacity)
            #print(p.usage)
            #print(p.usage/p.capacity)
        print("/cut1")
    for t in test2:
        for p in t:
            #print(p.capacity)
            #print(p.usage)
            print(p.usage / p.capacity)
        print("/cut2")

