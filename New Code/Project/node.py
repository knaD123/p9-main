class Node:

    def __init__(self, i):
        self.identity = i
        self.clinks = []
        return

    def newLink(self, n, i):
        l1 = Link(i, self, n)
        self.clinks.append(l1)
        n.clinks.append(l1)

class Link:
    def __init__(self, i, n1, n2):
        self.capacity = i
        self.usage = 0
        self.node1 = n1
        self.node2 = n2
        return

class Demand:
    def __init__(self, s, t, l):
        self.source = s
        self.target = t
        self.load = l
        return