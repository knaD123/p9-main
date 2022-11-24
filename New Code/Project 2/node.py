class Graph:
    def __init__(self, nodes = [], links = []):
        self.nodes = nodes
        self.links = links
        return

    def cleanup(self):
        for node in self.nodes:
            node.jumpsfromtarget = 0
        return

class Node:

    def __init__(self, i):
        self.identity = i
        self.clinks = []
        self.jumpsfromtarget = 0
        return

    def newLink(self, n, i1, i2 = None):
        l1 = Link(i1, self, n)
        if i2 != None:
            l2 = Link(i2, n, self)
        else:
            l2 = Link(i1, n, self)
        self.clinks.append(l1)
        n.clinks.append(l2)

class Link:
    def __init__(self, i, n1, n2):
        self.capacity = i
        self.usage = 0
        self.source_node = n1
        self.target_node = n2
        return

    def nextnode(self, node):
        if node == self.source_node:
            return self.target_node
        if node == self.target_node:
            return self.source_node
        return


class Demand:
    def __init__(self, s, t, l):
        self.source = s
        self.target = t
        self.load = l
        return