class Graph:
    def __init__(self, nodes = [], links = []):
        self.nodes = nodes
        #self.links = links
        return

    def cleanup(self):
        for node in self.nodes:
            node.jumpsfromtarget = 0
        return

    def get_node(self, id):
        for node in self.nodes:
            if node.identity == id:
                return node
        else:
            return None
class Node:

    def __init__(self, i):
        self.identity = i
        self.clinks = []
        self.jumpsfromtarget = 0
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

    def nextnode(self, node):
        if node == self.node1:
            return self.node2
        if node == self.node2:
            return self.node1
        return


class Demand:
    def __init__(self, s, t, l):
        self.source = s
        self.target = t
        self.load = l
        return