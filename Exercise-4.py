import pandas as pd
import numpy as np
import math
import random
from graphviz import Digraph
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
# class DecitionTree(self):


def B(q):
    if (q in [0, 1]):
        return 0
    return -(q*math.log(q, 2)+(1-q)*math.log(1-q, 2))


def Remainder(A, p, n, exs):
    r = 0
    grouped = exs.groupby(GOAL_ATTRIBUTE)[
        A].value_counts().unstack(fill_value=0).stack()

    for k in exs[A].unique():
        p_k = grouped[(1, k)]
        n_k = grouped[(0, k)]
        r += (p_k + n_k)/(p + n) * B(p_k / (p_k + n_k))
    return r


def Importance(A, exs):
    p = exs[GOAL_ATTRIBUTE].value_counts()[1]
    n = exs[GOAL_ATTRIBUTE].value_counts()[0]
    return B(p/(p+n)) - Remainder(A, p, n, exs)


class Node:
    def __init__(self, value):

        self.children = []
        self.value = value

    def addChild(self, node, state):
        node.parent = self
        node.state = state
        self.children.append(node)

    def __str__(self):
        return f"Value: {self.value}, Children: {self.children}"


def treeTest():
    n1 = Node("Patrons")
    n2 = Node(0)
    n3 = Node(1)
    n4 = Node("Hungry")
    n5 = Node(0)
    n6 = Node("Type")
    n1.addChild(n2, "none")
    n1.addChild(n3, "some")
    n1.addChild(n4, "Full")
    n4.addChild(n5, "No")
    n4.addChild(n6, "Yes")

    viz(n1)


def viztest():
    dot = Digraph(comment='The Round Table')
    dot.node('A', 'King Arthur')
    dot.node('B', 'Sir Bedevere the Wise')
    dot.node('L', "0")
    dot.edges(['AB', 'AL'])
    dot.edge('B', 'L', constraint='false', label="Hello")
    dot.render('test-output/round-table.gv', view=True)


def viz(root):
    dot = Digraph(comment="Decission Tree")  # Initialize graph
    # Queue for holding the next nodes to plot, containing tuples on the form (parentID, node)
    q = [(None, root)]
    counter = 0  # Using a counter as a node ID, since graphviz requires unique node IDs
    while q:
        parent, Node = q.pop()
        id = str(counter)
        dot.node(id, str(Node.value))

        if parent:
            dot.edge(parent, id, label=str(Node.state))
        for child in Node.children:
            # add children to the queue, using the unique node ID as parent id
            q.append((id, child))
        counter = counter + 1
    dot.render("graph.gv", view=True)


# treeTest()


# data = pd.read_csv("./train.csv", usecols=columns)


# p = data["Survived"].value_counts()[1]
# p = data["WillWait"].value_counts()[1]
# n = data["Survived"].value_counts()[0]
# n = data["WillWait"].value_counts()[0]

# g = B(p/(p+n))

# count = data.groupby("WillWait")[
#    "Patrons"].value_counts().unstack(fill_value=0).stack()
# a = "Patrons"
#vals = data.Patrons.unique()


def test(b):
    print(data.groupby("WillWait")
          [b].value_counts().unstack(fill_value=0).stack())


#print(Importance("Patrons", data))

def PluralityValueNode(exs):
    mode = exs[GOAL_ATTRIBUTE].mode()
    return Node(random.choice(mode))


def DTL(examples, attributes, parent_examples=None):
    if examples.empty:
        return PluralityValueNode(parent_examples)

    if len(examples[GOAL_ATTRIBUTE].unique()) == 1:
        val = examples[GOAL_ATTRIBUTE].unique()[0]
        # print(val)
        return Node(val)

    if not attributes:
        return PluralityValueNode(examples)

    A = ""
    v = 0
    for a in attributes:
        x = Importance(a, examples)
#        print(a, x)
        if x > v:
            A = a
            v = x
 #   print(attributes, A)
    attributes.remove(A)

    root = Node(A)

    for v in data[A].unique():
        exs = examples[examples[A] == v]
        subtree = DTL(exs, attributes.copy(), examples)
        root.addChild(subtree, v)
    return root


def Test(tree, data):
    for t in data:
        return


GOAL_ATTRIBUTE = "Survived"

columns = []
columns.append("Survived")
columns.append("Pclass")
# columns.append("Name")
columns.append("Sex")  # Relevant
# columns.append("Age") # Continuous - Relevant
# columns.append("SibSp") # Continuous - Relevant?
# columns.append("Parch") # Contniuous - Relevant?
# columns.append("Ticket") # Contniuous
# columns.append("Fare") # Contniuous
# columns.append("Cabin") # Contniuous
# columns.append("Embarked")


data = pd.read_csv("./train.csv", usecols=columns)
testData = pd.read_csv("./test.csv", usecols=columns)
attr = columns
attr.remove(GOAL_ATTRIBUTE)

tree = DTL(data, attr)


viz(tree)
