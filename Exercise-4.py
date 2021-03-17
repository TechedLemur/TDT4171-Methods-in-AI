import pandas as pd
import numpy as np
import math
import random
from graphviz import Digraph
import os
# Because windows...
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'


def B(q):  # Returns the entropy of a Boolean random variable that is true with probability q
    if (q in [0, 1]):
        return 0
    return -(q*math.log(q, 2)+(1-q)*math.log(1-q, 2))


def Remainder(A, p, n, exs, c=None):  # Returns expected remaining entropy.
    r = 0

    if A in continuous:
        left = exs[exs[A] <= c]  # Filter the examples for <= split
        p_k = left[GOAL_ATTRIBUTE].value_counts().reindex(
            data[GOAL_ATTRIBUTE].unique(), fill_value=0)[1]

        n_k = left[GOAL_ATTRIBUTE].value_counts().reindex(
            data[GOAL_ATTRIBUTE].unique(), fill_value=0)[0]

        if (p_k + n_k) > 0:
            r += (p_k + n_k)/(p + n) * B(p_k / (p_k + n_k))

        right = exs[exs[A] > c]  # Filter the examples for > split
        p_k = right[GOAL_ATTRIBUTE].value_counts(
        ).reindex(data[GOAL_ATTRIBUTE].unique(), fill_value=0)[1]

        n_k = right[GOAL_ATTRIBUTE].value_counts().reindex(
            data[GOAL_ATTRIBUTE].unique(), fill_value=0)[0]

        if (p_k + n_k) > 0:
            r += (p_k + n_k)/(p + n) * B(p_k / (p_k + n_k))
        return r

    else:  # Categorical variable
        grouped = exs.groupby(GOAL_ATTRIBUTE)[
            A].value_counts().unstack(fill_value=0).stack()

        for k in exs[A].unique():
            p_k = grouped[(1, k)]
            n_k = grouped[(0, k)]
            if (p_k + n_k) > 0:
                r += (p_k + n_k)/(p + n) * B(p_k / (p_k + n_k))
        return r


def Importance(A, exs):  # Implementation of the Gain function
    p = exs[GOAL_ATTRIBUTE].value_counts()[1]  # Survival count in examples
    n = exs[GOAL_ATTRIBUTE].value_counts()[0]  # Death count in exampes

    if A in continuous:
        split = score = 0

        candidates = exs[A].unique()
        candidates.sort()
        for c in candidates:  # Go through all possible split points, and select the best one.
            x = B(p/(p+n)) - Remainder(A, p, n, exs, c)
            if x >= score:
                split = c
                score = x
        return score, split

    return B(p/(p+n)) - Remainder(A, p, n, exs), None


class Node:  # Class used for building a tree
    def __init__(self, value):

        self.value = value
        self.children = {}

    def addChild(self, node, state, split=None):
        node.state = state  # the state that lead to this node
        self.children[state] = node  # The children of this node
        # The split value, if this is a continuous variable. Used in the Test method
        self.split = split

    def __str__(self):
        return f"Value: {self.value}, Children: {self.children}"


def viz(root):  # Use graphviz to plot the decission tree
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
        for child in Node.children.values():
            # add children to the queue, using the unique node ID as parent id
            q.append((id, child))
        counter = counter + 1
    dot.render("graph.gv", view=True)


# Returns the most frequent value for the goal attribute in the examples. If there are multiple values that are most frequent, choose randomly.
def PluralityValueNode(exs):
    mode = exs[GOAL_ATTRIBUTE].mode()  # Find the mode of the set
    return Node(random.choice(mode))


# Decission tree Learning. Creates a decission tree, and returns the root node.
def DTL(examples, attributes, parent_examples=None):
    if examples.empty:
        return PluralityValueNode(parent_examples)

    # All variables have the same classification
    if len(examples[GOAL_ATTRIBUTE].unique()) == 1:
        val = examples[GOAL_ATTRIBUTE].unique()[0]
        return Node(val)

    if not attributes:
        return PluralityValueNode(examples)

    A = ""
    val = 0
    split = None
    for a in attributes:  # Iterate through attributes, setting A to the attribute with highest information gain
        x, s = Importance(a, examples)
        if x >= val:
            A = a
            val = x
            split = s

    attributes.remove(A)

    root = Node(A)  # Create new tree

    if A in continuous:  # Use the split value to generate two subtrees and add them to the tree
        left = examples[examples[A] <= split]
        leftTree = DTL(left, attributes.copy(), examples)
        root.addChild(leftTree, f"<= {split}", split)
        right = examples[examples[A] > split]
        rightTree = DTL(right, attributes.copy(), examples)
        root.addChild(rightTree, f"> {split}", split)

    else:  # Categorical variable
        for v in data[A].unique():  # Make new subtree for each value the variable can take
            exs = examples[examples[A] == v]
            subtree = DTL(exs, attributes.copy(), examples)
            root.addChild(subtree, v)
    return root


def Test(tree, data):  # Validates a decission tree against a test set, returns the portion of correct predictions
    points = 0.0
    total = 0.0
    for index, row in data.iterrows():  # Iterate through all rows in the test set
        total += 1
        n = tree  # Start with the root node
        correct = row[GOAL_ATTRIBUTE]
        flag = True
        while(flag):  # Keep going until we reach a node with value 0 or 1

            if n.value in [0, 1]:
                if n.value == correct:
                    points += 1
                flag = False

            else:
                if n.value in continuous:  # Handle continuous variables
                    s = n.split
                    if row[n.value] > s:
                        n = n.children[f"> {s}"]
                    else:
                        n = n.children[f"<= {s}"]
                else:  # Categorical variable
                    state = row[n.value]
                    n = n.children[state]
    return points / total


def MissingValues(data):  # Function for finding columns with missing values
    columns = data.columns.values

    missing = []
    for c in columns:
        if (data[c].isnull().values.any()):
            missing.append(c)

    return missing


GOAL_ATTRIBUTE = "Survived"

continuous = ["SibSp", "Parch", "Fare"]

columns = []
columns.append("Survived")
columns.append("Pclass")  # Relevant?
# columns.append("Name")  # Not relevant
columns.append("Sex")  # Relevant
# columns.append("Age")  # Continuous - Relevant - missing
columns.append("SibSp")  # Continuous - Relevant?
columns.append("Parch")  # Continuous - Relevant?
# columns.append("Ticket") # Continuous - Not Relevant
columns.append("Fare")  # Continuous - Relevant?
# columns.append("Cabin") # Continuous - missing
# columns.append("Embarked")  # Not relevant


data = pd.read_csv("./train.csv")  # , usecols=columns)
testData = pd.read_csv("./test.csv")
attr = columns
attr.remove(GOAL_ATTRIBUTE)

tree = DTL(data, attr)

print(f"Accuracy is {Test(tree, testData)*100} %")
viz(tree)


# print(
#    f"The following columns are missing values in the training set: \n{MissingValues(data)}")
