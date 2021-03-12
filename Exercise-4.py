import pandas as pd
import numpy as np
import math
# class DecitionTree(self):


def B(q):
    if (q in [0, 1]):
        return 0
    return -(q*math.log(q, 2)+(1-q)*math.log(1-q, 2))


def Remainder(A):
    r = 0
    grouped = data.groupby("WillWait")[
        A].value_counts().unstack(fill_value=0).stack()

    for k in data[A].unique():
        p_k = grouped[(1, k)]
        n_k = grouped[(0, k)]
        r += (p_k + n_k)/(p + n) * B(p_k / (p_k + n_k))
    return r


def Gain(A):
    return g - Remainder(A)


class Node:
    def __init__(self, parent):

        self.parent = parent


columns = []
# columns.append("Survived")
# columns.append("Pclass")
# columns.append("Name")
# columns.append("Sex")  # Relevant
# columns.append("Age") # Continuous - Relevant
# columns.append("SibSp") # Continuous - Relevant?
# columns.append("Parch") # Contniuous - Relevant?
# columns.append("Ticket") # Contniuous
# columns.append("Fare") # Contniuous
# columns.append("Cabin") # Contniuous
# columns.append("Embarked")


# data = pd.read_csv("./train.csv", usecols=columns)
data = pd.read_csv("./book.csv")

# p = data["Survived"].value_counts()[1]
p = data["WillWait"].value_counts()[1]
# n = data["Survived"].value_counts()[0]
n = data["WillWait"].value_counts()[0]

g = B(p/(p+n))

# Remainder("Patrons")
count = data.groupby("WillWait")[
    "Patrons"].value_counts().unstack(fill_value=0).stack()
# for c in count.items():
#   print(c)
a = "Patrons"
vals = data.Patrons.unique()


def test(b):

    print(data.groupby("WillWait")
          [b].value_counts().unstack(fill_value=0).stack())


# for v in vals:
# test(a)
#    print(v)
#    print(count[(0, v)], "false")
#    print(count[(1, v)], "true")
# print(count[[0][0]])
# print(data)
# Remainder("Sex")
# print(g)

print(Gain("Patrons"))
