import pandas as pd
import numpy as np
import math
# class DecitionTree(self):


def B(q):
    if (q in [0, 1]):
        return 0
    return -(q*math.log(q, 2)+(1-q)*math.log(1-q, 2))


def Remainder(A, p, n):
    r = 0
    grouped = data.groupby(GOAL_ATTRIBUTE)[
        A].value_counts().unstack(fill_value=0).stack()

    for k in data[A].unique():
        p_k = grouped[(1, k)]
        n_k = grouped[(0, k)]
        r += (p_k + n_k)/(p + n) * B(p_k / (p_k + n_k))
    return r


def Importance(A, exs):
    p = exs[GOAL_ATTRIBUTE].value_counts()[1]
    n = exs[GOAL_ATTRIBUTE].value_counts()[0]
    return B(p/(p+n)) - Remainder(A, p, n)


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

GOAL_ATTRIBUTE = "WillWait"

# p = data["Survived"].value_counts()[1]
# p = data["WillWait"].value_counts()[1]
# n = data["Survived"].value_counts()[0]
# n = data["WillWait"].value_counts()[0]

# g = B(p/(p+n))

# count = data.groupby("WillWait")[
#    "Patrons"].value_counts().unstack(fill_value=0).stack()
# a = "Patrons"
# vals = data.Patrons.unique()


def test(b):
    print(data.groupby("WillWait")
          [b].value_counts().unstack(fill_value=0).stack())


print(Importance("Patrons", data))
