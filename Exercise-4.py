import pandas as pd
import numpy as np
import math
# class DecitionTree(self):


def B(q):
    return -(q*math.log(q, 2)+(1-q)*math.log(1-q, 2))


def Remainder(A):
    r = 0
    for k in data.groupby("Survived")[A]:
        print(k)


def Gain(A):
    return g - Remainder(A)


class Node:
    def __init__(self, parent):

        self.parent = parent


columns = []
columns.append("Survived")
# columns.append("Pclass")
columns.append("Name")
columns.append("Sex")  # Relevant
# columns.append("Age") # Continuous - Relevant
# columns.append("SibSp") # Continuous - Relevant?
# columns.append("Parch") # Contniuous - Relevant?
# columns.append("Ticket") # Contniuous
# columns.append("Fare") # Contniuous
# columns.append("Cabin") # Contniuous
# columns.append("Embarked")


data = pd.read_csv("./train.csv", usecols=columns)

p = data["Survived"].value_counts()[1]
n = data["Survived"].value_counts()[0]

g = B(p/(p+n))

# print(df.groupby("SibSp")["Name"].value_counts())

# Remainder("Sex")
print(g)
