import numpy as np
import matplotlib.pyplot as plt

T = np.array([[0.8, 0.2],
              [0.3, 0.7]])

B = np.array([[0.75, 0.23],
              [0.2, 0.8]])

O_1 = O_2 = O_4 = O_6 = np.array([[0.75, 0.0],
                                  [0.0, 0.2]])

O_3 = O_5 = np.array([[0.25, 0.0],
                      [0.0, 0.8]])
initial_probabilities = np.array([[0.5], [0.5]])

Evidence = [O_1, O_2, O_3, O_4, O_5, O_6]


def forward(t):

    if t == 0:
        return initial_probabilities

    x = Evidence[t-1].dot(T.transpose()).dot(forward(t-1))
    return x / np.sum(x)

# Filtering


def taskB():
    fig, ax = plt.subplots()
    x = []
    y = []
    y2 = []
    for i in range(1, 7):
        x.append(i)
        y.append(forward(i)[0][0])
        y2.append(forward(i)[1][0])
    ax.plot(x, y, label='P(xt | e1:t)')
    ax.plot(x, y2, label='P(~xt | e1:t)')
    ax.set_xlabel('t')  # Add an x-label to the axes.
    ax.set_ylabel('Probability')  # Add a y-label to the axes.
    ax.set_title("Simple Plot")  # Add a title to the axes.
    ax.legend()
    plt.show()

# Prediction


def taskC():
    fig, ax = plt.subplots()


taskB()
