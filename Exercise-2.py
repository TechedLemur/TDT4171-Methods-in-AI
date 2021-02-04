import numpy as np
import matplotlib.pyplot as plt

# Transition Matrix
T = np.array([[0.8, 0.2],
              [0.3, 0.7]])


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


def backward(k, t):
    if k == t + 1:
        return np.array([[1],
                         [1]])
    return T.dot(Evidence[k-1]).dot(backward(k+1, t))


# Prediction
def Prediction(t, k):
    if k == t:
        return forward(t)
    x = T.transpose().dot(Prediction(t, k-1))
    return x / np.sum(x)


# Smoothing
def Smoothing(k, t):
    x = forward(k)*backward(k+1, t)
    return x / np.sum(x)


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
    ax.plot(x, y2, label='P(¬xt | e1:t)')
    ax.set_xlabel('t')  # Add an x-label to the axes.
    ax.set_ylabel('Probability')  # Add a y-label to the axes.
    ax.set_title("Task B, Filtering")  # Add a title to the axes.
    ax.legend()
    plt.show()


def taskC():
    fig, ax = plt.subplots()
    x = []
    y = []
    y2 = []
    for k in range(7, 31):
        x.append(k)
        y.append(Prediction(6, k)[0][0])
        y2.append(Prediction(6, k)[1][0])

    ax.plot(x, y, label='P(xt | e1:6)')
    ax.plot(x, y2, label='P(¬xt | e1:6)')

    ax.set_xlabel('t')  # Add an x-label to the axes.
    ax.set_ylabel('Probability')  # Add a y-label to the axes.
    ax.set_title("Task C, Prediction")  # Add a title to the axes.
    ax.legend()
    plt.show()


def taskD():
    fig, ax = plt.subplots()
    x = []
    y = []
    y2 = []
    for k in range(6):
        s = Smoothing(k, 6)
        print(s)
        x.append(k)
        y.append(s[0][0])
        y2.append(s[1][0])

    ax.plot(x, y, '-o', label='P(xt | e1:6)')
    ax.plot(x, y2, '-o', label='P(¬xt | e1:6)')
    for a, b in zip(x, y):
        ax.annotate(s="{:.4f}".format(b),
                    xy=(a, b),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha=('center'))
    for a, b in zip(x, y2):
        ax.annotate(s="{:.4f}".format(b),
                    xy=(a, b),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha=('center'))
    ax.set_xlabel('t')  # Add an x-label to the axes.
    ax.set_ylabel('Probability')  # Add a y-label to the axes.
    ax.set_title("Task D, Smoothing")  # Add a title to the axes.
    ax.legend()
    plt.show()


taskB()
taskC()
taskD()
